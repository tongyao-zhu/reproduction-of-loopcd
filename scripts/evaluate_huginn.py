"""Zero-shot Huginn MC evaluation with native random initialization auditing.

Standard R=32 uses h6; standard R=16 uses h7. The explicit half-depth protocol
compares baseline R=32 with hidden R=16/h6/omega=0.5 (paper Table A8).
Every scored likelihood request gets one native initialization. No RNG state
is reset per request and no KV cache or one-token logit reuse is enabled.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MAX_LENGTH = 4096


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def resolve_protocol(config, mode, protocol="standard", loops=None, reference_loop=None, omega=None):
    if mode not in ("baseline", "hidden") or protocol not in ("standard", "half-depth"):
        raise ValueError("Unsupported mode or protocol")
    expected_loops = config["half_depth"]["guided_loops" if mode == "hidden" else "baseline_loops"] if protocol == "half-depth" else 32
    loops = expected_loops if loops is None else loops
    if isinstance(loops, bool) or loops not in (16, 32):
        raise ValueError("Only recurrent depths 16 and 32 are supported")
    if protocol == "half-depth" and loops != expected_loops:
        raise ValueError("Half-depth protocol requires baseline R32 or hidden R16")
    expected_reference = config["half_depth"]["reference_loop"] if protocol == "half-depth" else config["standard_reference_by_loops"][str(loops)]
    reference_loop = expected_reference if reference_loop is None else reference_loop
    if isinstance(reference_loop, bool) or not isinstance(reference_loop, int) or not 1 <= reference_loop <= loops:
        raise ValueError("reference_loop must identify a completed recurrent iteration")
    omega = config["fixed_omega"] if omega is None else omega
    if isinstance(omega, bool) or not isinstance(omega, (float, int)) or not math.isfinite(omega) or omega < 0:
        raise ValueError("omega must be finite and nonnegative")
    return {"mode": mode, "total_loops": loops, "reference_loop": reference_loop, "omega": float(omega),
            "protocol": protocol, "paper_guidance_parameters": mode == "baseline" or (reference_loop == expected_reference and omega == 0.5)}


def validate_requests(requests, max_length=MAX_LENGTH):
    """Validate before HFLM can perform its default left truncation."""
    for _, context, continuation in requests:
        if not context or not continuation:
            raise ValueError("Empty tokenized context or continuation")
        if len(continuation) > max_length or len(context) + len(continuation) > max_length + 1:
            raise ValueError(f"Request exceeds max_length={max_length}; refusing silent truncation")


class RequestAudit:
    """Record actual scoring order and random-initialization identity evidence."""
    def __init__(self, stream):
        self.stream = stream
        self.expected = []
        self.count = 0
        self.hashes = {key: hashlib.sha256() for key in (
            "request_plan_sha256", "request_order_sha256", "initialization_stream_sha256", "rng_boundary_stream_sha256")}

    def add_plan(self, requests, max_length=MAX_LENGTH):
        validate_requests(requests, max_length)
        # Matches HFLM._loglikelihood_tokens with batch_size=1/logits_cache=False.
        ordered = sorted(requests, key=lambda row: (-len(row[1] + row[2]), tuple(row[1] + row[2])))
        for _, context, continuation in ordered:
            self.expected.append(digest((context + continuation)[:-1]))
            self.hashes["request_plan_sha256"].update(canonical({"context": context, "continuation": continuation}) + b"\n")

    def record(self, tokens, before, after, initializations):
        token_hash = digest(tokens)
        if self.count >= len(self.expected) or token_hash != self.expected[self.count]:
            raise ValueError("Actual forward order differs from the recorded likelihood request plan")
        if len(initializations) != 1:
            raise ValueError("Each forward must execute exactly one native initialize_state")
        initialization = initializations[0]
        if before != initialization["rng_before"] or after != initialization["rng_after"]:
            raise ValueError("Random draws occurred outside native recurrent initialization")
        record = {"index": self.count, "input_ids_sha256": token_hash, "input_shape": [1, len(tokens)],
                  "rng_before": before, "rng_after": after, "initialization": initialization}
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.hashes["request_order_sha256"].update(canonical({"input_ids_sha256": token_hash, "shape": [1, len(tokens)]}) + b"\n")
        self.hashes["initialization_stream_sha256"].update(canonical(initialization) + b"\n")
        self.hashes["rng_boundary_stream_sha256"].update(canonical({"before": before, "after": after}) + b"\n")
        self.count += 1

    def summary(self, require_complete=False):
        complete = bool(self.count) and self.count == len(self.expected)
        if require_complete and not complete:
            raise ValueError("Likelihood request audit is incomplete")
        return {"request_count": self.count, "planned_requests": len(self.expected), "complete": complete,
                **{name: value.hexdigest() for name, value in self.hashes.items()},
                "pairing_requirement": "All counts and stream hashes must match across paired arms, including different recurrent depths."}


def make_audited_hflm(base_class, torch, audit, loops):
    """Construct a causal MC-only HFLM class without altering its scoring rules."""
    def rng_snapshot(device):
        result = {"cpu": hashlib.sha256(torch.random.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        if device.type == "cuda":
            result["model_device"] = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
        elif device.type != "cpu":
            raise ValueError("Random initialization auditing supports CPU and CUDA only")
        return result

    class AuditedHuginnHFLM(base_class):
        def _loglikelihood_tokens(self, requests, *positional, **kwargs):
            audit.add_plan(requests, self.max_length)
            return super()._loglikelihood_tokens(requests, *positional, **kwargs)

        def _model_call(self, inps, attn_mask=None, labels=None):
            if inps.ndim != 2 or inps.shape[0] != 1 or inps.shape[1] > MAX_LENGTH:
                raise ValueError("Huginn scoring requires one unpadded request within the context budget")
            if attn_mask is not None or labels is not None:
                raise ValueError("This runner only supports causal teacher-forced likelihood")
            device = inps.device
            before = rng_snapshot(device)
            initializations = []
            native_init = self.model.initialize_state
            existed = "initialize_state" in self.model.__dict__
            previous = self.model.__dict__.get("initialize_state")

            def observe_native_init(input_embeds, scale=1.0):
                if scale != 1.0:
                    raise ValueError("Preserve native initialization scale=1.0")
                init_before = rng_snapshot(device)
                state = native_init(input_embeds, scale=scale)
                initializations.append({
                    "rng_before": init_before, "rng_after": rng_snapshot(device),
                    "shape": list(state.shape), "dtype": str(state.dtype), "scale": scale,
                    "first_16_values_sha256": digest(state.detach().flatten()[:16].float().cpu().tolist()),
                })
                return state

            self.model.initialize_state = observe_native_init
            try:
                with torch.no_grad():
                    output = self.model(input_ids=inps, num_steps=loops, use_cache=False,
                                        past_key_values=None, input_states=None, init_scale=1.0)
            finally:
                if existed:
                    self.model.initialize_state = previous
                else:
                    self.model.__dict__.pop("initialize_state", None)
            after = rng_snapshot(device)
            audit.record(inps[0].detach().cpu().tolist(), before, after, initializations)
            return output.logits

        def _model_generate(self, *args, **kwargs):
            raise ValueError("This runner is restricted to multiple-choice likelihood scoring")

    return AuditedHuginnHFLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/huginn-0125"))
    parser.add_argument("--task", choices=("sciq", "piqa", "arc_challenge", "arc_easy", "hellaswag", "winogrande", "mmlu"), required=True)
    parser.add_argument("--mode", choices=("baseline", "hidden"), required=True)
    parser.add_argument("--protocol", choices=("standard", "half-depth"), default="standard")
    parser.add_argument("--loops", type=int, choices=(16, 32))
    parser.add_argument("--reference-loop", type=int)
    parser.add_argument("--omega", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Subset smoke test only; omit for full split")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-cache", type=Path, default=ROOT / ".cache/datasets")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/huginn_mc.json").read_text())
    try:
        protocol = resolve_protocol(config, args.mode, args.protocol, args.loops, args.reference_loop, args.omega)
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be a positive document count")
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "task": args.task, "limit": args.limit, "is_full_split": args.limit is None,
                "guidance": {key: protocol[key] for key in ("mode", "total_loops", "reference_loop", "omega")},
                "protocol": protocol["protocol"], "paper_guidance_parameters": protocol["paper_guidance_parameters"],
                "paper_config": config, "batch_size": 1, "seed": args.seed, "num_fewshot": 0,
                "chat_template": False, "max_length": MAX_LENGTH, "use_cache": False, "logits_cache": False,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def save_manifest():
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    save_manifest()
    started = time.monotonic()
    audit = None
    try:
        import torch
        import lm_eval
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import handle_non_serializable
        from check_mc_data import build_task_spec, build_local_task_spec, iter_leaf_tasks, validate_dataset_pin
        from loopcd_repro.huginn import load_huginn, HuginnHiddenConfig, HuginnHiddenGuidance
        from loopcd_repro.runtime import provenance, sha256
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        model, tokenizer = load_huginn(args.model, args.device)
        model.config.use_cache = False
        if getattr(model.config, "test_time_noise", 0) != 0:
            raise ValueError("Additional recurrent test-time noise is outside this protocol")
        std = float(model.config.init_values["std"])
        if std <= 0:
            raise ValueError("The checkpoint must retain nonzero native random initialization")
        manifest["native_initialization"] = {
            "method": "Native initialize_state: randn_like, then trunc_normal_ with native std and bounds +/-3*std, then native embedding scale",
            "std": std, "embedding_scale": float(model.emb_scale), "init_scale": 1.0,
            "test_time_noise": float(model.config.test_time_noise), "input_states": None,
            "per_request_reseed": False, "one_initialization_per_scored_request": True,
        }
        manifest["provenance"] = provenance(args.model, model)
        if manifest["provenance"]["model"]["revision"] != config["revision"]:
            raise ValueError("Huginn checkpoint does not match the paper-run pin")
        manifest["provenance"]["precision"] = "BF16 model; FP32 hidden blend cast back to BF16 before native coda; FP32 log_softmax"
        harness_root = Path(lm_eval.__file__).resolve().parent
        manifest["harness_files"] = {str(path.relative_to(harness_root)): sha256(path) for path in (
            harness_root / "models/huggingface.py", harness_root / "evaluator.py", harness_root / "api/task.py", harness_root / "utils.py")}
        registry = json.loads((ROOT / "configs/mc_datasets.json").read_text())
        entry = registry["tasks"][args.task]
        manager = TaskManager()
        spec = (build_local_task_spec(args.task, registry, manager, args.dataset_cache)
                if entry.get("group") else build_task_spec(args.task, registry))
        tree = manager.load_task_or_group([copy.deepcopy(spec)])
        objects = []
        manifest["datasets"] = {}
        for name, task in iter_leaf_tasks(tree):
            validate_dataset_pin(task, entry)
            task.EVAL_HARNESS_NAME = name
            objects.append(task)
            manifest["datasets"][name] = {
                "revision": entry["revision"], "raw_source": (task.config.metadata or {}).get("loopcd_dataset"),
                "splits": {split: {"rows": len(data), "fingerprint": data._fingerprint,
                                   "cache_files": [{"name": Path(item["filename"]).name, "sha256": sha256(item["filename"])} for item in data.cache_files]}
                           for split, data in task.dataset.items()},
            }
        if entry.get("expected_leaf_tasks") and len(objects) != entry["expected_leaf_tasks"]:
            raise ValueError("Missing MMLU subjects")
        settings = HuginnHiddenConfig(**manifest["guidance"])
        with (args.output / "request_trace.jsonl").open("x") as trace:
            audit = RequestAudit(trace)
            model_class = make_audited_hflm(HFLM, torch, audit, settings.total_loops)
            lm = model_class(pretrained=model, tokenizer=tokenizer, batch_size=1, max_length=MAX_LENGTH,
                             softmax_dtype=torch.float32, trust_remote_code=True, logits_cache=False, truncation=False)
            save_manifest()
            with HuginnHiddenGuidance(model, settings) as guidance:
                result = simple_evaluate(
                    model=lm, tasks=[copy.deepcopy(spec)] if entry.get("group") else objects, task_manager=manager,
                    num_fewshot=0, limit=args.limit, batch_size=1, bootstrap_iters=0, log_samples=True,
                    random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
                    fewshot_random_seed=args.seed, apply_chat_template=False,
                )
                manifest["last_guidance_observation"] = guidance.last_observation
            manifest["request_audit"] = audit.summary(require_complete=True)
        samples = result.pop("samples", {})
        for name, rows in samples.items():
            with (args.output / f"samples_{name}.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, default=handle_non_serializable) + "\n")
        (args.output / "results.json").write_text(json.dumps(result, indent=2, default=handle_non_serializable) + "\n")
        manifest.update(status="completed", results=result["results"], samples={name: len(rows) for name, rows in samples.items()})
        print(json.dumps({"output": str(args.output), "guidance": asdict(settings), "results": result["results"]}, default=str))
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if audit is not None:
            manifest["request_audit"] = audit.summary()
        manifest["elapsed_seconds"] = time.monotonic() - started
        save_manifest()


if __name__ == "__main__":
    main()
