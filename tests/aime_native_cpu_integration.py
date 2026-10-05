"""Exercise AIME generation plumbing with random, narrow native models on CPU.

This loads private model implementation and tokenizers, never checkpoint weights.
It is an interface diagnostic, not an AIME score or the required real GPU gate.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import generate_aime as gen
    import gpu_smoke_aime as smoke
    from loopcd_repro import aime_protocol as ap
    from loopcd_repro.guidance import GuidanceConfig
    if torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic with CUDA_VISIBLE_DEVICES empty")
    torch.set_num_threads(2)
    protocol = ap.load_protocol(args.project / "data/aime/protocol-v1.json")
    question = ap.load_questions(args.project / "data/aime/prepared-v1", 2024)[0]
    records = []
    for name, layers in (("Ouro-1.4B-Thinking", 24), ("Ouro-2.6B-Thinking", 48)):
        model_path = args.project / "models" / name
        identity = ap.read_model_identity(model_path, protocol)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        # Keep the native loop count/cache slot count and tokenizer vocabulary.
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_attention_heads = config.num_key_value_heads = 2
        config.head_dim = 16
        config.num_hidden_layers = layers
        config.layer_types = ["full_attention"] * layers
        config._attn_implementation = "sdpa"
        native = get_class_from_dynamic_module("modeling_ouro.OuroForCausalLM", str(model_path), local_files_only=True)
        gen.set_seed(1729)
        model = native(config).to(dtype=torch.bfloat16).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        prompt = ap.build_prompt(tokenizer, question, protocol)
        modes = {}
        for arm in ap.ARMS:
            row_config = {"guidance": ap.guidance_dict(arm, identity, protocol),
                          "samples_per_problem": 16, "model_identity": identity,
                          "generation_config": gen.generation_config(protocol, 3).to_dict(),
                          "is_full_split": False, "debug": {"random_cpu_model": True}}
            # Only device telemetry is mocked; forward, sampling, native cache,
            # hooks, guidance, tokenizer and row validation all execute normally.
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.cuda, "synchronize", return_value=None))
                stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats", return_value=None))
                stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
                stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=0))
                row = gen.generation_row(model, tokenizer, prompt, 0, row_config, "cpu")
            ap.validate_sample(row, row_config, prompt)
            modes[arm] = {key: row[key] for key in ("generated_token_ids", "stop_reason", "execution_observation", "adapter_observation")}
        ordinary = smoke.sampled_fixture(model, tokenizer, prompt, protocol, "cpu", cap=3)
        parity = {}
        for name_zero, settings in (("baseline", GuidanceConfig(mode="baseline")),
                                    ("fixed_zero", GuidanceConfig(mode="fixed", omega=0)),
                                    ("adaptive_zero", GuidanceConfig(mode="adaptive", omega_cap=0))):
            observed = smoke.sampled_fixture(model, tokenizer, prompt, protocol, "cpu", settings, cap=3)
            assert observed["generated_token_ids"] == ordinary["generated_token_ids"]
            parity[name_zero] = True
        eos = smoke.sampled_fixture(model, tokenizer, prompt, protocol, "cpu", GuidanceConfig(mode="baseline"), force_token=2, cap=3)
        capped = smoke.sampled_fixture(model, tokenizer, prompt, protocol, "cpu", GuidanceConfig(mode="baseline"), force_token=10, cap=3)
        assert eos["generated_token_ids"] == [2] and eos["stop_reason"] == "eos"
        assert capped["generated_token_ids"] == [10] * 3 and capped["cap_hit"]
        records.append({"native_implementation": name, "layers": layers, "hidden_size": 32,
                        "vocab_size": config.vocab_size, "random_init_seed": 1729,
                        "prompt_tokens": len(prompt["prompt_token_ids"]), "rows": modes,
                        "native_sampled_parity": parity, "forced_eos_and_cap_passed": True})
        del model
    report = {"status": "PASS", "created_at": ap.stamp(), "gpu_execution": False,
              "checkpoint_weights_loaded": False, "cuda_telemetry_mocked": True,
              "scope": "Random narrow CPU models; real native generation plumbing only. Not AIME model results or real GPU capacity gate.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "production_generation_config_sha256": ap.hash_json(gen.generation_config(protocol).to_dict()),
              "source_sha256": {str(p.relative_to(ROOT)): ap.sha256_file(p)
                                for folder in ("src", "scripts", "configs")
                                for p in sorted((ROOT / folder).rglob("*"))
                                if p.is_file() and p.suffix in (".py", ".json", ".yaml")},
              "records": records}
    ap.atomic_json(args.output, report)
    print(json.dumps({"status": "PASS", "models": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
