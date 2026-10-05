"""Inspect native module shapes on meta tensors; count dense 512-token prefill FLOPs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def summarize(linears, config, sequence_length=512):
    components = defaultdict(int)
    for name, dimensions in linears.items():
        if name == "lm_head":
            group = "head"
        elif name.startswith("transformer.adapter"):
            group = "adapter_per_loop"
        elif name.startswith("transformer.core_block."):
            group = "core_per_loop"
        elif name.startswith("transformer.prelude."):
            group = "prelude"
        elif name.startswith("transformer.coda."):
            group = "coda"
        else:
            raise ValueError(f"Unaccounted linear module: {name}")
        components[group] += 2 * sequence_length * dimensions[0] * dimensions[1]
    attention_block = 4 * sequence_length**2 * config["n_heads"] * config["head_dim"]
    for group, layers in (("prelude", config["n_layers_in_prelude"]),
                          ("core_per_loop", config["n_layers_in_recurrent_block"]),
                          ("coda", config["n_layers_in_coda"])):
        components[group] += attention_block * layers
    if components["adapter_per_loop"] != 4 * sequence_length * config["n_embd"]**2:
        raise ValueError("Unexpected recurrent injection projection")
    rows = {}
    for loops in (32, 16):
        without_adapter = components["prelude"] + loops * components["core_per_loop"] + components["coda"] + components["head"]
        full = without_adapter + loops * components["adapter_per_loop"]
        blend = 3 * sequence_length * config["n_embd"]
        rows[str(loops)] = {"paper_style_without_recurrent_adapter_flops": without_adapter,
                           "all_linear_and_attention_matmul_flops": full,
                           "hidden_vector_blend_flops": blend,
                           "hidden_multiplier_including_vector_blend": (full + blend) / full,
                           "logits_extra_coda_and_head_flops": components["coda"] + components["head"]}
    return {"components_flops": dict(components), "depths": rows,
            "half_depth_hidden_fraction_including_adapter_and_blend":
                (rows["16"]["all_linear_and_attention_matmul_flops"] + rows["16"]["hidden_vector_blend_flops"]) / rows["32"]["all_linear_and_attention_matmul_flops"],
            "half_depth_hidden_fraction_without_adapter":
                rows["16"]["paper_style_without_recurrent_adapter_flops"] / rows["32"]["paper_style_without_recurrent_adapter_flops"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from loopcd_repro.runtime import provenance, sha256
    config_data = json.loads((args.model / "config.json").read_text())
    preparation = json.loads((args.model / "model_provenance.json").read_text())
    if preparation["repo_id"] != "tomg-group-umd/huginn-0125" or sha256(args.model / "raven_modeling_minimal.py") != preparation["model_code_sha256"]:
        raise ValueError("Expected the prepared pinned Huginn source")
    config = AutoConfig.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    linears = {name: [module.in_features, module.out_features] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    result = summarize(linears, config_data)
    result.update(sequence_length=512, batch_size=1, linear_modules=linears, model_provenance=preparation,
                  config_sha256=sha256(args.model / "config.json"),
                  assumptions=["Two FLOPs per multiply-accumulate; all 512 positions projected by the head",
                               "Dense QK and attention-value matmuls counted at S by S, including causal-masked entries",
                               "Embedding lookup, normalization, rotary, nonlinearities, initialization and memory movement excluded",
                               "Hidden subtraction/multiply/add counted separately at 3*S*hidden_size",
                               "Meta instantiation inspects real module dimensions without allocating weights or using a GPU"],
                  paper_comparison={"table": "A7", "reported_tflops": {"32": 54.526, "16": 28.261},
                                    "finding": "The published rounded totals match dense transformer blocks plus head with the recurrent adapter omitted. The pinned native implementation also calls its 2d-to-d adapter once per loop, so both accounting conventions are reported."},
                  interpretation="Analytic prefill operation counts only. Accuracy recovery requires full paired MC results; these numbers are not measured speedups.")
    for loops, expected in result["paper_comparison"]["reported_tflops"].items():
        if round(result["depths"][loops]["paper_style_without_recurrent_adapter_flops"] / 1e12, 3) != expected:
            raise ValueError("The stated paper-accounting correspondence no longer holds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"depths": result["depths"], "half_depth_fraction": result["half_depth_hidden_fraction_including_adapter_and_blend"]}))


if __name__ == "__main__":
    main()
