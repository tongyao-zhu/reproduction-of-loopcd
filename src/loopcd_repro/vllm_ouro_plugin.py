"""Private per-run vLLM general-plugin; never installs into shared packages."""
def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model('OuroForCausalLM', 'loopcd_repro.vllm_ouro:LoopCDOuroForCausalLM')
