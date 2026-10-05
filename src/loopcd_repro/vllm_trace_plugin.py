"""Private general-plugin entry point; register the tracer in every worker."""
def register():
    import hashlib,json,os
    from pathlib import Path
    from vllm import ModelRegistry
    ModelRegistry.register_model('OuroForCausalLM','loopcd_repro.vllm_trace:TracedOuroForCausalLM')
    folder=Path(os.environ['LOOPCD_VLLM_TRACE'])
    (folder/('plugin_registered_'+str(os.getpid())+'.json')).write_text(json.dumps(dict(
        pid=os.getpid(),module=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))+'\n')
