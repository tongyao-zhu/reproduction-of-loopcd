"""Worker callbacks for a finite sampling-budget probe, not a task scorer."""
from dataclasses import asdict
import torch
from .guidance import GuidanceConfig
from .vllm_ouro import LoopCDOuroForCausalLM

class BudgetWorkerExtension:
    """Named RPC endpoints; only strings/numbers cross the engine boundary."""
    def loopcd_budget_configure(self, mode, cap):
        return configure(self.get_model(), mode, cap)

    def loopcd_budget_finish(self):
        return finish(self.get_model())

def configure(model, mode, cap):
    if type(model) is not LoopCDOuroForCausalLM or model.trace is not None:
        raise ValueError('Probe requires actual adapter, no forced-token/trace subclass')
    if getattr(model, '_budget_handles', []):
        raise RuntimeError('Previous generation counters still installed')
    model.settings = GuidanceConfig(mode=mode, omega_cap=cap)
    model._budget_counts = dict(forwards=0, loops=0, heads=0)
    def forward(*args): model._budget_counts['forwards'] += 1
    def loop(*args): model._budget_counts['loops'] += 1
    def head(*args): model._budget_counts['heads'] += 1
    model._budget_handles = [model.register_forward_hook(forward), model.model.norm.register_forward_hook(loop),
                             model.logits_processor.register_forward_hook(head)]
    torch.cuda.reset_peak_memory_stats()
    return dict(model_class=type(model).__module__+'.'+type(model).__name__, guidance=asdict(model.settings), total_loops=model.model.total_ut_steps)

def finish(model):
    counts = dict(model._budget_counts)
    for handle in model._budget_handles: handle.remove()
    model._budget_handles = []
    expected = 2 if model.settings.enabled else 1
    if counts['loops'] != 4*counts['forwards'] or counts['heads'] != expected*counts['forwards'] or counts['forwards'] == 0:
        raise ValueError('Unexpected recurrence/head count in actual generation: '+str(counts))
    return dict(counts=counts, max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved())
