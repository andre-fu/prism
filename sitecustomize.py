"""Auto-patch for vLLM + PyTorch 2.10 compatibility."""
try:
    import torch._inductor.standalone_compile as sc_module
    from torch._subclasses.fake_tensor import FakeTensorMode
    if not hasattr(sc_module, "FakeTensorMode"):
        sc_module.FakeTensorMode = FakeTensorMode
except Exception:
    pass
