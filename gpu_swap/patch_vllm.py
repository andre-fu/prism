"""
Monkey-patch to fix vLLM 0.18 + PyTorch 2.10 compilation compatibility.

PyTorch 2.10 doesn't export FakeTensorMode in torch._inductor.standalone_compile.
vLLM tries to mock.patch it there, which fails. This patch adds the missing symbol.

Import this module before starting vLLM:
    import gpu_swap.patch_vllm  # noqa
"""

import torch._inductor.standalone_compile as sc_module
from torch._subclasses.fake_tensor import FakeTensorMode

if not hasattr(sc_module, "FakeTensorMode"):
    sc_module.FakeTensorMode = FakeTensorMode
    print("[patch_vllm] Patched FakeTensorMode into torch._inductor.standalone_compile")
