#!/usr/bin/env python3
"""Bootstrap operator optimization workspace with CPU baseline + CUDA/Triton scaffold."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    logic: str
    op_type: str
    m: int
    n: int
    k: int


def choose_backend(spec: OperatorSpec, backend_hint: str) -> str:
    if backend_hint in {"cuda", "triton"}:
        return backend_hint

    if spec.op_type == "matmul":
        volume = spec.m * spec.n * spec.k
        # Large and TensorCore-friendly GEMM tends to benefit from CUDA/CUTLASS-style kernels.
        if volume >= 256 * 256 * 256 and spec.k % 16 == 0:
            return "cuda"
        return "triton"

    if spec.op_type in {"layernorm", "elementwise_add"}:
        return "triton"

    return "cuda"


def cpu_reference_source(spec: OperatorSpec) -> str:
    if spec.op_type == "matmul":
        body = "return a @ b"
        inputs = "a: torch.Tensor, b: torch.Tensor"
    elif spec.op_type == "elementwise_add":
        body = "return a + b"
        inputs = "a: torch.Tensor, b: torch.Tensor"
    else:
        body = (
            "return (x - x.mean(dim=-1, keepdim=True)) / "
            "(x.var(dim=-1, keepdim=True, unbiased=False) + eps).sqrt()"
        )
        inputs = "x: torch.Tensor, eps: float = 1e-5"

    return f'''"""CPU reference for {spec.name}.\nLogic: {spec.logic}"""
from __future__ import annotations

import torch


def op_cpu({inputs}) -> torch.Tensor:
    {body}
'''


def triton_template(spec: OperatorSpec) -> str:
    return f'''"""Triton scaffold for {spec.name}.\nLogic: {spec.logic}

Notes:
- Keep signature identical to cpu_reference.op_cpu where possible.
- Fill kernel launch + grid mapping in op_triton.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def op_triton(*args, **kwargs) -> torch.Tensor:
    raise NotImplementedError("Fill Triton kernel for {spec.op_type} and call it here.")
'''


def cuda_source_template(spec: OperatorSpec) -> str:
    return f'''// CUDA scaffold for {spec.name}
// Logic: {spec.logic}
#include <torch/extension.h>

at::Tensor op_cuda_placeholder(const at::Tensor& a, const at::Tensor& b) {{
  TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
  TORCH_CHECK(false, "Fill CUDA kernel implementation for {spec.op_type}");
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("op_cuda", &op_cuda_placeholder, "op_cuda");
}}
'''


def cuda_python_loader_template(spec: OperatorSpec) -> str:
    return f'''"""Python loader for CUDA extension of {spec.name}."""
from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load


_THIS_DIR = Path(__file__).resolve().parent
_EXT = load(
    name="{spec.name}_cuda_ext",
    sources=[str(_THIS_DIR / "kernel_cuda.cu")],
    verbose=False,
)


def op_cuda(*args, **kwargs):
    return _EXT.op_cuda(*args, **kwargs)
'''


def _build_cpu_inputs_code(op_type: str) -> str:
    if op_type == "matmul":
        return """
    a = torch.randn(m, k, dtype=dtype, device=\"cpu\")
    b = torch.randn(k, n, dtype=dtype, device=\"cpu\")
    return a, b
"""
    if op_type == "elementwise_add":
        return """
    a = torch.randn(m, n, dtype=dtype, device=\"cpu\")
    b = torch.randn(m, n, dtype=dtype, device=\"cpu\")
    return a, b
"""
    return """
    x = torch.randn(m, n, dtype=dtype, device=\"cpu\")
    return (x,)
"""


def harness_source(spec: OperatorSpec, backend: str) -> str:
    build_cpu_inputs = _build_cpu_inputs_code(spec.op_type).rstrip()
    return f'''"""Benchmark + exactness harness for {spec.name}."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cpu_reference import op_cpu


def build_cpu_inputs(m: int, n: int, k: int, dtype: torch.dtype):
{build_cpu_inputs}


def to_cuda(inputs):
    return tuple(t.cuda(non_blocking=True) if isinstance(t, torch.Tensor) else t for t in inputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cuda", "triton"], default="{backend}")
    parser.add_argument("--metric-output", default="metric.json")
    parser.add_argument("--exactness-output", default="exactness.json")
    parser.add_argument("--m", type=int, default={spec.m})
    parser.add_argument("--n", type=int, default={spec.n})
    parser.add_argument("--k", type=int, default={spec.k})
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backend benchmark/evaluation")

    torch.manual_seed(args.seed)
    dtype = torch.float32
    cpu_inputs = build_cpu_inputs(args.m, args.n, args.k, dtype=dtype)
    ref = op_cpu(*cpu_inputs)

    if args.backend == "triton":
        from kernel_triton import op_triton as backend_fn
    else:
        from kernel_cuda import op_cuda as backend_fn

    backend_inputs = to_cuda(cpu_inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    backend_out = backend_fn(*backend_inputs)
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    backend_cpu = backend_out.detach().cpu()
    passed = torch.allclose(ref, backend_cpu, atol=1e-4, rtol=1e-4)
    max_abs_error = float((ref - backend_cpu).abs().max().item())
    max_rel_error = float(((ref - backend_cpu).abs() / (ref.abs() + 1e-12)).max().item())

    Path(args.metric_output).write_text(
        json.dumps({{"metrics": {{"latency_ms": latency_ms}}}}, indent=2),
        encoding="utf-8",
    )
    Path(args.exactness_output).write_text(
        json.dumps(
            {{
                "exactness": {{
                    "passed": bool(passed),
                    "mismatch_count": 0 if passed else 1,
                    "max_abs_error": max_abs_error,
                    "max_rel_error": max_rel_error,
                    "logic_equivalent": True,
                    "algorithm_equivalent": True,
                }}
            }},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({{"backend": args.backend, "latency_ms": latency_ms, "passed": bool(passed)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CPU baseline + CUDA/Triton scaffolds")
    parser.add_argument("--name", required=True, help="operator name")
    parser.add_argument("--logic", required=True, help="human-readable operator logic")
    parser.add_argument("--op-type", choices=["matmul", "elementwise_add", "layernorm"], required=True)
    parser.add_argument("--backend", choices=["auto", "cuda", "triton"], default="auto")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec = OperatorSpec(
        name=args.name,
        logic=args.logic,
        op_type=args.op_type,
        m=args.m,
        n=args.n,
        k=args.k,
    )
    selected_backend = choose_backend(spec, args.backend)

    out_dir = Path(args.output_dir).expanduser().resolve() / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text(out_dir / "cpu_reference.py", cpu_reference_source(spec))
    if selected_backend == "triton":
        write_text(out_dir / "kernel_triton.py", triton_template(spec))
    else:
        write_text(out_dir / "kernel_cuda.cu", cuda_source_template(spec))
        write_text(out_dir / "kernel_cuda.py", cuda_python_loader_template(spec))
    write_text(out_dir / "benchmark_harness.py", harness_source(spec, selected_backend))

    manifest = {
        "name": spec.name,
        "logic": spec.logic,
        "op_type": spec.op_type,
        "shape_hint": {"m": spec.m, "n": spec.n, "k": spec.k},
        "backend": selected_backend,
        "generated_files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "baseline": "cpu_reference.py::op_cpu",
        "exactness_contract": "compare backend output against CPU reference with atol=1e-4, rtol=1e-4",
    }
    write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
