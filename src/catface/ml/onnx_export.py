"""Export a trained torch model to ONNX, simplify with onnxsim, and check
that the simplified graph's predictions match the source PyTorch model's.
Model-agnostic (any nn.Module + a matching sample input), so more than one
experiment script can reuse it.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from torch import nn


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_and_verify_onnx(
    model: nn.Module,
    sample_input: np.ndarray,
    out_dir: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int = 17,
    tol: float = 1e-4,
) -> dict:
    """Export `model` (already trained) to `out_dir/model.onnx`, simplify it
    to `out_dir/model.simplified.onnx`, then run both the PyTorch model and
    the simplified ONNX graph on `sample_input` and compare. Static batch=1
    export, so the ONNX side is run row by row. Writes and returns the
    result dict as `out_dir/onnx_export.json`. Not a generalization test --
    a numerical parity check between two graphs on real data."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "model.onnx"
    simplified_path = out_dir / "model.simplified.onnx"

    model.eval()
    dummy = torch.zeros(1, *sample_input.shape[1:], dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(raw_path), opset_version=opset,
        input_names=input_names, output_names=output_names, dynamo=False,
    )
    onnx_model = onnx.load(str(raw_path))
    simplified, sim_ok = onnxsim.simplify(onnx_model)
    onnx.save(simplified, str(simplified_path))

    with torch.no_grad():
        torch_out = model(torch.tensor(sample_input, dtype=torch.float32)).numpy()

    session = ort.InferenceSession(str(simplified_path))
    onnx_out = np.stack([
        session.run(output_names, {input_names[0]: row[None].astype(np.float32)})[0][0]
        for row in sample_input
    ])

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    match_fraction = float((torch_out.argmax(1) == onnx_out.argmax(1)).mean())

    result = {
        "success": bool(sim_ok) and max_abs_diff < tol,
        "onnxsim_success": bool(sim_ok),
        "tolerance": tol,
        "max_abs_logit_diff": max_abs_diff,
        "argmax_match_fraction": match_fraction,
        "n_samples": len(sample_input),
        "onnx_nodes_before_simplify": len(onnx_model.graph.node),
        "onnx_nodes_after_simplify": len(simplified.graph.node),
        "onnx_sha256": _sha256(raw_path),
        "onnx_simplified_sha256": _sha256(simplified_path),
        "opset": opset,
        "input_shape": [1, *sample_input.shape[1:]],
    }
    (out_dir / "onnx_export.json").write_text(json.dumps(result, indent=2))
    return result
