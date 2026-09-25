"""Build an interactive HTML report for Assignment 1.3.

Part A re-runs the NumPy Conv2D against ``torch.nn.functional.conv2d`` and records
the numerical error. Part B reads the saved ResNet50 run (``assignment_1_3B_outputs``)
and turns its history, test scores and wall-clock times into charts — no retraining,
the two 94 MB checkpoints are left untouched.

Usage:
    python visualize_1_3.py
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from report_kit import write as write_report  # noqa: E402

from conv2d_numpy import Conv2D  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = (
    "learning_curve_accuracy",
    "learning_curve_loss",
    "test_accuracy_comparison",
    "training_time_comparison",
)


# --------------------------------------------------------------------------- #
# part A — NumPy Conv2D vs PyTorch
# --------------------------------------------------------------------------- #
def collect_part_a() -> dict:
    rng = np.random.default_rng(42)
    cases = []

    def compare(label, shape, weight_shape, stride, padding, bias):
        x_np = rng.normal(size=shape).astype(np.float64)
        conv = Conv2D(
            shape[1],
            weight_shape[0],
            kernel_size=weight_shape[2:],
            stride=stride,
            padding=padding,
            bias=bias,
        )
        out_np = conv.forward(x_np)
        dout = rng.normal(size=out_np.shape).astype(np.float64)
        dx_np, dw_np, db_np = conv.backward(dout)

        # Conv2D initialises W/b in float32; compare in float64 so the tolerance is
        # testing the implementation rather than single-precision rounding.
        conv.W = conv.W.astype(np.float64)
        if bias:
            conv.b = conv.b.astype(np.float64)
        out_np = conv.forward(x_np)
        dx_np, dw_np, db_np = conv.backward(dout)

        x_t = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
        w_t = torch.tensor(conv.W, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(conv.b, dtype=torch.float64, requires_grad=True) if bias else None
        out_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding)
        out_t.backward(torch.tensor(dout))

        errors = {
            "forward": float(np.abs(out_np - out_t.detach().numpy()).max()),
            "dX": float(np.abs(dx_np - x_t.grad.numpy()).max()),
            "dW": float(np.abs(dw_np - w_t.grad.numpy()).max()),
        }
        if bias:
            errors["db"] = float(np.abs(db_np - b_t.grad.numpy()).max())
        cases.append(
            {
                "label": label,
                "input": list(shape),
                "weight": list(weight_shape),
                "stride": list(stride),
                "padding": list(padding),
                "bias": bias,
                "output": list(out_np.shape),
                "errors": errors,
                "passed": max(errors.values()) < 1e-9,
            }
        )

    compare("square kernel, stride 2", (2, 3, 7, 8), (4, 3, 3, 3), (2, 2), (1, 1), True)
    compare("asymmetric kernel, no bias", (1, 2, 5, 6), (3, 2, 2, 3), (2, 1), (1, 2), False)
    compare("1x1 kernel", (2, 4, 6, 6), (5, 4, 1, 1), (1, 1), (0, 0), True)
    compare("stride 1, padding 2", (1, 3, 9, 9), (2, 3, 5, 5), (1, 1), (2, 2), True)
    return {"cases": cases, "tolerance": 1e-9}


# --------------------------------------------------------------------------- #
# part B — ResNet50 scratch vs transfer
# --------------------------------------------------------------------------- #
def collect_part_b(out_dir: Path) -> dict:
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    runs = {}
    for name in ("scratch", "transfer"):
        entry = metrics[name]
        history = entry["history"]
        stages = []
        for row in history:
            if not stages or stages[-1]["stage"] != row["stage"]:
                stages.append({"stage": row["stage"], "first": row["epoch"], "last": row["epoch"]})
            else:
                stages[-1]["last"] = row["epoch"]
        runs[name] = {
            "history": history,
            "stages": stages,
            "epochs": len(history),
            "best_val_acc": entry["best_val_acc"],
            "best_val_epoch": entry["best_val_epoch"],
            "test_acc": entry["test_acc"],
            "test_loss": entry["test_loss"],
            "training_time_sec": entry["training_time_sec"],
            "sec_per_epoch": entry["training_time_sec"] / len(history),
        }

    plots = {}
    for name in PLOTS:
        path = out_dir / f"{name}.png"
        if path.exists():
            plots[name] = "data:image/png;base64," + base64.b64encode(
                path.read_bytes()
            ).decode("ascii")

    checkpoints = {
        path.name: path.stat().st_size
        for path in sorted(out_dir.glob("*.pt"))
    }
    return {
        "classes": metrics["classes"],
        "runs": runs,
        "plots": plots,
        "checkpoints": checkpoints,
        "source": str(out_dir.relative_to(HERE)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="assignment_1_3B_outputs")
    parser.add_argument("--out", default="outputs_viz/report_1_3.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = HERE / args.outputs
    if not (out_dir / "metrics.json").exists():
        raise FileNotFoundError(
            f"No metrics.json under {out_dir}. Unzip results.zip into this folder first."
        )

    print("part A: NumPy Conv2D vs torch.nn.functional.conv2d ...")
    part_a = collect_part_a()
    for case in part_a["cases"]:
        worst = max(case["errors"].values())
        print(f"  {case['label']:<28} max abs error {worst:.2e} "
              f"{'PASS' if case['passed'] else 'FAIL'}")

    print("part B: reading the saved ResNet50 run ...")
    part_b = collect_part_b(out_dir)
    for name, run in part_b["runs"].items():
        print(f"  {name:<9} test {run['test_acc']*100:.2f}% | "
              f"{run['epochs']} epochs | {run['training_time_sec']/60:.1f} min")

    out_path, artifact_path = write_report(
        HERE / args.out,
        HERE / "report_1_3_template.html",
        {"partA": part_a, "partB": part_b},
    )
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.0f} KB) - open this one")
    print(f"     {artifact_path.name} - body-only copy for publishing")


if __name__ == "__main__":
    main()
