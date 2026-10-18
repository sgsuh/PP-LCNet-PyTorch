"""Reproduce the paper's cumulative ablation (Table 3, 0.5x) on Imagenette.

Each step adds one technique on top of the previous one:
    BaseNet -> +H-Swish -> +5x5 -> +SE -> +1x1 conv -> +dropout

Runs are trained from scratch with tools/train.py, one JSON per (step, seed)
under --out-dir; existing results are skipped, so the script can be resumed.
CPU latency (batch 1) is measured once per step.

Example:
    python tools/ablation.py --root ~/data/imagenette2 --seeds 0 1 2
"""

import argparse
import json
import os
import statistics

from benchmark_latency import measure_latency
from train import build_model, get_parser, train

# (name, train.py flags, paper top-1 / latency(ms) at 0.5x on ImageNet)
STEPS = [
    ("BaseNet", ["--act", "relu", "--no-5x5", "--no-se", "--no-last-conv", "--dropout", "0"],
     (55.58, 1.61)),
    ("+H-Swish", ["--no-5x5", "--no-se", "--no-last-conv", "--dropout", "0"], (58.18, 1.66)),
    ("+5x5", ["--no-se", "--no-last-conv", "--dropout", "0"], (59.09, 1.70)),
    ("+SE", ["--no-last-conv", "--dropout", "0"], (59.91, 1.85)),
    ("+1x1 conv", ["--dropout", "0"], (62.50, 2.05)),
    ("+dropout", [], (63.14, 2.05)),
]


def run_name(step: str, seed: int) -> str:
    return step.strip("+").lower().replace(" ", "_").replace("-", "") + f"_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=10, help="CPU threads for latency")
    parser.add_argument("--out-dir", default="runs/ablation")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="additional flags passed to train.py")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base = ["--root", args.root, "--epochs", str(args.epochs), "--scale", str(args.scale)]
    for seed in args.seeds:
        for step, flags, _ in STEPS:
            out = os.path.join(args.out_dir, run_name(step, seed) + ".json")
            if os.path.exists(out):
                continue
            print(f"=== {step} seed={seed}", flush=True)
            train(get_parser().parse_args(base + flags + args.extra + ["--seed", str(seed), "--out", out]))

    rows = []
    for step, flags, (paper_top1, paper_ms) in STEPS:
        results = []
        for seed in args.seeds:
            with open(os.path.join(args.out_dir, run_name(step, seed) + ".json")) as f:
                results.append(json.load(f))
        top1 = [r["final_top1"] for r in results]
        model = build_model(get_parser().parse_args(base + flags), num_classes=10)
        rows.append({
            "step": step,
            "top1_mean": statistics.mean(top1),
            "top1_std": statistics.stdev(top1) if len(top1) > 1 else 0.0,
            "top1_runs": top1,
            "params_m": results[0]["params"] / 1e6,
            "latency_ms": measure_latency(model, threads=args.threads),
            "paper_top1": paper_top1,
            "paper_latency_ms": paper_ms,
        })

    lines = ["| Step | Imagenette top-1 | Runs | Params (M) | CPU latency (ms) | Paper top-1 | Paper latency (ms) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        runs = ", ".join(f"{v:.2f}" for v in r["top1_runs"])
        lines.append(f"| {r['step']} | {r['top1_mean']:.2f} ± {r['top1_std']:.2f} | {runs} | "
                     f"{r['params_m']:.2f} | {r['latency_ms']:.2f} | {r['paper_top1']:.2f} | "
                     f"{r['paper_latency_ms']:.2f} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
