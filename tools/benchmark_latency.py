"""Measure CPU inference latency (batch 1), as in the paper's setup.

The paper used an Intel Xeon Gold 6148 with MKLDNN and 10 threads; numbers
from other CPUs are only comparable relative to each other.

Example:
    python tools/benchmark_latency.py --scale 0.5 --threads 10
"""

import argparse
import statistics
import time

import torch

from pplcnet import PPLCNet


def measure_latency(model: torch.nn.Module, image_size: int = 224, threads: int = 10,
                    warmup: int = 50, runs: int = 300) -> float:
    """Median latency in milliseconds for one image on CPU."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        model = model.cpu().eval()
        x = torch.randn(1, 3, image_size, image_size)
        times = []
        with torch.inference_mode():
            for i in range(warmup + runs):
                start = time.perf_counter()
                model(x)
                if i >= warmup:
                    times.append((time.perf_counter() - start) * 1000)
        return statistics.median(times)
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    ms = measure_latency(PPLCNet(scale=args.scale), args.image_size, args.threads)
    print(f"PPLCNet scale={args.scale}: {ms:.2f} ms (batch 1, {args.threads} threads)")


if __name__ == "__main__":
    main()
