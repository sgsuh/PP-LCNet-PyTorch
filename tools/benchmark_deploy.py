"""Latency and accuracy of PP-LCNet vs. torchvision mobile models across runtimes.

Every model is exported to ONNX once (cached under --onnx-dir) and timed at
batch 1 with:
    torch-cpu  PyTorch eager, CPU
    ort-cpu    ONNX Runtime CPU (the closest to the paper's MKLDNN setup)
    ort-cuda   ONNX Runtime CUDA, FP32 (TF32 off)
    trt-fp32   ONNX Runtime TensorRT, FP32
    trt-fp16   ONNX Runtime TensorRT, FP16
plus trt-fp16-b64, TensorRT FP16 throughput (images/s) at batch 64, where GPU
efficiency of depthwise convs matters more than kernel-launch overhead.
GPU inputs and outputs stay on the device (IOBinding), so timings exclude PCIe copies.

With --imagenette, ImageNet top-1 (1000-way) on the Imagenette val split is
measured with ort-cpu (FP32 reference) and trt-fp16.

Example:
    python tools/benchmark_deploy.py --weights-dir weights/ --imagenette ~/data/imagenette2
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
import torchvision

import pplcnet
from pplcnet.convert import load_pretrained
from pplcnet.data import ImagenetteEval
from pplcnet.export import export_onnx

PPLCNET_SCALES = ["x0_25", "x0_35", "x0_5", "x0_75", "x1_0", "x1_5", "x2_0", "x2_5"]
TORCHVISION_MODELS = {
    "MobileNetV2-1.0": ("mobilenet_v2", "IMAGENET1K_V1"),
    "MobileNetV3-small": ("mobilenet_v3_small", "IMAGENET1K_V1"),
    "MobileNetV3-large": ("mobilenet_v3_large", "IMAGENET1K_V1"),
    "ShuffleNetV2-1.0": ("shufflenet_v2_x1_0", "IMAGENET1K_V1"),
    "ShuffleNetV2-2.0": ("shufflenet_v2_x2_0", "IMAGENET1K_V1"),
}
BACKENDS = ["torch-cpu", "ort-cpu", "ort-cuda", "trt-fp32", "trt-fp16", "trt-fp16-b64"]
EVAL_BATCH = 100
THROUGHPUT_BATCH = 64


def build_models(weights_dir):
    models = {}
    for s in PPLCNET_SCALES:
        model = getattr(pplcnet, f"PPLCNet_{s}")()
        path = os.path.join(weights_dir, f"PPLCNet_{s}_pretrained.pdparams") if weights_dir else None
        if path and os.path.exists(path):
            load_pretrained(model, path)
        elif weights_dir:
            raise FileNotFoundError(path)
        models[f"PP-LCNet-{s[1:].replace('_', '.')}x"] = model
    for name, (arch, weights) in TORCHVISION_MODELS.items():
        models[name] = getattr(torchvision.models, arch)(weights=weights)
    return {k: m.eval() for k, m in models.items()}


def ort_session(onnx_path, backend, threads, trt_cache):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if backend == "ort-cpu":
        opts.intra_op_num_threads = threads
        return ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    cuda = ("CUDAExecutionProvider", {"use_tf32": "0"})
    if backend == "ort-cuda":
        return ort.InferenceSession(onnx_path, opts, providers=[cuda])
    # TensorRT tunes kernels for the "opt" shape, so the throughput backend gets its own
    # profile and engine cache instead of reusing the batch-1 engine.
    opt = THROUGHPUT_BATCH if backend == "trt-fp16-b64" else 1
    cache = os.path.join(trt_cache, backend)
    os.makedirs(cache, exist_ok=True)
    trt = ("TensorrtExecutionProvider", {
        "trt_fp16_enable": str(backend.startswith("trt-fp16")),
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": cache,
        "trt_profile_min_shapes": "input:1x3x224x224",
        "trt_profile_opt_shapes": f"input:{opt}x3x224x224",
        "trt_profile_max_shapes": f"input:{EVAL_BATCH}x3x224x224",
    })
    return ort.InferenceSession(onnx_path, opts, providers=[trt, cuda])


PROVIDER_OF = {"ort-cpu": "CPUExecutionProvider", "ort-cuda": "CUDAExecutionProvider",
               "trt-fp32": "TensorrtExecutionProvider", "trt-fp16": "TensorrtExecutionProvider",
               "trt-fp16-b64": "TensorrtExecutionProvider"}


def check_provider(session, backend):
    # ONNX Runtime silently falls back to the next provider when one fails to load.
    active = session.get_providers()[0]
    if active != PROVIDER_OF[backend]:
        raise RuntimeError(f"{backend}: expected {PROVIDER_OF[backend]}, got {active}")


class OrtRunner:
    """Runs an ONNX Runtime session; GPU backends keep tensors on the device (IOBinding)."""

    def __init__(self, session, on_gpu: bool):
        self.session = session
        self.on_gpu = on_gpu

    def _bind(self, x):
        import onnxruntime as ort

        binding = self.session.io_binding()
        binding.bind_ortvalue_input("input", ort.OrtValue.ortvalue_from_numpy(x, "cuda", 0))
        binding.bind_output("logits", "cuda", 0)
        return binding

    def __call__(self, x):
        if not self.on_gpu:
            return self.session.run(None, {"input": x})[0]
        binding = self._bind(x)
        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def bench(self, x):
        """A zero-argument callable for timing; the input is uploaded once, outside the loop."""
        if not self.on_gpu:
            return lambda: self.session.run(None, {"input": x})
        binding = self._bind(x)

        def run():
            self.session.run_with_iobinding(binding)
            binding.synchronize_outputs()
        return run


def median_ms(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def load_eval_set(root):
    dataset = ImagenetteEval(root, "val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=EVAL_BATCH, num_workers=8)
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in loader])
    return xs, np.concatenate(ys)


def top1(run, xs, ys):
    preds = np.concatenate([run(np.ascontiguousarray(x)).argmax(1) for x in xs])
    return 100.0 * float((preds == ys).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", help="official PP-LCNet .pdparams (needed for accuracy)")
    parser.add_argument("--imagenette", help="imagenette2 directory; enables accuracy")
    parser.add_argument("--threads", type=int, default=10, help="CPU threads (paper: 10)")
    parser.add_argument("--backends", nargs="+", default=BACKENDS, choices=BACKENDS)
    parser.add_argument("--models", nargs="+", help="subset of model names")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--onnx-dir", default="runs/onnx")
    parser.add_argument("--out", default="runs/benchmark_deploy.json")
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(args.onnx_dir, exist_ok=True)
    trt_cache = os.path.join(args.onnx_dir, "trt_cache")
    os.makedirs(trt_cache, exist_ok=True)

    models = build_models(args.weights_dir)
    if args.models:
        models = {k: models[k] for k in args.models}
    eval_set = load_eval_set(args.imagenette) if args.imagenette else None
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    x_batch = rng.standard_normal((THROUGHPUT_BATCH, 3, 224, 224)).astype(np.float32)

    results = []
    for name, model in models.items():
        onnx_path = os.path.join(args.onnx_dir, name + ".onnx")
        if not os.path.exists(onnx_path):
            export_onnx(model, onnx_path)
        row = {"model": name, "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
        for backend in args.backends:
            if backend == "torch-cpu":
                torch.set_num_threads(args.threads)
                xt = torch.from_numpy(x1)
                with torch.inference_mode():
                    row[backend] = median_ms(lambda: model(xt), args.warmup, args.runs)
                continue
            session = ort_session(onnx_path, backend, args.threads, trt_cache)
            check_provider(session, backend)
            run = OrtRunner(session, on_gpu=backend != "ort-cpu")
            if backend == "trt-fp16-b64":
                ms = median_ms(run.bench(x_batch), args.warmup, args.runs)
                row[backend] = THROUGHPUT_BATCH * 1000.0 / ms
                continue
            row[backend] = median_ms(run.bench(x1), args.warmup, args.runs)
            if eval_set and backend in ("ort-cpu", "trt-fp16"):
                row[f"top1_{backend}"] = top1(run, *eval_set)
        results.append(row)
        print(json.dumps(row), flush=True)

    meta = {"cpu": platform.processor() or platform.machine(), "threads": args.threads,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)

    cols = ["params_m"] + args.backends + (["top1_ort-cpu", "top1_trt-fp16"] if eval_set else [])
    print("| Model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in results:
        print(f"| {r['model']} | " + " | ".join(
            f"{r[c]:.2f}" if c in r else "-" for c in cols) + " |")


if __name__ == "__main__":
    main()
