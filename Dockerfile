FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime AS dev

# The base image's Python is marked externally managed (PEP 668). This image
# is single-purpose, so installing on top of the preinstalled torch is fine.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

ENV PYTHONPATH=/workspace
CMD ["python", "-m", "pytest", "-q"]

# ONNX export and ONNX Runtime / TensorRT benchmarking:
#   docker build --target deploy -t pplcnet-deploy .
FROM dev AS deploy
COPY requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt
# Let ONNX Runtime's CUDA/TensorRT providers find the pip-installed libraries.
RUN python -c "import os, tensorrt_libs, nvidia; \
root = list(nvidia.__path__)[0]; \
print(os.path.dirname(tensorrt_libs.__file__)); \
print(os.path.join(root, 'cu13', 'lib')); print(os.path.join(root, 'cudnn', 'lib'))" \
    > /etc/ld.so.conf.d/pip-cuda.conf && ldconfig

# Keep the lightweight image as the default build target.
FROM dev
