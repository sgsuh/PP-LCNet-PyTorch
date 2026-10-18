FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

# The base image's Python is marked externally managed (PEP 668). This image
# is single-purpose, so installing on top of the preinstalled torch is fine.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

ENV PYTHONPATH=/workspace
CMD ["python", "-m", "pytest", "-q"]
