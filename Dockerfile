FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace/distrifuser

COPY . .

RUN pip install -e .

# RUN pip uninstall transformer-engine

