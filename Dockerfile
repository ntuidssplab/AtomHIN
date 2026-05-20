FROM nvcr.io/nvidia/dgl:24.11-py3

WORKDIR /workspace/AtomHIN

COPY . .

RUN pip install --upgrade pip && \
    pip install -e ".[scripts,precom,ray]"