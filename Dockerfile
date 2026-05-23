FROM nvcr.io/nvidia/dgl:24.11-py3

WORKDIR /workspace/AtomHIN

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e ".[scripts,precom,ray]"