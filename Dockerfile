FROM python:3.10.19-trixie

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install notebook jupyterlab

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
