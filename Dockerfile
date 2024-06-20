FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-rockylinux9 as base

RUN apt-get update -y && apt-get install git python3 pip3

WORKDIR /workspace
COPY src .

RUN git clone https://github.com/mistralai/mistral-finetune.git . \
    && pip install -r requirements.txt \
    && wget https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar \
    && tar -xf mistral-7B-v0.3.tar -C mistral_models \
    && python data.py

CMD ["torchrun", "--nproc-per-node", "4", "-m", "train", "8x7b.yaml"]
