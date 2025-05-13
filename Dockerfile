FROM docker.io/pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN apt update && apt install -y wget git python3 python3-pip python3-venv
RUN pip install matplotlib pillow scikit-learn timm

RUN git clone https://github.com/fkryan/gazelle.git /gazelle
WORKDIR /gazelle
RUN wget https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14.pt \
         https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt \
         https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt \
         https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt
RUN pip install -e .

COPY installAllModels.py /gazelle
RUN python3 installAllModels.py

COPY main.py /gazelle
CMD ["python3","main.py"]