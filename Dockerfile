# Use PyTorch base image with CUDA
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

ENV HF_HUB_OFFLINE=1

WORKDIR /.

COPY ./requirements.txt requirements.txt

# Fix numpy binary compatibility issue
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --force-reinstall numpy==1.24.4 && \
    python -m nltk.downloader punkt wordnet omw-1.4 && \
    python -m spacy download en_core_web_sm

COPY ./ .

#WORKDIR /code

ENTRYPOINT  ["main.py"]



 