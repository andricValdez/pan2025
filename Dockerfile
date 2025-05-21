FROM python:3.10-slim-buster

WORKDIR /.

COPY ./requirements.txt requirements.txt

RUN pip list
RUN pip install -r requirements.txt

COPY ./ .

#WORKDIR /code

ENTRYPOINT  ["main.py"]