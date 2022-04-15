FROM python:3.9

ENV PYTHONUNBUFFERED 1

COPY ./src /src
WORKDIR /src
RUN mkdir -p data
RUN mkdir -p log
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
