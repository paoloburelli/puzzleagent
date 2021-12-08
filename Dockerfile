FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y git

RUN mkdir /app
RUN mkdir /app/logs
RUN mkdir /app/logs/train
RUN mkdir /app/logs/test
RUN mkdir /app/logs/eval
COPY requirements.txt /app
WORKDIR /app

RUN python -m pip install -r requirements.txt --no-cache-dir

COPY . /app
RUN chmod +x /app/simulator/linux.x86_64
RUN chmod +x /app/scripts/*
CMD ./scripts/start_container.sh

