FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y htop python3-dev python3-pip git
RUN pip3 install virtualenv

RUN virtualenv venv

COPY data/raw KPE/data/raw
COPY src KPE/src
COPY requirements.txt KPE/
COPY README.md KPE/

RUN /bin/bash -c "source venv/bin/activate && pip install -r KPE/requirements.txt"