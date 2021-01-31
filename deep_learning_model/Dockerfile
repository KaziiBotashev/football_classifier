FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update

RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-setuptools python3-pip git ffmpeg libsm6 libxext6 curl

WORKDIR /usr/home/

COPY ./requirements.txt .

RUN pip3 install --upgrade pip
 
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8888






