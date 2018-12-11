FROM tensorflow/tensorflow:latest-gpu-py3

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

VOLUME ["/app"]
WORKDIR "/app"
