FROM nvcr.io/nvidia/pytorch:19.11-py3

RUN conda install -c anaconda pip -y

RUN mkdir /opt/app/
COPY requirements.txt /opt/app/
RUN pip install -r /opt/app/requirements.txt

COPY src /opt/app/src

WORKDIR /opt/app/
ENV PYTHONPATH=/opt/app

RUN mkdir /data
VOLUME /data

RUN mkdir /logs
VOLUME /logs

CMD "/bin/bash"
