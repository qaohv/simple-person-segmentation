FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN conda install -c anaconda pip -y

RUN mkdir /opt/app/
COPY requirements.txt /opt/app/
RUN pip install -r /opt/app/requirements.txt

COPY src /opt/app/src

WORKDIR /opt/app/
ENV PYTHONPATH=/opt/app

RUN git clone https://github.com/NVIDIA/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

RUN mkdir /data
VOLUME /data

RUN mkdir /logs
VOLUME /logs

CMD "/bin/bash"
