FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 5000
CMD [ "python" , "app.py"]
