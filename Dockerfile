FROM python:3.8.13-slim-buster
RUN mkdir model_files
COPY model_files/ /model_files
WORKDIR /model_files
RUN apt-get update && apt-get install -y procps
RUN apt-get install -y telnet
RUN pip install -r requirements.txt
#CMD ["tail","-f","/dev/null"]
CMD [ "python", "gateway_api.py"]
