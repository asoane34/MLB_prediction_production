FROM python:3.7-buster


COPY requirements.txt .
COPY ["deploy", "/usr/src/app/deploy"]
COPY ["models", "/usr/src/app/models"]
COPY ["data_collection", "/usr/src/app/data_collection"]

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

EXPOSE 5000
WORKDIR /usr/src/app/deploy

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

