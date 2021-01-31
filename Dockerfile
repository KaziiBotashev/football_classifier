FROM python:3.8.5-slim

WORKDIR /usr/home

COPY ./requirements.txt .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 curl -y
 
RUN pip install -r requirements.txt

RUN mkdir app
RUN mkdir response_dto
RUN mkdir -p deep_learning_model/predictions
RUN mkdir -p deep_learning_model/training
RUN mkdir -p deep_learning_model/trained_models



COPY app app
COPY response_dto response_dto
COPY deep_learning_model/predictions deep_learning_model/predictions
COPY deep_learning_model/training/img_transformations.py deep_learning_model/training/
COPY deep_learning_model/training/models.py deep_learning_model/training/
COPY deep_learning_model/training/config.py deep_learning_model/training/
COPY deep_learning_model/training/model_architecture.py deep_learning_model/training/
COPY deep_learning_model/trained_model deep_learning_model/trained_model 

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
