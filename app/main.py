from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

from PIL import Image
import io
import sys
import logging
import cv2
from cv2 import dnn_superres
import numpy as np
import os

from response_dto.prediction_response_dto import PredictionResponseDto
from deep_learning_model.predictions.classify_image import ImageClassifier


app = FastAPI()

image_classifier = ImageClassifier()


@app.post("/predict/", response_model=PredictionResponseDto)
async def predict(use_individual_models: bool, file: UploadFile = File(...)):
    # API main procedure. Takes image and predicts image label using ImageClassifier
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400,
            detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        if image.size[0] <= 100:
            image = np.array(image)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            sr_model_path = os.path.join(
                'deep_learning_model', 'trained_model', 'FSRCNN_x4.pb')
            sr = dnn_superres.DnnSuperResImpl_create()
            sr.readModel(sr_model_path)
            sr.setModel("fsrcnn", 4)
            image = sr.upsample(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        print(use_individual_models)
        predicted_class = image_classifier.predict(
            image, use_individual_models)

        logging.info(f"Predicted Class: {predicted_class}")
        return {
            "filename": file.filename,
            "contentype": file.content_type,
            "label": predicted_class,
        }
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
