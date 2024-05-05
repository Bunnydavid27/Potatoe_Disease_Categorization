import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Model = tf.keras.models.load_model(r"Model")
Class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


@app.get("/")
async def ping():
    return "Hello Running"

# @app.get("/ping")
# async def ping():
#     return "Hello Running"


def file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/prediction")
async def prediction():
    return "Running Prediction"


@app.post("/prediction")
async def predict(file: UploadFile = File(...)
):
    image = file_as_image(await file.read())
    image = np.resize(image, new_shape=(256,256,3))
    image_batch = np.expand_dims(image, 0)
    predictions = Model.predict(image_batch)
    predicted_class = Class_names[np.argmax(predictions[0])]
    print(predicted_class)
    return {
        'class' : predicted_class
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)