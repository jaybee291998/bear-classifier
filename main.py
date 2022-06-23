from typing import Union
from enum import Enum
# from webbrowser import get
import io

from numpy import argmax

import torch
import torchvision.transforms as transforms

import fastbook 
fastbook.setup_book()
from fastbook import *
from fastai.vision.all import *
# from fastai.vision import *

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from PIL import Image
import matplotlib.pyplot as plt

from utils import *
learn = load_learner('model/bearV3.pkl')
app = FastAPI()


vocab = ['black', 'grizzly', 'teddy']


def transform_image(im):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

    return my_transforms(im)

class ModelName(str, Enum):
    alexnet = "AlexNet"
    resnet = "ResNet"
    lenet = "LeNet"

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

@app.get('/')
async def root():
    return {"message":"hello world!"}


@app.get('/item/{item_id}')
async def item(item_id: int):
    return {'item id':item_id}


@app.get('/models/{model_name}')
async def models(model_name: ModelName):
    return {'model_name':model_name}


@app.get('/anime')
async def anime(title: str, q: Union[str, None]=None):
    if q: return {'q':q}
    return {'title': title}

@app.post('/create-item')
async def create_item(item: Item):
    return item

@app.post('/uploadfile')
async def upload(file: UploadFile):
    im = tensor(Image.open(file.file))
    # im = transform_image(image_bytes=file)
    # im.show()
    preds = learn.predict(im)
    # preds =' noene'
    probabilities = preds[2].tolist()
    bear_type = vocab[argmax(probabilities)]
    return {'preds':probabilities, 'bear_type':bear_type}