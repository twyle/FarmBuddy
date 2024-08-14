import torch
from torchvision import transforms 
import numpy as np
import os
from PIL import Image
import torch
import os
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Optional, Any
from torch import nn
import random
import numpy as np
from os import path
import json
from langchain_groq import ChatGroq
from langchain.base_language import BaseLanguageModel
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool, BaseTool
from langchain_core.messages import (
    AIMessage, FunctionMessage, ToolMessage, SystemMessage, BaseMessage, HumanMessage
)
from langchain_core.runnables import Runnable
import googlemaps

class TomatoModel():
    def __init__(self, model_path: str = '/home/lyle/Downloads/PestNet.pkl', device: str = 'cpu') -> None:
        self.device: str = device
        self.model = torch.load(model_path, map_location=device)
        
    def analyze_image(self, image: Image = None) -> tuple[str, dict[str, float]]:
        logits: list[float] = [random.uniform(0, 1) for _ in range(4)]
        exponentials: list[float] = np.exp(logits)
        predictions: list[str] = [value/sum(exponentials) for value in exponentials]
        prediction: int = np.argmax(predictions)
        labels: list[str] = [
            'Maize Leaf Rust',
            'Northern Leaf Blight',
            'Healthy',
            'Gray Leaf Spot'
        ]
        data = {
            'Maize Leaf Rust': round(float(predictions[0]), 2) * 100,
            'Northern Leaf Blight': round(float(predictions[1]) * 100, 2),
            'Healthy': round(float(predictions[2]), 2) * 100,
            'Gray Leaf Spot': round(float(predictions[3]) * 100, 2)
        }
        return {"prediction": labels[prediction], "predictions":data}

    def preprocess_image(self, image: Image):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        data_transform = transforms.Compose([
                transforms.RandomResizedCrop(224), # resize and crop image to 224 x 224 pixels
                transforms.RandomHorizontalFlip(), # flip the images horizontally
                transforms.ToTensor(), # convert to pytorch tensor data type
                transforms.Normalize(mean, std) # normalize the input image dataset.
            ])
        transformed_image = data_transform(image).to(self.device)
        transformed_image = torch.unsqueeze(transformed_image, 0)
        return transformed_image


    def evaluate_image(self, image: Image) -> tuple[str, dict[str, float]]:
        transformed_image = self.preprocess_image(image)
        labels = [
            'Bacterial Spot',
            'Early Blight',
            'Late Blight',
            'Leaf Mold',
            'Septoria Leaf Spot',
            'Two Spotted Spider Mite',
            'Target Spot',
            'Yellow Leaf Curl Virus',
            'Tomato Mosaic virus',
            'Healthy',
            ]
        self.model.eval()
        predictions = F.softmax(self.model(transformed_image), dim = 1)
        data = {label: round(float(predictions[0][i]), 4) * 100 for i, label in enumerate(labels) }
        prediction = predictions.argmax()
        return {"prediction": labels[prediction], "predictions": data}
    
    
model = TomatoModel("/ownloads/tomato.pt", "cpu")