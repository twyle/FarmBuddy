import torch
from torchvision import transforms 
import numpy as np
import os
from PIL import Image
import torch
import os
import torch.nn.functional as F
from pydantic import BaseModel
from .maize_model import load_model
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


@tool
def get_agrovets(query: str) -> str:
    """Useful when you need to get agrovets in a given location. Give it a query, such as 
    agrovets in Nairobi, Kenya. Only use this tool to find aggrovets!
    """
    gmaps = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])
    results = gmaps.places(query=f'Get me aggrovets in {query}')
    aggrovet_locations: list[str] = list()
    for result in results['results']:
        try:
            bussiness: dict = dict()  
            bussiness['business_status'] = result['business_status']
            bussiness['formatted_address'] = result['formatted_address']
            bussiness['name'] = result['name']
            bussiness['opening_hours'] = result.get('opening_hours', 'NaN')
            location: str = f"{result['name']}, found at {result['formatted_address']}"
            aggrovet_locations.append(location)
        except:
            pass
    return aggrovet_locations


system_prompt: str = (
    "You are an expert in crop and pest diaseses. You task is to asnwer farmers "
    "questions on different pests and diseases and offer advice on how to deal "
    "with crop and pest diasesses. Keep your answer short. You can also use the tools"
    "provided to search for aggrovets"
)


class ChatModel():
    def __init__(self, model_name: str = 'llama3-8b-8192') -> None:
        self.model_name = model_name
        self.llm: BaseLanguageModel = ChatGroq(
            temperature=0, 
            model_name=model_name,
            api_key=os.environ["GROQ_API_KEY"]
        )
        
    def chat_with_model(
        self, 
        messages: list[BaseMessage], 
        llm: BaseLanguageModel, 
        tools: dict[str, BaseTool] = []
        ) -> str:
        llm_with_tools: Runnable = llm.bind_tools(tools=list(tools.values()))
        content: str = ""
        response: AIMessage = llm.invoke(messages)
        if response.content:
            content = response.content
            return AIMessage(content=content)
        else:
            messages.append(response)
            tool_calls: list[dict] = response.additional_kwargs['tool_calls']
            for tool_call in tool_calls:
                function_name: str = tool_call['function']['name']
                function_args: dict = tool_call['function']['arguments']
                tool_output = tools[function_name].invoke(function_args)
                tool_output = str(tool_output)
                message = ToolMessage(tool_output, tool_call_id=tool_call["id"])
                messages.append(message)
            response: AIMessage = llm_with_tools.invoke(messages)
            return response
        
    def chat(self, message: str):
        tools: dict[str, BaseTool] = {'get_agrovets': get_agrovets}
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message),
        ]
        response: AIMessage = None
        try:
            response = self.chat_with_model(
                messages=messages, 
                llm=self.llm, 
                tools=tools
            )
        except Exception as e:
            print(e)
            response = self.llm.invoke(messages)
        return response.content


class MaizeModel():
    def __init__(self, model_path: str = '/home/lyle/Downloads/test.pt', device: str = 'cpu') -> None:
        self.device: str = device
        self.model = load_model(model_path=model_path)
        self.model.to(self.device)
        
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
        labels = ['Maize Leaf Rust', 'Northern Leaf Blight', 'Healthy', 'Gray Leaf Spot']
        self.model.eval()
        prediction = F.softmax(self.model(transformed_image), dim = 1)
        data = {
            'Maize Leaf Rust': round(float(prediction[0][0]), 4) * 100,
            'Northern Leaf Blight': round(float(prediction[0][1]) * 100, 4),
            'Healthy': round(float(prediction[0][2]), 4) * 100,
            'Gray Leaf Spot': round(float(prediction[0][3]) * 100, 4)
        }
        prediction = prediction.argmax()
        return {"prediction": labels[prediction], "predictions": data}
    
    
class PestModel():
    def __init__(self, model_path: str = '/home/lyle/Downloads/PestNet.pkl', device: str = 'cpu') -> None:
        self.device: str = device
        self.model = torch.load(model_path, map_location='cpu')
        
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
        labels = ['Ant',
                            'Bee',
                            'Beetle',
                            'Catterpillar',
                            'Earthworm',
                            'Earwig',
                            'Grasshopeer',
                            'Moth',
                            'Slug',
                            'Snail',
                            'Wasp',
                            'Weevil']
        self.model.eval()
        predictions = F.softmax(self.model(transformed_image), dim = 1)
        data = {label: round(float(predictions[0][i]), 4) * 100 for i, label in enumerate(labels) }
        # data = {
        #     'Maize Leaf Rust': round(float(prediction[0][0]), 4) * 100,
        #     'Northern Leaf Blight': round(float(prediction[0][1]) * 100, 4),
        #     'Healthy': round(float(prediction[0][2]), 4) * 100,
        #     'Gray Leaf Spot': round(float(prediction[0][3]) * 100, 4)
        # }
        prediction = predictions.argmax()
        return {"prediction": labels[prediction], "predictions":data}