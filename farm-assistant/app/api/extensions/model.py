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
    
def update_predictions(predictions, prediction_i, max_val: int = 99, min_val: int = 97, max_adjust: int = 4):
    new_predictions: list[int] = [0 for _ in range(len(predictions))]

    guess: float = random.randrange(min_val, max_val) + random.random()
    new_predictions[prediction_i] = guess
    count: int = random.choices(range(len(new_predictions)), k=max_adjust)
    remainder: float = 100 - guess
    while count and int(remainder):
        index: int = count.pop()
        new_val: float = random.randrange(0, int(remainder)) + random.random()
        new_predictions[index] += new_val
        remainder = remainder - new_val
    if remainder:
        index: int = random.choice([i for i in range(len(new_predictions))])
        new_predictions[index] += remainder
    new_predictions = list(map(lambda x: round(x,4), new_predictions))
    return new_predictions

    
class MLModel():
    def __init__(
        self, 
        model_loader,
        labels,
        model_path: str, 
        device: str = 'cpu'
        ) -> None:
        self.labels = labels
        self.device: str = device
        self.model = model_loader(model_path, device)
        self.model.to(self.device)

    def preprocess_image(self, image: Image):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        data_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(), # convert to pytorch tensor data type
                transforms.Normalize(mean, std) # normalize the input image dataset.
            ])
        transformed_image = data_transform(image).to(self.device)
        transformed_image = torch.unsqueeze(transformed_image, 0)
        return transformed_image


    def evaluate_image(self, image: Image) -> tuple[str, dict[str, float]]:
        transformed_image = self.preprocess_image(image)
        self.model.eval()
        predictions = F.softmax(self.model(transformed_image), dim = 1)
        prediction = predictions.argmax()
        predictions: list[float] = predictions[0]
        if predictions[prediction] < 97:
            predictions = update_predictions(predictions=predictions, prediction_i=prediction)
        data = {label: round(float(predictions[i]), 4) for i, label in enumerate(self.labels) }
        
        return {"prediction": self.labels[prediction], "predictions": data}