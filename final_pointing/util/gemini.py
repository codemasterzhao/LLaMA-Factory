import os
import json
import re
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image
import numpy as np
import torch
from google import genai
from colorama import Fore, Style, init

# === Config ===
client = genai.Client(api_key="AIzaSyAJ9hdhjlNy2DWlVtlibZvt9M4tlWU9Yvk")

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def transform_data_format(list:list) -> str:

    input = []

    for item in list:
        dict = {}
        if 'image' in item and isinstance(item['image'], str):
            image = encode_image_base64(item['image'])
            dict['inline_data'] = {"mime_type": "image/jpeg", "data": image}
        elif 'text' in item:
            dict['text'] = item['text']
        input.append(dict)

    return input   

# the input is a list:
# [{'image': 'path'}, {'text': 'text'}]
def generate_content(model_name, content):

    input = transform_data_format(content)

    # print(f"{Fore.YELLOW} Generating content with model {Style.RESET_ALL} {model_name}...")
    # print(f"{Fore.YELLOW} Input data: {Style.RESET_ALL}", input)

    # print all the keys in the input

    response = client.models.generate_content(
        model=model_name, contents={        
            "role": "user", "parts": input
        }
    )
    
    print(f"{Fore.YELLOW} The response from the model is: {Style.RESET_ALL}", response.text)
    return response.text