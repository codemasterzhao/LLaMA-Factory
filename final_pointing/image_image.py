from vlmeval.config import supported_VLM
import json
import re
import base64
import numpy as np
import torch
import time
import colorama
from util.prompt_generation import generate_question_prompt
from colorama import Fore, Style, init
import argparse

def fix_seed():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def evaluate_model(category_name, model_name, input_json_path, evaluate_output_path):
    
    fix_seed()

    # load input_json file
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)
    if not isinstance(input_data, list):
        raise ValueError(f"Input JSON file {input_json_path} should be a list of items.")

    evaluate_result = []

    for idx, item in enumerate(input_data):
        # only look for the correct category
        item_copy = item.copy()

        if category_name not in item.get("category", ""):
            continue
        
        question = item.get("question", "")
        options = item.get("options", [])
        image = item.get("image", None)

        for format in ["direct", "cot"]:
            question_prompt = generate_question_prompt(format, category_name, question, options, model_category="image")
            print(f"{Fore.YELLOW}Evaluating item {idx + 1}/{len(input_data)}: {item.get('id', idx)}{Style.RESET_ALL}")
            # print the input question prompt
            response = supported_VLM[model_name]().generate([image, question_prompt])

            item_copy[f"{format}_reply"] = response

        evaluate_result.append(item_copy)
        
    # save the evaluation result to a json file
    with open(evaluate_output_path, 'w') as f:
        json.dump(evaluate_result, f, indent=2, ensure_ascii=False)  

if __name__ == "__main__":
    # only evaluate model within the certain category in the input json file
    parser = argparse.ArgumentParser(description="Evaluate model on a specific category from a JSON file.")
    parser.add_argument("--model_name", type=str, required=True, help="The specific model name (e.g., claude-sonnet-4-20250514)")
    parser.add_argument("--input_json_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--category_name", type=str, required=True, help="Category to evaluate (e.g., pointing).")
    parser.add_argument("--model_series", type=str, required=True, choices=["gemini", "gpt", "claude"], help="Model family/series.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluation results. Default is auto-generated.")
    parser.add_argument("--evaluate_output_category", type=str, default=None, help="Path to save evaluation results. Default is auto-generated.")

    args = parser.parse_args()

    model_name = args.model_name
    input_json_path = args.input_json_path
    category_name = args.category_name
    model_series = args.model_series
    evaluate_category = args.evaluate_output_category

    # category_name 
    # action histroy understanding
    # next action prediction
    # path planning
    # relative direction
    # object localization 
    # trajectory
    # pointing
    # define more specific category

    evaluate_output_path = f"final_{model_name}_evaluate_{evaluate_category}.json"
    evaluate_model(category_name, model_name, input_json_path, evaluate_output_path)
