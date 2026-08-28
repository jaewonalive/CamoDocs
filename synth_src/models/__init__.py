from .GPT import GPT
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance.

    The released pipeline uses a single synthesizer, gpt-4o-mini
    (model_configs/gpt4o_mini_config.json, provider "gpt"). To plug in a
    different backend, add a Model subclass and a branch here.
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'gpt':
        model = GPT(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model
