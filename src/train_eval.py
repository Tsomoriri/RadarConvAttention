import os
import time
import configparser
import argparse
import ast
import json


from utils.train_eval_manager import TrainEvalManager
# Import your model classes here
from src.models.ConvLSTM import ConvLSTM
from src.models.ConvLSTM_Physics import ConvLSTM_iPINN as ConvLSTM_Physics
from src.models.AttentionConvLSTM import ConvLSTM as ConvLSTM_Attention
from src.models.AttentionConvLSTM_Physics import ConvLSTM as ConvLSTM_Attention_Physics

def load_config(config_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config file: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' does not exist.")
        return None
    
    config = configparser.ConfigParser()
    try:
        with open(config_file, 'r') as f:
            config.read_file(f)
    except IOError as e:
        print(f"Error reading config file: {e}")
        return None
    except configparser.Error as e:
        print(f"Error parsing config file: {e}")
        return None
    
    print("Sections found in config file:", config.sections())
    return config

def get_model_class(model_name):
    for name, obj in globals().items():
        if name.lower() == model_name.lower():
            return obj
    raise ValueError(f"Model class '{model_name}' not found")

def parse_params(params_str):
    def convert_to_tuple(obj):
        if isinstance(obj, list):
            return tuple(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_tuple(v) for k, v in obj.items()}
        return obj

    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        try:
            params = ast.literal_eval(params_str)
        except:
            print(f"Error parsing parameters: {params_str}")
            return None
    
    return convert_to_tuple(params)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments with configuration file")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to the configuration file")
    args = parser.parse_args()

    print(f"Attempting to load config file: {args.config}")
    config = load_config(args.config)

    if 'Models' not in config:
        print("Error: 'Models' section not found in config file.")
        print("Available sections:", config.sections())
        return

      # Load models configuration
    models_config = []
    for model_name, params in config['Models'].items():
        print(f"Processing model: {model_name}")
        try:
            model_class = get_model_class(model_name)
            model_params = parse_params(params)
            if model_params is None:
                continue
            print(f"Model Parameters: {model_params}")  # Add this line for debugging
            schemes = ast.literal_eval(config['Schemes'][model_name])
            models_config.append((model_name, model_class, model_params, schemes))
        except ValueError as e:
            print(f"Error processing model {model_name}: {e}")
        except KeyError as e:
            print(f"Error: Missing scheme for model {model_name}")

    # Load datasets configuration
    datasets_config = []
    for dataset_name, path in config['Datasets'].items():
        datasets_config.append((path, dataset_name))


    manager = TrainEvalManager(models_config, datasets_config)
    manager.run_all_experiments()

if __name__ == "__main__":
    main()
