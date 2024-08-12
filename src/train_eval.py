import os
import configparser
import argparse
import ast
import json


from utils.train_eval_manager import TrainEvalManager

# Import your model classes here


def load_config(config_file):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load config file: {config_file}")

    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' does not exist.")
        return None

    config = configparser.ConfigParser()
    try:
        with open(config_file, "r") as f:
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
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        try:
            params = ast.literal_eval(params_str)
        except:
            print(f"Error parsing parameters: {params_str}")
            return None

    # Convert tuples to lists for JSON compatibility
    def convert_tuples(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, list):
            return [convert_tuples(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_tuples(value) for key, value in obj.items()}
        return obj

    return convert_tuples(params)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run experiments with configuration file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    print(f"Attempting to load config file: {args.config}")
    config = load_config(args.config)

    if "Models" not in config:
        print("Error: 'Models' section not found in config file.")
        print("Available sections:", config.sections())
        return

    # Load models configuration
    models_config = []
    for model_name, params in config["Models"].items():
        print(f"Processing model: {model_name}")
        try:
            model_class = get_model_class(model_name)
            model_params = parse_params(params)
            if model_params is None:
                continue
            # Convert specific parameters to the correct type
            if "hidden_dim" in model_params:
                if isinstance(model_params["hidden_dim"], list):
                    model_params["hidden_dim"] = [
                        int(h) for h in model_params["hidden_dim"]
                    ]
                else:
                    model_params["hidden_dim"] = int(model_params["hidden_dim"])

            if "kernel_size" in model_params:
                model_params["kernel_size"] = tuple(model_params["kernel_size"])

            if "physics_kernel_size" in model_params:
                model_params["physics_kernel_size"] = tuple(
                    model_params["physics_kernel_size"]
                )

            print(f"Model Parameters: {model_params}")
            schemes = ast.literal_eval(config["Schemes"][model_name])
            models_config.append((model_name, model_class, model_params, schemes))
        except ValueError as e:
            print(f"Error processing model {model_name}: {e}")
        except KeyError:
            print(f"Error: Missing scheme for model {model_name}")

    # Load datasets configuration
    datasets_config = []
    for dataset_name, path in config["Datasets"].items():
        datasets_config.append((path, dataset_name))

    manager = TrainEvalManager(models_config, datasets_config)
    manager.run_all_experiments()


if __name__ == "__main__":
    main()
