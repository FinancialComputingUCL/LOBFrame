import os
import sys
from datetime import datetime

import yaml
import random
import string


def generate_id(name, target_stock):
    """
    Generate a unique experiment identifier based on the input `name` and the current timestamp in the format "YYYY-MM-DD_HH_MM_SS".
    Create a directory path using this identifier within the 'loggers/results'  directory relative to the script's location, and if
    it doesn't exist, create it.

    :param name: name of the DL model to be used in the experiment, (str).
    :return: experiment_id: unique experiment identifier, (str).
    """
    random_string_part = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(7))
    init_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    experiment_id = f"{target_stock}_{name}_{init_time}_{random_string_part}"

    root_path = sys.path[0]
    dir_path = f"{root_path}/loggers/results/{experiment_id}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return experiment_id


def find_save_path(model_id):
    """
    Find the directory path for saving results associated with a given `model_id`. This function constructs a directory path within the
    'loggers/results' directory relative to the script's location.

    :param model_id: model identifier, (str).
    :return: directory path, (str).
    """
    root_path = sys.path[0]
    dir_path = f"{root_path}/loggers/results/{model_id}"
    return dir_path


def logger(experiment_id, header, contents):
    """
    Log experimental results in a YAML file associated with the given `model_id`. If the file already exists, it appends new data to it;
    otherwise, it creates a new file.

    :param experiment_id: model identifier, (str).
    :param header: header for the data being logged, (str).
    :param contents: data to be logged, provided as a dictionary, (dict).
    """
    root_path = sys.path[0]
    file_path = f"{root_path}/loggers/results/{experiment_id}/data.yaml"

    contents = {header: contents}

    if os.path.exists(file_path):
        with open(file_path, "r") as yamlfile:
            current_yaml = yaml.safe_load(yamlfile)
            current_yaml.update(contents)
    else:
        current_yaml = contents
    with open(file_path, "w") as yamlfile:
        yaml.dump(current_yaml, yamlfile)


def read_log(model_id, header):
    """
    Read and retrieve data from a log file associated with the given `model_id`.

    :param model_id: Model identifier, (str).
    :param header: Header of the data to retrieve from the log, (str).
    :return: The data associated with the specified header from the log, (any type depending on data stored).
    """
    root_path = sys.path[0]
    file_path = f"{root_path}/loggers/results/{model_id}/log.yaml"

    with open(file_path, "r") as yamlfile:
        log = yaml.safe_load(yamlfile)

    return log[header]
