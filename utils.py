import glob
import os
import shutil
import argparse

import pandas as pd
import numpy as np
import yaml

from loggers import logger
from typing import List, Union, Any


def load_yaml(path: str, subsection: str) -> dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path (str): Path to the YAML file.
        subsection (str): Subsection to be considered (i.e. general, model, trading).

    Returns:
        A dictionary containing the YAML file.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config[subsection]


def data_split(
    dataset: str,
    training_stocks: list[str],
    target_stock: list[str],
    training_ratio: float,
    validation_ratio: float,
    include_target_stock_in_training: bool,
) -> None:
    """
    Split the data into training, validation and test sets based on the training, validation and test ratios.

    Args:
        dataset (str): The considered dataset (i.e. nasdaq, lse, ...).
        training_stocks (list):  The list of stocks to be used for training.
        target_stock (list):  The list of stocks to be used for validation and test.
        training_ratio (float):  The ratio of training data.
        validation_ratio (float): The ratio of validation data.
        include_target_stock_in_training (bool): Including or not the target stock in the training set.

    Returns:
        None.
    """
    # List of target_stocks contains stocks that must be split into training, validation and test sets.
    # If requested, target stocks are removed from the training set in a second stage.
    for stock in target_stock:
        # Sorted list of scaled data.
        files_scaled = sorted(glob.glob(f"./data/{dataset}/scaled_data/{stock}/*.csv"))
        # Sorted list of unscaled data.
        files_unscaled = sorted(
            glob.glob(f"./data/{dataset}/unscaled_data/{stock}/*.csv")
        )

        # Sanity check to make sure that the number of files in the scaled and unscaled folders is the same.
        assert len(files_scaled) == len(
            files_unscaled
        ), "The number of files in the scaled and unscaled folders must be the same."

        # Number of training files (based on training ratio).
        num_training_files = int(len(files_scaled) * training_ratio)
        # Number of validation files (based on validation ratio).
        num_validation_files = int(len(files_scaled) * validation_ratio)
        # Number of test files (based on test ratio).
        num_test_files = len(files_scaled) - num_training_files - num_validation_files

        # Create the training folder (scaled data) if it does not exist.
        if not os.path.exists(f"./data/{dataset}/scaled_data/training"):
            os.makedirs(f"./data/{dataset}/scaled_data/training")
        # Create the validation folder (scaled data) if it does not exist.
        if not os.path.exists(f"./data/{dataset}/scaled_data/validation"):
            os.makedirs(f"./data/{dataset}/scaled_data/validation")
        # Create the test folder (scaled data) if it does not exist.
        if not os.path.exists(f"./data/{dataset}/scaled_data/test"):
            os.makedirs(f"./data/{dataset}/scaled_data/test")

        # Create the training folder (unscaled data) if it does not exist.
        if not os.path.exists(f"./data/{dataset}/unscaled_data/training"):
            os.makedirs(f"./data/{dataset}/unscaled_data/training")
        # Create the validation folder (unscaled data) if it does not exist.
        if not os.path.exists(f"./data/{dataset}/unscaled_data/validation"):
            os.makedirs(f"./data/{dataset}/unscaled_data/validation")
        # Create the test folder (unscaled data) if it does not exist.
        if not os.path.exists(f"./data/{dataset}/unscaled_data/test"):
            os.makedirs(f"./data/{dataset}/unscaled_data/test")

        # Move the files to the training folder (scaled data).
        # If requested, target stocks are removed from the training set.
        for i in range(num_training_files):
            destination_folder = f"./data/{dataset}/scaled_data/training"
            file = files_scaled[i]
            if include_target_stock_in_training:
                shutil.move(file, destination_folder)
            else:
                if target_stock not in file:
                    shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Move the files to the validation folder (scaled data).
        for i in range(num_validation_files):
            destination_folder = f"./data/{dataset}/scaled_data/validation"
            file = files_scaled[i + num_training_files]
            shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Move the files to the test folder (scaled data).
        for i in range(num_test_files):
            destination_folder = f"./data/{dataset}/scaled_data/test"
            file = files_scaled[i + num_training_files + num_validation_files]
            shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Move the files to the training folder (unscaled data).
        # If requested, target stocks are removed from the training set.
        for i in range(num_training_files):
            destination_folder = f"./data/{dataset}/unscaled_data/training"
            file = files_unscaled[i]
            if include_target_stock_in_training:
                shutil.move(file, destination_folder)
            else:
                if target_stock not in file:
                    shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Move the files to the validation folder (unscaled data).
        for i in range(num_validation_files):
            destination_folder = f"./data/{dataset}/unscaled_data/validation"
            file = files_unscaled[i + num_training_files]
            shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Move the files to the test folder (unscaled data).
        for i in range(num_test_files):
            destination_folder = f"./data/{dataset}/unscaled_data/test"
            file = files_unscaled[i + num_training_files + num_validation_files]
            shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Delete the folders containing the original processed LOB data.
        shutil.rmtree(f"./data/{dataset}/scaled_data/{stock}")
        shutil.rmtree(f"./data/{dataset}/unscaled_data/{stock}")

    # Until now, only the data belonging to target_stocks have been treated.
    # Now, all the other stocks need to be treated.
    # Perform the set difference operation between the training_stocks and target_stock sets.
    difference_set = list(set(training_stocks).difference(set(target_stock)))

    # Stocks in difference_set are training-only data.
    for stock in difference_set:
        # Get the sorted list of scaled LOB files.
        files_scaled = sorted(glob.glob(f"./data/{dataset}/scaled_data/{stock}/*.csv"))
        # Get the sorted list of unscaled LOB files.
        files_unscaled = sorted(
            glob.glob(f"./data/{dataset}/unscaled_data/{stock}/*.csv")
        )

        # Sanity check to make sure that the number of files in the scaled and unscaled folders is the same.
        assert len(files_scaled) == len(
            files_unscaled
        ), "The number of files in the scaled and unscaled folders must be the same."

        # Move the files to the training folder (scaled data).
        for i in range(len(files_scaled)):
            destination_folder = f"./data/{dataset}/scaled_data/training"
            file = files_scaled[i]
            shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Move the files to the training folder (unscaled data).
        for i in range(len(files_unscaled)):
            destination_folder = f"./data/{dataset}/unscaled_data/training"
            file = files_unscaled[i]
            shutil.move(file, destination_folder)
            print(f"{file} --> {destination_folder}")

        # Delete the folders containing the original processed LOB data.
        shutil.rmtree(f"./data/{dataset}/scaled_data/{stock}")
        shutil.rmtree(f"./data/{dataset}/unscaled_data/{stock}")

    # When dealing with multiple stocks, we want to maintain the same number of files for each of them in the training folder.
    print("Aligning data...")
    target_stock_dates = set()
    other_dates = set()
    # As a first step, we check the number of representatives of the target_stock in the training folder.
    for stock in target_stock:
        files = sorted(
            glob.glob(f"./data/{dataset}/unscaled_data/training/{stock}_*.csv")
        )
        for file in files:
            date = file.split("/")[-1].split("_")[-1].split(".")[0]
            target_stock_dates.add(date)
    # As a second step, we check the number of representatives of the other stocks in the training folder.
    # As a third step, we remove redundant files (if any) from both scaled and unscaled data folder.
    for stock in training_stocks:
        files = sorted(
            glob.glob(f"./data/{dataset}/unscaled_data/training/{stock}_*.csv")
        )
        for file in files:
            date = file.split("/")[-1].split("_")[-1].split(".")[0]
            other_dates.add(date)
    dates_to_remove = list(other_dates.difference(target_stock_dates))
    for date in dates_to_remove:
        files = sorted(
            glob.glob(f"./data/{dataset}/unscaled_data/training/*_{date}.csv")
        )
        for file in files:
            os.remove(file)
        files = sorted(glob.glob(f"./data/{dataset}/scaled_data/training/*_{date}.csv"))
        for file in files:
            os.remove(file)
    print("Data aligned.")


def save_dataset_info(
    experiment_id: str,
    general_hyperparameters: dict[str, Any],
) -> None:
    """
    Save all the days used in the training, validation and test sets.
    Args:
        experiment_id (str): ID of the experiment.
        general_hyperparameters (dict): General hyperparameters.

    Returns:
        None.
    """
    # Access the training data folder and list all the files.
    training_days_temp = glob.glob(
        f"./data/{general_hyperparameters['dataset']}/scaled_data/training/*.csv"
    )
    # Access the validation data folder and list all the files.
    validation_days_temp = glob.glob(
        f"./data/{general_hyperparameters['dataset']}/scaled_data/validation/*.csv"
    )
    # Access the test data folder and list all the files.
    test_days_temp = glob.glob(
        f"./data/{general_hyperparameters['dataset']}/scaled_data/test/*.csv"
    )

    training_days = []
    validation_days = []
    test_days = []

    # Extract the dates from the file names (training data).
    for i in training_days_temp:
        i = i.split("/")[-1].split("_")[-1]
        training_days.append(i)

    # Extract the dates from the file names (validation data).
    for i in validation_days_temp:
        i = i.split("/")[-1].split("_")[-1]
        validation_days.append(i)

    # Extract the dates from the file names (test data).
    for i in test_days_temp:
        i = i.split("/")[-1].split("_")[-1]
        test_days.append(i)

    # Create a dictionary containing the training, validation and test days.
    dataset_info = {
        "training_days": sorted(set(training_days)),
        "validation_days": sorted(set(validation_days)),
        "test_days": sorted(set(test_days)),
    }

    # Save the dictionary as a YAML file.
    logger.logger(
        experiment_id=experiment_id,
        header="dataset_info",
        contents=dataset_info,
    )


def get_best_levels_prices_and_labels(
    dataset: str,
    target_stocks: str,
    history_length: int,
    all_horizons: list[int],
    prediction_horizon: int,
    threshold: float,
) -> tuple[Any, ...]:
    """
    Get the best levels (bid and ask) prices and the corresponding discretized labels.
    Args:
        dataset (str): Name of the dataset to be used (e.g. nasdaq, lse, ...).
        history_length (int): Length of the history (each model's sample is a 2D array of shape (<history_length>, <features>)).
        all_horizons (list): List all horizons computed in the preprocessing stage.
        prediction_horizon (int): Horizon to be considered.
        threshold (float): Threshold to be used to discretize the labels.

    Returns:
        A tuple containing the best levels (bid and ask) prices and the corresponding discretized labels.
    """

    # List the test files.
    test_files = sorted(glob.glob(f"./data/{dataset}/unscaled_data/test/*{target_stocks[0]}*.csv"))

    best_levels_prices = pd.DataFrame()

    # Get the position of the prediction horizon in the list of all horizons.
    position = next(
        (
            index
            for index, value in enumerate(all_horizons)
            if value == prediction_horizon
        ),
        None,
    )
    all_labels_temp = []

    for file in test_files:
        # Load the file.
        df = pd.read_csv(file).iloc[history_length:, :]
        # Reset the index.
        df.reset_index(drop=True, inplace=True)
        # Get all the labels.
        label_df = df.iloc[:, 41:]
        # Get the label corresponding to the prediction horizon.
        label = label_df.iloc[:, position]
        # Get the best levels (ask and bid) prices and the datetime corresponding to each tick.
        best_levels_prices = pd.concat(
            [best_levels_prices, df[["seconds", "ASKp1", "BIDp1"]]]
        )
        # Append the label to the list of labels.
        all_labels_temp = all_labels_temp + label.tolist()

    # Discretize the labels (0: downtrend, 1: no trend, 2: uptrend).
    all_labels = [
        2 if label >= threshold else 0 if label <= -threshold else 1
        for label in all_labels_temp
    ]

    return best_levels_prices, all_labels


def detect_changing_points(
    target: int, cumulative_lengths: list[int]
) -> Union[int, None]:
    """
    Detect the last index of the file containing the target value.
    Args:
        target (int): Target index.
        cumulative_lengths (list): List of cumulative lengths.

    Returns:
        0 if the target value is in the first file, the last index of the file containing the target value otherwise.
    """
    for i, length in enumerate(cumulative_lengths):
        if target <= length:
            if i == 0:
                return 0
            else:
                return cumulative_lengths[i - 1]
    return None


def wandb_hyperparameters_saving(
    wandb_logger: Any,
    general_hyperparameters: dict[str, Any],
    model_hyperparameters: dict[str, Any],
) -> None:
    """
    Save the general/model hyperparameters in the Weights & Biases dashboard.
    Args:
        wandb_logger (any): Wandb logger.
        general_hyperparameters (dict): General hyperparameters.
        model_hyperparameters (dict): Model hyperparameters.

    Returns:
        None.
    """
    wbl = wandb_logger
    for key in general_hyperparameters:
        wbl.experiment.config[key] = general_hyperparameters[key]
    for key in model_hyperparameters:
        wbl.experiment.config[key] = model_hyperparameters[key]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> Any:
    """
    Parser for input arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Hyperparameters acquisition.")

    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="ID of the experiment (if any). This argument is used to resume older experiments or partially re-run experiments.",
    )

    # General hyperparameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="nasdaq",
        help="The dataset to be used (e.g. nasdaq, lse, ...). Each dataset has a different raw data format which needs to be correctly handled.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deeplob",
        help="The model to be used (e.g. deeplob, ...).",
    )
    parser.add_argument(
        "--training_stocks",
        type=str,
        default="XYZ",
        help="Stock to be used for training (e.g., 'CSCO').",
    )
    parser.add_argument(
        "--target_stocks",
        type=str,
        default="XYZ",
        help="The stock to be used in the validation and test sets (it is always unique)",
    )
    parser.add_argument(
        "--normalization_window",
        type=int,
        default=5,
        help="Number of files to be used for rolling data normalization.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="10,50,100",
        help="Horizon(s) to be considered (to be expressed in this format: '10,50,100').",
    )
    parser.add_argument(
        "--training_ratio",
        type=float,
        default=0.6,
        help="Training data proportion."
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.2,
        help="Validation data proportion.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test data proportion."
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="data_processing",
        help="Stage(s) to be run (to be expressed in this format: 'training,evaluation').",
    )  # data_processing | torch_dataset_preparation | torch_dataset_preparation_backtest | training,evaluation | backtest,post_trading_analysis
    parser.add_argument(
        "--include_target_stock_in_training",
        type=str2bool,
        default=True,
        help="Including or not the target stock in the training set.",
    )
    parser.add_argument(
        "--targets_type",
        type=str,
        default='raw',
        help="Type of targets to be used (i.e. smooth, raw).",
    )

    # Model hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=5,
        help="Number of workers to be used by the dataloader.",
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=100,
        help="Length of the history to be used (each model's sample is a 2D array of shape (<history_length>, <features>).",
    )
    parser.add_argument(
        "--shuffling_seed",
        type=int,
        default=428,
        help="Seed to be used for data shuffling.",
    )
    parser.add_argument(
        "--lighten",
        type=str2bool,
        default=False,
        help="Lighten the model's input (10 -> 5 levels).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold to be used to discretize the labels.",
    )
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=10,
        help="Horizon to be considered in the inference stage.",
    )
    parser.add_argument(
        "--balanced_sampling",
        type=str2bool,
        default=True,
        help="Either or not using a balanced sampling approach in the training stage.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience to be used in the training stage.",
    )

    # Trading hyperparameters
    parser.add_argument(
        "--initial_cash",
        type=int,
        default=1000,
        help="Initial cash to be used in the trading simulation.",
    )
    parser.add_argument(
        "--trading_fee",
        type=float,
        default=0.0001,
        help="Trading fee to be used in the trading simulation.",
    )
    parser.add_argument(
        "--mid_side_trading",
        type=str,
        default="mid_to_mid",
        help="Trading strategy to be used in the trading simulation.",
    )
    parser.add_argument(
        "--simulation_type",
        type=str,
        default="with_fees",
        help="Either or not applying trading fees in the trading simulation.",
    )
    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=0.65,
        help="Threshold used to decide if exploiting or ignoring a signal in the trading simulation.",
    )

    args = parser.parse_args()
    return args


def create_hyperparameters_yaml(experiment_id: str, args: Any) -> None:
    """
    Create and save a YAML file containing the hyperparameters as part of an experiment.
    Args:
        experiment_id (str): ID of the experiment.
        args (any): Stage's arguments.

    Returns:
        None.
    """
    training_stocks = list(
        args.training_stocks.split(",")
    )  # Parsing of 'training_stocks' input argument.
    target_stocks = list(
        args.target_stocks.split(",")
    )  # Parsing of 'target_stocks' input argument.
    horizons = list(
        map(int, args.horizons.split(","))
    )  # Parsing of 'horizons' input argument.
    stages = list(args.stages.split(","))  # Parsing of 'stages' input argument.

    # Create a dictionary (YAML structure) containing the hyperparameters.
    data = {
        "general": {
            "dataset": args.dataset,
            "model": args.model,
            "training_stocks": training_stocks,
            "target_stocks": target_stocks,
            "normalization_window": args.normalization_window,
            "horizons": horizons,
            "training_ratio": args.training_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "stages": stages,
            "include_target_stock_in_training": args.include_target_stock_in_training,
            "targets_type": args.targets_type,
        },
        "model": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "history_length": args.history_length,
            "shuffling_seed": args.shuffling_seed,
            "lighten": args.lighten,
            "threshold": args.threshold,
            "prediction_horizon": args.prediction_horizon,
            "balanced_sampling": args.balanced_sampling,
            "patience": args.patience,
        },
        "trading": {
            "initial_cash": args.initial_cash,
            "trading_fee": args.trading_fee,
            "mid_side_trading": args.mid_side_trading,
            "simulation_type": args.simulation_type,
            "probability_threshold": args.probability_threshold,
        },
    }

    # Specify the file path where saving the YAML file.
    file_path = f"{logger.find_save_path(experiment_id)}/hyperparameters.yaml"

    # Write the data to the YAML file.
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def create_tree(path: str) -> None:
    """
    Create folders recursively.
    Args:
        path (str): Tree of folders to be created.

    Returns:
        None.
    """
    # Recursively create a tree of folders. If the path already exists, delete it and create a new one.
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def get_training_test_stocks_as_string(general_hyperparameters):
    training_stocks = general_hyperparameters["training_stocks"]
    general_training_string = ""
    for s in training_stocks:
        general_training_string += s + "_"
    general_training_string = general_training_string[:-1]

    test_stocks = general_hyperparameters["target_stocks"]
    general_test_string = ""
    for s in test_stocks:
        general_test_string += s + "_"
    general_test_string = general_test_string[:-1]

    return general_training_string, general_test_string