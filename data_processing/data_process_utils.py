import os

from data_processing import data_process


class DataUtils:
    def __init__(self, ticker, dataset, experiment_id, horizons, normalization_window):
        self.ticker = ticker  # Ticker of the stock to be processed.
        self.dataset = dataset  # Dataset to be used.
        self.experiment_id = experiment_id  # Experiment ID.
        self.horizons = horizons  # Horizons to be used when computing labels.
        self.normalization_window = normalization_window  # Normalization window to be used when normalizing data.

        self.__raw_data_path = None  # Path containing the raw LOB data.
        self.__processed_data_path_unscaled_data = (
            None  # Path containing the processed but unscaled LOB data.
        )
        self.__processed_data_path_scaled_data = (
            None  # Path containing the processed and scaled LOB data.
        )
        self.__logs_path = None  # Path containing the logs of the data processing.

    def __set_raw_data_path(self):
        # Set the raw data path according to the dataset.
        if self.dataset == "nasdaq":
            self.__raw_data_path = f"./data/{self.dataset}/raw/{self.ticker}"

    def __set_processed_data_path_unscaled_data(self):
        # Set the path containing the processed but unscaled LOB data according to the dataset.
        if self.dataset == "nasdaq":
            self.__processed_data_path_unscaled_data = (
                f"./data/{self.dataset}/unscaled_data/{self.ticker}"
            )

    def __set_processed_data_path_scaled_data(self):
        # Set the path containing the processed and scaled LOB data according to the dataset.
        if self.dataset == "nasdaq":
            self.__processed_data_path_scaled_data = (
                f"./data/{self.dataset}/scaled_data/{self.ticker}"
            )

    def __set_logs_path(self):
        # Set the path containing the logs of the data processing according to the experiment ID.
        self.__logs_path = (
            f"./loggers/results/{self.experiment_id}/data_processing_logs"
        )

    def generate_data_folders(self):
        self.__set_raw_data_path()  # Set the raw data path.
        self.__set_processed_data_path_unscaled_data()  # Set the path containing the processed but unscaled LOB data.
        self.__set_processed_data_path_scaled_data()  # Set the path containing the processed and scaled LOB data.
        self.__set_logs_path()  # Set the path containing the logs of the data processing.

        # Create the folder for the processed but unscaled LOB data if it does not exist.
        if not os.path.exists(self.__processed_data_path_unscaled_data):
            os.makedirs(self.__processed_data_path_unscaled_data)

        # Create the folder for the processed and scaled LOB data if it does not exist.
        if not os.path.exists(self.__processed_data_path_scaled_data):
            os.makedirs(self.__processed_data_path_scaled_data)

        # Create the folder for the logs of the data processing if it does not exist.
        if not os.path.exists(self.__logs_path):
            os.makedirs(self.__logs_path)

    # Process the data to obtain scaled and unscaled data.
    def process_data(self):
        data_process.process_data(
            ticker=self.ticker,
            input_path=self.__raw_data_path,
            output_path=self.__processed_data_path_unscaled_data,
            logs_path=self.__logs_path,
            horizons=self.horizons,
            normalization_window=self.normalization_window,
            scaling=False,
        )

        data_process.process_data(
            ticker=self.ticker,
            input_path=self.__raw_data_path,
            output_path=self.__processed_data_path_scaled_data,
            logs_path=self.__logs_path,
            horizons=self.horizons,
            normalization_window=self.normalization_window,
            scaling=True,
        )
