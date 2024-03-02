import glob
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import tqdm
import matplotlib.pyplot as plt

from utils import detect_changing_points


class CustomDataset(Dataset):
    def __init__(
        self,
        dataset,
        learning_stage,
        window_size,
        shuffling_seed,
        cache_size,
        lighten,
        threshold,
        all_horizons,
        prediction_horizon,
        targets_type,
        balanced_dataloader=False,
        backtest=False,
        training_stocks=None,
        validation_stocks=None,
        target_stocks=None
    ):
        self.learning_stage = learning_stage  # The current learning stage (training, validation or testing).
        self.shuffling_seed = (
            shuffling_seed  # The seed for the random shuffling of the datasets.
        )
        self.balanced_dataloader = balanced_dataloader  # Whether to use a balanced dataloader or not. This option is available only for training.
        self.backtest = backtest
        self.targets_type = targets_type

        if self.learning_stage == "training":
            file_patterns = [f"./data/{dataset}/scaled_data/{self.learning_stage}/{element}_orderbooks*.csv" for element in training_stocks]
            self.csv_files = []
            for pattern in file_patterns:
                self.csv_files.extend(glob.glob(pattern.format(dataset=dataset, self=self)))

            random.seed(self.shuffling_seed)
            random.shuffle(self.csv_files)
        else:
            # During the validation and testing stages it is fundamental to read the datasets in chronological order.
            if self.learning_stage == 'validation':
                file_patterns = [f"./data/{dataset}/scaled_data/{self.learning_stage}/{element}_orderbooks*.csv" for element in validation_stocks]
            else:
                file_patterns = [f"./data/{dataset}/scaled_data/{self.learning_stage}/{element}_orderbooks*.csv" for element in target_stocks]

            self.csv_files = []
            for pattern in file_patterns:
                self.csv_files.extend(glob.glob(pattern.format(dataset=dataset, self=self)))
            self.csv_files = sorted(self.csv_files)

        self.window_size = window_size  # The number of time steps in each window.
        self.lighten = lighten  # Whether to use the light version of the dataset.
        self.threshold = threshold  # The threshold for the classification task.
        self.prediction_horizon = (
            prediction_horizon  # The prediction horizon for the classification task.
        )
        self.all_horizons = (
            all_horizons  # List of all the possible prediction horizons.
        )

        self.cumulative_lengths = [0]  # Store cumulative lengths of datasets.
        self.cache_size = cache_size  # The number of datasets to cache in memory.
        self.cache_data = [
            None
        ] * self.cache_size  # Initialize a cache with <cache_size> empty slots.
        self.cache_indices = [
            None
        ] * self.cache_size  # Initialize the indices for the cache.
        self.current_cache_index = -1

        self.glob_indices = []

        if self.balanced_dataloader:
            print(f"BALANCED dataset construction...")
        else:
            print(f"UNBALANCED dataset construction...")
        for csv_file in tqdm.tqdm(self.csv_files):
            df = pd.read_csv(csv_file)
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(df) - window_size
            )  # Store the lengths of all datasets.

            # If requested by the user, during the training stage, we create balanced dataloaders.
            if self.learning_stage == "training" and self.balanced_dataloader is True:
                temp_labels = (
                    []
                )  # This is a temporary variable which stores the (discretized) labels (i.e. classes) for each sample in each input dataset.

                if self.targets_type == "raw":
                    labels = df.iloc[:-window_size, :][
                        f"Raw_Target_{self.prediction_horizon}"
                    ]  # Extract the raw, continuous labels (i.e. returns) from the current dataset.
                else:
                    labels = df.iloc[:-window_size, :][
                        f"Smooth_Target_{self.prediction_horizon}"
                    ]  # Extract the raw, continuous labels (i.e. returns) from the current dataset.

                # For each file, we must know the corresponding index. This is the reason why we access the cumulative lengths list.
                for label, index in zip(
                    labels,
                    range(self.cumulative_lengths[-2], self.cumulative_lengths[-1]),
                ):
                    # The discretization is performed using the provided threshold. Temporary labels are tuples of the form (class, index).
                    if label > self.threshold:
                        temp_labels.append((2, index))
                    elif label < -self.threshold:
                        temp_labels.append((0, index))
                    else:
                        temp_labels.append((1, index))

                # Group data by class representatives
                class_groups = {}
                for item in temp_labels:
                    (
                        class_representative,
                        index,
                    ) = item  # Unpack the tuple (class, index).

                    # Understand what is the un-cumulative index of each sample.
                    corresponding_cumulative_length = detect_changing_points(
                        index, self.cumulative_lengths
                    )
                    if corresponding_cumulative_length is not None:
                        # If the current sample does not belong to the first dataset, we must subtract the cumulative length of the previous dataset.
                        temp_index = index - corresponding_cumulative_length
                    else:
                        # If the current sample belongs to the first dataset, we do not need to subtract anything.
                        temp_index = index

                    # Even having a balanced dataloader, labels would be messed up once computing models' inputs.
                    # Indeed, given an index 'i', the input rows are the ones from 'i' to 'i + window_size' and the label to be used is the one at 'i + window_size'.
                    # Therefore, we must subtract the window size from the index of each sample.
                    if temp_index >= self.window_size:
                        if class_representative in class_groups:
                            class_groups[class_representative].append(
                                index - self.window_size
                            )
                        else:
                            class_groups[class_representative] = [
                                index - self.window_size
                            ]
                    else:
                        pass

                # Determine the desired number of samples per class (pseudo-balanced). We use the size of the less represented class.
                min_samples_class = min(
                    len(indices) for indices in class_groups.values()
                )
                if min_samples_class > 5000:
                    min_samples_class = 5000

                # We randomly select indices from each class to create the subsample.
                subsample_indices = []
                for class_representative, indices in class_groups.items():
                    random.seed(self.shuffling_seed)
                    subsample_indices.extend(random.sample(indices, min_samples_class))

                # We store the chosen indices in the 'global_indices_list'.
                random.seed(self.shuffling_seed)
                random.shuffle(subsample_indices)
                self.glob_indices.extend(subsample_indices)

            # If requested by the user, during the training stage, we use all the available samples distributed across input datasets.
            if self.learning_stage == "training" and self.balanced_dataloader is False:
                temp_labels = (
                    []
                )  # This is a temporary variable which stores the (discretized) labels (i.e. classes) for each sample in each input dataset.

                if self.targets_type == "raw":
                    labels = df.iloc[:-window_size, :][
                        f"Raw_Target_{self.prediction_horizon}"
                    ]  # Extract the raw, continuous labels (i.e. returns) from the current dataset.
                else:
                    labels = df.iloc[:-window_size, :][
                        f"Smooth_Target_{self.prediction_horizon}"
                    ]  # Extract the raw, continuous labels (i.e. returns) from the current dataset.

                # For each file, we must know the corresponding index. This is the reason why we access the cumulative lengths list.
                for label, index in zip(
                        labels,
                        range(self.cumulative_lengths[-2], self.cumulative_lengths[-1]),
                ):
                    # The discretization is performed using the provided threshold. Temporary labels are tuples of the form (class, index).
                    if label > self.threshold:
                        temp_labels.append((2, index))
                    elif label < -self.threshold:
                        temp_labels.append((0, index))
                    else:
                        temp_labels.append((1, index))

                # Group data by class representatives
                class_groups = {}
                for item in temp_labels:
                    (
                        class_representative,
                        index,
                    ) = item  # Unpack the tuple (class, index).

                    # Understand what is the un-cumulative index of each sample.
                    corresponding_cumulative_length = detect_changing_points(
                        index, self.cumulative_lengths
                    )
                    if corresponding_cumulative_length is not None:
                        # If the current sample does not belong to the first dataset, we must subtract the cumulative length of the previous dataset.
                        temp_index = index - corresponding_cumulative_length
                    else:
                        # If the current sample belongs to the first dataset, we do not need to subtract anything.
                        temp_index = index

                    # Even having a balanced dataloader, labels would be messed up once computing models' inputs.
                    # Indeed, given an index 'i', the input rows are the ones from 'i' to 'i + window_size' and the label to be used is the one at 'i + window_size'.
                    # Therefore, we must subtract the window size from the index of each sample.
                    if temp_index >= self.window_size:
                        if class_representative in class_groups:
                            class_groups[class_representative].append(
                                index - self.window_size
                            )
                        else:
                            class_groups[class_representative] = [
                                index - self.window_size
                            ]
                    else:
                        pass

                # We randomly select indices from each class to create the subsample.
                subsample_indices = []
                for class_representative, indices in class_groups.items():
                    random.seed(self.shuffling_seed)
                    subsample_indices.extend(random.sample(indices, int(len(indices) * 0.1)))

                # We store the chosen indices in the 'global_indices_list'.
                random.seed(self.shuffling_seed)
                random.shuffle(subsample_indices)
                self.glob_indices.extend(subsample_indices)

    def __len__(self):
        # This is the cumulative length of all input datasets.
        return self.cumulative_lengths[-1]

    def cache_dataset(self, dataset_index):
        if self.current_cache_index >= 0:
            # Remove the least recently used cache entry
            self.cache_data[self.current_cache_index] = None
            self.cache_indices[self.current_cache_index] = None

        # Select a random cache slot for the new dataset
        self.current_cache_index = random.randint(0, self.cache_size - 1)

        # Cache the data from the CSV file
        df = pl.read_csv(self.csv_files[dataset_index]).to_pandas()

        self.cache_data[self.current_cache_index] = df.values[:, 1:].astype(np.float32)
        self.cache_indices[self.current_cache_index] = dataset_index

    def __getitem__(self, index):
        try:
            dataset_index = 0
            while index >= self.cumulative_lengths[dataset_index + 1]:
                dataset_index += 1

            if self.cache_indices[self.current_cache_index] != dataset_index:
                # Cache the dataset if it's not already cached.
                self.cache_dataset(dataset_index)

            # Retrieve the un-cumulative index of the current sample.
            start_index = (
                index
                if dataset_index == 0
                else index - self.cumulative_lengths[dataset_index]
            )

            if self.lighten:
                # If the "lighten" option is enabled, we use only the first 5 levels of the orderbook (i.e. 4_level_features * 5_levels = 20_orderbook_features).
                window_data = self.cache_data[self.current_cache_index][
                    start_index: start_index + self.window_size, :20
                ]
            else:
                # If the "lighten" option is not enabled, we use all the 10 levels of the orderbook (i.e. 4_level_features * 10_levels = 40_orderbook_features).
                window_data = self.cache_data[self.current_cache_index][
                    start_index: start_index + self.window_size, :40
                ]

            # Determine the position of the prediction horizon in the list of all horizons.
            position = next(
                (
                    index
                    for index, value in enumerate(self.all_horizons)
                    if value == self.prediction_horizon
                ),
                None,
            )
            # Extract the label from the dataset given its position.
            label = self.cache_data[self.current_cache_index][
                start_index + self.window_size, 40:
            ][position]
            # Discretize the label using the provided threshold.
            if self.backtest is False:
                if label > self.threshold:
                    label = 2
                elif label < -self.threshold:
                    label = 0
                else:
                    label = 1

            return torch.tensor(window_data).unsqueeze(0), torch.tensor(label)
        except Exception as e:
            print(f"Exception in DataLoader worker: {e}")
            raise e


'''
if __name__ == "__main__":
    # Create dataset and DataLoader with random shuffling
    dataset = CustomDataset(
        dataset="nasdaq",
        learning_stage="training",
        window_size=100,
        shuffling_seed=42,
        cache_size=1,
        lighten=True,
        threshold=32,
        targets_type="raw",
        all_horizons=[5, 10, 30, 50, 100],
        prediction_horizon=100,
        balanced_dataloader=False,
        training_stocks=["CHTR"],
        validation_stocks=["CHTR"],
        target_stocks=["CHTR"]
    )

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=True, sampler=dataset.glob_indices
    )

    print(len(dataloader))

    complete_list = []
    # Example usage of the DataLoader
    for batch_data, batch_labels in dataloader:
        # Train your model using batch_data and batch_labels
        # print(batch_labels.tolist())
        complete_list.extend(batch_labels.tolist())
        #print(batch_data.shape, batch_labels.shape)

    plt.hist(complete_list)
    plt.show()
'''