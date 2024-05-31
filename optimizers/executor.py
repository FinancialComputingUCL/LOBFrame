import shutil
from torch.utils.data import DataLoader
import torch

from loaders.custom_dataset import CustomDataset
from models.DeepLob.deeplob import DeepLOB
from models.iTransformer.itransformer import ITransformer
from models.Transformer.transformer import Transformer
from models.LobTransformer.lobtransformer import LobTransformer
from models.DLA.DLA import DLA
from models.CNN1.cnn1 import CNN1
from models.CNN2.cnn2 import CNN2
from models.AxialLob.axiallob import AxialLOB
from models.TABL.bin_tabl import BiN_BTABL, BiN_CTABL
from models.CompleteHCNN.complete_hcnn import Complete_HCNN
from optimizers.lightning_batch_gd import BatchGDManager
from loggers import logger
from utils import create_tree, get_training_test_stocks_as_string


class Executor:
    def __init__(self, experiment_id, general_hyperparameters, model_hyperparameters, torch_dataset_preparation=False, torch_dataset_preparation_backtest=False):
        self.manager = None
        self.model = None
        self.experiment_id = experiment_id
        self.torch_dataset_preparation = torch_dataset_preparation
        self.torch_dataset_preparation_backtest = torch_dataset_preparation_backtest

        self.training_stocks_string, self.test_stocks_string = get_training_test_stocks_as_string(general_hyperparameters)

        if self.torch_dataset_preparation:
            create_tree(f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/")

        if general_hyperparameters["model"] == "deeplob":
            self.model = DeepLOB(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "transformer":
            self.model = Transformer(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "itransformer":
            self.model = ITransformer(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "lobtransformer":
            self.model = LobTransformer(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "dla":
            self.model = DLA(lighten=model_hyperparameters["lighten"])
        elif general_hyperparameters["model"] == "cnn1":
            self.model = CNN1()
        elif general_hyperparameters["model"] == "cnn2":
            self.model = CNN2()
        elif general_hyperparameters["model"] == "binbtabl":
            self.model = BiN_BTABL(120, 40, 100, 5, 3, 1)
        elif general_hyperparameters["model"] == "binctabl":
            self.model = BiN_CTABL(120, 40, 100, 5, 120, 5, 3, 1)
        elif general_hyperparameters["model"] == "axiallob":
            self.model = AxialLOB()
        elif general_hyperparameters["model"] == "hlob":
            homological_structures = torch.load(f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/complete_homological_structures.pt")
            self.model = Complete_HCNN(lighten=model_hyperparameters["lighten"], homological_structures=homological_structures)
        
        if self.torch_dataset_preparation:
            # Prepare the training dataloader.
            dataset = CustomDataset(
                dataset=general_hyperparameters["dataset"],
                learning_stage="training",
                window_size=model_hyperparameters["history_length"],
                shuffling_seed=model_hyperparameters["shuffling_seed"],
                cache_size=1,
                lighten=model_hyperparameters["lighten"],
                threshold=model_hyperparameters["threshold"],
                all_horizons=general_hyperparameters["horizons"],
                prediction_horizon=model_hyperparameters["prediction_horizon"],
                targets_type=general_hyperparameters["targets_type"],
                balanced_dataloader=model_hyperparameters["balanced_sampling"],
                training_stocks=general_hyperparameters["training_stocks"],
                validation_stocks=general_hyperparameters["target_stocks"],
                target_stocks=general_hyperparameters["target_stocks"]
            )
            torch.save(dataset, f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/training_dataset.pt")
        elif self.torch_dataset_preparation is False and self.torch_dataset_preparation_backtest is False:
            dataset = torch.load(f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/training_dataset.pt")
            self.train_loader = DataLoader(
                dataset,
                batch_size=model_hyperparameters["batch_size"],
                shuffle=False,
                num_workers=model_hyperparameters["num_workers"],
                sampler=dataset.glob_indices,
            )

        if self.torch_dataset_preparation:
            # Prepare the validation dataloader.
            dataset = CustomDataset(
                dataset=general_hyperparameters["dataset"],
                learning_stage="validation",
                window_size=model_hyperparameters["history_length"],
                shuffling_seed=model_hyperparameters["shuffling_seed"],
                cache_size=1,
                lighten=model_hyperparameters["lighten"],
                threshold=model_hyperparameters["threshold"],
                all_horizons=general_hyperparameters["horizons"],
                targets_type=general_hyperparameters["targets_type"],
                prediction_horizon=model_hyperparameters["prediction_horizon"],
                training_stocks=general_hyperparameters["training_stocks"],
                validation_stocks=general_hyperparameters["target_stocks"],
                target_stocks=general_hyperparameters["target_stocks"]
            )

            torch.save(dataset, f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/validation_dataset.pt")
        elif self.torch_dataset_preparation is False and self.torch_dataset_preparation_backtest is False:
            dataset = torch.load(f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/validation_dataset.pt")
            self.val_loader = DataLoader(
                dataset,
                batch_size=model_hyperparameters["batch_size"],
                shuffle=False,
                num_workers=model_hyperparameters["num_workers"],
            )

        if self.torch_dataset_preparation is False and self.torch_dataset_preparation_backtest:
            dataset = CustomDataset(
                dataset=general_hyperparameters["dataset"],
                learning_stage="test",
                window_size=model_hyperparameters["history_length"],
                shuffling_seed=model_hyperparameters["shuffling_seed"],
                cache_size=1,
                lighten=model_hyperparameters["lighten"],
                threshold=model_hyperparameters["threshold"],
                all_horizons=general_hyperparameters["horizons"],
                targets_type=general_hyperparameters["targets_type"],
                prediction_horizon=model_hyperparameters["prediction_horizon"],
                backtest=True,
                training_stocks=general_hyperparameters["training_stocks"],
                validation_stocks=general_hyperparameters["target_stocks"],
                target_stocks=general_hyperparameters["target_stocks"]
            )
            torch.save(dataset, f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/test_dataset_backtest.pt")
        elif self.torch_dataset_preparation and self.torch_dataset_preparation_backtest is False:
            dataset = CustomDataset(
                dataset=general_hyperparameters["dataset"],
                learning_stage="test",
                window_size=model_hyperparameters["history_length"],
                shuffling_seed=model_hyperparameters["shuffling_seed"],
                cache_size=1,
                lighten=model_hyperparameters["lighten"],
                threshold=model_hyperparameters["threshold"],
                all_horizons=general_hyperparameters["horizons"],
                targets_type=general_hyperparameters["targets_type"],
                prediction_horizon=model_hyperparameters["prediction_horizon"],
                training_stocks=general_hyperparameters["training_stocks"],
                validation_stocks=general_hyperparameters["target_stocks"],
                target_stocks=general_hyperparameters["target_stocks"]
            )
            torch.save(dataset, f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/test_dataset.pt")
        elif self.torch_dataset_preparation is False and self.torch_dataset_preparation_backtest is False:
            dataset = torch.load(f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{self.training_stocks_string}_test_{self.test_stocks_string}/{model_hyperparameters['prediction_horizon']}/test_dataset.pt")
            self.test_loader = DataLoader(
                dataset,
                batch_size=model_hyperparameters["batch_size"],
                shuffle=False,
                num_workers=model_hyperparameters["num_workers"],
            )

        if self.torch_dataset_preparation is False and self.torch_dataset_preparation_backtest is False:
            self.manager = BatchGDManager(
                experiment_id=experiment_id,
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                epochs=model_hyperparameters["epochs"],
                learning_rate=model_hyperparameters["learning_rate"],
                patience=model_hyperparameters["patience"],
                general_hyperparameters=general_hyperparameters,
                model_hyperparameters=model_hyperparameters,
            )

    def execute_training(self):
        self.manager.train()

    def execute_testing(self):
        self.manager.test()

    def logger_clean_up(self):
        folder_path = f"{logger.find_save_path(self.experiment_id)}/wandb/"
        try:
            shutil.rmtree(folder_path)
        except:
            pass


