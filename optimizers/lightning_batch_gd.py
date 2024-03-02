import copy
import os
import pickle
import wandb
import shutil
import stat

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim
from torchmetrics import Accuracy, F1Score
from lightning.pytorch.loggers import WandbLogger

from loggers import logger
from utils import get_best_levels_prices_and_labels, wandb_hyperparameters_saving
import sys

class LOBLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        experiment_id,
        learning_rate,
        general_hyperparameters,
        model_hyperparameters,
    ):
        super().__init__()
        self.model = model
        self.experiment_id = experiment_id
        self.learning_rate = learning_rate
        self.general_hyperparameters = general_hyperparameters
        self.model_hyperparameters = model_hyperparameters

        self.loss = nn.CrossEntropyLoss()

        self.training_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.training_f1 = F1Score(task="multiclass", num_classes=3, average="macro")
        self.validation_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.validation_f1 = F1Score(task="multiclass", num_classes=3, average="macro")

        self.batch_loss_training = []
        self.batch_accuracies_training = []
        self.batch_f1_scores_training = []
        self.batch_loss_validation = []
        self.batch_accuracies_validation = []
        self.batch_f1_scores_validation = []
        self.batch_loss_test = []
        self.test_outputs = []
        self.test_targets = []
        self.test_probs = []

        self.csv_path = f"{logger.find_save_path(experiment_id)}/metrics.csv"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss(logits, targets)
        outputs = nn.functional.softmax(logits, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        train_acc = self.training_accuracy(outputs, targets)
        train_f1 = self.training_f1(outputs, targets)

        self.batch_loss_training.append(loss.item())
        self.batch_accuracies_training.append(train_acc.item())
        self.batch_f1_scores_training.append(train_f1.item())

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss(logits, targets)
        outputs = nn.functional.softmax(logits, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        val_acc = self.validation_accuracy(outputs, targets)
        val_f1 = self.validation_f1(outputs, targets)

        self.batch_loss_validation.append(loss.item())
        self.batch_accuracies_validation.append(val_acc.item())
        self.batch_f1_scores_validation.append(val_f1.item())

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.loss(logits, targets)
        outputs = nn.functional.softmax(logits, dim=1)

        saving_probs = copy.copy(outputs)
        self.test_probs.extend(saving_probs.tolist())

        outputs = torch.argmax(outputs, dim=1).tolist()
        self.test_outputs.extend(outputs)
        self.test_targets.extend(targets.tolist())

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.model_hyperparameters["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=1e-1,
        )
        return optimizer

    def on_validation_epoch_end(self):
        # Calculate the average accuracy, F1, and MCC for the epoch (assuming you have them stored in a list).
        avg_loss_training = np.mean(
            self.batch_loss_training
        )  # Average of the batch-level losses (training).
        avg_accuracy_training = np.mean(
            self.batch_accuracies_training
        )  # Average of the batch-level accuracies (training).
        avg_f1_score_training = np.mean(
            self.batch_f1_scores_training
        )  # Average of the batch-level F1 scores (training).

        avg_loss_validation = np.mean(
            self.batch_loss_validation
        )  # Replace with your batch-level loss list (validation).
        avg_accuracy_validation = np.mean(
            self.batch_accuracies_validation
        )  # Replace with your batch-level accuracy list (validation).
        avg_f1_score_validation = np.mean(
            self.batch_f1_scores_validation
        )  # Replace with your batch-level F1 score list (validation).

        self.log(
            "loss",
            torch.tensor(avg_loss_training),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "acc",
            torch.tensor(avg_accuracy_training),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "f1",
            torch.tensor(avg_f1_score_training),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_loss",
            torch.tensor(avg_loss_validation),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_acc",
            torch.tensor(avg_accuracy_validation),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            torch.tensor(avg_f1_score_validation),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        self.training_accuracy.reset()
        self.training_f1.reset()
        self.validation_accuracy.reset()
        self.validation_f1.reset()

        # Append metrics to the list
        metrics_data = [
            avg_loss_training,
            avg_accuracy_training,
            avg_f1_score_training,
            avg_loss_validation,
            avg_accuracy_validation,
            avg_f1_score_validation,
        ]

        # Save metrics to a CSV file
        if not os.path.exists(self.csv_path):
            header = [
                "Training_Loss",
                "Training_Accuracy",
                "Training_F1",
                "Validation_Loss",
                "Validation_Accuracy",
                "Validation_F1",
            ]
            with open(self.csv_path, "w") as file:
                file.write(",".join(header) + "\n")

        with open(self.csv_path, "a") as file:
            file.write(",".join(map(str, metrics_data)) + "\n")

    def on_test_end(self):
        best_levels_prices, sanity_check_labels = get_best_levels_prices_and_labels(
            dataset=self.general_hyperparameters["dataset"],
            target_stocks=self.general_hyperparameters["target_stocks"],
            history_length=self.model_hyperparameters["history_length"],
            prediction_horizon=self.model_hyperparameters["prediction_horizon"],
            all_horizons=self.general_hyperparameters["horizons"],
            threshold=self.model_hyperparameters["threshold"],
        )
        with open(
            os.path.join(logger.find_save_path(self.experiment_id), "prediction.pkl"),
            "wb",
        ) as f:
            pickle.dump(
                [
                    best_levels_prices,
                    sanity_check_labels,
                    np.array(self.test_targets),
                    np.array(self.test_outputs),
                    np.array(self.test_probs),
                ],
                f,
            )


class BatchGDManager:
    def __init__(
        self,
        experiment_id,
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        learning_rate,
        patience,
        general_hyperparameters,
        model_hyperparameters,
    ):
        self.experiment_id = experiment_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.general_hyperparameters = general_hyperparameters
        self.model_hyperparameters = model_hyperparameters
        self.lob_lightning_module = None
        self.trainer = None
        self.deleted_run = None

    def delete_run(self):
        api = wandb.Api()
        project_path = "<Specify here the name of WB project>" # Specify here the name of WB project.
        runs = api.runs(path=project_path)
        print('Deleting runs...')
        while len(runs) < 1:
            runs = api.runs(path=project_path)
        for run in runs:
            input_list = run.metadata
            if input_list is not None:
                input_list = input_list['args']
                result_dict = {input_list[i][2:]: input_list[i + 1] for i in range(0, len(input_list), 2)}
                modified_dict = result_dict
                if modified_dict['model'] == str(self.general_hyperparameters['model']) and modified_dict['prediction_horizon'] == str(self.model_hyperparameters['prediction_horizon']) and modified_dict['training_stocks'] == str(self.general_hyperparameters['training_stocks'][0]) and modified_dict['target_stocks'] == str(self.general_hyperparameters['target_stocks'][0]):
                    self.deleted_run = run.name
                    run.delete()
                    print(f"Run succesfully deleted from WanDB: {run.name}.")

    def train(self):
        self.lob_lightning_module = LOBLightningModule(
            self.model,
            experiment_id=self.experiment_id,
            learning_rate=self.learning_rate,
            general_hyperparameters=self.general_hyperparameters,
            model_hyperparameters=self.model_hyperparameters,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=logger.find_save_path(self.experiment_id),
            filename="best_val_model",
            save_top_k=1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping("val_loss", patience=self.patience, min_delta=0.003)

        os.environ["WANDB_API_KEY"] = ""  # Insert API key
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        try:
            wandb_logger = WandbLogger(
                project="Limit_Order_Book",
                name=self.experiment_id,
                save_dir=logger.find_save_path(self.experiment_id),
            )
            wandb_hyperparameters_saving(
                wandb_logger=wandb_logger,
                general_hyperparameters=self.general_hyperparameters,
                model_hyperparameters=self.model_hyperparameters,
            )
            self.trainer = pl.Trainer(
                max_epochs=self.epochs,
                callbacks=[checkpoint_callback, early_stopping_callback],
                logger=wandb_logger,
                num_sanity_val_steps=0,
            )
            self.trainer.fit(self.lob_lightning_module, self.train_loader, self.val_loader)
            wandb.finish()
        except:
            root_path = sys.path[0]
            dir_path = f"{root_path}/loggers/results/{self.experiment_id}"
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Folder {self.experiment_id} deleted successfully.")
            else:
                print(f"Unable to delete folder {self.experiment_id}.")

            self.delete_run()

            model = self.general_hyperparameters['model']
            horizon = self.model_hyperparameters['prediction_horizon']
            training_stocks = self.general_hyperparameters['training_stocks']
            target_stocks = self.general_hyperparameters['target_stocks']
            errors_string = f"{model} {horizon} {training_stocks} {target_stocks} {self.deleted_run}\n"
            with open("errors.txt", 'r+') as file:
                content = file.read()

                # If the string does not exist in the file, append it
                if errors_string.strip() not in content:
                    # Move the cursor to the end of the file before appending
                    file.write(errors_string)
                    print("String appended successfully.")
                else:
                    print("String already exists in the file.")
            #raise Exception

    def test(self):
        if self.trainer is None:
            self.lob_lightning_module = LOBLightningModule(
                self.model,
                experiment_id=self.experiment_id,
                learning_rate=self.learning_rate,
                general_hyperparameters=self.general_hyperparameters,
                model_hyperparameters=self.model_hyperparameters,
            )
            self.trainer = pl.Trainer()
            try:
                best_model = self.lob_lightning_module.load_from_checkpoint(
                    checkpoint_path=f"{logger.find_save_path(self.experiment_id)}/best_val_model.ckpt",
                    model=self.model,
                    experiment_id=self.experiment_id,
                    learning_rate=self.learning_rate,
                    general_hyperparameters=self.general_hyperparameters,
                    model_hyperparameters=self.model_hyperparameters,
                )
            except:
                best_model = self.lob_lightning_module.load_from_checkpoint(
                    checkpoint_path=f"{logger.find_save_path(self.experiment_id)}/best_val_model.ckpt",
                    map_location=torch.device('cpu'),
                    model=self.model,
                    experiment_id=self.experiment_id,
                    learning_rate=self.learning_rate,
                    general_hyperparameters=self.general_hyperparameters,
                    model_hyperparameters=self.model_hyperparameters,
                )
            self.trainer.test(best_model, dataloaders=self.test_loader)
        else:
            best_model_path = (
                f"{logger.find_save_path(self.experiment_id)}/best_val_model.ckpt"
            )
            self.trainer.test(ckpt_path=best_model_path, dataloaders=self.test_loader)
