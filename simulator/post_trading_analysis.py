from itertools import cycle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from loggers import logger
from utils import get_training_test_stocks_as_string


def __get_fees_free_pnl__(trading_simulation):
    df = trading_simulation
    profit_list = []
    for index, row in df.iterrows():
        profit_no_fees = 0
        if row.Type == 'Long':
            local_profit = (row.Price_Exit_Long - row.Price_Entry_Long)
            profit_no_fees += local_profit
        elif row.Type == 'Short':
            local_profit = (row.Price_Entry_Short - row.Price_Exit_Short)
            profit_no_fees += local_profit

        profit_list.append(profit_no_fees)
    return profit_list


def __get_pnl_with_fees__(trading_simulation, trading_hyperparameters):
    df = trading_simulation
    profit_list = []
    for index, row in df.iterrows():
        profit_no_fees = 0
        if row.Type == 'Long':
            local_profit = (row.Price_Exit_Long - row.Price_Entry_Long) - (row.Price_Exit_Long * trading_hyperparameters['trading_fee']) - (row.Price_Entry_Long * trading_hyperparameters['trading_fee'])
            profit_no_fees += local_profit
        elif row.Type == 'Short':
            local_profit = (row.Price_Entry_Short - row.Price_Exit_Short) - (row.Price_Entry_Short * trading_hyperparameters['trading_fee']) - (row.Price_Exit_Short * trading_hyperparameters['trading_fee'])
            profit_no_fees += local_profit

        profit_list.append(profit_no_fees)
    return profit_list


def __get_long_short_indices__(trading_simulation):
    long_indices = []
    short_indices = []
    for index, row in trading_simulation.iterrows():
        if row.Type == 'Long':
            long_indices.append(pd.to_datetime(row.Entry_Long))
        elif row.Type == 'Short':
            short_indices.append(pd.to_datetime(row.Entry_Short))

    return long_indices, short_indices


def post_trading_analysis(experiment_id, general_hyperparameters, trading_hyperparameters, model_hyperparameters):
    prediction = pd.read_pickle(f"{logger.find_save_path(experiment_id)}/prediction.pkl")
    trading_simulation = pd.read_pickle(f"{logger.find_save_path(experiment_id)}/trading_simulation.pkl")

    training_stocks_string, test_stocks_string = get_training_test_stocks_as_string(general_hyperparameters)

    dataset = torch.load(
        f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{training_stocks_string}_test_{test_stocks_string}/{model_hyperparameters['prediction_horizon']}/test_dataset_backtest.pt")
    print(f"Reading test (backtest version) dataset...")
    test_loader = DataLoader(
        dataset,
        batch_size=model_hyperparameters["batch_size"],
        shuffle=False,
        num_workers=model_hyperparameters["num_workers"],
    )
    returns_labels_list = []
    for data, labels in tqdm(test_loader):
        returns_labels_list.extend(labels.tolist())

    targets = prediction[2].tolist()
    predictions = prediction[3].tolist()

    print(classification_report(targets, predictions))

    distributions_dataset = pd.DataFrame({"Predictions": predictions, "PCs": returns_labels_list})
    distribution_label_0 = distributions_dataset[distributions_dataset['Predictions'] == 0].PCs
    distribution_label_1 = distributions_dataset[distributions_dataset['Predictions'] == 1].PCs
    distribution_label_2 = distributions_dataset[distributions_dataset['Predictions'] == 2].PCs

    plt.hist(distribution_label_0, label='Label 0', alpha=0.5, bins=10)
    plt.hist(distribution_label_1, label='Label 1', alpha=0.5, bins=10)
    plt.hist(distribution_label_2, label='Label 2', alpha=0.5, bins=10)

    plt.title("Predictions' distribution")
    plt.xlabel("PCs Values")
    plt.ylabel("Frequency")
    plt.legend(title="Labels")
    plt.show()

    label_binarizer = LabelBinarizer().fit(targets)
    y_onehot_test = label_binarizer.transform(targets)
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    fig, ax = plt.subplots(figsize=(10, 8))
    for class_id, color in zip(range(0, 3), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            prediction[-1][:, class_id],
            name=f"ROC curve for class: {class_id}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2),
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()

    print(f"Matthews Correlation Coefficient: {round(matthews_corrcoef(targets, predictions), 2)}")
    print(f"Macro-average AUC-ROC (ovr): {round(roc_auc_score(targets, prediction[-1].tolist(), average='macro', multi_class='ovr'), 2)}")
    print(f"Macro-average AUC-ROC (ovo): {round(roc_auc_score(targets, prediction[-1].tolist(), average='macro', multi_class='ovo'), 2)}")
    print(f"Top-k (with k=2) Accuracy Score: {round(top_k_accuracy_score(targets, prediction[-1], k=2), 2)}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax in axs.flat:
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])

    # Confusion matrix plot.
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot(ax=axs[0, 0], cmap='Blues')
    axs[0, 0].set_title('Confusion Matrix')

    # P&L distribution plot.
    ax = fig.add_subplot(2, 2, 2)
    if trading_hyperparameters['simulation_type'] == 'no_fees':
        plt.hist(__get_fees_free_pnl__(trading_simulation), bins=30)
    elif trading_hyperparameters['simulation_type'] == 'with_fees':
        plt.hist(__get_pnl_with_fees__(trading_simulation, trading_hyperparameters), bins=30)
    axs[0, 1].set_title('P&L Distribution')

    # P&L cumsum plot.
    ax = fig.add_subplot(2, 2, 3)
    if trading_hyperparameters['simulation_type'] == 'no_fees':
        plt.plot(np.cumsum(__get_fees_free_pnl__(trading_simulation)))
    elif trading_hyperparameters['simulation_type'] == 'with_fees':
        plt.plot(np.cumsum(__get_pnl_with_fees__(trading_simulation, trading_hyperparameters)))
    axs[1, 0].set_title('P&L cumsum')

    # Mid price
    df = prediction[0].reset_index(drop=True)
    seconds = pd.to_datetime(df.seconds)
    mid = (df.BIDp1 + df.ASKp1) / 2
    trading_df = pd.DataFrame()
    trading_df['seconds'] = seconds
    trading_df['mid'] = mid

    long_indices, short_indices = __get_long_short_indices__(trading_simulation)
    trading_df.drop_duplicates(inplace=True, keep='first', subset='seconds')
    trading_df.set_index('seconds', inplace=True)

    ax = fig.add_subplot(2, 2, 4)
    plt.plot(trading_df.mid)
    for datetime in long_indices:
        y_value = trading_df.loc[datetime, 'mid']
        ax.plot(datetime, y_value, marker='^', color='green', markersize=5)
    for datetime in short_indices:
        y_value = trading_df.loc[datetime, 'mid']
        ax.plot(datetime, y_value, marker='v', color='red', markersize=5)

    axs[1, 1].set_title('Mid price')

    plt.tight_layout()
    plt.show()