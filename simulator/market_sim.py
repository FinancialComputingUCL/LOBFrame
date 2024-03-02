import pandas as pd
import numpy as np
from tqdm import tqdm
from loggers import logger
from simulator.trading_agent import Trading
from utils import load_yaml
from datetime import timedelta


def __get_data__(experiment_id):
    path = f"{logger.find_save_path(experiment_id)}/prediction.pkl"
    all_prices, sanity_check_labels, all_targets, all_predictions, all_probs = pd.read_pickle(path)
    all_prices.reset_index(drop=True, inplace=True)
    return all_prices, sanity_check_labels, all_targets.tolist(), all_predictions.tolist(), all_probs


def backtest(experiment_id, trading_hyperparameters):
    prices, sanity_check_labels, targets, predictions, probs = __get_data__(experiment_id)
    TradingAgent = Trading(trading_hyperparameters)

    prices["Mid"] = (prices["BIDp1"] + prices["ASKp1"]) / 2
    prices["seconds"] = pd.to_datetime(prices["seconds"])

    prices['Predictions'] = predictions
    prices.reset_index(drop=True, inplace=True)
    indices_to_delete = prices[prices['Predictions'] == 1].index
    prices = prices.drop(indices_to_delete)
    mask = (prices['Predictions'] != prices['Predictions'].shift()) | (prices.index == 0)
    prices = prices[mask]
    prices = prices.reset_index(drop=True)
    predictions = prices['Predictions'].tolist()
    prices = prices.drop(columns=['Predictions'])
    prices.reset_index(drop=True, inplace=True)

    dates = prices['seconds'].dt.date
    day_changed_indices = dates.ne(dates.shift())
    new_day_indices = day_changed_indices.index[day_changed_indices].tolist()
    end_of_day_indices = [element - 1 for element in new_day_indices]
    end_of_day_indices.append(len(prices) - 1)
    end_of_day_indices = end_of_day_indices[1:]

    for i in tqdm(range(len(predictions))):
        mid_price = prices.at[i, "Mid"]
        best_bid_price = prices.at[i, "BIDp1"]
        best_ask_price = prices.at[i, "ASKp1"]
        timestamp = prices.at[i, "seconds"]
        prediction = predictions[i]
        probability = np.max(probs[i])

        if trading_hyperparameters['mid_side_trading'] == 'mid_to_mid':
            if i in end_of_day_indices:
                if TradingAgent.long_inventory > 0:
                    TradingAgent.exit_long(mid_price, timestamp)
                if TradingAgent.short_inventory > 0:
                    TradingAgent.exit_short(mid_price, timestamp)
            else:
                if prediction == 2 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(mid_price, timestamp)
                    elif TradingAgent.long_inventory == 0 and TradingAgent.short_inventory > 0:
                        TradingAgent.exit_short(mid_price, timestamp)
                        TradingAgent.long(mid_price, timestamp)
                elif prediction == 0 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(mid_price, timestamp)
                    elif TradingAgent.short_inventory == 0 and TradingAgent.long_inventory > 0:
                        TradingAgent.exit_long(mid_price, timestamp)
                        TradingAgent.short(mid_price, timestamp)
        elif trading_hyperparameters['mid_side_trading'] == 'side_market_orders':
            if i in end_of_day_indices:
                if TradingAgent.long_inventory > 0:
                    TradingAgent.exit_long(best_bid_price, timestamp)
                if TradingAgent.short_inventory > 0:
                    TradingAgent.exit_short(best_ask_price, timestamp)
            else:
                if prediction == 2 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(best_ask_price, timestamp)
                    elif TradingAgent.long_inventory == 0 and TradingAgent.short_inventory > 0:
                        TradingAgent.exit_short(best_ask_price, timestamp)
                        TradingAgent.long(best_ask_price, timestamp)
                elif prediction == 0 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(best_bid_price, timestamp)
                    elif TradingAgent.short_inventory == 0 and TradingAgent.long_inventory > 0:
                        TradingAgent.exit_long(best_bid_price, timestamp)
                        TradingAgent.short(best_bid_price, timestamp)
        elif trading_hyperparameters['mid_side_trading'] == 'side_limit_orders':
            if i in end_of_day_indices:
                if TradingAgent.long_inventory > 0:
                    TradingAgent.exit_long(best_ask_price, timestamp)
                if TradingAgent.short_inventory > 0:
                    TradingAgent.exit_short(best_bid_price, timestamp)
            else:
                if prediction == 2 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.long(best_bid_price, timestamp)
                    elif TradingAgent.long_inventory == 0 and TradingAgent.short_inventory > 0:
                        TradingAgent.exit_short(best_bid_price, timestamp)
                        TradingAgent.long(best_bid_price, timestamp)
                elif prediction == 0 and probability >= trading_hyperparameters['probability_threshold']:
                    if TradingAgent.long_inventory == 0 and TradingAgent.short_inventory == 0:
                        TradingAgent.short(best_ask_price, timestamp)
                    elif TradingAgent.short_inventory == 0 and TradingAgent.long_inventory > 0:
                        TradingAgent.exit_long(best_ask_price, timestamp)
                        TradingAgent.short(best_ask_price, timestamp)

    trading_history_dataframe = pd.DataFrame(TradingAgent.trading_history)
    save_path = f"{logger.find_save_path(experiment_id)}/trading_simulation.pkl"
    trading_history_dataframe.to_pickle(save_path)
