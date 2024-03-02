import glob
import os
import yaml

import pandas as pd
import numpy as np
from sklearn.metrics import *
from scipy.stats import skew, kurtosis
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import warnings
import multiprocessing
import pickle
import gzip

def find_substrings_in_string(string_list, main_string):
    return [s for s in string_list if s in main_string]

def calculate_log_returns(series, step=50):
    return np.log(series / series.shift(step)).dropna().reset_index(drop=True)

def optimized_rolling_diff(series, window_size):
    return series.rolling(window=window_size).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False).shift(-(window_size - 1))

def process_file(f):
    df = pd.read_csv(f)

    best_ask_price = df.ASKp1 / 10000
    best_bid_price = df.BIDp1 / 10000
    local_mids = (best_ask_price + best_bid_price) / 2
    local_spreads = best_ask_price - best_bid_price
    volatility_10 = np.std(calculate_log_returns(local_mids, 10))
    volatility_50 = np.std(calculate_log_returns(local_mids, 50))
    volatility_100 = np.std(calculate_log_returns(local_mids, 100))
    levels_ask_side = ((df.ASKp10 / 10000 - df.ASKp1 / 10000) / 0.01).tolist()
    levels_bid_side = ((df.BIDp1 / 10000 - df.BIDp10 / 10000) / 0.01).tolist()
    df['seconds'] = pd.to_datetime(df['seconds'])
    secs = df['seconds'].astype(int) / 10**9

    seconds_in_horizon_10 = optimized_rolling_diff(secs, 10).dropna().tolist()
    seconds_in_horizon_50 = optimized_rolling_diff(secs, 50).dropna().tolist()
    seconds_in_horizon_100 = optimized_rolling_diff(secs, 100).dropna().tolist()

    print(f"Finished {f}.")
    return {
        'Mids': local_mids.tolist(),
        'Spreads': local_spreads.tolist(),
        'Best_Ask_Volume': df.ASKs1.tolist(),
        'Best_Bid_Volume': df.BIDs1.tolist(),
        'Volatility_10': [volatility_10],
        'Volatility_50': [volatility_50],
        'Volatility_100': [volatility_100],
        'Levels_Ask_Side': levels_ask_side,
        'Levels_Bid_Side': levels_bid_side,
        'Seconds_Horizon_10': seconds_in_horizon_10,
        'Seconds_Horizon_50': seconds_in_horizon_50,
        'Seconds_Horizon_100': seconds_in_horizon_100
    }

def process_stock_files(file_list):
    stock_data = {
        'Mids': [], 'Spreads': [], 'Best_Ask_Volume': [], 'Best_Bid_Volume': [],
        'Volatility_10': [], 'Volatility_50': [], 'Volatility_100': [],
        'Levels_Ask_Side': [], 'Levels_Bid_Side': [], 'Seconds_Horizon_10': [],
        'Seconds_Horizon_50': [], 'Seconds_Horizon_100': []
    }
    for f in file_list:
        file_data = process_file(f)
        for key in stock_data:
            stock_data[key].extend(file_data[key])
    return stock_data

def process_stock(s):
    files = sorted(glob.glob(f"../data/nasdaq/unscaled_data/{s}/*"))
    num_workers = 10

    # Splitting files into chunks for each process
    file_chunks = np.array_split(files, num_workers)

    with multiprocessing.Pool(num_workers) as pool:
        chunk_results = pool.map(process_stock_files, file_chunks)

    # Aggregating results from all chunks
    stock_data = {
        'Mids': [], 'Spreads': [], 'Best_Ask_Volume': [], 'Best_Bid_Volume': [],
        'Volatility_10': [], 'Volatility_50': [], 'Volatility_100': [],
        'Levels_Ask_Side': [], 'Levels_Bid_Side': [], 'Seconds_Horizon_10': [],
        'Seconds_Horizon_50': [], 'Seconds_Horizon_100': []
    }
    for chunk in chunk_results:
        for key in stock_data:
            stock_data[key].extend(chunk[key])

    return s, stock_data

if __name__ == "__main__":
    stocks = ["BAC", "CHTR", "CSCO", "GOOG", "GS", "IBM", "MCD", "NVDA", "ORCL", "PFE", "PM", "VZ"] #"ABBV", "KO", "AAPL", 

    for s in stocks:
        stock_dictionary = {}
        try:
            s, stock_data = process_stock(s)
            stock_dictionary[s] = stock_data
            print(f"Completed processing for stock: {s}")
        except Exception as e:
            print(f"Error processing stock: {s} with error {e}")

        with open(f'../statistical_analysis/{s}.pkl', 'wb') as f:
            pickle.dump(stock_dictionary, f, protocol=-1)