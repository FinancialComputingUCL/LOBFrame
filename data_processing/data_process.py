import datetime
import glob
import random
import re
from datetime import datetime, time

import numpy as np
import pandas as pd


def process_data(
        ticker: str,
        input_path: str,
        output_path: str,
        logs_path: str,
        horizons: list[int],
        normalization_window: int,
        time_index: str = "seconds",
        features: str = "orderbooks",
        scaling: bool = True,
) -> None:
    """
    Function to pre-process LOBSTER data. The data must be stored in the input_path directory as 'daily message LOB' and 'orderbook' files.

    The data are treated in the following way:
    - Orderbook's states with crossed quotes are removed.
    - Each state in the orderbook is time-stamped, with states occurring at the same time collapsed onto the last occurring state.
    - The first and last 10 minutes of market activity (inside usual opening times) are dropped.
    - Rolling z-score normalization is applied to the data, i.e. the mean and standard deviation of the previous 5 days is used to normalize current day's data.
      Hence, the first 5 days are dropped.
    - Smoothed returns at the requested horizons (in orderbook's changes) are returned:
       - if smoothing = "horizon": l = (m+ - m)/m, where m+ denotes the mean of the next h mid-prices, m(.) is current mid-price.
       - if smoothing = "uniform": l = (m+ - m)/m, where m+ denotes the mean of the k+1 mid-prices centered at m(. + h), m(.) is current mid-price.

    A log file is produced tracking:
    - Orderbook's files with problems.
    - Message orderbook's files with problems.
    - Trading days with unusual opening - closing times.
    - Trading days with crossed quotes.

    A statistics.csv file summarizes the following (daily) statistics:
    - # Updates (000): the total number of changes in the orderbook file.
    - # Trades (000): the total number of trades, computed by counting the number of message book events corresponding to the execution of (possibly hidden)
                      limit orders ('event_type' 4 or 5 in LOBSTER orderbook's message file).
    - # Price Changes (000): the total number of price changes per day.
    - # Price (USD): average price on the day, weighted average by time.
    - # Spread (bps): average spread on the day, weighted average by time.
    - # Volume (USD MM): total volume traded on the day, computed as the sum of the volumes of all the executed trades ('event_type' 4 or 5 in LOBSTER orderbook's message file).
                         The volume of a single trade is given by size*price.
    - # Tick size: the fraction of time that the bid-ask spread is equal to one tick for each stock.

    Args:
        ticker (str): The ticker to be considered.
        input_path (str): The path where the order book and message book files are stored, order book files have shape (:, 4*levels):
                       ["ASKp1", "ASKs1", "BIDp1",  "BIDs1", ..., "ASKp10", "ASKs10", "BIDp10",  "BIDs10"].
        output_path (str): The path where we wish to save the processed datasets.
        logs_path (str): The path where we wish to save the logs.
        time_index (str): The time-index to use ("seconds" or "datetime").
        horizons (list): Forecasting horizons for labels.
        normalization_window (int): Window for rolling z-score normalization.
        features (str): Whether to return 'orderbooks' or 'orderflows'.
        scaling (bool): Whether to apply rolling z-score normalization.

    Returns:
        None.
    """

    csv_file_list = glob.glob(
        f"{input_path}/*.csv"
    )  # Get the list of all the .csv files in the input_path directory.

    csv_orderbook = [
        name for name in csv_file_list if "orderbook" in name
    ]  # Get the list of all the orderbook files in the input_path directory.
    csv_orderbook.sort()  # Sort the list of orderbook files.
    csv_message = [
        name for name in csv_file_list if "message" in name
    ]  # Get the list of all the message files in the input_path directory.
    csv_message.sort()  # Sort the list of message files.

    # Check if exactly half of the files are order book and exactly half are messages.
    assert len(csv_message) == len(csv_orderbook)
    assert len(csv_file_list) == len(csv_message) + len(csv_orderbook)

    print(f"Data preprocessing loop started. SCALING: {str(scaling)}.")

    # Initialize the dataframe containing logs.
    logs = []
    df_statistics = pd.DataFrame(
        [],
        columns=[
            "Updates (000)",
            "Trades (000)",
            "Price Changes (000)",
            "Price (USD)",
            "Spread (bps)",
            "Volume (USD MM)",
            "Tick Size",
        ],
        dtype=float,
    )

    # Initialize dataframes for dynamic Z-score normalization.
    mean_df = pd.DataFrame()
    mean2_df = pd.DataFrame()
    nsamples_df = pd.DataFrame()

    for orderbook_name in csv_orderbook:
        print(orderbook_name)

        # Read orderbook files and keep a record of problematic files.
        df_orderbook = None
        try:
            df_orderbook = pd.read_csv(orderbook_name, header=None)
        except:
            logs.append(f"{orderbook_name} skipped. Error: failed to read orderbook.")

        levels = int(
            df_orderbook.shape[1] / 4
        )  # Verify that the number of columns is a multiple of 4.
        feature_names_raw = [
            "ASKp",
            "ASKs",
            "BIDp",
            "BIDs",
        ]  # Define sorted raw features' names.
        feature_names = []
        for i in range(1, levels + 1):
            for j in range(4):
                feature_names += [
                    feature_names_raw[j] + str(i)
                ]  # Add to raw features' names the level number.
        df_orderbook.columns = (
            feature_names  # Rename the columns of the orderbook dataframe.
        )

        # Divide prices by 10000.
        target_columns = [col for col in df_orderbook.columns if "ASKp" in col or "BIDp" in col]
        df_orderbook[target_columns] = df_orderbook[target_columns].astype(int)  # / 10000

        df_orderbook.insert(
            0, "mid_price", (df_orderbook["ASKp1"] + df_orderbook["BIDp1"]) / 2
        )  # Add the mid-price column to the orderbook dataframe.
        df_orderbook.mid_price = df_orderbook.mid_price.astype(int)

        # Extract the date from the orderbook file's name.
        match = re.findall(r"\d{4}-\d{2}-\d{2}", orderbook_name)[-1]
        date = datetime.strptime(match, "%Y-%m-%d")

        # Read message files and keep a record of problematic files.
        message_name = orderbook_name.replace("orderbook", "message")
        df_message = None
        try:
            df_message = pd.read_csv(
                message_name, usecols=[0, 1, 2, 3, 4, 5], header=None
            )
        except:
            logs.append(f"{message_name} skipped. Error: failed to read message file.")

        # Check the two dataframes created before have the same length.
        assert len(df_message) == len(df_orderbook)

        # Rename the columns of the message dataframe.
        df_message.columns = [
            "seconds",
            "event_type",
            "order ID",
            "volume",
            "price",
            "direction",
        ]

        # Remove trading halts.
        trading_halts_start = df_message[
            (df_message["event_type"] == 7) & (df_message["price"] == -1)
            ].index
        trading_halts_end = df_message[
            (df_message["event_type"] == 7) & (df_message["price"] == 1)
            ].index
        trading_halts_index = np.array([])
        for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
            trading_halts_index = np.append(
                trading_halts_index,
                df_message.index[
                    (df_message.index >= halt_start) & (df_message.index < halt_end)
                    ],
            )
        if len(trading_halts_index) > 0:
            for halt_start, halt_end in zip(trading_halts_start, trading_halts_end):
                logs.append(
                    f"Warning: trading halt between {str(df_message.loc[halt_start, 'seconds'])} and {str(df_message.loc[halt_end, 'seconds'])} in {orderbook_name}."
                )
        df_orderbook = df_orderbook.drop(trading_halts_index)
        df_message = df_message.drop(trading_halts_index)

        # Remove crossed quotes.
        crossed_quotes_index = df_orderbook[
            (df_orderbook["BIDp1"] > df_orderbook["ASKp1"])
        ].index
        if len(crossed_quotes_index) > 0:
            logs.append(
                f"Warning: {str(len(crossed_quotes_index))} crossed quotes removed in {orderbook_name}."
            )
        df_orderbook = df_orderbook.drop(crossed_quotes_index)
        df_message = df_message.drop(crossed_quotes_index)

        # Add the 'seconds since midnight' column to the orderbook from the message book.
        df_orderbook.insert(0, "seconds", df_message["seconds"])

        # One conceptual event (e.g. limit order modification which is implemented as a cancellation followed by an immediate new arrival,
        # single market order executing against multiple resting limit orders) may appear as multiple rows in the message file, all with
        # the same timestamp. We hence group the order book data by unique timestamps and take the last entry.
        df_orderbook = df_orderbook.groupby(["seconds"]).tail(1)
        df_message = df_message.groupby(["seconds"]).tail(1)

        # Check market opening times for strange values.
        market_open = (int(df_orderbook["seconds"].iloc[0] / 60) / 60)  # Open at minute before first transaction.
        market_close = (int(df_orderbook["seconds"].iloc[-1] / 60) + 1) / 60  # Close at minute after last transaction.

        if not (market_open == 9.5 and market_close == 16):
            logs.append(
                f"Warning: unusual opening times in {orderbook_name}: {str(market_open)} - {str(market_close)}."
            )

        if time_index == "seconds":
            # Drop values outside of market hours using seconds
            df_orderbook = df_orderbook.loc[
                (df_orderbook["seconds"] >= 34200) & (df_orderbook["seconds"] <= 57600)
                ]
            df_message = df_message.loc[
                (df_message["seconds"] >= 34200) & (df_message["seconds"] <= 57600)
                ]

            # Drop first and last 10 minutes of trading using seconds.
            market_open_seconds = market_open * 60 * 60 + 10 * 60
            market_close_seconds = market_close * 60 * 60 - 10 * 60
            df_orderbook = df_orderbook.loc[
                (df_orderbook["seconds"] >= market_open_seconds)
                & (df_orderbook["seconds"] <= market_close_seconds)
                ]
            df_message = df_message.loc[
                (df_message["seconds"] >= market_open_seconds)
                & (df_message["seconds"] <= market_close_seconds)
                ]
        else:
            raise Exception("time_index must be seconds.")

        # Save statistical information.
        if len(df_orderbook) > 0:
            updates = df_orderbook.shape[0] / 1000
            trades = (
                    np.sum(
                        (df_message["event_type"] == 4) | (df_message["event_type"] == 5)
                    )
                    / 1000
            )
            price_changes = np.sum(~(np.diff(df_orderbook["mid_price"]) == 0.0)) / 1000
            time_deltas = np.append(
                np.diff(df_orderbook["seconds"]),
                market_close_seconds - df_orderbook["seconds"].iloc[-1],
            )
            price = np.average(df_orderbook["mid_price"] / 10 ** 4, weights=time_deltas)
            spread = np.average(
                (df_orderbook["ASKp1"] - df_orderbook["BIDp1"])
                / df_orderbook["mid_price"]
                * 10000,
                weights=time_deltas,
            )
            volume = (
                    np.sum(
                        df_message.loc[
                            (df_message["event_type"] == 4)
                            | (df_message["event_type"] == 5)
                            ]["volume"]
                        * df_message.loc[
                            (df_message["event_type"] == 4)
                            | (df_message["event_type"] == 5)
                            ]["price"]
                        / 10 ** 4
                    )
                    / 10 ** 6
            )
            tick_size = np.average(
                (df_orderbook["ASKp1"] - df_orderbook["BIDp1"]) == 100.0,
                weights=time_deltas,
            )

            df_statistics.loc[date] = [
                updates,
                trades,
                price_changes,
                price,
                spread,
                volume,
                tick_size,
            ]

        if features == "orderbooks":
            pass
        elif features == "orderflows":
            # Compute bid and ask multilevel orderflow.
            ASK_prices = df_orderbook.loc[:, df_orderbook.columns.str.contains("ASKp")]
            BID_prices = df_orderbook.loc[:, df_orderbook.columns.str.contains("BIDp")]
            ASK_sizes = df_orderbook.loc[:, df_orderbook.columns.str.contains("ASKs")]
            BID_sizes = df_orderbook.loc[:, df_orderbook.columns.str.contains("BIDs")]

            ASK_price_changes = ASK_prices.diff().dropna().to_numpy()
            BID_price_changes = BID_prices.diff().dropna().to_numpy()
            ASK_size_changes = ASK_sizes.diff().dropna().to_numpy()
            BID_size_changes = BID_sizes.diff().dropna().to_numpy()

            ASK_sizes = ASK_sizes.to_numpy()
            BID_sizes = BID_sizes.to_numpy()

            ASK_OF = (
                    (ASK_price_changes > 0.0) * (-ASK_sizes[:-1, :])
                    + (ASK_price_changes == 0.0) * ASK_size_changes
                    + (ASK_price_changes < 0) * ASK_sizes[1:, :]
            )
            BID_OF = (
                    (BID_price_changes < 0.0) * (-BID_sizes[:-1, :])
                    + (BID_price_changes == 0.0) * BID_size_changes
                    + (BID_price_changes > 0) * BID_sizes[1:, :]
            )

            # Remove all price-volume features and add in orderflow.
            df_orderbook = df_orderbook.drop(feature_names, axis=1).iloc[1:, :]
            mid_seconds_columns = list(df_orderbook.columns)
            feature_names_raw = ["ASK_OF", "BID_OF"]
            feature_names = []
            for feature_name in feature_names_raw:
                for i in range(1, levels + 1):
                    feature_names += [feature_name + str(i)]
            df_orderbook[feature_names] = np.concatenate([ASK_OF, BID_OF], axis=1)

            # Re-order columns.
            feature_names_reordered = [[]] * len(feature_names)
            feature_names_reordered[::2] = feature_names[:levels]
            feature_names_reordered[1::2] = feature_names[levels:]
            feature_names = feature_names_reordered

            df_orderbook = df_orderbook[mid_seconds_columns + feature_names]
        else:
            raise ValueError("Features must be 'orderbooks' or 'orderflows'.")

        # Dynamic z-score normalization.
        orderbook_mean_df = pd.DataFrame(
            df_orderbook[feature_names].mean().values.reshape(-1, len(feature_names)),
            columns=feature_names,
        )
        orderbook_mean2_df = pd.DataFrame(
            (df_orderbook[feature_names] ** 2)
            .mean()
            .values.reshape(-1, len(feature_names)),
            columns=feature_names,
        )
        orderbook_nsamples_df = pd.DataFrame(
            np.array([[len(df_orderbook)]] * len(feature_names)).T,
            columns=feature_names,
        )

        if len(mean_df) < normalization_window:
            logs.append(
                f"{orderbook_name} skipped. Initializing rolling z-score normalization."
            )
            # Don't save the first <normalization_window> days as we don't have enough days to normalize.
            mean_df = pd.concat([mean_df, orderbook_mean_df], ignore_index=True)
            mean2_df = pd.concat([mean2_df, orderbook_mean2_df], ignore_index=True)
            nsamples_df = pd.concat(
                [nsamples_df, orderbook_nsamples_df], ignore_index=True
            )
            continue
        else:
            z_mean_df = pd.DataFrame(
                (nsamples_df * mean_df).sum(axis=0) / nsamples_df.sum(axis=0)
            ).T  # Dynamically compute mean.
            z_stdev_df = pd.DataFrame(
                np.sqrt(
                    (nsamples_df * mean2_df).sum(axis=0) / nsamples_df.sum(axis=0)
                    - z_mean_df ** 2
                )
            )  # Dynamically compute standard deviation.

            # Broadcast to df_orderbook size.
            z_mean_df = z_mean_df.loc[z_mean_df.index.repeat(len(df_orderbook))]
            z_stdev_df = z_stdev_df.loc[z_stdev_df.index.repeat(len(df_orderbook))]
            z_mean_df.index = df_orderbook.index
            z_stdev_df.index = df_orderbook.index
            if scaling is True:
                df_orderbook[feature_names] = (df_orderbook[feature_names] - z_mean_df) / z_stdev_df  # Apply normalization.

            # Roll forward by dropping first rows and adding most recent mean and mean2.
            mean_df = mean_df.iloc[1:, :]
            mean2_df = mean2_df.iloc[1:, :]
            nsamples_df = nsamples_df.iloc[1:, :]

            mean_df = pd.concat([mean_df, orderbook_mean_df], ignore_index=True)
            mean2_df = pd.concat([mean2_df, orderbook_mean2_df], ignore_index=True)
            nsamples_df = pd.concat(
                [nsamples_df, orderbook_nsamples_df], ignore_index=True
            )

        # Create labels with simple delta prices.
        rolling_mid = df_orderbook["mid_price"]
        rolling_mid = rolling_mid.to_numpy().flatten()
        for h in horizons:
            delta_ticks = rolling_mid[h:] - df_orderbook["mid_price"][:-h]
            df_orderbook[f"Raw_Target_{str(h)}"] = delta_ticks

        # Create labels applying smoothing.
        for h in horizons:
            rolling_mid_minus = df_orderbook['mid_price'].rolling(window=h, min_periods=h).mean().shift(h)
            rolling_mid_plus = df_orderbook["mid_price"].rolling(window=h, min_periods=h).mean().to_numpy().flatten()
            smooth_pct_change = rolling_mid_plus - rolling_mid_minus
            df_orderbook[f"Smooth_Target_{str(h)}"] = smooth_pct_change

        # Drop the mid-price column and transform seconds column into a readable format.
        df_orderbook = df_orderbook.drop(["mid_price"], axis=1)
        pattern = r"\d{4}-\d{2}-\d{2}"
        match = re.search(pattern, orderbook_name)
        date_temp = match.group()
        df_orderbook.seconds = df_orderbook.apply(
            lambda row: get_datetime_from_seconds(row["seconds"], date_temp), axis=1
        )

        # Drop elements which cannot be used for training.
        df_orderbook = df_orderbook.dropna()
        df_orderbook.drop_duplicates(inplace=True, keep='last', subset='seconds')

        # Save processed files.
        output_name = f"{output_path}/{ticker}_{features}_{str(date.date())}"
        df_orderbook.to_csv(f"{output_name}.csv", header=True, index=False)

        logs.append(f"{orderbook_name} completed.")

    print(f"Data preprocessing loop finished. SCALING: {str(scaling)}.")

    with open(f"{logs_path}/{features}_processing_logs.txt", "w") as f:
        for log in logs:
            f.write(log + "\n")

    print("Please check processing logs.")

    df_statistics.to_csv(
        f"{logs_path}/{features}_statistics.csv", header=True, index=False
    )  # Save statistics.


def get_datetime_from_seconds(seconds_after_midnight, date_str):
    # Convert the date_str to a datetime.date object.
    dt_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    # Calculate the time component from seconds_after_midnight.
    hours = int(seconds_after_midnight // 3600)
    minutes = int((seconds_after_midnight % 3600) // 60)
    seconds = int(seconds_after_midnight % 60)
    microseconds = int(
        (seconds_after_midnight % 1) * 1e6
    )  # Convert decimal part to microseconds.

    # Create a datetime.time object for the time component.
    dt_time = time(hour=hours, minute=minutes, second=seconds, microsecond=microseconds)

    # Combine the date and time to create the datetime.datetime object.
    dt_datetime = datetime.combine(dt_date, dt_time)

    return dt_datetime
