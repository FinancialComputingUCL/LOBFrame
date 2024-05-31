import glob
import concurrent.futures
from itertools import chain

import pandas as pd
import numpy as np
import polars as pl
from typing import *

import networkx as nx
from fast_tmfg import *
from sklearn.metrics import mutual_info_score

from utils import get_training_test_stocks_as_string
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def compute_pairwise_mi(df: pd.DataFrame, n_bins: int = 3000) -> pd.DataFrame:
    """
    Compute the pairwise mutual information between the columns of a dataframe.

    Parameters
    ----------
    df : pandas.Dataframe
        The pandas dataframe to compute the pairwise mutual information for.
    n_bins: int
        The number of bins to use for discretization.

    Returns
    ----------
    mi_matrix: pandas.Dataframe
        The pairwise mutual information matrix.

    """

    shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # Shuffle the dataset.
    sampled_df = shuffled_df.sample(n=len(df), replace=True)  # Perform bootstrapping.
    df = sampled_df.copy()  # Copy the dataset into a variable called 'df'.
    df.reset_index(drop=True, inplace=True)  # Reset the indices.
    del sampled_df  # Delete an unused variable.

    flat_series = df.values.flatten()  # Flat the df to perform a binning on all the values (not feature-by-feature).
    bins = pd.cut(flat_series, bins=n_bins, labels=False, retbins=True)  # Perform the binning.
    # Apply the binning to each feature of the original dataset.
    for column in df.columns:
        df[column] = pd.cut(df[column], bins=bins[1], labels=False, include_lowest=True)
    del flat_series  # Delete an unused variable.

    discretized_df = df.copy()  # Copy the dataset into a variable called 'discretized_df'.
    del df  # Delete an unused variable.

    # Initialize an empty Mutual Information (MI) matrix and fill it with 0s.
    n_features = discretized_df.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    # Compute the pairwise MI and fill the MI matrix consequently.
    for i in range(n_features):
        for j in range(i, n_features):
            mi_value = mutual_info_score(
                discretized_df.iloc[:, i], discretized_df.iloc[:, j]
            )
            mi_matrix[i, j] = mi_value
            mi_matrix[j, i] = mi_value

    mi_matrix = pd.DataFrame(mi_matrix)  # Transform the MI matrix into a Pandas dataframe.
    return mi_matrix  # Return the MI matrix in the form of a Pandas dataframe.


def process_file(
        file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, nx.Graph, nx.Graph, nx.Graph]:
    """
    Compute the TMFG for volumes of a given orderbook file.

    Parameters
    ----------
    file : str
        The path to the file to compute the TMFG for.

    Returns
    ----------
    sim_ask : pandas.DataFrame
        The pairwise mutual information matrix for the ask volumes.
    sim_bid : pandas.DataFrame
        The pairwise mutual information matrix for the bid volumes.
    sim_all : pandas.DataFrame
        The pairwise mutual information matrix for the ask and bid volumes.
    net_ask : networkx.Graph
        The TMFG for the ask volumes.
    net_bid : networkx.Graph
        The TMFG for the bid volumes.
    net_all : networkx.Graph
        The TMFG for the ask and bid volumes.
    """

    print(f"Computing structure for file: {file}...")
    # Read the file using polars to accelerate the process.
    df = pl.read_csv(file)
    df = df.to_pandas()

    # Extract the volumes for the ask and bid sides.
    volumes_all = df.iloc[:, 1:41].iloc[:, 1::2]

    # Compute the pairwise mutual information matrices.
    sim_all = compute_pairwise_mi(volumes_all)

    # Compute the TMFGs.
    model_all = TMFG()
    cliques_all, seps_all, adj_matrix_all = model_all.fit_transform(
        sim_all, output="weighted_sparse_W_matrix"
    )

    # Convert the adjacency matrices to networkx graphs.
    net_all = nx.from_numpy_array(adj_matrix_all)

    return sim_all, net_all, file


def mean_tmfg(sm_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Compute the average similarity matrix for a list of similarity matrices.

    Parameters
    ----------
    sm_list : List[pandas.DataFrame]
        The list of similarity matrices to compute the average for.

    Returns
    ----------
    average_matrix : pandas.DataFrame
        The average similarity matrix.
    """

    # Stack the matrices along a new axis (axis=0)
    stacked_matrices = np.stack(sm_list, axis=0)

    # Calculate the entry-wise average along the new axis
    average_matrix = np.mean(stacked_matrices, axis=0)
    np.fill_diagonal(average_matrix, 0)

    average_matrix = pd.DataFrame(average_matrix)

    '''
    plt.figure(figsize=(10, 8))  # Optional: Adjusts the size of the figure
    sns.heatmap(average_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title("Correlation Matrix Heatmap")
    plt.show()
    '''

    return average_matrix


def extract_components(
        cliques: List[List[int]], separators: List[List[int]], adjacency_matrix: np.ndarray
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Given the cliques, separators and adjacency matrix of a TMFG, extract the b-cliques of size 2 (edges), 3 (triangles) and 4 (tetrahedra).

    Parameters
    ----------
    cliques : List[int]
        The list of cliques of the TMFG.
    separators : List[int]
        The list of separators of the TMFG.
    adjacency_matrix : numpy.ndarray
        The adjacency matrix of the TMFG.

    Returns
    ----------
    final_b_cliques_4 : List[List[int]]
        The final list of tetrahera.
    final_b_cliques_3 : List[List[int]]
        The final list of triangles.
    final_b_cliques_2 : List[List[int]]
        The final list of edges.
    """

    # Extract edges.
    edges = []
    adjacency_matrix = nx.from_numpy_array(adjacency_matrix)

    for i in nx.enumerate_all_cliques(adjacency_matrix):
        if len(i) == 2:
            edges.append(sorted(i))

    b_cliques_4 = []
    b_cliques_3 = []
    b_cliques_2 = []

    b_cliques_all = nx.enumerate_all_cliques(adjacency_matrix)

    for i in b_cliques_all:
        if len(i) == 2:
            b_cliques_2.append(sorted(i))
        if len(i) == 3:
            b_cliques_3.append(sorted(i))
        if len(i) == 4:
            b_cliques_4.append(sorted(i))

    final_b_cliques_4 = b_cliques_4

    final_b_cliques_3 = b_cliques_3

    final_b_cliques_2 = edges

    final_b_cliques_4 = [[(x * 2) + 1 for x in sublist] for sublist in final_b_cliques_4]
    final_b_cliques_4 = [[x, x - 1] for sublist in final_b_cliques_4 for x in sublist]
    final_b_cliques_4 = list(chain.from_iterable(final_b_cliques_4))
    final_b_cliques_4 = [final_b_cliques_4[i:i + 8] for i in range(0, len(final_b_cliques_4), 8)]
    final_b_cliques_4 = [sorted(sublist) for sublist in final_b_cliques_4]

    final_b_cliques_3 = [[(x * 2) + 1 for x in sublist] for sublist in final_b_cliques_3]
    final_b_cliques_3 = [[x, x - 1] for sublist in final_b_cliques_3 for x in sublist]
    final_b_cliques_3 = list(chain.from_iterable(final_b_cliques_3))
    final_b_cliques_3 = [final_b_cliques_3[i:i + 6] for i in range(0, len(final_b_cliques_3), 6)]
    final_b_cliques_3 = [sorted(sublist) for sublist in final_b_cliques_3]

    final_b_cliques_2 = [[(x * 2) + 1 for x in sublist] for sublist in final_b_cliques_2]
    final_b_cliques_2 = [[x, x - 1] for sublist in final_b_cliques_2 for x in sublist]
    final_b_cliques_2 = list(chain.from_iterable(final_b_cliques_2))
    final_b_cliques_2 = [final_b_cliques_2[i:i + 4] for i in range(0, len(final_b_cliques_2), 4)]
    final_b_cliques_2 = [sorted(sublist) for sublist in final_b_cliques_2]

    return final_b_cliques_4, final_b_cliques_3, final_b_cliques_2


def execute_pipeline(file_patterns, general_hyperparameters):
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern.format(dataset={general_hyperparameters['dataset']})))

    max_threads = 5
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        results = list(executor.map(process_file, files))

    nets_all = []
    sm_all = []
    files_all = []

    for result in results:
        sim_all, net_all, file = result
        nets_all.append(net_all)
        sm_all.append(sim_all)
        files_all.append(file)

    del results

    model_all = TMFG()
    cliques_all, seps_all, adj_matrix_all = model_all.fit_transform(
        mean_tmfg(sm_all), output="weighted_sparse_W_matrix"
    )

    c4, c3, c2 = extract_components(cliques_all, seps_all, adj_matrix_all)
    c4 = list(chain.from_iterable(c4))
    c3 = list(chain.from_iterable(c3))
    c2 = list(chain.from_iterable(c2))

    original_cliques_all = list(chain.from_iterable(cliques_all))
    original_seps_all = list(chain.from_iterable(seps_all))

    return c4, c3, c2, original_cliques_all, original_seps_all, adj_matrix_all, sm_all, files_all


def get_complete_homology(
        general_hyperparameters: Dict[str, Any],
        model_hyperparameters: Dict[str, Any],
) -> Dict[str, List[List[int]]]:
    """
    Compute the homological structures to be used in the HCNN building process.

    Parameters
    ----------
    general_hyperparameters : Dict[str, Any]
        The general hyperparameters of the experiment.

    Returns
    ----------
    homological_structures : Dict[str, List[List[int]]]
    """

    file_patterns_training = [f"./data/{general_hyperparameters['dataset']}/unscaled_data/training/*{element}*.csv" for element in
                              general_hyperparameters['training_stocks']]
    c4_training, c3_training, c2_training, original_cliques_all_training, original_seps_all_training, adj_matrix_all_training, sm_all_training, files_all_training = execute_pipeline(
        file_patterns_training, general_hyperparameters)

    file_patterns_validation = [f"./data/{general_hyperparameters['dataset']}/unscaled_data/validation/*{element}*.csv" for element in
                                general_hyperparameters['training_stocks']]
    _, _, _, _, _, adj_matrix_all_validation, sm_all_validation, files_all_validation = execute_pipeline(file_patterns_validation, general_hyperparameters)

    file_patterns_test = [f"./data/{general_hyperparameters['dataset']}/unscaled_data/test/*{element}*.csv" for element in
                          general_hyperparameters['target_stocks']]
    _, _, _, _, _, adj_matrix_all_test, sm_all_test, files_all_test = execute_pipeline(file_patterns_test, general_hyperparameters)

    homological_structures = {"tetrahedra": c4_training,
                              "triangles": c3_training,
                              "edges": c2_training,
                              "original_cliques": original_cliques_all_training,
                              "original_separators": original_seps_all_training,
                              "adj_matrix_training": adj_matrix_all_training,
                              "similarity_matrices_training": sm_all_training,
                              "files_training": files_all_training,
                              "adj_matrix_validation": adj_matrix_all_validation,
                              "similarity_matrices_validation": sm_all_validation,
                              "files_validation": files_all_validation,
                              "adj_matrix_test": adj_matrix_all_test,
                              "similarity_matrices_test": sm_all_test,
                              "files_test": files_all_test
                              }

    training_stocks_string, test_stocks_string = get_training_test_stocks_as_string(general_hyperparameters)
    print(training_stocks_string, test_stocks_string)
    torch.save(homological_structures,
               f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/training_{training_stocks_string}_test_{test_stocks_string}/complete_homological_structures.pt")
    # torch.save(homological_structures,
    #           f"./torch_datasets/threshold_{model_hyperparameters['threshold']}/batch_size_{model_hyperparameters['batch_size']}/homological_structures_large_tick_stocks.pt")
    print('Homological structures have been saved.')

# get_homology({'dataset': 'nasdaq'})
