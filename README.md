# LOBFrame

We release `LOBFrame' (see the [paper](https://arxiv.org/abs/2403.09267v1)), a novel, open-source code base which presents a renewed way to process large-scale Limit Order Book (LOB) data. This framework integrates all the latest cutting-edge insights from scientific research (see [Lucchese et al.](https://www.sciencedirect.com/science/article/pii/S0169207024000062), [Prata et al.](https://arxiv.org/pdf/2308.01915.pdf)) into a cohesive system. Its strength lies in the comprehensive nature of the implemented pipeline, which includes the data transformation and processing stage, an ultra-fast implementation of the training, validation, and testing steps, as well as the evaluation of the quality of a model's outputs through trading simulations. Moreover, it offers flexibility by accommodating the integration of new models, ensuring adaptability to future advancements in the field.

## Introduction

In this tutorial, we show how to replicate the experiments presented in the paper titled __"Deep Limit Order Book Forecasting: A microstructural guide"__.

Before starting, please remember to **ALWAYS CITE OUR WORK** as follows:

```
@article{briola2024deep,
  title={Deep Limit Order Book Forecasting},
  author={Briola, Antonio and Bartolucci, Silvia and Aste, Tomaso},
  journal={arXiv preprint arXiv:2403.09267},
  year={2024}
}
```

## Pre-requisites

Install the required packages:

```bash
pip3 install -r requirements.txt
```

## Data
All the code in this repository exploits [LOBSTER](https://lobsterdata.com) data. To have an overview on their structure, please refer
to the official documentation available at the following [link](https://lobsterdata.com/info/DataStructure.php).

# Preliminary operations
Before starting any experiment:
- Open the ```lightning_batch_gd.py``` file and insert the [Weights & Biases](https://wandb.ai/site) project's name and API key (search for TODOs).
- Open the ```utils.py``` file and set the default values of the parameters.

## Usage
To start an experiment from scratch, you need to follow these steps:
- Place the raw data in the `data/nasdaq/raw` folder. The data must be in the LOBSTER format and each folder must be named with the asset's name (e.g. AAPL for Apple stock).
- Run the following command to pre-process data:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "data_processing"
  ```
- Run the following command to prepare the torch datasets (this allows to reduce the training time):
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "torch_dataset_preparation" --prediction_horizon 10
  ```
  If you are interested also in performing the backtest stage, run the following command:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "torch_dataset_preparation,torch_dataset_preparation_backtest" --prediction_horizon 10
  ```
- Run the following command to train the model:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "training"
  ```
- Run the following command to evaluate the model:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --experiment_id "<experiment_id_generated_in_the_training_stage>" --stages "evaluation"
  ```
- Run the following command to analyze the results:
  ```bash
    python3 main --training_stocks "CSCO" --target_stocks "CSCO" --experiment_id "<experiment_id_generated_in_the_training_stage>" --stages "backtest,post_trading_analysis"
  ```

Multiple (compatible) stages can be executed at the same time. Consider the following example:
```bash
python3 main --training_stocks "CSCO" --target_stocks "CSCO" --stages "data_processing,torch_dataset_preparation,torch_dataset_preparation_backtest,training,evaluation,backtest,post_trading_analysis"
```

Each experiment can be resumed and re-run by specifying its ID in the `experiment_id` parameter.

We now provide the typical structure of a folder before an experiment's run:

```bash
.
├── README.md
├── data
│   └── nasdaq
│        ├── raw_data
│             ├── <Stock1_Name>
│             └── <Stock1_Name>
│        ├── scaled_data
│             ├── test
│             ├── training
│             └── validation
│        └── unscaled_data
│             ├── test
│             ├── training
│             └── validation
├── data_processing
│   ├── data_process.py
│   └── data_process_utils.py
├── loaders
│   └── custom_dataset.py
├── loggers
│   ├── logger.py
│   └── results
├── main.py
├── models
│   ├── AxialLob
│         └── axiallob.py
│   ├── CNN1
│         └── cnn1.py
│   ├── CNN2
│         └── cnn2.py
│   ├── DeepLob
│         └── deeplob.py
│   ├── DLA
│         └── DLA.py
│   ├── iTransformer
│         └── itransformer.py
│   ├── LobTransformer
│         └── lobtransformer.py
│   ├── TABL
│         ├── bin_nn.py
│         ├── bin_tabl.py
│         ├── bl_layer.py
│         └── tabl_layer.py
│   ├── Transformer
│         └── transformer.py
├── optimizers
│   ├── executor.py
│   └── lightning_batch_gd.py
├── requirements.txt
├── simulator
│   ├── market_sim.py
│   ├── post_trading_analysis.py
│   └── trading_agent.py
├── torch_datasets
│   └── threshold_1e-05
│       └── batch_size_32
│           └── 10
│               ├── test_dataset.pt
│               ├── test_dataset_backtest.pt
│               ├── training_dataset.pt
│               └── validation_dataset.pt
├── results
└── utils.py
```

# License

Copyright 2024 Antonio Briola, Silvia Bartolucci, Tomaso Aste.

Licensed under the CC BY-NC-ND 4.0 Licence (the "Licence"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

```
https://creativecommons.org/licenses/by-nc-nd/4.0/
```

Software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the provided link for the specific language governing permissions and limitations under the License.