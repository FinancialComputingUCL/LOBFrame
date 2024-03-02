from data_processing import data_process_utils
from loggers import logger
from optimizers.executor import Executor
from simulator import market_sim, post_trading_analysis
from utils import (
    data_split,
    load_yaml,
    save_dataset_info,
    parse_args,
    create_hyperparameters_yaml,
)

if __name__ == "__main__":
    # Parse input arguments.
    args = parse_args()
    wb_error_detection = False

    if args.experiment_id is None:
        # If no experiment ID is passed, generate a new one.
        experiment_id = logger.generate_id(args.model, args.target_stocks)
        # Create a new configuration file containing the hyperparameters.
        create_hyperparameters_yaml(experiment_id, args)
    else:
        # If an experiment ID is passed, use it.
        experiment_id = args.experiment_id
        # Replace the hyperparameters file with the new arguments passed as input.
        # create_hyperparameters_yaml(experiment_id, args)

    # Load the configuration file containing the hyperparameters.
    hyperparameters_path = (
        f"{logger.find_save_path(experiment_id)}/hyperparameters.yaml"
    )

    # Load the configuration file (general hyperparameters).
    general_hyperparameters = load_yaml(hyperparameters_path, "general")
    # Load the configuration file (model's hyperparameters).
    model_hyperparameters = load_yaml(hyperparameters_path, "model")
    # Load the configuration file (trading hyperparameters).
    trading_hyperparameters = load_yaml(hyperparameters_path, "trading")

    if args.experiment_id is not None:
        general_hyperparameters['stages'] = args.stages.split(",")

    # Handle the data processing stage.
    if "data_processing" in general_hyperparameters["stages"]:
        # Make the list of training stocks a set to avoid duplicates.
        training_stocks = set(general_hyperparameters["training_stocks"])
        # Make the list of target stocks a set to avoid duplicates.
        target_stocks = set(general_hyperparameters["target_stocks"])
        # Iterate over stocks after performing the union of sets operation (a stock can occur both in training_stocks and target_stocks).
        for stock in list(training_stocks.union(target_stocks)):
            data_utils = data_process_utils.DataUtils(
                ticker=stock,
                dataset=general_hyperparameters["dataset"],
                experiment_id=experiment_id,
                horizons=general_hyperparameters["horizons"],
                normalization_window=general_hyperparameters["normalization_window"],
            )
            # Generate the data folders.
            data_utils.generate_data_folders()
            # Transform the data.
            data_utils.process_data()
        # Split the data into training, validation and test sets.
        data_split(
            dataset=general_hyperparameters["dataset"],
            training_stocks=general_hyperparameters["training_stocks"],
            target_stock=general_hyperparameters["target_stocks"],
            training_ratio=general_hyperparameters["training_ratio"],
            validation_ratio=general_hyperparameters["validation_ratio"],
            include_target_stock_in_training=general_hyperparameters[
                "include_target_stock_in_training"
            ],
        )

    # Instantiate the executor as None.
    executor = None
    # For 'torch_dataset_preparation' stage, instantiate the executor with proper arguments.
    if "torch_dataset_preparation" in general_hyperparameters["stages"]:
        executor = Executor(
            experiment_id, general_hyperparameters, model_hyperparameters, torch_dataset_preparation=True
        )

    if "torch_dataset_preparation_backtest" in general_hyperparameters["stages"]:
        executor = Executor(
            experiment_id, general_hyperparameters, model_hyperparameters, torch_dataset_preparation=False, torch_dataset_preparation_backtest=True
        )

    # For the 'training' and 'evaluation' stages, instantiate the executor with proper arguments.
    if (
        "training" in general_hyperparameters["stages"]
        or "evaluation" in general_hyperparameters["stages"]
    ):
        executor = Executor(
            experiment_id, general_hyperparameters, model_hyperparameters
        )

    if "training" in general_hyperparameters["stages"]:
        try:
            # Keep track of the files used in the training, validation and test sets.
            save_dataset_info(
                experiment_id=experiment_id,
                general_hyperparameters=general_hyperparameters,
            )
            # Train the model.
            executor.execute_training()
            # Clean up the experiment folder from wandb logging files.
            executor.logger_clean_up()
        except:
            print('Exception detected')
            wb_error_detection = True

    if "evaluation" in general_hyperparameters["stages"] and wb_error_detection is False:
        # Out-of-sample test of the model.
        executor.execute_testing()
        # Clean up the experiment folder from wandb logging files.
        executor.logger_clean_up()

    if "backtest" in general_hyperparameters["stages"]:
        # Backtest the model.
        market_sim.backtest(
            experiment_id=experiment_id, trading_hyperparameters=trading_hyperparameters
        )

    if "post_trading_analysis" in general_hyperparameters["stages"]:
        # Perform a post-trading analysis.
        post_trading_analysis.post_trading_analysis(
            experiment_id=experiment_id, general_hyperparameters=general_hyperparameters, trading_hyperparameters=trading_hyperparameters, model_hyperparameters=model_hyperparameters
        )
