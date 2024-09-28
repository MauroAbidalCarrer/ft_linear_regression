import pandas as pd
from pandas import DataFrame as DF
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def main():
    cli_kwargs = parse_arguments()
    dataset = pd.read_csv(cli_kwargs["path_to_dataset"]).dropna(how="any")
    theta_0, theta_1 = fit_lr(dataset, **cli_kwargs)
    with open("lr_params.json", "w") as param_file:
        param_file.write(json.dumps({"theta_0": theta_0, "theta_1": theta_1}, indent=1))
    if cli_kwargs["print_scores"]:
        print_scores(dataset, theta_0, theta_1)

def fit_lr(dataset:DF, plt_fitting=False, plt_final_fitting=False, learning_rate=0.1, nb_epochs=20, **kwargs) -> tuple[float, float]:
    # apply std standardization on dataset
    means = dataset.mean()
    stds = dataset.std()
    std_scaled_dataset:DF = (dataset - means) / stds
    dataset_size = len(dataset)

    theta_0 = 0
    theta_1 = 0

    if plt_fitting:
        plt_dataset_and_lr(std_scaled_dataset, theta_0, theta_1)


    for _ in range(nb_epochs):
        std_scaled_dataset:DF = (
            std_scaled_dataset
            .eval(f"estimated_price = {theta_0} + {theta_1} * km")
            .eval(f"residual = estimated_price - price")
        )
        # Compute loss gradients
        tmp_theta_0 = learning_rate * std_scaled_dataset["residual"].sum() / dataset_size
        tmp_theta_1 = learning_rate * std_scaled_dataset.eval("residual * km").sum() / dataset_size
        # substract gradients to parameters
        theta_0 -= tmp_theta_0
        theta_1 -= tmp_theta_1

        if plt_fitting:
            plt_dataset_and_lr(std_scaled_dataset, theta_0, theta_1)
    # Undo std scaling
    real_theta_1 = (stds["price"] * theta_1) / stds["km"]
    real_theta_0 = stds["price"] * theta_0 + means["price"] - real_theta_1 * means["km"]

    if plt_final_fitting:
        plt_dataset_and_lr(dataset, real_theta_0, real_theta_1)
    
    return real_theta_0, real_theta_1


def plt_dataset_and_lr(dataset:DF, theta_0:float, theta_1:float):
    def predict(theta_0:float, theta_1:float, km:float) -> float:
        return theta_0 + theta_1 * km

    ax = dataset.plot.scatter(x="km", y="price")
    ax.plot(
        [dataset["km"].min(), dataset["km"].max()],
        [
            predict(theta_0, theta_1, dataset["km"].min()),
            predict(theta_0, theta_1, dataset["km"].max())
        ]
    )
    plt.show()

def print_scores(dataset:DF, theta_0:float, theta_1:float):
    dataset:DF = (
        dataset
        .eval(f"estimated_price = {theta_0} + {theta_1} * km")
        .eval(f"residual = estimated_price - price")
    )
    print(dataset)
    print("mean residual:", dataset["residual"].mean())

def parse_arguments():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Linear Regression with optional settings.")
    
    # Add the required arguments
    parser.add_argument(
        "--path_to_dataset", 
        type=str, 
        default="data.csv", 
        help="Path to dataset."
    )
    
    parser.add_argument(
        "--plt_fitting", 
        action="store_true", 
        default=False, 
        help="Plot dataset and fit line after each epoch."
    )
    
    parser.add_argument(
        "--plt_final_fitting", 
        action="store_true", 
        default=False, 
        help="Plot dataset and fit line after fitting."
    )
    
    parser.add_argument(
        "--nb_epochs", 
        type=int, 
        default=20, 
        help="Number of epochs."
    )
    
    parser.add_argument(
        "--print_scores", 
        action="store_true", 
        default=False, 
        help="Print the predictions and their corresponding ground truth, as well as the mean accuracy."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Return a dictionary of the parsed arguments
    return {
        "path_to_dataset": args.path_to_dataset,
        "plt_fitting": args.plt_fitting,
        "plt_final_fitting": args.plt_final_fitting,
        "nb_epochs": args.nb_epochs,
        "print_scores": args.print_scores
    }

if __name__ == "__main__":
    main()