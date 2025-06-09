import pickle

import pandas as pd
from tabulate import tabulate

from baseline import baseline_most_frequent
from logistic_regression import train_logistic_regression
from data_split import load_split
from compute_cost import CLASSES as TARGET_CLASSES
from evaluation_functions import evaluate_cost
from file_locations import DATASET_PATH

if __name__ == '__main__':
    X_train, Y_train, train_files = load_split("train")
    X_test, Y_test, test_files = load_split("test")
    X_val, Y_val, val_files = load_split("val")

    bl_inference_funcs = baseline_most_frequent(Y_train, TARGET_CLASSES)
    lr_inference_funcs = train_logistic_regression(
        X_train, Y_train, TARGET_CLASSES
    )

    # baseline inference on test set
    bl_total, bl_breakdown = evaluate_cost(
        test_files,
        DATASET_PATH,
        TARGET_CLASSES,
        X_test,
        bl_inference_funcs
    )

    # logistic regression inference on test set
    lr_total, lr_breakdown = evaluate_cost(
        test_files,
        DATASET_PATH,
        TARGET_CLASSES,
        X_test,
        lr_inference_funcs
    )

    # shuffle around format for pretty print

    # Convert breakdowns into dict[class → cost]
    bl_costs = {cls: d["cost"] for cls, d in bl_breakdown.items()}
    lr_costs = {cls: d["cost"] for cls, d in lr_breakdown.items()}

    # Add total cost
    bl_costs["TOTAL"] = bl_total
    lr_costs["TOTAL"] = lr_total

    with open("RNN/rnn_512x2_costs.pkl", "rb") as file:
        gru_512x2_costs = pickle.load(file)
    with open("RNN/rnn_512x3_costs.pkl", "rb") as file:
        gru_512x3_costs = pickle.load(file)
    with open("RNN/rnn_512x4_costs.pkl", "rb") as file:
        gru_512x4_costs = pickle.load(file)
    with open("RNN/rnn_1024x2_costs.pkl", "rb") as file:
        gru_1024x2_costs = pickle.load(file)
    with open("RNN/rnn_1024x3_costs.pkl", "rb") as file:
        gru_1024x3_costs = pickle.load(file)
    with open("RNN/rnn_1024x4_costs.pkl", "rb") as file:
        gru_1024x4_costs = pickle.load(file)

    # Create a DataFrame for comparison
    cost_df = pd.DataFrame({
        "Baseline": bl_costs,
        "Logistic Regression": lr_costs,
        "GRU_512x2": gru_512x2_costs,
        "GRU_512x3": gru_512x3_costs,
        "GRU_512x4": gru_512x4_costs,
        "GRU_1024x2": gru_1024x2_costs,
        "GRU_1024x3": gru_1024x3_costs,
        "GRU_1024x4": gru_1024x4_costs,
    }).round(2)

    print(tabulate(cost_df.reset_index().values,
                   headers=["Class",
                            "Baseline",
                            "Logistic Regression",
                            "GRU_512x2",
                            "GRU_512x3",
                            "GRU_512x4",
                            "GRU_1024x2",
                            "GRU_1024x3",
                            "GRU_1024x4"],
                   tablefmt="github"))

    with open("comparison_table.txt", "w") as file:  # Specify your desired file name and extension
        file.write(tabulate(cost_df.reset_index().values,
                            headers=["Class",
                                     "Baseline",
                                     "Logistic Regression",
                                     "GRU_512x2",
                                     "GRU_512x3",
                                     "GRU_512x4",
                                     "GRU_1024x2",
                                     "GRU_1024x3",
                                     "GRU_1024x4"],
                            tablefmt="github"))

    # Save the DataFrame to a CSV file
    cost_df.to_csv("comparison_table.csv", index=True)

