import sys

sys.path.append("../")


import pandas as pd

pd.options.mode.chained_assignment = None

import numpy as np
import itertools
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from src.utils import get_table_from_report

from dotenv import load_dotenv

load_dotenv()

RESULTS_PATH = os.getenv("RESULTS_PATH")


if __name__ == "__main__":
    # compare the true results wwith the classying results
    df = pd.read_csv(RESULTS_PATH + "/annotations/clean_annotations.csv", index_col=[0])

    grouped = df.groupby("annotateur")["comment_id"].apply(list)
    common_ids = list(set.intersection(*map(set, grouped)))

    df_common = df[df["comment_id"].isin(common_ids)]
    df_common = df_common.drop_duplicates().reset_index(drop=True)
    df_class = df.drop("annotateur", axis=1).drop_duplicates().reset_index(drop=True)
    df_class.columns = [
        "comment_id",
        "true_antisémitisme",
        "true_islamophobie",
        "true_racisme",
        "true_hate",
    ]

    df_hate = pd.read_csv(RESULTS_PATH + "/data_hate.csv", index_col=[0])
    df_hate["pred_hate"] = 1

    df_hate_unique = df_hate.pivot(
        index="comment_id", columns="type", values="pred_hate"
    )

    df_class_global = df_class[["comment_id", "true_hate"]]
    df_hate_global = df_hate[["comment_id", "pred_hate"]]

    df_class_hate = pd.merge(
        df_class_global, df_hate_global, on="comment_id", how="left"
    )
    df_class_hate = df_class_hate.fillna(0)
    df_class_hate.columns = ["comment_id", "true", "pred"]

    pred_labels = list(df_class_hate["pred"])
    true_labels = list(df_class_hate["true"])

    report = classification_report(true_labels, pred_labels, output_dict=True)
    df_report = get_table_from_report(report)
    plt.savefig(RESULTS_PATH + f"/performances/global_hate.png", dpi=300)

    hate_types = ["antisémitisme", "islamophobie", "racisme"]
    for hate_type in hate_types:
        df_score = df_hate_unique[hate_type].dropna().reset_index()
        df_score = df_score.rename(columns={hate_type: "pred_hate"})
        df_score = pd.merge(
            df_class[["comment_id", f"true_{hate_type}"]],
            df_score,
            on="comment_id",
            how="left",
        )
        df_score = df_score.fillna(0)
        df_score.columns = ["comment_id", "true", "pred"]

        pred_labels = list(df_score["pred"])
        true_labels = list(df_score["true"])

        report = classification_report(true_labels, pred_labels, output_dict=True)
        df_report = get_table_from_report(report)
        plt.savefig(
            RESULTS_PATH + f"/performances/{hate_type}.png", dpi=300
        )  # Save as PNG with 300 dpi resolution
