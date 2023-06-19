import os
import pandas as pd
import matplotlib.pyplot as plt
from src.agreement import (
    get_kappa_matrix,
    get_percent_agreement_matrix,
    get_mean,
    plot_table,
)

from dotenv import load_dotenv

load_dotenv()

RESULTS_PATH = os.getenv("RESULTS_PATH")

if __name__ == "__main__":
    df = pd.read_csv(RESULTS_PATH + "/annotations/clean_annotations.csv", index_col=[0])

    grouped = df.groupby("annotateur")["comment_id"].apply(list)
    common_ids = list(set.intersection(*map(set, grouped)))

    df_common = df[df["comment_id"].isin(common_ids)]
    df_common = df_common.drop_duplicates().reset_index(drop=True)

    hate_types = ["antisémitisme", "islamophobie", "racisme", "global hate"]

    for hate_type in hate_types:
        df_bet = df_common[["comment_id", "annotateur", hate_type]]
        df_bet = df_bet.pivot(
            index="comment_id", columns="annotateur", values=hate_type
        )
        df_bet = df_bet.astype(int)

        coder1 = list(df_bet["Benjamin"])
        coder2 = list(df_bet["Charles"])
        coder3 = list(df_bet["Florian"])

        coders = [coder1, coder2, coder3]
        # labels = ["Benjamin", "Charles", "Florian"]
        labels = ["coder 1", "coder 2", "coder 3"]

        # Compute Percent Agreement
        df_percent_agreement = get_percent_agreement_matrix(coders, labels)
        overall_mean = round(get_mean(df_percent_agreement), 1)
        overall_mean = f"overall mean: {overall_mean}"
        fig = plot_table(df_percent_agreement, hate_type, overall_mean)
        target_path = RESULTS_PATH + "/agreement/percent"
        plt.savefig(target_path + f"/{hate_type}.png", bbox_inches="tight", dpi=300)

        # Compute Kappa
        df_kappa_matrix = get_kappa_matrix(coders, labels)
        df_kappa_matrix = round(df_kappa_matrix, 2)
        overall_mean = round(get_mean(df_kappa_matrix), 1)
        overall_mean = f"overall mean: {overall_mean}"
        fig = plot_table(df_kappa_matrix, hate_type, overall_mean)
        target_path = RESULTS_PATH + "/agreement/kappa/"
        plt.savefig(target_path + f"/{hate_type}.png", bbox_inches="tight", dpi=300)
