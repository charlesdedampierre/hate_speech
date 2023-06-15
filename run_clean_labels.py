import pandas as pd
import os

pd.options.mode.chained_assignment = None

from dotenv import load_dotenv

load_dotenv()

RESULTS_PATH = os.getenv("RESULTS_PATH")


if __name__ == "__main__":
    # compare the true results wwith the classying results

    df = pd.read_csv(RESULTS_PATH + "/annotations/CNCDH 2 - Annotation - data.csv")
    df = df.fillna(0)
    df = df.drop("text", axis=1)
    df = df.replace("#", 0)
    df[["antisémitisme", "islamophobie", "racisme"]] = df[
        ["antisémitisme", "islamophobie", "racisme"]
    ].astype(float)
    df = df.replace(2, 1)
    hate_types = ["antisémitisme", "islamophobie", "racisme"]
    df["global hate"] = df[hate_types].max(axis=1)
    df = df.replace("#", 0)
    # get the one that are done with the task
    df = df[df["annotateur"].isin(["Florian", "Charles", "Benjamin"])]
    df.to_csv(RESULTS_PATH + "/annotations/clean_annotations.csv")
