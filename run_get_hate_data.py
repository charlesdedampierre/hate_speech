import polars as pl
import os

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
RESULTS_PATH = os.getenv("RESULTS_PATH")


if __name__ == "__main__":
    data = pl.read_csv(DATA_PATH + "/data_pca_scaled_comment_id.csv")
    data = data.to_pandas()
    data = data.drop("terms", axis=1)
    data = data.drop_duplicates()
    hate_type = ["antisémitisme", "islamophobie", "racisme"]
    data_filter = data[data["type"].isin(hate_type)]

    data_hate = data_filter[data_filter["hate_index_scaled"] >= 0.8]
    data_hate = data_hate.drop(["", "hate_index", "hate_index_scaled"], axis=1)
    data_hate = data_hate.drop_duplicates("comment_id", keep="first")
    data_hate.to_csv(RESULTS_PATH + "/data_hate.csv")
