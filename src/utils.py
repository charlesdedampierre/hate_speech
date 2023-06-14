import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

pd.options.mode.chained_assignment = None


def get_table_from_report(report: dict):
    df_report = pd.DataFrame(report).T
    df_report = df_report.reset_index()
    df_report = round(df_report, 2)
    df_report["precision"][df_report["index"] == "accuracy"] = ""
    df_report["recall"][df_report["index"] == "accuracy"] = ""
    df_report = df_report.set_index("index")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table_data = table(
        ax,
        df_report,
        loc="center",
        cellLoc="center",
        colWidths=[0.1, 0.1, 0.1, 0.1],
    )
    table_data.scale(2, 2)
    table_data.auto_set_font_size(False)
    table_data.set_fontsize(11)

    return table_data
