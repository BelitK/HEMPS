from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# load parquet
df = pd.read_parquet("slp_pv.parquet", engine="pyarrow")

print(df.head())
def plot_pv():
    # plot time series
    plt.figure()
    plt.plot(df["timestamp"], df["Profilwert"])
    plt.xlabel("Time")
    plt.ylabel("Profilwert")
    plt.title("PV / Demand Time Series (15-min resolution)")
    plt.tight_layout()
    plt.show()



print(df['Profilwert'].max(), "\n",
      df['Profilwert'].min())