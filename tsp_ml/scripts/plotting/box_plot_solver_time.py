# -*- coding: utf-8 -*-
# generates a box/scatter plot from the results
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, package_folder_path)

from paths import RESULTS_FOLDER_PATH

# load solver time CSV
csv_filepath = RESULTS_FOLDER_PATH / "recursive_solver_time.csv"
df = pd.read_csv(csv_filepath)


# remove duplicates
df = df.drop_duplicates("instance_id")

# consider only num_nodes from 1 to 15
# df = df[df['num_nodes'] <= 15]

# breakpoint()

# box plot
plot = sns.boxplot(df, x="num_nodes", y="prediction_elapsed_time")
# scatter plot
# plot = sns.scatterplot(df, x="num_nodes", y="prediction_elapsed_time")

# axis labels
plot.set_ylabel("Solver time in seconds")
plot.set_xlabel("Number of nodes")

plt.show()
