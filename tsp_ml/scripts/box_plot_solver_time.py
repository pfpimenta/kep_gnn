# -*- coding: utf-8 -*-
# generates a box/scatter plot from the results
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# to allow imports from outside the tsp_ml/datasets/ package
package_folder_path = str(Path(__file__).parent.parent)
sys.path.insert(0, package_folder_path)

from paths import RESULTS_FOLDER_PATH

# load solver time CSV
csv_filepath = RESULTS_FOLDER_PATH / "recursive_solver_time.csv"
df = pd.read_csv(csv_filepath)


# remove duplicates
df = df.drop_duplicates("instance_id")
# breakpoint()

# box plot
plot = sns.boxplot(df, x="num_nodes", y="prediction_elapsed_time")
# scatter plot
# plot = sns.scatterplot(df, x="num_nodes", y="prediction_elapsed_time")

plot.set_ylabel("solver time in seconds")

plt.show()
