# tsp_ml
The goal of this repository is to solve the Traveling Salesperson Problem (TSP) with Machine Learning (ML) methods.

## Installation

[Optional] It is recommended that you install the dependencies in a virtual environment. To do so, you can use the command in the 'create_venv.sh' file.

You can install the dependencies by running the install_dependencies.sh bash script:
```bashrc
sh install_dependencies.sh
```

Or by using pip to install the dependencies specified in the 'requirements.txt' file:
```bashrc
pip install -r requirements.txt
```

To test the installation, you can run the 'import_test.py' script:
```bashrc
python3 tsp_ml/import_test.py
```
If there are no errors and the script prints a message saying that all imports worked, the installation is complete!

## How to run
TODO little text describing what you can do with the project

#### * Generating the TSP dataset
```bashrc
python3 tsp_ml/datasets/generate_tsp_dataset.py
```

#### * Generating the DTSP dataset
```bashrc
python3 tsp_ml/datasets/generate_dtsp_dataset.py
```

#### * Training a model
First, choose the model architecture, the dataset to be used and the hyperparameter values by changing the variables in the beginning of the train.py script. Then, run the Python script:
```bashrc
python3 tsp_ml/train.py
```

#### * Evaluating a model
First, choose the trained model by changing the 'TRAINED_MODEL_NAME'. The trained model should be at the trained models folder. Then, run the evaluate Python script:
```bashrc
python3 tsp_ml/evaluate.py
```

#### * Using a model to predict the TSP route of a single graph
TODO

#### * Visualizing route predictions
TODO
