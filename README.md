# kep_gnn

Repository for the code for solving the Kidney Exchange Problem (KEP) with Graph Neural Network (GNN) methods.
The results of the experiments were summarized into a scientific article available at https://arxiv.org/abs/2304.09975.
The results are fully reproducible with only the code in this repository, as all the data used is artificially generated and the model training and evaluation is done by code contained in this repository.

A short paper was published at the 2023 IEEE ICTAI conference, and a full paper has also been submitted to a journal.

*WARNING*: the rest of this README file below will be updated soon with clear instructions on how to install, setup, use, reproduce results and extend the work.

## Project description

The goal of this repository was, initially, to solve the __Traveling Salesperson Problem__ (TSP) with Machine Learning (ML) methods.

The decision variant of the TSP problem, the __Decision TSP__ (or DTSP) is also addressed. A description of the problem and a ML method to solve it can be found at the article "_Learning to Solve NP-Complete Problems - A Graph Neural Network for Decision TSP_" (https://arxiv.org/abs/1809.02721).

Another problem, the __Kidney Exchange Problem__ (KEP), is also addressed in a similar fashion. The problem definition can be found at the article "_Finding long chains in kidney exchange using the traveling salesman problem_" (https://www.pnas.org/doi/10.1073/pnas.1421853112).

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
To be able to train the model, you must first have training, validation, and test data, which can be generated through the following scripts:

#### * Generating the TSP dataset
```bashrc
python3 tsp_ml/scripts/dataset_generation/generate_tsp_dataset.py
```

#### * Generating the DTSP dataset
```bashrc
python3 tsp_ml/scripts/dataset_generation/generate_dtsp_dataset.py
```

#### * Generating the KEP dataset
```bashrc
python3 tsp_ml/scripts/dataset_generation/generate_kep_dataset.py
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

#### * Visualizing predictions
TODO
