# make sure the venv is activated
source venv_tsp_gnn/bin/activate

# pytorch geometric
# GNNs and geometric deep learning lib
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# networkx
# graph lib
pip install networkx

# for printing (drawing) graphs
pip install matplotlib

# TSP solver
pip install python-tsp

# Jupyter
pip install -U ipykernel

# pre-commit (for development only)
pip install pre-commit
