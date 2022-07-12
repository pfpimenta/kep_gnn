
create_venv:
	python3 -m venv tcc_venv

activate_venv:
	source tcc_venv/bin/activate

generate_dataset:
	python3 tsp_gnn/generate_dataset.py

train:
	python3 tsp_gnn/train_gcn_tsp.py

evaluate:
	python3 tsp_gnn/evaluate.py
