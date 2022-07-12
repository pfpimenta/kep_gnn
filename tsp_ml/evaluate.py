from typing import Tuple
import torch
from tqdm import tqdm

import sys
from models.tsp_ggcn import TSP_GGCN
from tsp_dataset import TSPDataset
from torch_geometric.loader import DataLoader
from definitions import TRAINED_MODELS_FOLDER_PATH, TRAIN_DATASET_FOLDER_PATH, TEST_DATASET_FOLDER_PATH


# MODEL_FILENAME = "tsp_gcn_model.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_05_16h31.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_07_17h56.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_07_19h40.pt"
# MODEL_FILENAME = "TSP_GGCN_2022_07_08_17h00.pt"
MODEL_FILENAME = "TSP_GGCN_2022_07_12_12h27.pt"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

def confusion(prediction: torch.Tensor, truth: torch.Tensor) -> Tuple[int, int, int, int]:
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def evaluate(model: torch.nn.Module, dataset: TSPDataset):
    dataloader = DataLoader(dataset, shuffle=True,batch_size=batch_size,pin_memory=True,num_workers=4)
    model.eval() # set the model to evaluation mode
    total_corrects = 0
    total_non_route_edges_truth = 0
    total_route_edges_pred = 0
    total_non_route_edges_pred = 0
    total_route_edges_truth = 0
    total_edges = 0
    accuracy_list = []
    TP, FP, TN, FN = 0, 0, 0, 0
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluation",file=sys.stdout)):
        batch = batch.to(device)
        label = batch.y
        label = label.to(torch.float32)
        scores = model(batch)
        pred = torch.argmax(scores, 1).to(int)

        # confusion matrix
        batch_TP, batch_FP, batch_TN, batch_FN = confusion(pred, batch.y)
        # print(f"batch_TP, batch_FP, batch_TN, batch_FN: {batch_TP, batch_FP, batch_TN, batch_FN}")
        TP += batch_TP
        FP += batch_FP
        TN += batch_TN
        FN += batch_FN

        correct_predictions = int((pred.to(int) == batch.y).sum())
        total_non_route_edges_truth += batch.num_edges - batch.y.sum()
        total_route_edges_truth += batch.y.sum()
        total_non_route_edges_pred += pred.to(int).sum()
        total_route_edges_pred += pred.to(int).sum()
        batch_accuracy = correct_predictions / batch.num_edges
        accuracy_list.append(batch_accuracy)
        total_corrects += correct_predictions
        total_edges += batch.num_edges
        if i == 30:
            pass
            # import pdb
            # pdb.set_trace()


    print(f"TP, FP: {TP, FP}")
    print(f"TN, FN: {TN, FN}")
    precision = TP / (TP + FP)
    print(f"precision: {precision}")
    recall = TP /(TP + FN)
    print(f"recall: {recall}")
    print(f'total_route_edges_truth: {total_route_edges_truth}')
    print(f'total_route_edges_pred: {total_route_edges_pred}')
    print(f'total_corrects: {total_corrects}')
    dataset_size = len(dataloader.dataset)
    print(f'dataset_size: {dataset_size}')
    print(f'total_edges: {total_edges}')
    avg_num_edges = int(total_edges) / int(dataset_size)
    print(f'avg_num_edges: {avg_num_edges}')
    accuracy = total_corrects / int(total_edges)
    print(f'Accuracy: {accuracy:.4f}')

# load model
model = TSP_GGCN().to(device)
model_filepath = TRAINED_MODELS_FOLDER_PATH / MODEL_FILENAME
print(f"...Loading model from file {model_filepath}")
model.load_state_dict(torch.load(model_filepath, map_location=device))

# setup data
batch_size = 10
train_dataset = TSPDataset(dataset_folderpath=TRAIN_DATASET_FOLDER_PATH)
test_dataset = TSPDataset(dataset_folderpath=TEST_DATASET_FOLDER_PATH)

print("\n\nEvaluating the model on the train dataset")
evaluate(model=model, dataset=train_dataset)
print("\n\nEvaluating the model on the test dataset")
evaluate(model=model, dataset=test_dataset)