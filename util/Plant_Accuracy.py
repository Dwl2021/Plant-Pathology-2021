from torchmetrics import Metric
import torch


class Plant_Accuracy(Metric):
    def __init__(self):
        super().__init__()
        # Adds two states to the instance of the class: "correct" and "total"
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # A method that takes in predictions and target values and updates the state of the class
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Applies a threshold of 0.5 to the predictions and converts them to integers
        preds = (torch.sigmoid(preds) > 0.50).int()
        self.correct += torch.sum((preds == target).all(dim=1))
        self.total += target.shape[0]

    def compute(self):
        # Calculates the accuracy as the ratio of the number of correct predictions to the total number of predictions
        return self.correct.float() / self.total