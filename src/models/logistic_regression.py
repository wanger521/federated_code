import torch
import torch.nn as nn
from src.models import BaseModel


class Model(BaseModel):
    def __init__(self, feature_dimension, num_classes, **kwargs):
        super(Model, self).__init__(feature_dimension=feature_dimension, num_classes=num_classes)
        # Logistic regression is equivalent to using a fully connected layer and a Sigmoid layer
        self.linear = nn.Linear(feature_dimension, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.flatten(x, 1)
        out1 = self.linear(x1)
        out2 = self.sigmoid(out1)
        return out2
