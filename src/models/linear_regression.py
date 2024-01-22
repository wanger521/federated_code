import torch
import torch.nn as nn
from src.models import BaseModel


class Model(BaseModel):
    def __init__(self, feature_dimension=1, num_classes=1, **kwargs):
        super(Model, self).__init__(feature_dimension=feature_dimension, num_classes=num_classes)
        # Linear regression is equivalent to using a fully connected layer
        self.linear = nn.Linear(in_features=feature_dimension, out_features=num_classes)

    def forward(self, x):
        out1 = torch.flatten(x, 1)
        out = self.linear(out1)
        return out
