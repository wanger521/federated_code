import torch
import torch.nn as nn
from src.models import BaseModel


class Model(BaseModel):
    def __init__(self, feature_dimension, num_classes, **kwargs):
        super(Model, self).__init__(feature_dimension=feature_dimension, num_classes=num_classes)
        # LSoftmax regression is equivalent to using a fully connected layer and a softmax layer
        self.linear = nn.Linear(in_features=feature_dimension, out_features=num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x1 = torch.flatten(x, 1)
        out1 = self.linear(x1)
        out2 = self.softmax(out1)
        return out2
