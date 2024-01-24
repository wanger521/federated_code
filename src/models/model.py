import importlib
from os import path

from torch import nn
from src.library.logger import create_logger
from src.library.tool import adapt_model_type

logger = create_logger()


class BaseModel(nn.Module):
    def __init__(self, feature_dimension=-1, num_classes=10):
        self.feature_dimension = feature_dimension
        self.num_classes = num_classes
        super(BaseModel, self).__init__()


def load_model(model_name: str):
    dir_path = path.dirname(path.realpath(__file__))
    model_file = path.join(dir_path, "{}.py".format(model_name))
    if not path.exists(model_file):
        logger.error("Please specify a valid model.")
    model_path = "src.models.{}".format(model_name)
    model_lib = importlib.import_module(model_path)
    model = getattr(model_lib, "Model")
    # TODO: maybe return the model class initiator
    return model
