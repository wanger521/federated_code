import random
import time
import traceback

import torch

from src.datas.make_data import FEATURE_TYPE


def log(*k, **kw):
    timeStamp = time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime())
    print(timeStamp, end='')
    print(*k, **kw)
    # sys.stdout.flush()


def adapt_model_type(model):
    if FEATURE_TYPE == torch.float64:
        return model.type(torch.DoubleTensor)#model.double()
    elif FEATURE_TYPE == torch.float32:
        return model.type(torch.FloatTensor)
    elif FEATURE_TYPE == torch.float16:
        return model.half()


# function decorator: fix seed
# TODO: the 'construc_rng_pack' function in class 'distributedOptimizer'
#      should consider GPU and cpu generators
# Warning: having used in the code. Develops are recommanded to use the RngPack
#          given in ByrdLab.library.RandomNumberGenerator to control the random
#          seed
def fix_seed(run):
    def wrapper(self, *args, **kw):
        # fit seed
        if self.fix_seed:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                self._cuda_deterministic = torch.backends.cudnn.deterministic
                torch.backends.cudnn.deterministic = True
        result = run(self, *args, **kw)
        # reset the random seed
        if self.fix_seed:
            seed = time.time()
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = self._cuda_deterministic
        return result

    return wrapper


# function decorator: program won't stop even when exception raise
def no_exception_blocking(func):
    def wrapper(*args, **kw):
        try:
            return func(*args, **kw)
        except Exception as e:
            traceback.print_exc()

    return wrapper


def get_model_param(model, use_str=True):
    para = sum([x.nelement() for x in model.parameters()])
    if not use_str:
        return para
    elif para >= 1 << 30:
        return '{:.2f}G'.format(para / (1 << 30))
    elif para >= 1 << 20:
        return '{:.2f}M'.format(para / (1 << 20))
    elif para >= 1 << 10:
        return '{:.2f}K'.format(para / (1 << 10))
    else:
        return str(para)


def naive_local_avg(graph):
    node_size = graph.number_of_nodes()
    W = torch.zeros((node_size, node_size), dtype=FEATURE_TYPE)
    for i in range(node_size):
        neigbor_size = graph.neighbor_sizes[i] + 1
        for j in range(node_size):
            if i != j and not graph.has_edge(j, i):
                continue
            W[i][j] = 1 / neigbor_size
    return W
