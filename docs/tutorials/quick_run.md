## High-level Introduction

RobustFL provides numerous existing models and datasets. Models include `rnn`, `resnet`, `resnet18`, `resnet50`, `vgg9`, `simple_cnn`, `softmax_regression`, `linear_regression`, `logistic_regression`. Datasets include`Mnist`, `Cifar10`, `Cifar100`, `Synthetic`.
This note will present how to start training with these existing models and standard datasets.

RobustFL provides three types of high-level APIs: **registration**, **initialization**, and **execution**.
Registration is for registering customized components, which we will introduce in the following notes.
In this note, we focus on **initialization** and **execution**.

## Simplest Run

We can run robust federated learning with only two lines of code (not counting the import statement).
It executes training with default configurations: simulating 10 nodes with the MNIST dataset and select all nodes for training in each training iteration.
We explain more about the configurations in [another note](2.config.md).

Note: we package default partitioning of Mnist data to avoid downloading the whole dataset.

```python
import src

# Initialize federated learning with default configurations.
src.init()
# Execute federated learning training.
src.run()
```

## Run with Configurations

You can specify configurations to overwrite the default configurations.

```python
import src

# Define part of customized configs.
config = {
    "data": {"dataset": "Cifar10", "partition_type": "noniid_class"},
    "controller": {"rounds_iterations": 2000, "nodes_per_round": 10},
    "node": {"epoch_or_iteration": "iteration", "local_iteration": 1},
    "model": "resnet18"
}

# Define part of configs in a yaml file.
config_file = "config.yaml"
# Load and combine these two configs.
config = src.load_config(config_file, config)
# Initialize RobustFL with the new config.
src.init(config)
# Execute federated learning training.
src.run()
```

In the example above, we run training with model ResNet-18 and CIFAR-10 dataset that is partitioned into 10 nodes by label `class`.
It runs training with 10 nodes per iteration for 2000 iterations. In each iteration, each node trains 1 iteration.
