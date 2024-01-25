<p><small>Project based on the <a target="──blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template
</a> and <a target="──blank" href="https://easyfl.readthedocs.io/en/latest/introduction.html">EasyFL: an easy-to-use federated learning platform
</a> and <a target="──blank" href="https://github.com/Zhaoxian-Wu/IOS">IOS
</a>. #cookiecutterdatascience</small></p>
<div align="center">
  <h1 align="center">Robust Federated Learning Framework</h1>
  </div>

*<small>This project is a Byzantine-robust federated learning framework, including both centralized and decentralized training architectures.<small>*

---

## Introduction

**RobustFL** is a Byzantine-robust federated learning (FL) platform based on PyTorch. It aims to enable users with various levels of expertise to experiment and prototype FL applications with little/no coding. 
We designed the federated architecture based on EasyFL, and data processing, attack and aggregation based on IOS, etc.

You can use it for:
* Robust FL Research on algorithm and system
* Learning FL implementations 

## Major Features

**Easy to Start**

RobustFL is easy to install and easy to learn, just the same as EasyFL. It does not have complex dependency requirements. You can run EasyFL on your personal computer with only three lines of code ([Quick Start](docs/en/quick_run.md)).

**Out-of-the-box Functionalities**

RobustFL provides many out-of-the-box functionalities, including datasets, models, Byzantine attacks, aggregation rules, graphs, distributed architectures, learning rate controllers and Local training ways. With simple configurations, you simulate different FL scenarios using the popular datasets. We support both statistical heterogeneity simulation and system heterogeneity simulation.

**Flexible, Customizable, and Reproducible**

RobustFL is flexible and to be customized according to your needs. You can easily migrate existing robust federated learning applications into the manner by writing the PyTorch codes that you are most familiar with. 

**One Training Mode**

RobustFL only supports **standalone training**. We use a single card for pseudo-distributed training.

## Getting Started

You can refer to [Get Started](docs/en/get_started.md) for installation and [Quick Run](docs/en/quick_run.md) for the simplest way of using RobustFL.

For more advanced usage, we provide a list of tutorials on:
* [High-level APIs](docs/en/tutorials/high-level_apis.md)
* [Configurations](docs/en/tutorials/config.md)
* [Datasets](docs/en/tutorials/dataset.md)
* [Models](docs/en/tutorials/model.md)
* [Customize Controller and Node](docs/en/tutorials/customize_controller_and_node.md)
* [Aggregations](docs/en/tutorials/aggregations.md)
* [Byzantine-attacks](docs/en/tutorials/byzantine-attacks.md)
* [Others](docs/en/tutorials/other-tools.md)
* [Code-Structure](docs/en/tutorials/structure.md)


## Projects & Papers

We have released the source code for the following papers under the `applications` folder:

- byrd_saga: [[code]](https://github.com/wanger521/federated_code/tree/master/applications/byrd_saga) for [Federated Variance-Reduced Stochastic Gradient Descent With Robustness to Byzantine Attacks](https://ieeexplore.ieee.org/abstract/document/9153949) (_TSP_)
- d_ogd: [[code]](https://github.com/wanger521/federated_code/tree/master/applications/d_ogd) for two papers: [Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment](https://ieeexplore.ieee.org/abstract/document/10354032) (_TSP_) and [Collaborative Unsupervised Visual Representation Learning From Decentralized Data](https://ieeexplore.ieee.org/document/10095178) (_ICASSP'2023_)
- rsa: [[code]](https://github.com/wanger521/federated_code/tree/master/applications/rsa) for two papers: [RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets](https://ojs.aaai.org/index.php/AAAI/article/view/3968) (_AAAI'2019_) and [Byzantine-robust decentralized stochastic optimization over static and time-varying networks](https://www.sciencedirect.com/science/article/pii/S0165168421000591) (_SP_)


## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use this platform or related projects in your research, please cite this project.


## Main Contributors

Xingrong Dong [:octocat:](https://github.com/wanger521) 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
        └── tutorials      <- The detailed tutorials for this code
    │
    ├── saved_models       <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    │
    ├── src                <- Source code for use in this project.
    │   ├── ────init────.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── aggregations       <- Scripts to aggregate messages for the nodes in the graph.
    │   │
    │   │
    │   ├── attacks            <- Scripts to attack messages for the byzantine nodes in the graph.
    │   │
    │   │
    │   ├── compressions       <- Scripts to compression model for comunication.
    │   │
    │   │
    │   ├── datas              <- Scripts to download or generate data
    │   │   └── federated_dataset.py
    │   │
    │   ├── library            <- Scripts to some useful functions, like graph.
    │   │
    │   │
    │   ├── models             <- Scripts to the model you trained.
    │   │
    │   │
    │   ├── optimizations      <- Scripts to the overwrite optimization, likes SGD.
    │   │
    │   │
    │   ├── tracking           <- Scripts to some common constant variables.
    │   │
    │   │
    │   ├── train              <- The core of federated learning.
    │   │   ├── controller     <- Controller like a god,  emulates aggregation, attack, record, save model and scheduling nodes.
    │   │   └── nodes          <- Each node train and test, sends the message to the controller.
    │   │
    │   └── visualization      <- Scripts to create exploratory and results-oriented visualizations
    │   │   └── metric_plotter.py
    │   │
    │   ├── config.yaml        <- Scripts to the static config file.
    │   │
    │   │
    │   ├── config_operate.py  <- Merge the config.yaml and input dictionary config.
    │   │
    │   │
    │   └── coordinate.py      <- The most important file includes the logic of how to starte and form a training session.
    │   
    │
    ├── applications      <- The applications of this code.
    │   │
    │   ├── byrd_saga          <- Federated Variance-Reduced Stochastic Gradient Descent with Robustness to Byzantine Attacks.
    │   │   └── main.py   
    │   │
    │   ├── d_ogd              <- Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment.
    │   │   └── main.py
    │   │
    │   └── rsa                <- RSA: Byzantine-Robust Stochastic Aggregation Methods for Distributed Learning from Heterogeneous Datasets.
    │       └── main.py
    │
    │
    ├── main.py            <- The code entry.
    │
    │
    ├── test.py            <- The code test, you can use this to test some methods.
    │
    │    
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------