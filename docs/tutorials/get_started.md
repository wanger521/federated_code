## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.10+
- PyTorch 1.13+
- CUDA 11.6+ (If you run using GPU)

## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n federated-env python=3.10 -y
    conda activate federated-env
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ```
    or
    ```shell
    pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
    ```

4. _You can skip the following CUDA-related content if you plan to run it on CPU._ Make sure that your compilation CUDA version and runtime CUDA version match. 

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.,` 1. If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g.,` 2. If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt package,
    you can use more CUDA versions such as 9.0.

### Get RobustFL and install package

```shell
git clone https://github.com/wanger521/federated_code.git
cd federated_code
python -m pip install -r requirements.txt
```

### A from-scratch setup script

Assuming that you already have CUDA 11.6 installed, here is a full script for setting up MMDetection with conda.

```shell
conda create -n federated-env python=3.10 -y
conda activate federated-env

# Without GPU
conda install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -y

# With GPU
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# install RobustFL
git clone https://github.com/wanger521/federated_code.git
cd federated_code
python -m pip install -r requirements.txt
```

## Verification

To verify whether RobustFL is installed correctly, we can run the following sample code to test.

```python
import src

src.init()
```

The above code is supposed to run successfully after you finish the installation.
