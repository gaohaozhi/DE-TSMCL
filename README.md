# DETSMCL

This repository contains the official implementation for the paper: Distillation Enhanced Time Series Forecasting Network with
Momentum Contrastive Learning.

## Requirements

The recommended requirements for DETSMCL are specified as follows:
* Python 3.6/3.8
* torch==1.10.3
* torchvision==0.11.2
* scikit_learn==0.24.2
* scipy==1.6.1
* numpy==1.21.5
* numpy-base==1.23.5
* pandas==1.0.1
* Bottleneck==1.3.1


The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/ETTh1.csv`, `datasets/ETTh2.csv` and `datasets/ETTm1.csv`.
* [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) should be preprocessed using `datasets/preprocess_electricity.py` and placed at `datasets/electricity.csv`.


## Usage

To train and evaluate DETSMCL on a dataset, run the following command:

```train & evaluate
python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
```

