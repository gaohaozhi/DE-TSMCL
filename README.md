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
The detailed descriptions about the arguments are as following:
| Parameter name | Description of parameter |
| --- | --- |
| dataset_name | The dataset name |
| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader | The data loader used to load the experimental data. This can be set to `forecast_csv`, `forecast_csv_univar`|
| batch_size | The batch size (defaults to 4) |
| repr_dims | The representation dimensions (defaults to 320) |
| gpu | The gpu no. used for training and inference (defaults to 0) |
| eval | Whether to perform evaluation after training |
| momentum | The momentum update value (defaults to 0.999) |

(For descriptions of more arguments, run `python train.py -h`.)

After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. 

**Scripts:** The scripts for reproduction are provided in `scripts/` folder.


