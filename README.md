# MDSR-tf

## Introduction

this project implements super resolution algorithm MDSR and EDSR.

## prepare dataset

the super resolution algorithm is trained on DIV2K. download and prepare the dataset by executing

```shell
python3 create_dataset.py
```

## how to train

train with command

```shell
python3 train.py --model (EDSR|MDSR) --batch_size <batch size>
```

## how to save model

save the trained model with the command

```shell
python3 train.py --save_model --model (EDSR|MDSR)
```

