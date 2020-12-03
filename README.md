# mrc_event_detection


A reading comprehension based event detection framework.

## Data
This repository uses ACE2005 for event detection dataset. 
ACE2005 dataset can be found at https://catalog.ldc.upenn.edu/LDC2006T06.
While it is not freely available, 
we provide data preprocessing code at ACE/preprocessing that generates the format used by this repo.
The resulting files:
- train_process.json,
- test_process.json,
- dev_process.json
should be found in ACE2005.

## Code base strucutre
In *base* directory is the bulk of our framework.

- In package base.nn, there are some modules and utilities that might be used in the framework.
- In package base.model, there are the model scripts based on pytorch_lightning. 
- In package base.data, there are data loading and processing classes.
- In package base.exp, there contains a basic class for scheduling experiments. 

## Model
We implement our models and training framework based on hugginface-transformers and pytorch_lighting. One can find model files in *base/model*.

## Experiments
In *tests*, we saved experiment files. One should move the desired script to the root directory and then execute it. It will train models according to configurations specified in the script. Experiment scripts are compatible with fully supervised and few-shot learning and their evaluations.

## Remarks
This repository is still under construction; we will add more experiment scripts/results and comments on files overtime.
