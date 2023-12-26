# Project 4: Text Summarization
This is a text summarization task. We need to summarize the medical texts as accurately as possible.

&nbsp;

## Requirements

There are some environment and packages for this project. If you are willing to run the codes, it is strongly recommended to make preparations as follows.

### Environment: Python

Python is compulsory for the codes.

### Platform: AutoDL

As we are training large models, normal equipment will provide extremely low efficiency. In order to fine-tune the model more efficiently, we rent 2 GPUs of RTX 4090 on the platform called **AutoDL**.

The URL of AutoDL is https://www.autodl.com/home.

### Packages

There are some necessary packages you are supposed to install.

- numpy
- pandas
- rouge
- torch
- sklearn
- nltk

It is strongly recommended to install the packages in **requirements.txt**.

### How to Run

There are 2 ways to run the codes.

1. Try running .py files. You can use certain commands to run the codes. Examples are as follows.

`python BART.py`

`python T5.py`

2. If you can not run the .py files successfully, you can try run the .ipynb files.

&nbsp;

## Work

### Step 1: 

Use the package **Transformers** from **HuggingFace**, specifically T5 and BART.

### Step 2: 

Customize the data. 

Data should be in a certain kind of format.

### Step 3:

Train the large models and run a test. The measurement **Rouge, BLEU, METEOR** will be collected.

&nbsp;

## Files

- T5.ipynb: The .ipynb file for T5 large model.
- BART.ipynb: The .ipynb file for BART large model.
- T5.py: The .py file for T5 large model.
- BART.py: The .py file for BART large model.
- train.csv: The training dataset for this task.
- test.csv: The test dataset for this task.
- predictions1(, 2, 3, 4, 5).csv: The prediction file of T5's 5-fold cross-validation.
- predictions.csv: The prediction of test.csv when using T5.
- predictions_bart: The prediction of test.csv when using BART.
- 10215501434李睿恩实验四.pdf: The report for Assignment 4.
