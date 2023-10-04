# Project 1: Text Classification
This is a text classification task. We need to classify thousands of texts into 10 categories.

&nbsp;

## Requirements

There are some indispensable environment and packages for this project. If you are willing to run the codes, you have to make preparations as follows.

### Environment: Python

Python is compulsory for the codes. DataSpell, PyCharm or VSCode is highly recommended.

### Packages

- numpy
- time
- gensim
- nltk
- sklearn
- tensorflow
- keras

&nbsp;

## Work

### Step 1: 

Convert the texts into vectors, making the texts classifiable.

**TF-IDF**, **Word2Vec** and **Tokenizer** (from Keras for TextCNN) are conducted in order to achieve this step.


### Step 2: 

Choose an appropriate classification algorithm.

As traditional machine learning algorithms, **Logistic Regression**, **Support Vector Machine** (SVM), and **Decision Tree** are tried in this step.

As deep learning (neural networks) algorithms, **Multilayer Perceptron** (MLP) and **TextCNN** are tried in this step.


### Step 3:

Predict the classification results of a given file.

The results are shown in the file **results.txt**.

&nbsp;

## Results

**TF-IDF** is highly preferred in this project, as it performs the best among those algorithms for converting texts into vectors.

**Multilayer Perceptron**'s performance is greater than others; therefore, it is selected as the best classifier for this project.

The results are shown in the file **results.txt**.
