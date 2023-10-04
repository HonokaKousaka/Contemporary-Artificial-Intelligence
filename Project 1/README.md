# Project 1: Text Classification
This is a text classification task. We need to classify thousands of texts into 10 categories.

The data for training is in the file **train_data.txt**, and the data for test is in the file **test.txt**.

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

&nbsp;

## Files

- train_data.txt: The data for training.
- test.txt: The data for test.
- results.txt: The prediction results.
- labelList.npy: A numpy list which stores the labels of training data. Should be loaded by **numpy**.
- textVectorList.npy: A numpy list which stores the vector-converted training data. Should be loaded by **numpy**.
- rawList.npy: A numpy list which stores the training texts. Should be loaded by **numpy**.
- result_list.npy: A numpy list which stores the vector-converted test data. Should be loaded by **numpy**.
- TFIDF_DecisionTree.ipynb: Decision Tree as the classifier, TF-IDF as the converter. Should start by **Jupyter Notebook**.
- TFIDF_Logistic.ipynb: Logistic Regression as the classifier, TF-IDF as the converter. Should start by **Jupyter Notebook**.
- TFIDF_MLP_sklearn.ipynb: Multilayer Perceptron as the classifier, TF-IDF as the converter. Should start by **Jupyter Notebook**.
- TFIDF_SVM.ipynb: Support Vector Machine as the classifier, TF-IDF as the converter. Should start by **Jupyter Notebook**.
- TextCNN.ipynb: TextCNN as the classifier, TF-IDF and Tokenizer as the converters. Should start by **Jupyter Notebook**.
- Word2Vec_Logistic.ipynb: Logistic Regression as the classifier, Word2Vector as the converter. Should start by **Jupyter Notebook**.
- 10215501434李睿恩实验一.pdf: The report of the project.
