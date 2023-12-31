# Project 3: Image Classification and Classical Convolutional Neural Networks
This is an image classification task. We need to classify MNIST dataset as correctly as possible.

&nbsp;

## Requirements

There are some indispensable environment and packages for this project. If you are willing to run the codes, you have to make preparations as follows.

### Environment: Python

Python is compulsory for the codes. DataSpell, PyCharm or VSCode is highly recommended.

The codes were originally run on CoLaboratory with the help of T4 GPU.

### Packages

- argparse
- time
- numpy
- tensorflow

It is strongly recommended to install the packages in **requirements.txt**.

### How to Run

There are 2 ways to run the codes.

1. Run the .ipynb files directly.

2. If running .ipynb file is not convenient for you, you can try running .py files. The package **argparse** is used in .py file; therefore, you have to use certain commands to run the codes. Examples are as follows.

`python LeNet.py --lr 0.001 --dropout 0.0`

`python AlexNet.py --lr 0.001 --dropout 0.5`

`python ResNet.py --lr 0.001 --dropout 0.0`

`python VGGNet.py --lr 0.001 --dropout 0.0`

`python GoogLeNet.py --lr 0.001 --dropout 0.4`

`python Inceptionv2.py --lr 0.001 --dropout 0.4`

In these commands, --lr means learning rate, while --dropout menas the dropout rate.

Although these are only examples, we **strongly recommend you to run these 6 .py files with these parameters**. It is proved that these parameters are useful enough.

&nbsp;

## Work

### Step 1: 

Convert the image data and label data into an approriate shape.

### Step 2: 

Construct diverse convolutional neural networks.

In this assignment, LeNet, AlexNet, ResNet, VGGNet, GoogLeNet, InceptionV2 are constructed.

### Step 3:

Train the neural networks and run a test. The time spent and the accuracy will be collected.

&nbsp;

## Files

- LeNet.ipynb: The .ipynb file for LeNet.
- AlexNet.ipynb: The .ipynb file for AlexNet.
- ResNet.ipynb: The .ipynb file for ResNet.
- VGGNet.ipynb: The .ipynb file for VGGNet.
- GoogLeNet.ipynb: The .ipynb file for GoogLeNet.
- Inceptionv2.ipynb: The .ipynb file for Inceptionv2.
- LeNet.py: The .py file for LeNet.
- AlexNet.py: The .py file for AlexNet.
- ResNet.py: The .py file for ResNet.
- VGGNet.py: The .py file for VGGNet.
- GoogLeNet.py: The .py file for GoogLeNet.
- Inceptionv2.py: The .py file for Inceptionv2.
- 10215501434李睿恩实验三: The report for Assignment 3.
