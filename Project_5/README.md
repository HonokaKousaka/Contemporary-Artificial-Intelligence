# Project 5: Multimodal Sentiment Analysis

This is the repository for Multimodal Sentiment Analysis.

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- numpy

- PIL

- pandas

- sklearn

- nltk

- torch

- transformers

- torchvision

- argparse

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- data # It should be here!!! It isn't here because the dataset is too big.
|-- figures # Some figures showing the results of the codes
|-- model # The models of the project
    |-- BERT_ALEX.py # BERT + AlexNet
    |-- BERT_RESNET.py # BERT + ResNet
    |-- BERT_VGG.py # BERT + VGGNet
    |-- image.py # ResNet
    |-- text.py # BERT
|-- predictions # Contains the .txt file for predictions
    |-- predictions.txt # The file for predictions
|-- report # Contains the report for the project
    |-- 10215501434李睿恩实验五.pdf # The report for the project
|-- BERT_ALEXNET.ipynb # BERT + AlexNet
|-- BERT_RESNET.ipynb # BERT + ResNet
|-- BERT_VGGNET.ipynb # BERT + VGGNet
|-- main.py # The main code
|-- requirements.txt # The environment for the codes
|-- test_without_label.txt # The data for testing
|-- train.py # The code for training
|-- train.txt # The data for training
```

## Run the codes
Before you run the codes, you need to make sure the dataset exists.

You can download the dataset through Baidu Netdisk:

```python
URL: https://pan.baidu.com/s/12BJmCCbnvPM2Qzw_NSkzFQ?pwd=dase
Password: dase
```

Please add the directory 'data' to this project as I have claimed above in **Repository structure**.

You have basically two ways to run the codes.

1. 

## Attribution

Parts of this code are based on the following repositories:

- [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

- [PYGCN](https://github.com/tkipf/pygcn)

- [WRGAT](https://github.com/susheels/gnns-and-local-assortativity/tree/main/struc_sim)


## Citation

If you find this code working for you, please cite:

```python
@article{li2022finding,
  title={Finding Global Homophily in Graph Neural Networks When Meeting Heterophily},
  author={Li, Xiang and Zhu, Renyu and Cheng, Yao and Shan, Caihua and Luo, Siqiang and Li, Dongsheng and Qian, Weining},
  journal={arXiv preprint arXiv:2205.07308},
  year={2022}
}
```
