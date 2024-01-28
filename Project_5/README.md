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

## Run pipeline for big-scale datasets
1. Entering the large-scale directory and download 6 big-scale datasets from the repository of [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale). Notice, you should rename the datasets and place them in the right directory.
```python
cd large-scale
```

2. You can run any models implemented in 'models.py'. For examples, you can run our model on 'genius' dataset by the script:
```python
python main.py --dataset genius --sub_dataset None --method mlpnorm
```
And you can run other models, such as 
```python
python main.py --dataset genius --sub_dataset None --method acmgcn
```
For more experiments running details, you can ref the running sh in the 'experiments/' directory.

3. You can reproduce the experimental results of our method by running the scripts:
```python
bash run_glognn_sota_reproduce_big.sh
bash run_glognn++_sota_reproduce_big.sh
```



## Run pipeline for small-scale datasets
1. Entering the large-scale directory and we provide the original datasets with their splits.
```python
cd small-scale
```

2. You can run our model like the script in the below:
```python
python main.py --no-cuda --model mlp_norm --dataset chameleon --split 0
```
Notice, we run all small-scale datasets on CPUs.
For more experiments running details, you can ref the running sh in the 'sh/' directory.


3. You can reproduce the experimental results of our method by running the scripts:
```python
bash run_glognn_sota_reproduce_small.sh
bash run_glognn++_sota_reproduce_small.sh
```


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
