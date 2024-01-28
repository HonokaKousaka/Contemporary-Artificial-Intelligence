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

1. Try running **main.py**. You can use certain commands to run the codes. Examples are as follows.

`python main.py --model bertAlexModel --lr 1e-5 --epoch_num 10`

`python main.py --model bertResModel --lr 1e-5 --epoch_num 10`

`python main.py --model bertVGGModel --lr 1e-5 --epoch_num 10`

`python main.py --image --lr 1e-5 --epoch_num 10`

`python main.py --text --lr 1e-5 --epoch_num 10`

If you use `--model`, then you can follow it with `bertAlexModel`, `bertResModel` or `bertVGGModel`. These are the three models combined BERT with CNN.

If you use `--image`, then you are only using the model ResNet34. 

If you use `--text`, then you are only using the model BERT.

Although `--lr` can change the learning rate and `--epoch_num` can change the number of epoches, it is strongly recommended to follow the commands given above, as they are the most effective parameters through tests.

## Attribution

Parts of this code are based on the following repositories:

- [abhimishra91](https://github.com/abhimishra91/transformers-tutorials)

- [guitld](https://github.com/guitld/Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis/tree/main)

- [Miaheeee](https://github.com/Miaheeee/AI_lab5)
