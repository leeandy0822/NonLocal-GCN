# NonLocal-GCN

In this work, we present non-local operator combination with the GCN network for mnist number classification. The mnist dataset will first turn into graph by the top-75 light pixels.We want to test the difference and the performance of different non-local operator comparing to the classifcal GCN network.

![](https://i.imgur.com/1zts09v.png)

## Presentation
[LINK](https://docs.google.com/presentation/d/1iiyQROx4b8Xbw9EEzydcFCRy7YKd4TD2fby7ncrDBvg/edit?usp=sharing)
![](https://i.imgur.com/ZYynkPe.png)
![](https://i.imgur.com/lI8McVZ.png)

## Usage
- install pytorch-geometric (You may need to use conda)

- The training code is inside train.py , you can change the network architecture if you want 

## Citations: 
- [MNIST GCN](https://github.com/dna1980drys/mnistGNN)
    - (https://qiita.com/DNA1980/items/8c8258c9a566ea9ea5fc)
- [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)
    - Implementation of [**Non-local Neural Block**](https://arxiv.org/abs/1711.07971).
