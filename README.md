## Overview

CGML is a library for general-purpose Machine Learning. Underlying is the notion of a computational graph, which can be specified to do various ML tasks such as: 
- classification
- regression
- dimensionality reduction
- reinforcement learning

## Installation

python setup.py install 


## Specifying a Computational Graph

Graph can be of type 
- classifier 
- regressor
- autoencoder
- reinforcement-learner

Graph needs to start with an input layer, and end with an output layer

Graphs are declared as:
layer -> trans -> layer -> trans -> ... -> trans -> layer

Possible transformations are:
- linear
- sigmoid
- tanh
- softmax

An example graph to classify the 28x28 pixel MNIST images to 10 classes: 
```
description: MNIST digit classifier as Computational Graph
type: classifier
n_in: 784
n_out: 10
randomInit: True
graph:
- tanh 300
- softmax 10
```

## Usage
```
cgml --cg mnist_classifier.cg --trainData mnist_train.tsv --testData mnist_test.tsv --learnRate 0.01 --nPasses 10
```
