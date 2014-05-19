## Overview

CGML is a library for general-purpose Machine Learning. Underlying is the notion of a computational graph, which can be specified to do various ML tasks such as: 
- classification
- regression
- dimensionality reduction
- reinforcement learning

## Installation

python setup.py install 

## Usage

To see Logistic Regression in action, try:
```
cgml --cg cg/mnist_logreg_classifier.cg \
     --trainData mnist_train.tsv \
     --testData mnist_test.tsv \
     --learnRate 0.01 \
     --nPasses 10 \
     | bin/score_classification
```

To see Convolutional Neural Network with dropout in action, try:
```
cgml --cg cg/mnist_cnn_dropout_classifier.cg \
     --trainData mnist_train.tsv \
     --testData mnist_test.tsv \
     --learnRate 0.01 \
     --nPasses 10 \
     --momentum 0.5 \
     --batchSize 10
     | bin/score_classification
```
