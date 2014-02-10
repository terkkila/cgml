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

Directory cg/ has some example computational graphs:
```
cg/mnist_logreg_classifier.cg
cg/mnist_mlp_classifier.cg
```

## Usage

To see Logistic Regression in action, try:
```
cgml --cg mnist_logreg_classifier.cg --trainData mnist_train.tsv --testData mnist_test.tsv --learnRate 0.01 --nPasses 10 | bin/score_classification
```

To see Multilayer Perceptron in action, try:
```
cgml --cg mnist_mlp_classifier.cg --trainData mnist_train.tsv --testData mnist_test.tsv --learnRate 0.01 --nPasses 10 | bin/score_classification
```
