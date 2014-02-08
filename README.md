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
- class (classification)
- reg   (regression)
- ae    (autoencoder)
- rl    (reinforcement learning)

Graph needs to start with an input layer, and end with an output layer

Graphs are declared as:
layer -> trans -> layer -> trans -> ... -> trans -> layer

Possible transformations are:
- linear
- sigmoid
- tanh
- softmax

An example graph to classify 4-dimensional input to 8 classes:
```
#classifier.cg
graph class
layer input 4
trans tanh
layer hidden 10
trans softmax
layer output 8
```

An example graph to regress on 4-dimensional input:
```
#regressor.cg
graph reg
layer input 4
trans tanh
layer hidden 10
trans linear
layer output 1
```

And example graph to encode 4-dimensional input to 2 dimensions:
```
#autoencoder.cg
graph ae
layer input 4
trans tanh 
layer output 2
```

An example graph for reinforcement learning 8-dimensional actions from
4-dimensional state space:
```
#rl.cg
graph rl
layer input 4
trans tanh
layer hidden 10
trans softmax
layer output 8
```

