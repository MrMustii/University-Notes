# Learning Paradigms
## Supervised Learning
The model is trained on a dataset where the correct output or “label” is already provided for each input. 
**Use cases**
* Image classification
* Text classification
* Natural Language Processing (NLP)
* Predictive modeling
* Medical diagnosis
* Object detection in videos and images
**Examples of algorithms**
1. Linear Regression
2. Logistic Regression
3. Decision Trees
4. Random Forest
5. Support Vector Machines (SVM)
6. k-Nearest Neighbors (kNN)
----
The goal of supervised learning is given a labeled training set and minimize the training error that the network does on the training outputs. this is done by defining a cost function and minimize that by a stochastic gradients descent method
## Unsupervised Learning
The model is given a dataset without any labels or output. The model must then find patterns and structure within the data on its own. 
**Use cases**
- Clustering
- Dimensionality reduction
- Anomaly detection
- Generative models
**Examples of algorithms**
1. K-MeansK-Means
2. Hierarchical Clustering
3. PCA(Principal Component Analysis)
4. t-SNE ( t-Distributed Stochastic Neighbor Embedding)

## Reinforcement learning
The model learns from the consequences of its actions. The model receives feedback on its performance, and uses that information to adjust its actions and improve its performance over time.
**Use cases**
- Robotics
- Game playing
- Autonomous vehicles
- Industrial control
- Healthcare
- Finance
**Examples of algorithms**
1. Q-Learning
2. SARSA
3. DQN
4. A3C
# Bias Term in Neural Networks
![[Pasted image 20240903133421.png]]![[Pasted image 20240903133512.png]]
![[Pasted image 20240903133625.png]]
Add a bias term to the input and hidden layers
# Activation Functions 
Activation functions have to be non-linear function because a linear activation function will give a linear network. _non-linear_ means that the output cannot be reproduced from a linear combination of the inputs (which is not the same as output that renders to a straight line--the word for this is _affine_). If all activation functions in a neural network are linear, each layer performs a linear transformation of the input. Mathematically, a series of linear transformations is equivalent to a single linear transformation. This means that no matter how many layers you have, the network can only represent linear relationships, which is no more powerful than a single-layer linear model. To model complex, non-linear patterns, you need non-linear activation functions -gpt
$$
\mathbf h^{(2)} = \mathbf W^{(2)}\mathbf h^{(1)} +\mathbf h^{(2)}
$$
$$
=\mathbf W^{(2)} (\mathbf W^{(1)}\mathbf x +\mathbf b^{(1)} ) + \mathbf b^{(1)}
$$

$$
= (\mathbf W^{(2)} \mathbf W^{(1)}) \mathbf x +(\mathbf W^{(2)}\mathbf b^{(1)} + \mathbf b^{(2)})
$$
$$
=\mathbf W{'} \mathbf x + \mathbf b'
$$
Examples of non leaner activation functions 
- Logistic functions $\sigma(a) = \frac {1}{1+e^{-a}}$  used for binary classification
- Hyperbolic tangent $\tanh(a)=\frac{e^a-e^{-a}}{e^a+e^{-a}}$ 
- Rectified linear $relu(a)=max(0,a)$  
- Softmax $softmax(z)=\frac{\exp(z_i)}{\sum_j\exp(z_j)}$  has property to that we can interpret the outputs as probabilities  moreover
	- $softmax(\boldsymbol z)_i\ge0$ 
	- $\sum_i softmax(\boldsymbol z)_i=1$ 