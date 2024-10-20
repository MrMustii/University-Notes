NOTES FROM [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)  \[NNDL\]
NOTES FROM [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) \[SLP\]
NOTES FROM [Deep Learning](https://www.deeplearningbook.org/) \[DLB\]
# \[NNDL\] Chapter 1 Using neural nets to recognize handwritten digits 

## Perceptrons 
- **Input and Output**: Perceptrons take multiple binary inputs, denoted as $x_1$, $x_2$, etc (0 or 1) and produce a single binary output (0 or 1)
- **Weights and Threshold**: Each input to the perceptron has an associated weight, ($w_1$, $w_2$, etc.), a real number that represents the importance of that input. Additionally, a perceptron has a threshold value, also a real number.
- **Weighted Sum and Decision:** The perceptron calculates the weighted sum of its inputs, meaning it multiplies each input by its corresponding weight and adds those products together $\text{output} = \begin{cases} 0 & \text{if } \sum_j w_j x_j \leq \text{threshold} \ 1 & \text{if } \sum_j w_j x_j > \text{threshold} \end{cases}$ 
- **Output Based on Threshold:** If this weighted sum is less than or equal to the threshold, the perceptron outputs 0; if the weighted sum exceeds the threshold, the perceptron outputs 1.
### The Bias Term
To simplify the perceptron model, the book introduce the concept of a bias term. The bias, denoted as b, is defined as the negative of the threshold: $b \equiv -\text{threshold}$.
Using the bias term, the perceptron rule can be rewritten as:

$\text{output} = \begin{cases} 0 & \text{if } w \cdot x + b \leq 0\ 1 & \text{if } w \cdot x + b > 0 \end{cases}$ where: $w \cdot x$ is the dot product of the weight vector w and the input vector x. The bias term represents how easy it is to get the perceptron to output a 1. A larger bias makes it easier for the perceptron to fire, while a smaller (more negative) bias makes it harder.

### Constructing More Complex Networks
The book also discuss the possibility of building multi-layer perceptron networks to handle more complex tasks. These networks are arranged with:

● An input layer that receives the initial data.
● One or more hidden layers that perform intermediate computations.
● An output layer that produces the final result.

The idea is that each subsequent layer of perceptrons can make decisions at a higher level of abstraction by weighing the outputs of the previous layer.

## Sigmoid neurons
While perceptrons can be powerful computational tools, their reliance on a step function activation leads to a significant limitation. As previously noted in our conversation, small changes in the weights or bias of a perceptron can cause its output to flip completely, making the training process challenging. This is because the step function introduces a discontinuity. Sigmoid neurons address the limitations of perceptrons, specifically their abrupt, step-like activation function, by employing a smoother, continuous activation function called the sigmoid function. This allows for more nuanced output and a more stable learning process, crucial for gradient descent algorithms.

The sigmoid function, denoted as $\sigma(z)$, takes any real number as input and outputs a value between 0 and 1. It is mathematically defined as: $\sigma(z) \equiv \frac{1}{1 + e^{-z}}$. The graph of the sigmoid function has a characteristic S-shape, gradually transitioning between 0 and 1. This smooth transition allows for a more continuous and subtle response to changes in the weighted input, unlike the abrupt step-like behaviour of perceptrons. 
![[Pasted image 20241013173017.png]]
Similar to perceptrons, a sigmoid neuron receives multiple inputs ($x_1$, $x_2$, etc.), each with an associated weight ($w_1$, $w_2$, etc.). It also has a bias term, b. However, instead of using a step function, the sigmoid neuron applies the sigmoid function to the weighted sum of its inputs plus the bias.

The output of a sigmoid neuron is therefore given by: $\text{output} = \sigma(w \cdot x + b) = \sigma(\sum_j w_j x_j + b) = \frac{1}{1 + \text{exp}(-(\sum_j w_j x_j + b))}$

When the weighted input to a sigmoid neuron ($z = w \cdot x + b$) is a large positive number, $e^{-z}$ becomes very small, making the output of the sigmoid function approximately 1, similar to a perceptron firing. Conversely, when z is a large negative number, $e^{-z}$ becomes very large, and the output of the sigmoid function approaches 0, similar to a perceptron not firing. The smooth transition of the sigmoid function translates to small changes in the weights and bias producing small changes in the output of the neuron. This relationship can be approximated using calculus: $\Delta \text{output} \approx \sum_j \frac{\partial \text{output}}{\partial w_j} \Delta w_j + \frac{\partial \text{output}}{\partial b} \Delta b$ where:
● $\Delta w_j$ and $\Delta b$ represent small changes in the weights and bias, respectively.
● $\frac{\partial \text{output}}{\partial w_j}$ and $\frac{\partial \text{output}}{\partial b}$ are the partial derivatives of the output with respect to the weights and bias.
The smooth and continuous nature of the sigmoid function makes sigmoid neurons well-suited for gradient descent learning algorithms which we will discussed later. These algorithms rely on calculating the gradient of the cost function with respect to the weights and biases to determine how to adjust them to minimize the cost. With a step function, the cost function can change drastically with small weight adjustments, making it difficult for gradient descent to find the optimal settings.

## Learning with gradient descent
### The Cost Function
A key concept in the learning process is the cost function, which measures how well the network is performing on the training data. A common choice for the cost function, discussed extensively in the previous turn of our conversation, is the quadratic cost function, denoted as C. The quadratic cost function calculates the average squared difference between the network's output and the desired output for each training example. Mathematically, the quadratic cost function is given by: $C = \frac{1}{2n} \sum_x ||y(x)-a^L(x)||^2$ where: 
- n is the total number of training examples.
- y(x) is the desired output for input x.
- $a^L(x)$ is the actual output of the network for input x.
- The sum is taken over all training examples x.
The goal of learning is to find weights and biases that minimize the cost function. A lower cost indicates that the network's outputs are closer to the desired outputs, implying better performance on the training data.
### Minimizing the Cost with Gradient Descent
The idea behind gradient descent is to iteratively adjust the weights and biases in the direction that most rapidly decreases the cost. To understand gradient descent, it's helpful to visualize the cost function as a surface in a high-dimensional space. Each dimension of this space corresponds to a weight or bias in the network. ![[Pasted image 20241013182137.png]]
Okay, let’s suppose we’re trying to minimize some function, $C(v)$. This could be any real-valued function of many variables, $v = v_1,v_2,....$ Note that I’ve replaced the w and b notation by v to emphasize that this could be any function– we’re not specifically thinking in the neural networks context any more.

The goal is to find the lowest point on this surface, which represents the minimum cost. Gradient descent works by:
1. Starting at a random point on the cost surface.
2.   Calculating the gradient of the cost function at that point. The gradient indicates the direction of the steepest ascent on the surface.
3. Moving a small step in the opposite direction of the gradient. This step is determined by the learning rate, denoted as η.
4. Repeating steps 2 and 3 until the cost function stops decreasing significantly.
Mathematically, the weight update rule for gradient descent is:  $w_k \rightarrow w_k' = w_k - \eta \frac{\partial C}{\partial w_k}$ where:
- $w_k$ represents the current weight.
- $w_k'$ represents the updated weight.
- $\eta$ is the learning rate, which controls the size of the step taken in the direction of the negative gradient.
- $\frac{\partial C}{\partial w_k}$ is the partial derivative of the cost function with respect to the weight $w_k$. It represents the rate of change of the cost with respect to that weight.
### Stochastic Gradient Descent
In practice, calculating the gradient of the cost function using the entire training set can be computationally expensive. To address this, the book introduce the concept of stochastic gradient descent, which involves estimating the gradient using a small random sample of the training data, called a mini-batch. Stochastic gradient descent offers several advantages:
- **Computational Efficiency**: Calculating the gradient using a mini-batch is much faster than using the entire training set.
- **Faster Learning**: The frequent updates based on mini-batches allow the network to learn more quickly.
- **Escape from Local Minima**: The noise introduced by using mini-batches can help the algorithm escape from local minima in the cost function and find a better global minimum.
The weight update rule for stochastic gradient descent using a mini-batch of size m is similar to that of regular gradient descent, but the gradient is calculated using the average over the mini-batch: $w_k \rightarrow w_k' = w_k - \frac{\eta}{m} \sum_{j=1}^m \frac{\partial C_{x_j}}{\partial w_k}$ where:
- The sum is taken over the m training examples in the mini-batch.
- $C_{x_j}$ is the cost for the individual training example $x_j$. 
While powerful, gradient descent comes with challenges:
- **Choosing the Learning Rate**: A learning rate that is too large can cause the algorithm to overshoot the minimum, while a learning rate that is too small can result in slow learning.
- **Local Minima**: Gradient descent can sometimes get stuck in local minima of the cost function, failing to find the global minimum.
# \[NNDL\] Chapter 2: How the backpropagation algorithm works
## The two assumptions we need about the cost function
1. The cost function can be expressed as an average over the costs of individual training examples. Mathematically, this means that the overall cost function, denoted as C, can be written as: $C = \frac{1}{n} \sum_x C_x$ where 
	1. n represents the total number of training examples.
	2. The sum is taken over individual training examples, denoted as x.
	3. $C_x$ signifies the cost associated with a single training example x.
2. The cost function can be expressed as a function of the outputs from the neural network.
The first assumption aligns with the practical implementation of backpropagation, where we often compute the gradient of the cost function for a single training example or a small mini-batch of examples and then average those gradients to obtain an estimate of the overall gradient. By breaking down the cost function into individual contributions, backpropagation can efficiently compute the necessary gradients.
The second assumption implies that the cost depends solely on the activations in the output layer, which are denoted as $a^L$ (where L represents the output layer). The specific form of this dependence is not constrained, allowing for various cost functions, such as the quadratic cost or the cross-entropy cost. This assumption is vital because backpropagation aims to determine how changes in the weights and biases of the network influence the output activations, and ultimately, the cost function. By ensuring that the cost depends directly on the outputs, backpropagation can trace the impact of weight and bias modifications through the network, leading to efficient gradient calculations.
## The Hadamard product or Schur product, $s\odot t$
The backpropagation algorithm is based on common linear algebraic operations – things like vector addition, multiplying a vector by a matrix, and soon. But one of the operations is a little less commonly used. In particular, suppose s and t are two vectors of the same dimension. Then we use $s\odot t$ to denote the element wise product of the two vectors. Thus the components of s t are just $(s \odot t)j=s_j*t_j$.As an example
$$
\begin{bmatrix}  
1\\  
2  
\end{bmatrix}
\odot
\begin{bmatrix}  
3\\  
4  
\end{bmatrix}
= 
\begin{bmatrix}  
1*3\\  
2*4  
\end{bmatrix}
=
\begin{bmatrix}  
3\\  
8  
\end{bmatrix}
$$
## The four fundamental equations behind backpropagation
Before diving into the equations, the book establish the notion of error, represented as $\delta^l_j$, which quantifies the sensitivity of the cost function C to changes in the weighted input $z^l_j$ of the j-th neuron in the l-th layer.
### Equation 1: Quantifying Output Error
(BP1) $\delta^L = \nabla_a C \odot \sigma'(z^L)$
This equation focuses on the output layer (denoted by L) and connects the error $\delta^L$ with how the cost function C changes with respect to the output activations. Let's break down the components:
- $\nabla_a C$: This represents the gradient of the cost function with respect to the output activations. It's a vector where each element quantifies how much the cost changes when a specific output activation is altered.
- $\sigma'(z^L)$: This term represents the vector obtained by applying the derivative of the sigmoid function to the weighted input $z^L$ of the output layer. It captures how sensitive the output activation is to changes in the weighted input.
- $\odot$: The symbol denotes the Hadamard product, an element-wise multiplication of the two vectors.
Equation (BP1) reveals that the error in the output layer is determined by how sensitive the cost function is to the output activations and how much the output activations change in response to changes in their weighted inputs.
### Equation 2: Backpropagating the Error
(BP2) $\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$
This equation allows us to propagate the error backward through the network, moving from the output layer toward the input layer. It links the error in a given layer l with the error in the subsequent layer l+1.
- $w^{l+1}$: This is the weight matrix governing the connections between neurons in layer l and layer l+1.
- $(w^{l+1})^T$: Represents the transpose of the weight matrix. 
Equation (BP2) reveals how the error in a layer is influenced by:
- The error in the subsequent layer ($\delta^{l+1}$).
- The weights connecting the two layers ($w^{l+1}$).
- The sensitivity of the activation function to changes in the weighted input ($\sigma'(z^l)$).
By combining (BP2) with (BP1) we can compute the error $\delta^{l}$ for any layer in the network. We start by using (BP1) to compute L, then apply Equation (BP2) to compute $\delta^{L-1}$, then Equation (BP2) again to compute $\delta^{L-2}$, and so on, all the way back through the network. An equation for the rate of change of the cost with respect to any bias in the net
### Equation 3: Cost Sensitivity to Bias
(BP3) $\frac{\partial C}{\partial b^l_j} = \delta^l_j$
Equation (BP3) highlights the direct relationship between the error $\delta^l_j$ and the rate at which the cost function C changes with respect to the bias $b^l_j$ of a neuron. It tells us that the error of a neuron directly quantifies how much the bias of that neuron affects the overall cost. 
### Equation 4: Cost Sensitivity to Weight
(BP4) $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j$
Equation (BP4) reveals how sensitive the cost function C is to changes in a specific weight $w^l_{jk}$ connecting the k-th neuron in layer l-1 to the j-th neuron in layer l. The equation shows that this sensitivity depends on:
- The activation of the neuron feeding into the weight ($a^{l-1}_k$).
- The error of the neuron receiving the input from that weight ($\delta^l_j$). 
This equation is crucial for updating the weights during the learning process. It shows that the weight update depends not only on the error of the receiving neuron but also on the activation of the sending neuron.
## The backpropagation algorithm
The algorithm begins with a single training example, x, and proceeds through the following steps:
1. Input Initialization: 
	1. The activation $a^1$ of the input layer is set to the input example x.
2. Feedforward Computation:
	1. For each layer l from 2 to L (where L represents the output layer), the algorithm computes:
		1. The weighted input to the neurons: $z^l = w^l a^{l-1} + b^l$.
		2. The activation of the neurons: $a^l = \sigma(z^l)$, where $\sigma$ represents the activation function (typically the sigmoid function).
3. Output Error Calculation:
	1. The error in the output layer, $\delta^L$, is computed using the first fundamental equation (BP1): $\delta^L = \nabla_a C \odot \sigma'(z^L)$.
4. Error Backpropagation
	1. The error is propagated backward through the network, layer by layer, from L-1 down to 2, using the second fundamental equation (BP2): $\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$.
5. Gradient Computation
	1. The partial derivatives of the cost function with respect to each weight and bias are calculated using the third (BP3) and fourth (BP4) fundamental equations:
		1.  $\frac{\partial C}{\partial b^l_j} = \delta^l_j$
		2. $\frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j$
### Backpropagation with Stochastic Gradient Descent
In practice, backpropagation is typically combined with stochastic gradient descent (SGD). Instead of computing the gradient for a single training point, SGD uses a mini-batch of m training examples to estimate the overall gradient. This approach offers significant computational advantages, as discussed in our previous conversation. The backpropagation algorithm for a mini-batch proceeds as follows:
- Input a Set of Training Examples:
	- A mini-batch of training examples is selected.
- Gradient Calculation for Each Example:
	- For each training example x in the mini-batch, the algorithm performs steps 1-5 described above to compute the gradient for that example.
- Gradient Descent Update:
	- For each layer l from L down to 2, the weights and biases are updated using the average gradient over the mini-batch:
		- $w^l \rightarrow w^l - \frac{\eta}{m} \sum_x \delta^{x,l}(a^{x,l-1})^T$
		- $b^l \rightarrow b^l - \frac{\eta}{m} \sum_x \delta^{x,l}$
where: 
- $\eta$ is the learning rate.
- The sums are taken over all training examples x in the mini-batch.
- $\delta^{x,l}$ and $a^{x,l-1}$ represent the error and activation for example x in layer l, respectively


# \[NNDL\] Chapter 3: Improving the way neural networks learn
## The Cross-Entropy Cost Function
### The Learning Slowdown Problem
![[Pasted image 20241014162003.png]]
Single Neuron Toy Model
It shows that with the quadratic cost function, the neuron learns very slowly when the initial weight and bias are set such that the neuron's initial output is far from the desired output. Specifically, the graph in shows that learning starts slowly when both the weight and bias are initialized to 2.0. This slow learning occurs because the partial derivative of the quadratic cost function with respect to the weights includes a term, σ′(z), which becomes very small when the neuron is saturated, that is, when the output is close to 0 or 1. 
![[Pasted image 20241014162621.png]] 
Starts the weight and bias 0.6 and 0.9 respectively
![[Pasted image 20241014162804.png]]
the starting weight and bias 2.0 
The desired output for both is 0
### Introducing the Cross-Entropy Cost Function
To overcome this learning slowdown, the subchapter introduces the cross-entropy cost function. For a single neuron with multiple inputs, the cross-entropy cost function is defined as:

$C = -\frac{1}{n} \sum_x [y \ln a + (1 - y) \ln(1 - a)]$ (57) Where
- n is the total number of training data items.
- The sum is over all training inputs, x.
- y is the corresponding desired output.
- a is the actual output of the neuron.
#### Benefits 
- Non-negativity: The cross-entropy, C, is always greater than or equal to 0. This is because all the individual terms in the sum in (57) are negative (as the logarithms are of numbers between 0 and 1) and there is a minus sign in front of the sum.
- Approaches zero as neuron's performance improves: As the neuron gets better at computing the desired output, y, for all training inputs, x, the cross-entropy cost, C, decreases towards zero. This behavior is consistent with what is intuitively expected of a cost function.
- Avoids the learning slowdown problem: The cross-entropy cost function overcomes the learning slowdown problem. This is because the $\sigma′(z)$ term, which was responsible for the slowdown with the quadratic cost function, is canceled out when the partial derivative of the cross-entropy cost with respect to the weights is calculated. This cancellation results in a simplified expression: $\frac{\partial C}{\partial w_j} = \frac{1}{n} \sum_x x_j (\sigma(z) - y)$ 
This expression demonstrates that the rate at which a weight learns is determined by the error in the output, denoted by (σ(z) - y). Thus, the larger the error, the faster the learning. The equation for the partial derivative of the cost with respect to the bias also avoids the $\sigma′(z)$ term, further contributing to overcoming the learning slowdown.

One way to understand the origin of the cross-entropy is by considering the objective of finding a cost function that eliminates the $\sigma′(z)$ term from the gradient. By establishing equations for the desired partial derivatives of the cost function, applying the chain rule, and integrating, we can naturally derive the form of the cross-entropy function.

The cross-entropy can also be interpreted through the lens of information theory as a measure of “surprise.” In this view, the neuron's output is considered its estimated probability distribution for the desired output. The cross-entropy then quantifies the average "surprise" experienced upon learning the true output.
### Softmax
The softmax function normalizes a vector of input values into a probability distribution where each value's probability is proportional to the exponential of the input value. When used with the log-likelihood cost function, a softmax output layer exhibits similar behavior to a sigmoid output layer using the cross-entropy cost, effectively preventing learning slowdown.

## Overfitting and regularization 
### The Problem of Overfitting
The book begin by highlighting that neural networks, particularly large ones, can have a large number of parameters (weights and biases). For instance, the 30 hidden neuron network for MNIST digit classification has nearly 24,000 parameters, while the 100 hidden neuron network has nearly 80,0001. This raises concerns about the network's ability to generalize well to new, unseen data.

![[Pasted image 20241014170711.png]]
This looks encouraging, showing a smooth decrease in the cost, just as we expect. Note that I've only shown training epochs 200 through 399. This gives us a nice up-close view of the later stages of learning, which, as we'll see, turns out to be where the interesting action is.
Let's now look at how the classification accuracy on the test data changes over time:

![[Pasted image 20241014170745.png]]
The accuracy rises to just under 82 percent. The learning then gradually slows down. Finally, at around epoch 280 the classification accuracy pretty much stops improving. Later epochs merely see small stochastic fluctuations near the value of the accuracy at epoch 280. Contrast this with the earlier graph, where the cost associated to the training data continues to smoothly drop. If we just look at that cost, it appears that our model is still getting "better". But the test accuracy results show the improvement is an illusion. Just like the model that Fermi disliked, what our network learns after epoch 280 no longer generalizes to the test data. And so it's not useful learning. We say the network is _overfitting_ or _overtraining_ beyond epoch 280.

What would happen if we compared the cost on the training data with the cost on the test data, so we're comparing similar measures? Or perhaps we could compare the classification accuracy on both the training data and the test data? In fact, essentially the same phenomenon shows up no matter how we do the comparison. The details do change, however. For instance, let's look at the cost on the test data:
![[Pasted image 20241014171135.png]]
Another sign of overfitting may be seen in the classification accuracy on the training data
![[Pasted image 20241014171233.png]]
The accuracy rises all the way up to 100 percent. That is, our network correctly classifies all 1,000training images! Meanwhile, our test accuracy tops out at just 82.27 percent. So our network really is learning about peculiarities of the training set, not just recognizing digits in general. It's almost as though our network is merely memorizing the training set, without understanding digits well enough to generalize to the test set.

### Detect overfitting
keeping track of accuracy on the test data as our network trains. If we see that the accuracy on the test data is no longer improving, then we should stop training. Of course, strictly speaking, this is not necessarily a sign of overfitting. It might be that accuracy on the test data and the training data both stop improving at the same time. Still, adopting this strategy will prevent overfitting.

### Early stopping
Using the validation data we'll compute the classification accuracy on the validation data at the end of each epoch. Once the classification accuracy on the validation data has saturated, we stop training. Of course, in practice we won't immediately know when the accuracy has saturated. Instead, we continue training until we're confident that the accuracy has saturated. Keep in mind that neural networks sometimes plateau for a while in training, before continuing to improve  

### Regularization
Regularization techniques are designed to prevent this overfitting by adding constraints to the learning process, essentially guiding the network towards learning more general patterns rather than memorizing specific examples.
Regularization works by :
- **Penalizing Complexity:** Most regularization techniques work by adding a penalty term to the network's cost function. The cost function measures how well the network is performing; a lower cost generally means better performance. By adding a penalty term, we make the network pay a price for having large weights, which often correspond to overly complex models
- **Encouraging Simplicity**: This penalty encourages the network to find a balance between minimizing the original cost (fitting the training data well) and keeping the weights small (avoiding excessive complexity). This balance helps the network learn more general patterns, leading to better generalization
### Types of Regularization

- L2 Regularization (Weight Decay)
	- The Formula: In L2 regularization, we add a term to the cost function that is proportional to the sum of the squares of all the weights in the network. This term is scaled by a factor called the regularization parameter (λ).
	- The Effect: This penalty discourages the network from having large weights, effectively pushing the weights towards smaller values. Smaller weights generally correspond to simpler models that are less prone to overfitting.
	- The Intuition: Imagine pulling on a rubber band connected to each weight in the network. The rubber band tries to pull each weight towards zero. The stronger the rubber band (larger λ), the harder it pulls, leading to smaller weights.
- L1 Regularization
	- The Formula: Similar to L2 regularization, but instead of squaring the weights, we add the sum of their absolute values to the cost function.
	- The Effect: L1 regularization also encourages small weights, but it tends to drive many weights all the way to zero, effectively creating a sparse network where only a few connections are strong. This can be useful for feature selection, as it highlights the most important connections
- Dropout 
	- The Concept: Dropout is a more radical approach where we randomly "drop out" (deactivate) a portion of the hidden neurons during each training iteration.
	- The Effect: This forces the network to learn more robust features that are not dependent on any single neuron. By learning to generalize even with missing neurons, the network becomes less reliant on specific examples and more adaptable to new data
	- The Intuition: It's like training a team where each member has to be prepared to step up if others are unavailable. This encourages individual strength and adaptability, making the team more resilient overall.
- Artificially Expanding the Training Data
	- The Idea: Instead of modifying the network or the cost function, we can increase the size of the training dataset by creating variations of the existing examples
	- The Approach: This can involve techniques like rotating, skewing, or translating images, adding noise to audio data, or generating synthetic data based on the existing examples.
	- The Benefit: By exposing the network to a more diverse range of examples, we help it learn more robust and generalizable features, reducing overfitting.



## Weight initialization 
Saturation occurs when the output of an activation function is very close to its maximum or minimum value, resulting in very small gradients and slow learning. Consider a network with 1,000 input neurons where normalized Gaussians are used to initialize the weights connecting to the first hidden layer. If the inputs to the network are also normalized Gaussian random variables, each of the 1,000 weighted inputs to the first hidden layer will also be a Gaussian random variable with a mean of 0 and a standard deviation of approximately 1. This means that it is highly probable for a large number of weighted inputs to the activation function to fall in the region where the sigmoid function is very flat, leading to very slow learning. This issue of saturation and slow learning is not resolved by using a different cost function like cross-entropy, as this solution only addresses saturated output neurons and not saturated hidden neurons. Saturation in later hidden layers can also occur if weights are initialized using normalized Gaussians.

An Improved initialization approach to prevent saturation and the resulting slowdown in learning is for a neuron with n<sub>in</sub> input weights, this method initializes those weights as Gaussian random variables with a mean of 0 and a standard deviation of 1/√n<sub>in</sub>. This squashes the Gaussians, making saturation less likely. The bias can continue to be chosen as a Gaussian with a mean of 0 and a standard deviation of 1. Using this method with the example 1,000 input neuron network results in each of the weighted sums being a Gaussian random variable with a mean of 0 and a standard deviation of √3/2, which is much less likely to saturate than a neuron with a standard deviation of 1.

In the improved initialization method, the bias can be initialized as Gaussian random variables with a mean of 0 and a standard deviation of 1 because this does not significantly increase the likelihood of saturation. In fact, the initialization of the biases doesn't have a large impact as long as saturation is avoided.

## How to choose a neural network's hyper-parameters?
**Broad strategy:** When using neural networks to attack a new problem the first challenge is to get _any_ non-trivial learning, i.e., for the network to achieve results better than chance. This can be surprisingly difficult, especially when confronting a new class of problem. Let's look at some strategies you can use if you're having this kind of trouble.

Example, in the MNIST  Get rid of all the training and validation images except images which are 0s or 1s. Then try to train a network to distinguish 0s from 1s. Not only is that an inherently easier problem than distinguishing all ten digits, it also reduces the amount of training data by 80 percent, speeding up training by a factor of 5. That enables much more rapid experimentation, and so gives you more rapid insight into how to build a good network.

Start a smaller networks. If you believe a \[784, 10\] network can likely do better-than-chance classification of MNIST digits, then begin your experimentation with such a network. It'll be much faster than training a \[784, 30, 10\] network, and you can build back up to the latter.

You can get another speed up in experimentation by increasing the frequency of monitoring.  We can get feedback more quickly by monitoring the validation accuracy more often, say, after every 1,000 training images. instead of every epoch (in MNIST there are 50000 images). Furthermore, instead of using the full 10,000 image validation set to monitor performance, we can get a much faster estimate using just 100 validation images.

Once we've explored to find an improved value for η, then we move on to find a good value for λ. Then experiment with a more complex architecture, say a network with 10 hidden neurons. Then adjust the values for η and λ again. Then increase to 20 hidden neurons. And then adjust other hyper-parameters some more.

### Learning rate
**Learning rate:** Suppose we run three MNIST networks with three different learning rates, η=0.025η=0.025, η=0.25η=0.25 and η=2.5η=2.5, respectively. We'll set the other hyper-parameters as for the experiments in earlier sections, running over 30 epochs, with a mini-batch size of 10, and with λ=5.0λ=5.0. We'll also return to using the full 50,00050,000 training images. Here's a graph showing the behaviour of the training cost as we train
![[Pasted image 20241014185714.png]]
 To understand the reason for the oscillations, recall that stochastic gradient descent is supposed to step us gradually down into a valley of the cost function, However, if η is too large then the steps will be so large that they may actually overshoot the minimum, causing the algorithm to climb up out of the valley instead. 

First, we estimate the threshold value for η at which the cost on the training data immediately begins decreasing, instead of oscillating or increasing. This estimate doesn't need to be too accurate. You can estimate the order of magnitude by starting with η=0.01. If the cost decreases during the first few epochs, then you should successively try η=0.1,1.0,…until you find a value for η where the cost oscillates or increases during the first few epochs. This gives us an estimate for the threshold value of η.

Obviously, the actual value of η that you use should be no larger than the threshold value. In fact, if the value of η is to remain usable over many epochs then you likely want to use a value for η that is smaller, say, a factor of two below the threshold. Such a choice will typically allow you to train for many epochs, without causing too much of a slowdown in learning.

### The number of training epochs
Early stopping means that at the end of each epoch we should compute the classification accuracy on the validation data. When that stops improving, terminate. This makes setting the number of epochs very simple. In particular, it means that we don't need to worry about explicitly figuring out how the number of epochs depends on the other hyper-parameters. Instead, that's taken care of automatically. Furthermore, early stopping also automatically prevents us from overfitting. This is, of course, a good thing, although in the early stages of experimentation it can be helpful to turn off early stopping, so you can see any signs of overfitting, and use it to inform your approach to regularization.

To implement early stopping we need to say more precisely what it means that the classification accuracy has stopped improving. As we've seen, the accuracy can jump around quite a bit, even when the overall trend is to improve. If we stop the first time the accuracy decreases then we'll almost certainly stop when there are more improvements to be had. A better rule is to terminate if the best classification accuracy doesn't improve for quite some time. Then we might elect to terminate if the classification accuracy hasn't improved during the last ten epochs. This ensures that we don't stop too soon, in response to bad luck in training, but also that we're not waiting around forever for an improvement that never comes. using the no-improvement-in-ten rule for initial experimentation, and gradually adopting more lenient rules, as you better understand the way your network trains: no-improvement-in-twenty, no-improvement-in-fifty, and so on. Of course, this introduces a new hyper-parameter to optimize!. 

### Learning rate schedule 
 We've been holding the learning rate ηη constant. However, it's often advantageous to vary the learning rate. Early on during the learning process it's likely that the weights are badly wrong. And so it's best to use a large learning rate that causes the weights to change quickly. Later, we can reduce the learning rate as we make more fine-tuned adjustments to our weights.

Many approaches are possible. One natural approach is to use the same basic idea as early stopping. The idea is to hold the learning rate constant until the validation accuracy starts to get worse. Then decrease the learning rate by some amount, say a factor of two or ten. We repeat this many times, until, say, the learning rate is a factor of 1,024 (or 1,000) times lower than the initial value. Then we terminate.

### The regularization parameter, λ
Start initially with no regularization (λ=0.0), and determining a value for η, as above. Using that choice of η, we can then use the validation data to select a good value for λ. Start by trialling λ=1.0 and then increase or decrease by factors of 10, as needed to improve performance on the validation data. Once you've found a good order of magnitude, you can fine tune your value of λ. That done, you should return and re-optimize η again. 

### Mini-batch size
 first suppose that we're doing online learning, i.e., that we're using a mini-batch size of 1. The obvious worry about online learning is that using mini-batches which contain just a single training example will cause significant errors in our estimate of the gradient. In fact, though, the errors turn out to not be such a problem. The reason is that the individual gradient estimates don't need to be super-accurate. All we need is an estimate accurate enough that our cost function tends to keep decreasing. It's as though you are trying to get to the North Magnetic Pole, but have a wonky compass that's 10-20 degrees off each time you look at it. Provided you stop to check the compass frequently, and the compass gets the direction right on average, you'll end up at the North Magnetic Pole just fine.

Choosing the best mini-batch size is a compromise. Too small, and you don't get to take full advantage of the benefits of good matrix libraries optimized for fast hardware. Too large and you're simply not updating your weights often enough. What you need is to choose a compromise value which maximizes the speed of learning. Fortunately, the choice of mini-batch size at which the speed is maximized is relatively independent of the other hyper-parameters (apart from the overall architecture).  The way to go is therefore to use some acceptable (but not necessarily optimal) values for the other hyper-parameters, and then trial a number of different mini-batch sizes, scaling η as above. Plot the validation accuracy versus _time_ (as in, real elapsed time, not epoch!), and choose whichever mini-batch size gives you the most rapid improvement in performance. With the mini-batch size chosen you can then proceed to optimize the other hyper-parameters.


**Summing up:** Following the rules-of-thumb described won't give you the absolute best possible results from your neural network. But it will likely give you a good start and a basis for further improvements. In particular, I've discussed the hyper-parameters largely independently. In practice, there are relationships between the hyper-parameters. You may experiment with η, feel that you've got it just right, then start to optimize for λ, only to find that it's messing up your optimization for η. In practice, it helps to bounce backward and forward, gradually closing in good values. Above all, keep in mind that the heuristics I've described are rules of thumb, not rules cast in stone. You should be on the lookout for signs that things aren't working, and be willing to experiment. In particular, this means carefully monitoring your network's behaviour, especially the validation accuracy.

## Other techniques

While stochastic gradient descent (SGD) is a powerful and widely used algorithm for training neural networks, the book acknowledges that other optimization techniques may offer superior performance in some situations.

- Hessian Technique: This technique utilizes the Hessian matrix, which contains information about the second-order partial derivatives of the cost function. This additional information can help the Hessian approach converge faster and avoid some issues that can arise with standard gradient descent. However, the source notes that computing the Hessian matrix can be computationally expensive, particularly for large networks.

- Momentum Technique: This technique introduces a "momentum" term that helps the algorithm build up speed as it moves down the gradient. This can help the algorithm avoid getting stuck in local minima and converge more quickly. The momentum term is controlled by a hyper-parameter, µ, which is typically set to a value between 0 and 1.

The book also briefly mentions other optimization techniques, including conjugate gradient descent, the BFGS method, and Nesterov's accelerated gradient technique. However, the source states that plain SGD, particularly with the momentum technique, works well for many problems and remains the primary optimization method used throughout the book.



# \[NNDL\] Chapter 6 Deep learning 
## Convolutional Neural Networks
These networks use a special architecture which is particularly well-adapted to classify images. Using this architecture makes convolutional networks fast to train. This, in turn, helps us train deep, many-layer networks, which are very good at classifying images.  Convolutional neural networks use three basic ideas: local receptive fields, shared weights, and pooling. 
### Local receptive fields
In an feed forward neural network , the inputs were depicted as a vertical line of neurons. In a convolutional net, it’ll help to think instead of the inputs as a 28 × 28 (matrix) square of neurons, whose values correspond to the 28 × 28 pixel intensities we’re using as inputs (the 28 × 28 are the images form MNIST).
![](Pasted%20image%2020240921144154.png)
connect the input pixels to a layer of hidden neurons. But we won’t connect every input pixel to every hidden neuron. Instead, we only make connections in small, localized regions of the input image.  To be more precise, each neuron in the first hidden layer will be connected to a small region of the input neurons, say, for example, a 5 × 5 region, corresponding to 25 input pixels. So, for a particular hidden neuron, we might have connections that look like this:
![](Pasted%20image%2020240921144305.png)
That region in the input image is called the local receptive field for the hidden neuron. It’s a little window on the input pixels. Each connection learns a weight. And the hidden neuron learns an overall bias as well. You can think of that particular hidden neuron as learning to analyze its particular local receptive field.

We then slide the local receptive field across the entire input image. For each local receptive field, there is a different hidden neuron in the first hidden layer. To illustrate this concretely, let’s start with a local receptive field in the top-left corner Then we slide the local receptive field over by one pixel to the right (i.e., by one neuron), to connect to a second hidden neuron:
![](Pasted%20image%2020240921144544.png)
Note that if we have a 28×28 input image, and 5 × 5 local receptive fields, then there will be 24 × 24 neurons in the hidden layer. 

I’ve shown the local receptive field being moved by one pixel at a time. In fact, sometimes a different stride length is used. For instance, we might move the local receptive field 2 pixels to the right (or down), in which case we’d say a stride length of 2 is used. In this chapter we’ll mostly stick with stride length 1, but it’s worth knowing that people sometimes experiment with different stride lengths. if we’re interested in trying different stride lengths then we can use validation data to pick out the stride length which gives the best performance. The same approach may also be used to choose the size of the local receptive field – there is, of course, nothing special about using a 5 × 5 local receptive field. In general, larger local receptive fields tend to be helpful when the input images are significantly larger than the 28 × 28 pixel MNIST images.

### Shared weights and biases
I’ve said that each hidden neuron has a bias and 5 × 5 weights connected to its local receptive field. we’re going to use the same weights and bias for each of the 24 × 24 hidden neurons. In other words, for the j, k-th hidden neuron, the output is: 
![](Pasted%20image%2020240921145548.png)
Here, σ is the neural activation function. b is the shared value for the bias. $w_{l,m}$ is a 5 × 5 array of shared weights. And, finally, we use $a_{x, y}$ to denote the input activation at position x, y.

This means that all the neurons in the first hidden layer detect exactly the same feature , just at different locations in the input image. To see why this makes sense, suppose the weights and bias are such that the hidden neuron can pick out, say, a vertical edge in a particular local receptive field. That ability is also likely to be useful at other places in the image. And so it is useful to apply the same feature detector everywhere in the image. To put it in slightly more abstract terms, convolutional networks are well adapted to the translation invariance of images: move a picture of a cat (say) a little ways, and it’s still an image of a cat.

For this reason, we sometimes call the map from the input layer to the hidden layer a feature map. We call the weights defining the feature map the shared weights. And we call the bias defining the feature map in this way the shared bias. The shared weights and bias are often said to define a kernel or filter

The network structure I’ve described so far can detect just a single kind of localized feature. To do image recognition we’ll need more than one feature map. And so a complete convolutional layer consists of several different feature maps:
![](Pasted%20image%2020240921151544.png)
In the example shown, there are 3 feature maps. Each feature map is defined by a set of 5 × 5 shared weights, and a single shared bias. The result is that the network can detect 3 different kinds of features, with each feature being detectable across the entire image. However, in practice convolutional networks may use more (and perhaps many more) feature maps. 
![](Pasted%20image%2020240921151940.png)
The 20 images correspond to 20 different feature maps (or filters, or kernels). Each map is represented as a 5 × 5 block image, corresponding to the 5 × 5 weights in the local receptive field. Whiter blocks mean a smaller weight and darker blocks mean a larger weight, so the feature map responds more to the corresponding input ., it’s difficult to see what these feature detectors are learning. Certainly, we’re not learning (say) the Gabor filters which have been used in many traditional approaches to image recognition.

A big advantage of sharing weights and biases is that it greatly reduces the number of parameters involved in a convolutional network. For each feature map we need 25 = 5 × 5 shared weights, plus a single shared bias. So each feature map requires 26 parameters. If we have 20 feature maps that’s a total of 20 × 26 = 520 parameters defining the convolutional layer. By comparison, suppose we had a fully connected first layer, with 784 = 28 × 28 input neurons, and a relatively modest 30 hidden neurons.

Incidentally, the name convolutional comes from the fact that the operation in equation above is sometimes known as a convolution. A little more precisely, people sometimes write that equation as $a^1 = σ(b + w ∗ a^0)$, where $a^1$ denotes the set of output activations from one feature map, $a^0$ is the set of input activations, and ∗ is called a convolution operation.

In short, each "kernel"  for the feature detection has some weights, these weights are the same when using that kernel across the whole input. meaning that when applying it on the top right comer the weights are the same as when applying them at the bottom left. when trying to detected multiple features, each kernel is applied to the input layer.  

### Pooling layers
Pooling layers are usually used immediately after convolutional layers. What the pooling layers do is simplify the information in the output from the convolutional layer. In detail, a pooling layer takes each feature map output from the convolutional layer and prepares a condensed feature map. For instance, each unit in the pooling layer may summarize a region of (say) 2 × 2 neurons in the previous layer. As a concrete example, one common procedure for pooling is known as max-pooling. In max-pooling, a pooling unit simply outputs the maximum activation in the 2 × 2 input region, as illustrated in the following diagram.![](Pasted%20image%2020240921160017.png)
Note that since we have 24 × 24 neurons output from the convolutional layer, after pooling we have 12 × 12 neurons. 
As mentioned above, the convolutional layer usually involves more than a single feature map. We apply max-pooling to each feature map separately. So if there were three feature maps, the combined convolutional and max-pooling layers would look like:
![](Pasted%20image%2020240921160111.png)
We can think of max-pooling as a way for the network to ask whether a given feature is found anywhere in a region of the image. It then throws away the exact positional information. The intuition is that once a feature has been found, its exact location isn’t as important as its rough location relative to other features. A big benefit is that there are many fewer pooled features, and so this helps reduce the number of parameters needed in later layers. 

Max-pooling isn’t the only technique used for pooling. Another common approach is known as L2 pooling. Here, instead of taking the maximum activation of a 2 × 2 region of neurons, we take the square root of the sum of the squares of the activations in the 2 × 2 region. While the details are different, the intuition is similar to max-pooling: L2 pooling is a way of condensing information from the convolutional layer. In practice, both techniques have been widely used. And sometimes people use other types of pooling operation. 

### Putting it all together
![](Pasted%20image%2020240921160611.png)
The network begins with 28×28 input neurons, which are used to encode the pixel intensities for the MNIST image. This is then followed by a convolutional layer using a 5 × 5 local receptive field and 3 feature maps. The result is a layer of 3×24×24 hidden feature neurons. The next step is a max-pooling layer, applied to 2 × 2 regions, across each of the 3 feature maps. The result is a layer of 3 × 12 × 12 hidden feature neurons.
The final layer of connections in the network is a fully-connected layer. That is, this layer connects every neuron from the max-pooled layer to every one of the 10 output neurons. This fully-connected architecture is the same as we used in earlier chapters. Note, however, that in the diagram above, I’ve used a single arrow, for simplicity, rather than showing all the connections.

We will train our network using stochastic gradient descent and backpropagation. This mostly proceeds in exactly the same way as in earlier chapters. However, we do need to make a few modifications to the backpropagation procedure. The reason is that our earlier derivation of backpropagation was for networks with fully-connected layers. 

##  Convolutional Neural Networks in Practice
USING CHAT GPT 
The chapter delves into how convolutional neural networks (CNNs) work in practice, focusing on the reasoning behind the architectural choices and the iterative improvements made to achieve higher accuracy in image classification tasks. While the implementations revolve around the MNIST digit classification problem, the insights provided are generally applicable to any deep learning task that uses CNNs.

The initial approach starts with a shallow network containing a single hidden layer of 100 neurons, trained over 60 epochs. This baseline achieves an accuracy of about 97.8%, which is close to previous experiments but lacks more advanced regularization techniques and improvements. One key improvement in the updated implementation is the replacement of the sigmoid activation function and cross-entropy loss function with softmax and log-likelihood, respectively, which aligns more with modern image classification methods.

The next logical step to improve performance involves the introduction of a convolutional layer at the start of the network. This convolutional layer learns local spatial structures from the input images, extracting features using small 5x5 receptive fields with a stride of 1 and 20 feature maps. Following the convolutional layer, a max-pooling layer reduces the spatial dimensions, aggregating features with 2x2 pooling windows.

The convolutional and pooling layers allow the network to capture local features while reducing the complexity before passing the data to a fully connected layer. This helps the network integrate global information from the image and increases classification accuracy. The first modification leads to an accuracy of 98.78%, a significant improvement over the earlier shallow network, reducing the error rate by more than a third.

An interesting aspect discussed is the decision to treat convolution and pooling layers as a single unit. While they could be separated, combining them simplifies the network architecture without affecting performance. This design can be replicated in other CNN architectures, especially when you aim to simplify the overall codebase while retaining the learning of local spatial structures.

To further improve the model, a second convolutional-pooling layer is added between the first convolutional-pooling layer and the fully connected layers. The reasoning behind this is that the second layer can now learn abstracted features from the condensed output of the first layer. This abstraction retains spatial structure but represents more complex patterns, boosting accuracy to 99.06%.

A key consideration with deeper layers is how they handle the input from previous layers. Each neuron in the second convolutional layer learns from all the feature maps of the previous layer but only within a localized receptive field, providing the network with more detailed information without overwhelming complexity. This approach allows the model to handle multiple inputs while learning more intricate spatial relationships.

Moving from traditional activation functions like sigmoid to rectified linear units (ReLUs) was another vital step. ReLUs generally speed up training because they don't saturate for large inputs, allowing the network to continue learning even with high values of the input. This shift resulted in a modest improvement in accuracy to 99.23%, but more importantly, it made training more efficient. This improvement demonstrates why ReLUs have become a popular choice in modern CNNs.

An additional performance boost was achieved by expanding the training data. By algorithmically shifting the MNIST images (e.g., translating them by one pixel in each direction), the training set size was increased fivefold. This helped mitigate overfitting and further enhanced the model's ability to generalize. Training on this expanded dataset resulted in an accuracy of 99.37%. This technique can be applied in other contexts where training data is limited, especially in cases where overfitting is a concern.

Finally, the chapter explores the concept of adding dropout to the fully connected layers, where random neurons are "dropped" during training. This regularization method prevents the network from relying too heavily on specific neurons, making it more robust. Dropout was not applied to the convolutional layers because the weight-sharing mechanism in these layers already provides resistance to overfitting. Applying dropout to the fully connected layers led to a significant improvement, achieving 99.60% accuracy.

This approach was then combined with an ensemble method, where multiple networks were trained, and their outputs were combined through voting. This led to an accuracy of 99.67%, showing that combining predictions from several models can reduce individual errors.

### Application in Other Deep Learning Networks

The key insights from this chapter can be applied to a wide range of deep learning models:

- **Use of Convolutional Layers**: The hierarchical structure of CNNs, starting with convolutional layers to capture local spatial relationships, is crucial for any task involving structured data such as images. This principle is applicable in tasks like object detection, facial recognition, and even some natural language processing tasks where spatial structure matters.
    
- **Pooling Layers for Dimensionality Reduction**: Max-pooling layers are effective at reducing the dimensionality of data while preserving essential information. This can help manage computational complexity in any deep learning task with high-dimensional inputs.
    
- **Rectified Linear Units (ReLUs)**: Switching to ReLUs as the activation function is a good practice for faster training in any neural network, especially when training deep models. ReLUs prevent the vanishing gradient problem, making them a default choice in deep learning.
    
- **Data Augmentation**: The technique of expanding training data algorithmically can be adapted to other domains where gathering large datasets is challenging. For instance, in text classification, data augmentation techniques could involve paraphrasing sentences or injecting noise into the inputs.
    
- **Dropout Regularization**: Dropout remains a powerful regularization technique, especially in networks prone to overfitting. It can be applied to fully connected layers in almost any neural network architecture to improve generalization. However, it is unnecessary in convolutional layers where shared weights provide enough regularization.
    
- **Ensemble Methods**: Combining several trained models to vote on predictions is a general technique applicable to most machine learning problems, particularly when marginal gains in accuracy are needed.



# Language Modelling and Transformers 
## Text to vectors 
### Tokenization
Text can be decomposed into various types of units or *tokens*: characters, syllables, words or even sentences. Each tokenization system comes with vocabulary $\mathcal{V}$ that references all known symbols. 

The choice of tokenizer is a tradeoff between the size of the vocabulary and the number of tokens required to encode a sentence. For instance, character-level tokenizers result in a smaller vocabulary size (only 128 character when using ASCII encoding) than other tokenizers. Word-based tokenizers encode text using fewer tokens than the other tokenizers but require a much larger vocabulary, which still might miss words seen at test time. Sub-words tokenizers such as [WordPiece](https://arxiv.org/abs/2012.15524) and [byte-pair encoding (BPE)](https://arxiv.org/abs/1508.07909) are a tradeoff between character-level and word-level encoding. They have progressively taken over the field as they provide two main advantages: (i) good tradeoff between vocabulary size and encoding length, (ii) open-ended vocabulary.


### Embeddings
A tokenizer transforms fragments of text into list of integers that maps a vocabulary. We assign one vector of dimension $d$ to each item in the vocabulary of size $N_\mathcal{V}$, this results in a matrix $E$ of dimension ${N_\mathcal{V} \times d}$. Converting a fragment of text into a sequence of vector representations can be done by tokenizing the text, and then looking up the embedding vector for each token, which is equivalent to *one-hot encoding* the tokens and performing a matrix multiplication using $E$. Given $\mathbf{t}_1, \ldots, \mathbf{t}_L$ the sequence of one-hot encoded tokens, this is equivalent to

$$

\mathbf{w}_i = E  \mathbf{t}_i ,

$$

The first step is to break down a text into  tokens. Each token is then mapped to a unique integer from the tokenizer's vocabulary. For example, if the vocabulary has 50,000 words, each token gets assigned an integer between 1 and 50,000.

For instance, consider the sentence:

**"Hello World!"**

After tokenization, it might be converted into tokens like `["hello", "world", "!"]`, and each token is mapped to an integer (Using BPE):
- "hello" → 7592
- "world" → 2088
- "!" → 999
Once we have these tokens as integers, we use an **embedding matrix** to convert them into vectors. The embedding matrix, denoted as $E$, is a large matrix where each row corresponds to the vector representation of a token in the vocabulary. If the vocabulary contains $N_\mathcal{V}$ tokens, and each token is represented by a vector of length $d$ (the embedding dimension), the matrix $E$ will have dimensions $N_\mathcal{V} \times d$. For example, if we have a vocabulary size of 50,000 and we want 300-dimensional embeddings, the matrix $E$ will be $50,000 \times 300$. 

To convert tokens into their vector representations, we perform an operation similar to **one-hot encoding**. Each token is represented as a one-hot vector, where all entries are 0 except for the position corresponding to the token's index in the vocabulary.

Once we have the one-hot encoded vector, we multiply it by the embedding matrix $E$ to get the corresponding embedding vector. The one-hot vector essentially "selects" the row of the matrix that corresponds to the token's index. This multiplication can be written as:

$$
\mathbf{w}_i = E  \mathbf{t}_i 
$$
where - $\mathbf{t}_i$ is the one-hot encoded vector for the $i$-th token.

For example, for the token "hello", this multiplication retrieves the 7529-th row of the embedding matrix $E$, which is the embedding vector corresponding to "hello". This vector might look something like this: $W_{hello}=[0.2,−0.5,0.1,…,0.7]$
### World vectors
![[Pasted image 20241016161815.png]]
Word2vec converts words into vector representations, which are learned using the Skip-Gram algorithm. Intuitively, The algorithm is based on the idea that words that appear together are related to each other.

The word vector space allows to use the inner product to compare words (dot product), and arithmetic operations to manipulate word representations. For instance, in a well-defined word vector space, the concept "king" can be translated into "queen" by applying a linear transformation and the vector. `vec("captial") - vec("country")` was found to correspond to the relative concept `"capital city of a country"` . 
## Language Models 
### Language Modelling
**Autoregressive factorization** Language models aim at grasping the underlying linguistic structure of a text fragment: whereas word vectors model words independently of each others, a language model tracks the grammatical and semantic relationships between word tokens. Given a piece of text encoded into tokens $\mathbf{w}_{1:T} = [\mathbf{w_1}, \ldots, \mathbf{w}_T]$ a *left-to-right* language model describes $\mathbf{w}_{1:T}$ with the following factorization:

$$

 p_\theta(\mathbf{w}_{1:T}) = \prod_{t=1}^T p_\theta(\mathbf{w}_t \mid \mathbf{w}_{<t}) \ ,

$$

where $\theta$ is a model parameter. The above *autoregressive* factorization describes a *recursive* function $p_\theta(\mathbf{w}_t \mid \mathbf{w}_{<t})$, which is shared across all the time steps. In the above figure, we represent a left-to-right language model with dependencies represented by arrows for fixed steps $t=3$ and $t=4$. Because of this choice of factorization, a language model defines a graphical model where each step $t$ depends on all the previous steps $<t$ and the conditional $p_\theta(\mathbf{w}_t \mid \mathbf{w}_{<t})$ models the dependendies between the context $\mathbf{w}_{<t}$ and the variable $\mathbf{w}_t$.
?????????????????????????????????????????????????????????????????????????????????????????

## \[SLP\] Recurrent Neural Networks 
A recurrent neural network (RNN) is any network that contains a cycle within its network connections, meaning that the value of some unit is directly, or indirectly, dependent on its own earlier outputs as an input. 

### Basic RNN
![[Pasted image 20241016194722.png]]

The figure above illustrates the structure of an RNN. As with ordinary feedforward networks, an input vector representing the current input, $x_t$ , is multiplied by a weight matrix and then passed through a non-linear activation function to compute the values for a layer of hidden units. This hidden layer is then used to calculate a corresponding output, $y_t$ . Sequences are processed by presenting one item at a time to the network. We’ll use subscripts to represent time, thus $x_t$ will mean the input vector x at time t. The key difference from a feedforward network lies in the recurrent link shown in the figure with the dashed line. This link augments the input to the computation at the hidden layer with the value of the hidden layer from the preceding point in time.

The hidden layer from the previous time step provides a form of memory, or context, that encodes earlier processing and informs the decisions to be made at later points in time. Critically, this approach does not impose a fixed-length limit on this prior context.

The significant part about RNN lies in the new set of weights, U, that connect the hidden layer from the previous time step to the current hidden layer. These weights determine how the network makes use of past context in calculating the output for the current input. As with the other weights in the network, these connections are trained via backpropagation

### Forward step
![]({F04D1A92-FE6A-49B6-B9AF-AA7E764CE1B7}.png)
Where :
- $W_{hh}$ is the weight matrix of from the previous time step with size of some $d_h X d_h$
- $W_{hx}$ the weight matrix going to the hidden layer  with size of $d_hX d_{input}$ 
- $W_{hy}$ the weight matrix going to the output  layer $d_{out}X d_{h}$ 


you can kind of think of recurrent neural networks in two ways.
- One is this concept of having a hidden state that feeds back at itself, recurrently
- Unrolling this computational graph for multiple time steps.  And this makes the data flow of the hidden states and the inputs and the outputs and the weights maybe a little bit more clear.

![]({61D5E856-0CDD-47E8-90FE-819C3CF14C0B}.png)
initialize the first hidden state to 0 for most contexts. you are reusing the weight matrix for all hidden function. Sometimes, you compute the loss of the output at each time step $L_t$ and the sum of the loss is L.  This is for many to many.
### RNNs as Language Models

Language models predict the next word in a sequence given some preceding context. For example, if the preceding context is “Thanks for all the” and we want to know how likely the next word is “fish” we would compute:
$$
P(fish|\text{Thanks for all the})
$$
Language models give us the ability to assign such a conditional probability to every possible next word, giving us a distribution over the entire vocabulary. We can also assign probabilities to entire sequences by combining these conditional probabilities with the chain rule:
$$
P(W_{1:n})=\prod_{i=1}^n P(w_i|w_{<i})
$$
RNN language models (Mikolov et al., 2010) process the input sequence one word at a time, attempting to predict the next word from the current word and the previous hidden state. RNNs thus don’t have the limited context problem that n-gram models have, or the fixed context that feedforward language models have, since the hidden state can in principle represent information about all of the preceding words all the way back to the beginning of the sequence. 

The input sequence $X = [x_1;...;x_t ;...;x_N]$ consists of a series of word embeddings each represented as a one-hot vector of size |V| × 1, and the output prediction, y, is a vector representing a probability distribution over the vocabulary.

At each step, the model uses the word embedding matrix E to retrieve the embedding for the current word, and then combines it with the hidden layer from the previous step to compute a new hidden layer. This hidden layer is then used to generate an output layer which is passed through a softmax layer to generate a probability distribution over the entire vocabulary. The probability that a particular word i in the vocabulary is the next word is represented by $yt [i]$, the i-th component of $yt$ :
$$
P(w_{t+1}=i|w_1,...,w_t)= yt [i]
$$

The probability of an entire sequence is just the product of the probabilities of each item in the sequence, where we’ll use $yi [wi ]$ to mean the probability of the true word $wi$ at time step $i$. 
$$
P(W_{1:n})=\prod_{i=1}^n P(w_i|w_{1:i-1})=\prod_{i=1}^n yt [i]
$$

### Stacked RNN Architectures
In our examples thus far, the inputs to our RNNs have consisted of sequences of word or character embeddings (vectors) and the outputs have been vectors useful for predicting words. However, nothing prevents us from using the entire sequence of outputs from one RNN as an input sequence to another one.
Stacked RNNs consist of multiple networks where the output of one layer serves as the input to a subsequent layer, as shown below 
![[Pasted image 20241016204009.png]]

Stacked RNNs generally outperform single-layer networks. One reason for this success seems to be that the network induces representations at differing levels of abstraction across layers. Just as the early stages of the human visual system detect edges that are then used for finding larger regions and shapes, the initial layers of stacked networks can induce representations that serve as useful abstractions for further layers—representations that might prove difficult to induce in a single RNN. The optimal number of stacked RNNs is specific to each application and to each training set. However, as the number of stacks is increased the training costs rise quickly.
### Bidirectional RNNs

The RNN uses information from the left (prior) context to make its predictions at time t. But in many applications we have access to the entire input sequence; in those cases we would like to use words from the context to the right of t. One way to do this is to run two separate RNNs, one left-to-right, and one right-to-left, and concatenate their representations. 

In the left-to-right RNNs we’ve discussed so far, the hidden state at a given time t represents everything the network knows about the sequence up to that point. The state is a function of the inputs $x_1,...,x_t$ and represents the context of the network to the left of the current time.  $h_t^f = RNN_{forward}(x_1,...,x_t)$. This new notation $h_t^f$ simply corresponds to the normal hidden state at time t, representing everything the network has gleaned from the sequence so far.  To take advantage of context to the right of the current input, we can train an RNN on a reversed input sequence. With this approach, the hidden state at time t represents information about the sequence to the right of the current input: $h_t^b = RNN_{backwards}(x_1,...,x_t)$ Here, the hidden state h b t represents all the information we have discerned about the sequence from t to the end of the sequence.

A bidirectional RNN (Schuster and Paliwal, 1997) combines two independent RNNs, one where the input is processed from the start to the end, and the other from the end to the start. We then concatenate the two representations computed by the networks into a single vector that captures both the left and right contexts of an input at each point in time. Here we use either the semicolon ”;” or the equivalent symbol ⊕ to mean vector concatenation:
$$
h_f=[h_t^f;h_t^b] = h_t^f ⊕ h_t^b
$$
![[Pasted image 20241016204706.png]]
The figure above illustrates such a bidirectional network that concatenates the outputs of the forward and backward pass. Other simple ways to combine the forward and backward contexts include element-wise addition or multiplication. The output at each step in time thus captures information to the left and to the right of the current input. In sequence labeling applications, these concatenated outputs can serve as the basis for a local labeling decision.

### The LSTM

One reason for the inability of RNNs to carry forward critical information is that the hidden layers, and, by extension, the weights that determine the values in the hidden layer, are being asked to perform two tasks simultaneously: provide information useful for the current decision, and updating and carrying forward information required for future decisions. A second difficulty with training RNNs arises from the need to backpropagate the error signal back through time. The hidden layer at time t contributes to the loss at the next time step since it takes part in that calculation. As a result, during the backward pass of training, the hidden layers are subject to repeated multiplications, as determined by the length of the sequence. A frequent result of this process is that the gradients are eventually driven to zero, a situation called the vanishing gradients problem.

The long short-term memory (LSTM) network (Hochreiter and Schmidhuber, 1997). LSTMs divide the context management problem into two subproblems: removing information no longer needed from the context, and adding information likely to be needed for later decision making. The key to solving both problems is to learn how to manage this context rather than hard-coding a strategy into the architecture. LSTMs accomplish this by first adding an explicit context layer to the architecture (in addition to the usual recurrent hidden layer), and through the use of specialized neural units that make use of gates to control the flow of information into and out of the units that comprise the network layers. These gates are implemented through the use of additional weights that operate sequentially on the input, and previous hidden layer, and previous context layers

The first gate we’ll consider is the forget gate. The purpose of this gate is to delete information from the context that is no longer needed. The forget gate computes a weighted sum of the previous state’s hidden layer and the current input and passes that through a sigmoid. This mask is then multiplied element-wise by the context vector to remove the information from context that is no longer required.

Next, we generate the mask for the add gate to select the information to add to the current context. we add this to the modified context vector to get our new context vector. 

The final gate we’ll use is the output gate which is used to decide what information is required for the current hidden state (as opposed to what information needs to be preserved for future decisions).

![[Pasted image 20241016210143.png]]
### The Encoder-Decoder Model with RNNs
Different models for different combinations of input and output size.
![]({BB5D6B0F-7937-45FB-AF2E-9518C804B2A5}.png)
- One to one: vanilla neural networks
- One to many: e.g image captioning image > sequence of words  out put is variable in length 
- Many to one: e.g. sentiment classification. sequnce of words -> sentiment. Input is variable in length 
- Many to many: e.g. machine translation seq. of words > seq. of words. Both variable in length and the input and output do not have to be in the same length.
- Many to many: e.g. video classification on frame level, same length of input and output. they are  both variable. 


The encoder-decoder model is used when we are taking an input sequence and translating it to an output sequence that is of a different length than the input, and doesn’t align with it in a word-to-word way. 
Encoder-decoder models are used especially for tasks like machine translation, where the input sequence and output sequence can have different lengths and the mapping between a token in the input and a token in the output can be very indirect. 

The key idea underlying these networks is the use of an encoder network that takes an input sequence and creates a contextualized representation of it, often called the context. This representation is then passed to a decoder which generates a task-specific output sequence. Encoder-decoder networks consist of three components:
- An encoder that accepts an input sequence, $x^n_1$ , and generates a corresponding sequence of contextualized representations, $h^n_1$ . LSTMs, convolutional networks, and Transformers can all be employed as encoders.
- A context vector, c, which is a function of $h^n_1$ , and conveys the essence of the input to the decoder.
- A decoder, which accepts c as input and generates an arbitrary length sequence of hidden states $h^m_1$ , from which a corresponding sequence of output states $y^m_1$ , can be obtained. Just as with encoders, decoders can be realized by any kind of sequence architecture. 






![[Pasted image 20241016211854.png]]
The elements of the network on the left process the input sequence x and comprise the encoder. While our simplified figure shows only a single network layer for the encoder, stacked architectures are the norm, where the output states from the top layer of the stack are taken as the final representation. A widely used encoder design makes use of stacked biLSTMs where the hidden states from top layers from the forward and backward passes are concatenated to provide the contextualized representations for each time step. 

The entire purpose of the encoder is to generate a contextualized representation of the input. This representation is embodied in the final hidden state of the encoder, $h^e_n$ . This representation, also called c for context, is then passed to the decoder. The decoder network on the right takes this state and uses it to initialize the first hidden state of the decoder. That is, the first decoder RNN cell uses c as its prior hidden state $h^d_0$ . The decoder autoregressively generates a sequence of outputs, an element at a time, until an end-of-sequence marker is generated.


### Attention Mechanism
In the model as we’ve described it so far, this context vector is $h_n$, the hidden state of the last (n th) time step of the source text. This final hidden state is thus acting as a bottleneck: it must represent absolutely everything about the meaning of the source text, since the only thing the decoder knows about the source text is what’s in this context vector. 

The attention mechanism is a solution to the bottleneck problem, a way of attention mechanism allowing the decoder to get information from all the hidden states of the encoder, not just the last hidden state.

In the attention mechanism, as in the vanilla encoder-decoder model, the context vector c is a single vector that is a function of the hidden states of the encoder, that is, $c = f(h^e_1 ...h^e_n )$. Because the number of hidden states varies with the size of the input, we can’t use the entire set of encoder hidden state vectors directly as the context for the decoder. 

The idea of attention is instead to create the single fixed-length vector c by taking a weighted sum of all the encoder hidden states. The weights focus on (‘attend to’) a particular part of the source text that is relevant for the token the decoder is currently producing. Attention thus replaces the static context vector with one that is dynamically derived from the encoder hidden states, different for each token in decoding.  This context vector, $c_i$ , is generated anew with each decoding step i and takes all of the encoder hidden states into account in its derivation. We then make this context available during decoding by conditioning the computation of the current decoder hidden state on it (along with the prior hidden state and the previous output generated by the decoder), as we see in this equation $h^d_i = g(\hat y_{i-1},h^d_{i-1},c_i)$ .

![[Pasted image 20241017163123.png]]

The first step in computing $c_i$ is to compute how much to focus on each encoder state, how relevant each encoder state is to the decoder state captured in $h^d_{i−1}$ . We capture relevance by computing— at each state i during decoding—a $score(h^d_{i−1} ,h^e_{j} )$ for each encoder state j. 
The simplest such score, called dot-product attention, implements relevance as similarity: measuring how similar the decoder hidden state is to an encoder hidden state, by computing the dot product between them.  The score that results from this dot product is a scalar that reflects the degree of similarity between the two vectors. The vector of these scores across all the encoder hidden states gives us the relevance of each encoder state to the current step of the decoder. 

To make use of these scores, we’ll normalize them with a softmax to create a vector of weights, $α_{i j}$, that tells us the proportional relevance of each encoder hidden state j to the prior hidden decoder state, $h^d_{i−1}$.  Finally, given the distribution in α, we can compute a fixed-length context vector for the current decoder state by taking a weighted average over all the encoder hidden states. 

With this, we finally have a fixed-length context vector that takes into account information from the entire encoder state that is dynamically updated to reflect the needs of the decoder at each step of decoding. 
![[Pasted image 20241017163807.png]]

## \[SLP\] The Transformer
![[Pasted image 20241017164443.png]]
The figure above sketches the transformer architecture. A transformer has three major components. At the centre are columns of transformer blocks. Each block is a multilayer network (a multi-head attention layer, feedforward networks and layer normalization steps) that maps an input vector $x_i$ in column i (corresponding to input token i) to an output vector $h_i$ . The set of n blocks maps an entire context window of input vectors ($x_1,...,x_n$) to a window of output vectors ($h_1,...,h_n$) of the same length. A column might contain from 12 to 96 or more stacked blocks.

The column of blocks is preceded by the input encoding component, which processes an input token (like the word thanks) into a contextual vector representation, using an embedding matrix E and a mechanism for encoding token position. Each column is followed by a language modeling head, which takes the embedding output by the final transformer block, passes it through an unembedding matrix U and a softmax over the vocabulary to generate a single token for that column.

### Attention 
word2vec and other static embeddings, the representation of a word’s meaning is always the same vector irrespective of the context: the word chicken, for example, is always represented by the same fixed vector. So a static vector for the word it might somehow encode that this is a pronoun used for animals and inanimate entities. But in context it has a much richer meaning.

These contextual words that help us compute the meaning of words in context can be quite far away in the sentence or paragraph. Transformers can build contextual representations of word meaning, contextual embeddings, by integrating the meaning of these helpful contextual words. In a transformer, layer by layer, we build up richer and richer contextualized representations of the meanings of input tokens. At each layer, we compute the representation of a token i by combining information about i from the previous layer with information about the neighboring tokens to produce a contextualized representation for each word at each position. 

Attention is the mechanism in the transformer that weighs and combines the representations from appropriate other tokens in the context from layer k−1 to build the representation for tokens in layer k.
![[Pasted image 20241017182111.png]]
#### Attention more formally
Attention takes an input representation $x_i$ corresponding to the input token at position i, and a context window of prior inputs $x_1..x_{i−1}$, and produces an output $a_i$. In causal, left-to-right language models, the context is any of the prior words. That is, when processing $x_i$ , the model has access to $x_i$ as well as the representations of all the prior tokens in the context window but no tokens after i. 


Simplified version of attention At its heart, attention is really just a weighted sum of context vectors, with a lot of complications added to how the weights are computed and what gets summed. For pedagogical purposes let’s first describe a simplified intuition of attention, in which the attention output ai at token position i is simply the weighted sum of all the representations  $x_i$ , for all j ≤ i; we’ll use αi j to mean how much $x_i$ should contribute to  $a_i$ :
$$
a_i=\sum_{j\le i} \alpha_{ij}x_i
$$
Each $α_{ij}$ is a scalar used for weighing the value of input  $x_i$ when summing up the inputs to compute $a_i$. In attention we weight each prior embedding proportionally to how similar it is to the current token i. So the output of attention is a sum of the embeddings of prior tokens weighted by their similarity with the current token embedding. We compute similarity scores via dot product. 

### A single attention head using query, key, and value matrices
The attention head allows us to distinctly represent three different roles that each input embedding plays during the course of the attention process:
- As the current element being compared to the preceding inputs. We’ll refer to  this role as a query
- In its role as a preceding input that is being compared to the current element to determine a similarity weight. We’ll refer to this role as a key
- And finally, as a value of a preceding element that gets weighted and summed up to compute the output for the current element

To capture these three different roles, transformers introduce weight matrices $W^Q$, $W^K$, and $W^V$. These weights will project each input vector $x_i$ into a representation of its role as a key, query, or value: $q_i=x_iW^Q$ , $k_i=x_iW^K$, $v_i=x_iW^V$ .

Given these projections, when we are computing the similarity of the current element $x_i$ with some prior element $x_j$ , we’ll use the dot product between the current element’s query vector $q_i$ and the preceding element’s key vector $k_j$ . Furthermore, the result of a dot product can be an arbitrarily large (positive or negative) value, and exponentiating large values can lead to numerical issues and loss of gradients during training. To avoid this, we scale the dot product by a factor related to the size of the embeddings, via diving by the square root of the dimensionality of the query and key vectors ($d_k$)
![[Pasted image 20241017184205.png]]
![[Pasted image 20241017184234.png]]
The input to attention $x_i$ and the output from attention $a_i$ both have the same dimensionality 1 × d. We’ll have a dimension $d_k$ for the key and query vectors. The query vector and the key vector are both dimensionality 1×$d_k$. We’ll have a separate dimension $d_v$ for the value vectors. 
### Multi-head Attention
Equations 9.11-9.13 describe a single attention head. But actually, transformers use multiple attention heads. The intuition is that each head might be attending to the context for different purposes: heads might be specialized to represent different linguistic relationships between context elements and the current token, or to look for particular kinds of patterns in the context

So in multi-head attention we have h separate attention heads that reside in parallel layers at the same depth in a model, each with its own set of parameters that allows the head to model different aspects of the relationships among inputs. Thus each head i in a self-attention layer has its own set of key, query and value matrices: $W^{Ki}$ , $W^{Qi}$ and $W^{Vi}$. These are used to project the inputs into separate key, value, and query embeddings for each head. 
![[Pasted image 20241017185140.png]]
The output of each of the h heads is of shape 1 × $d_v$, and so the output of the multi-head layer with h heads consists of h vectors of shape 1×$d_v$. These are concatenated to produce a single output with dimensionality 1×hdv. Then we use yet another linear projection $W^O ∈ \mathbb{R} ^{hd_v×d}$ to reshape it, resulting in the multi-head attention vector ai with the correct output shape \[1xd\] at each input i.
![[Pasted image 20241017190053.png]]
### Transformer Block
In addition to the self-attention layer, The transformer block includes three other kinds of layers: (1) a feedforward layer, (2) residual connections, and (3) normalizing layers (colloquially called “layer norm”).


![[Pasted image 20241017190641.png]]

In the residual stream viewpoint, we consider the processing of an individual token i through the transformer block as a single stream of d-dimensional representations for token position i. This residual stream starts with the original input vector, and the various components read their input from the residual stream and add their output back into the stream.

The input at the bottom of the stream is an embedding for a token, which has dimensionality d. This initial embedding gets passed up (by residual connections), and is progressively added to by the other components of the transformer. Thus the initial vector is passed through a layer norm and attention layer, and the result is added back into the stream, in this case to the original input vector $x_i$. And then this summed vector is again passed through another layer norm and a feedforward layer, and the output of those is added back into the residual, and we’ll use $h_i$ to refer to the resulting output of the transformer block for token i. 

Feedforward layer The feedforward layer is a fully-connected 2-layer network, i.e., one hidden layer, two weight matrices. The weights are the same for each token position i , but are different from layer to layer. It is common to make the dimensionality $d_{ff}$ of the hidden layer of the feedforward network be larger than the model dimensionality d. Layer Norm At two stages in the transformer block we normalize the vector This process, called layer norm (short for layer normalization), is one of many forms of normalization that can be used to improve training performance in deep neural networks. Layer norm is a variation of the z-score from statistics, applied to a single vector in a hidden layer, thus, the input to layer norm is a single vector of dimensionality d and the output is that vector normalized, again of dimensionality d. The first step in layer normalization is to calculate the mean, µ, and standard deviation, σ, over the elements of the vector to be normalized. Given an embedding vector x of dimensionality d, these values are calculated as follows $\mu = \frac{1}{d}\sum^d_{i=1}x_i$  $\sigma = \sqrt{\frac{1}{d}\sum^d_{i=1}(x_i-\mu)^2}$. 
Given these values, the vector components are normalized by subtracting the mean from each and dividing by the standard deviation. The result of this computation is a new vector with zero mean and a standard deviation of one. Finally, in the standard implementation of layer normalization, two learnable parameters, γ and β, representing gain and offset values, are introduced. 
$$
LayerNorm(x) = \gamma\frac{(x-\mu)}{\sigma}+\beta
$$
Putting it all together The function computed by a transformer block can be expressed by breaking it down with one equation for each component computation, using t (of shape \[1 × d\]) to stand for transformer and superscripts to demarcate each computation inside the block:
![[Pasted image 20241017191732.png]]
Notice that the only component that takes as input information from other tokens (other residual streams) is multi-head attention, which (as we see from (9.27)) looks at all the neighboring tokens in the context. The output from attention, however, is then added into this token’s embedding stream. In fact, Elhage et al. (2021) show that we can view attention heads as literally moving information from the residual stream of a neighboring token into the current stream. The high-dimensional embedding space at each position thus contains information about the current token and about neighboring tokens, albeit in different subspaces of the vector space
![[Pasted image 20241017191900.png]]
Crucially, the input and output dimensions of transformer blocks are matched so they can be stacked. Each token vector xi at the input to the block has dimensionality d, and the output hi also has dimensionality d. Transformers for large language models stack many of these blocks, from 12 layers (used for the T5 or GPT-3-small language models) to 96 layers (used for GPT-3 large), to even more for more recent models. 
### Parallelizing Computation
This description of multi-head attention and the rest of the transformer block has been from the perspective of computing a single output at a single time step i in a single residual stream. the attention computation performed for each token to compute $a_i$ is independent of the computation for each other token, and that’s also true for all the computation in the transformer block computing $h_i$ from the input $x_i$ . That means we can easily parallelize the entire computation, taking advantage of efficient matrix multiplication routines. 

We do this by packing the input embeddings for the N tokens of the input sequence into a single matrix X of size \[N × d\]. Each row of X is the embedding of one token of the input
#### Parallelizing attention
Let’s first see this for a single attention head and then turn to multiple heads, and then add in the rest of the components in the transformer block. For one head we multiply X by the key, query, and value matrices $W^Q$ of shape \[d ×$d_k$ \], $W^K$ of shape \[d ×$d_k$ \], and $W^V$ of shape \[d ×dv\], to produce matrices Q of shape \[N ×$d_k$ \], $K ∈ \mathbb{R}^ {N×d_k}$ , and $V ∈ \mathbb{R}^ {N×d_v}$ , containing all the key, query, and value vectors: $Q=XW^Q$ $K=XW^K$  $V=XW^V$ 

Given these matrices we can compute all the requisite query-key comparisons simultaneously by multiplying Q and $K^T$ in a single matrix multiplication. The product is of shape N ×N.
![[Pasted image 20241017194858.png]]
Once we have this $QK^T$ matrix, we can very efficiently scale these scores, take the softmax, and then multiply the result by V resulting in a matrix of shape N ×d: a vector embedding representation for each token in the input. We’ve reduced the entire self-attention step for an entire sequence of N tokens for one head to the following computation:
$$
A=softmax(mask(\frac{QK^T}{\sqrt{d_k}}))V
$$
#### Masking out the future
You may have noticed that we introduced a mask function in equation above. This is because the self-attention computation as we’ve described it has a problem: the calculation in $QK^T$ results in a score for each query value to every key value, including those that follow the query. To fix this, the elements in the upper-triangular portion of the matrix are zeroed out (set to −∞), thus eliminating any knowledge of words that follow in the sequence.
![[Pasted image 20241017195819.png]]
![[Pasted image 20241017195833.png]]
#### Parallelizing multi-head attention
In multi-head attention, as with self-attention, the input and output have the model dimension d, the key and query embeddings have dimensionality $d_k$ , and the value embeddings are of dimensionality $d_v$. Thus for each head i, we have weight layers $W^Q_i ∈ \mathbb{R}^{d×d_k}$ , $W^K_i ∈ \mathbb{R}^{d×d_k}$ , and $W^V_i ∈ \mathbb{R}^{d×d_v}$ , and these get multiplied by the inputs packed into X to produce $Q ∈ \mathbb{R}^ {N×d_k}$, $K ∈ \mathbb{R}^ {N×d_k}$ , and $V ∈ \mathbb{R}^{N×d_v}$ . The output of each of the h heads is of shape N × $d_v$, and so the output of the multi-head layer with h heads consists of h matrices of shape N×$d_v$. To make use of these matrices in further processing, they are concatenated to produce a single output with dimensionality $N × hd_v$. Finally, we use yet another linear projection $W^O ∈ \mathbb{R}^{hd_v×d}$ , that reshape it to the original output dimension for each token. Multiplying the concatenated $N ×hd_v$ matrix output by $W^O ∈ \mathbb{R}^ {hd_v×d}$ yields the self-attention output A of shape \[N ×d\].
![[Pasted image 20241017201549.png]]

Putting it all together with the parallel input matrix X The function computed in parallel by an entire layer of N transformer block over the entire N input tokens can be expressed as:![[Pasted image 20241017201718.png]]
Or we can break it down with one equation for each component computation, using T (of shape \[N × d\]) to stand for transformer and superscripts to demarcate each computation inside the block:
![[Pasted image 20241017201743.png]]

# Tricks of the trade and data science challenge
## Faster training and convergence 
### Initialization 
Initialize the weights $W_{ij} \approx \sqrt{\frac{1}{n_j+n_i} }\mathcal{N}(0,1)$   (Glorot Initialization).  where i and j are the number of inputs and outputs. 
### Gradient clipping
![[Pasted image 20241018160739.png]]
- Highly nonlinear model: Gradient update can catapult parameters very far.
- Heuristic: Clip the magnitude of the gradient.
solution truncate the gradients at some value 

### Batch Normalization 
Normalization is a data pre-processing tool used to bring the numerical data to a common scale without distorting its shape. Batch normalization is a process to make neural networks faster and more stable through adding extra layers in a deep neural network. The new layer performs the standardizing and normalizing operations on the input of a layer coming from a previous layer.![[image-74.webp]]
Initially, our inputs X1, X2, X3, X4 are in normalized form as they are coming from the pre-processing stage. When the input passes through the first layer, it transforms, as a sigmoid function applied over the dot product of input X and the weight matrix W.  Although, our input X was normalized with time the output will no longer be on the same scale. As the data go through multiple layers of the neural network and L activation functions are applied, it leads to an internal co-variate shift in the data.

Normalization is the process of transforming the data to have a mean zero and standard deviation one.  the next step is to calculate the standard deviation of the hidden activations.
![[Screenshot-from-2021-03-09-11-41-50.webp]]
Further, as we have the mean and the standard deviation ready. We will normalize the hidden activations using these values. For this, we will subtract the mean from each input and divide the whole value with the sum of standard deviation and the smoothing term (_ε_). The smoothing term(_ε_) assures numerical stability within the operation by stopping a division by a zero value.
![[Screenshot-from-2021-03-09-11-43-08-1.webp]]
In the final operation, the re-scaling and offsetting of the input take place. Here two components of the BN algorithm come into the picture, γ(gamma) and β (beta). These parameters are used for re-scaling (γ) and shifting(β) of the vector containing values from the previous operations.
![[Screenshot-from-2021-03-09-12-51-12.webp]]
These two are learnable parameters, during the training neural network ensures the optimal values of γ and β are used. That will enable the accurate normalization of each batch.
## Better Performance
### Regularization 
Goals
- Easy to perform great on the training set (overfitting) in a neural networks.
- Regularization improves generalization to new data at the expense of increased training error.
- Use held-out validation data to choose hyperparameter (e.g. regularization strength).
- Use held-out test data to evaluate performance.
#### Regularization Methods
• Limited size of network
• Early stopping
• Weight decay
• Data augmentation
• Injecting noise
• Parameter sharing (e.g. convolutional)
• Sparse representations
• Ensemble methods
• Auxiliary tasks (e.g. unsupervised)
• Probabilistic treatment (e.g. variational methods)
• Adversarial training, ...
##### Limited size of network
- Rule of thumb: When the number of parameters is ten times less than the number of outputs times the number of examples, overfitting will not be severe.
- Reducing input dimensionality (e.g. by PCA) helps in reducing parameters
- Easy. Low computational complexity
- Other methods give better accuracy
	- Data augmentation increases the number of examples
	- Parameter sharing decreases the number of parameters
	- Auxiliary tasks increases the number of outputs
##### Early stopping 
• Monitor validation performance during training
• Stop when it starts to deteriorate
• With other regularization, it might never start
• Keeps solution close to the initialization
##### Weight decay 
- Penalizes the complexity of the model
- Add a penalty term to the training cost $C = ··· + \Omega (\theta)$ Note: only a function of parameters $\theta$, not data.
- $L^2$ regularization: $\Omega(\theta)=\frac{\lambda}{2}||\theta||^2$ hyperparameter for strength. Gradient: $\frac{\delta  \Omega(\theta)}{\delta \theta_i} = \lambda\theta_i$ 
- $L^1$ regularization: $\Omega(\theta) =\lambda/2||\theta||_1$ Gradient: $\frac{\delta\Omega(\theta)} {\delta\theta_i} = \lambda sign(\theta_i)$. Induces sparsity: Often many params become zero.
- Max-norm: Constrain row vectors $w_i$ of weight matrices to $||w_i||^2 \le c$.

How to set hyperparameter $\lambda$ 
- Split data into training, validation, and test sets
- Choose a number of settings train separately
- Use validation performance to pick the best $\lambda$
- (Retrain using both training and validation sets)
- Evaluate final performance on test data
##### Data Augmentation 
Augmented data by image-specific transformations. E.g. cropping just 2 pixels gets you 9 times the data!. Tilt the images and other stuff. 
##### Injecting noise
- Inject random noise during training separately in each epoch
- Can be applied to input data, to hidden activations, or to weights
- Can be seen as data augmentation
- Simple end effective
##### Parameter sharing 
- Force sets of parameters to be equal
- Reduces the number of (unique) parameters
- Important in convolutional networks (CNNs, this lecture)
- Auto-encoders sometimes share weights between encoder and decoder (Unsupervised learning lecture)
##### Ensemble methods
- Train several models and take average of their outputs Instead of one point representing $P(\theta|X)$, use several
- Also known as bagging or model averaging
- It helps to make individual models different by
	- varying models or algorithms
	- varying hyperparameters
	- varying data (dropping examples or dimensions)
	- varying random seed
- It is possible to train a single final model to mimick the performance of the ensemble, for test-time computational efficiency
##### Dropout 
- Each time we present data example x, randomly delete each hidden node with 0.5 probability
- Can be seen as injecting noise or as ensemble:
	- Multiplicative binary noise
	- Training an ensemble of $2^{|h|}$ networks with weight sharing
- At test time, use all nodes but divide weights by 2
![[Pasted image 20241018170959.png]]
##### Adversarial training
![[Pasted image 20241018171356.png]]
- Search for an input $x'$ near a datapoint x that would have very different output $y'$ from y
- Adversaries can be found surprisingly close!
- Look at the panda input look at the cost function then change the image in the direction where the cost function increases the most. 
## Tricks of the trade 
- **Become one with the data**. The first step is to thoroughly inspect the data you will be using to train your model. Look for:
    - Duplicate examples
    - Corrupted images or labels
    - Data imbalances and biases
    - Think about how you would classify the data yourself.
    - Search/filter/sort the data by whatever you can think of (e.g. type of label, size of annotations, number of annotations, etc.) and visualize their distributions and the outliers along any axis.
- **Set up the end-to-end training/evaluation skeleton + get dumb baselines**. Before you start building a complex model, set up a basic training and evaluation pipeline with a simple model, such as a linear classifier or a small convolutional neural network. We’ll want to train it, visualize the losses, any other metrics (e.g. accuracy), model predictions, and perform a series of ablation experiments with explicit hypotheses along the way. This will help ensure the correctness of the pipeline.
- Here are some tips for this stage:
    - **Fix the random seed**. This guarantees that the same results will be obtained if the code is run twice.
    - **Simplify**. Make sure that any unnecessary complexity is disabled, such as data augmentation.
    - **Add significant digits to the evaluation**. Run the evaluation over the entire (large) test set when plotting the test loss. Don't just plot test losses over batches.
    - **Verify loss at initialization**. Verify that the loss begins at the correct loss value. If you initialize your final layer correctly you should measure $-log(1/n_classes)$ on a softmax at initialization.
    - **Initialize weights well**. For example, if regressing values that have a mean of 50, initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization.
    - **Set a human baseline**. Monitor metrics other than loss that are human interpretable and checkable (for example, accuracy). Evaluate your own (human) accuracy whenever possible and compare it to this baseline.
    - **Set an input-independent baseline**. For example, set all inputs to zero. This should perform worse than plugging in the data. If not, the model is not learning to extract information from the input.
    - **Overfit one batch**. Overfit a single batch of only a few examples. Increase the model capacity by adding layers or filters and make sure that the lowest achievable loss is reached.
    - **Verify decreasing training loss**. If you're working with a toy model at this point, you should be underfitting your dataset. Try to increase its capacity slightly. Did your training loss decrease as expected?
    - **Visualize just before the network**. Visualize the data immediately before `y_hat = model(x)`. Visualize _exactly_ what goes into the network by decoding the raw tensor of data and labels.
    - **Visualize prediction dynamics**. Visualize model predictions on a fixed test batch during training. The "dynamics" of how these predictions move will provide a good understanding of how the training progresses.
    - **Use backpropagation to chart dependencies**. Deep learning code frequently contains complex, vectorized, and broadcasted operations. One way to debug this  is to set the loss to be something trivial like the sum of all outputs of example **i**, run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the **i-th** input.
    - **Generalize a special case**. A relatively common bug is that people attempt to write relatively general functionality from scratch. It may be useful to write a very specific function for the task at hand, ensure that it works, and then generalize it later, ensuring that the same result is obtained.
- **Overfit**. Once you have a good understanding of the dataset and a working training and evaluation pipeline, you can start iterating on a good model. Start with a model that is large enough to overfit the training data  and then regularize it appropriately. The reason for these two stages is that if no model can achieve a low error rate, there may be problems, bugs, or misconfiguration.
    - **Picking the model**. Resist the temptation of getting crazy and creative in stacking up the lego blocks of the neural net toolbox in various exotic architectures in the early stages of your project. Use an appropriate architecture for your data. **Don’t be a hero** . Find the most related paper to your task and copy and paste their simplest architecture that performs well. For example, for image classification tasks, just copy and paste ResNet-50 for the first run.
    - **Adam is safe**. In the early stages of establishing baselines, use Adam with a learning rate of 3e-4. Adam is much more forgiving to hyperparameters, including a bad learning rate. For convolutional neural networks, well-tuned stochastic gradient descent will almost always outperform Adam, but the optimal learning rate region is much narrower and more problem-specific. Note: If you are using recurrent neural networks (RNNs) and related sequence models, Adam is more commonly used.
    - **Complexify only one at a time**. If you have multiple signals to plug into your classifier, connect them one at a time, ensuring that you get the expected performance boost.
    - **Do not trust learning rate decay defaults**. If you're re-purposing code from another domain, be very careful with learning rate decay. ImageNet, for example, would decay by 10 on epoch 30. If you're not training ImageNet, you almost certainly don't want to use that. If you're not careful, your code could secretly be driving your learning rate to zero too early, preventing your model from converging.
- **Regularize**. Once you have a model that can overfit the training data, you need to regularize it to prevent overfitting. 
	- **Get more data**: Adding more real training data is the best and preferred way to regularize a model in any practical setting. A common mistake is to spend a lot of engineering effort trying to get the most out of a small dataset when you could be collecting more data.
	- **Data augmentation**: Use more aggressive data augmentation if collecting more real data isn't an option.
	- **Creative augmentation**: People are finding creative ways to expand datasets, such as domain randomization, using simulation, or even using generative adversarial networks (GANs).
	- **Pretrain**: Even if you have enough data, it's rarely a bad idea to use a pre-trained network.
	- **Stick with supervised learning**: Unsupervised pretraining has not, as far as is known, reported strong results in modern computer vision (though natural language processing seems to be doing pretty well with BERT and similar models these days, most likely due to the more deliberate nature of text and a higher signal-to-noise ratio).
	- **Smaller input dimensionality**: Eliminate features that may contain spurious signals.
	- **Smaller model size**: Use domain knowledge constraints on the network to decrease its size in many cases.
	- **Decrease the batch size**: Smaller batch sizes correspond to stronger regularization due to the normalization inside batch normalization.
	- **Drop**: Add dropout. Use dropout2d (spatial dropout) for convolutional neural networks (CNNs). Use this sparingly/carefully because dropout does not seem to play nice with batch normalization.
	- **Weight decay**: Increase the weight decay penalty.
	- **Early stopping**: Stop training based on the measured validation loss to catch the model just as it is about to overfit.
	- **Try a larger model**: While larger models will eventually overfit much more, their "early stopped" performance can often be much better than that of smaller models.
	- **L1 Regularization**: Add the sum of the absolute values of the weights to the unregularized cost function. L1 regularization tends to concentrate the weight of the network in a relatively small number of high-importance connections, while the other weights are driven toward zero.

- **Tune**. Once you have a regularized model, you can fine-tune the hyperparameters to get the best possible performance. 
	-  **Random over grid search**: Random search is better than grid search for tuning multiple hyperparameters simultaneously because neural networks are more sensitive to some parameters than others.
	
- **Squeeze out the juice:** Once you find the best types of architectures and hyper-parameters you can still use a few more tricks to squeeze out the last pieces of juice out of the system:
	- **Ensembles**: Model ensembles are a guaranteed way to gain 2% accuracy on any task. If you can’t afford the computation at test time look into distilling your ensemble into a network using dark knowledge.
	- **leave it training**: Networks train for a surprisingly long time.

The sources also mention some of the common problems you may encounter while developing a deep learning model. The **vanishing gradient problem** can occur in deep neural networks, making it difficult to train the early layers of the network. This problem happens when the gradients in the early layers become very small, preventing the weights from being updated effectively. Another potential issue is using a **learning rate that is too high**, which can prevent the model from converging. Finally, **overfitting** can occur when the model is too complex and learns the training data too well, resulting in poor performance on new data.

# \[DLB\] Un- and semi-supervised learning
## Unsupervised Learning
 Deep learning today:
	• Mostly about pure supervised learning
	• Requires a lot of labelled data: expensive to collect
Deep learning in the future:
	• Unsupervised, more human-like

Data is just $x'$ , not input-output pairs $x, y$.

Possible goals:
	• Model P(x'), or
	• Representation $f : x' → h$.
Comparisons to supervised learning $P(y|x)$:
- See data as $x' = y$, model $P(y|x = ∅)$
- No right output y given, invent your own output h
- Concatenate inputs and outputs to $x' = [x; y]$, prepare to answer any query, including $P(y|x)$.
### Approaches to unsupervised learning
Besides kernel density estimation, virtually all unsupervised learning approaches use variables h.
- Discrete h (cluster index, hidden state of HMM, map unit of SOM)
- Binary vector h (most Boltzmann machines)
- Continuous vector h (PCA, ICA, NMF, sparse coding, autoencoders, state-space models, . . . )
Vocabulary:
- Encoder function $f : x → h$
- Decoder function $g : h → \hat x$
- Reconstruction $\hat x$ 
![[Pasted image 20241020183624.png]]
### PCA as an autoencoder
Assume linear encoder and decoder:
- $f(x) = W^{(1)}x + b^{(1)}$
- $g(x) = W^{(2)}x + b^{(2)}$
PCA solution minimizes criterion $C=\mathbb{E} \bigg[ ||x-\hat x||^2     \bigg]$ Note: Solution is not unique, even if restricting $W^{(2)} = W^{(1)T}$. 


## Autoencoders
### Undercomplete Autoencoders 
An autoencoder whose code dimension is less than the input dimension is called **undercomplete**. Learning an undercomplete representation forces the autoencoder to capture the most salient features of the training data. The learning process is described simply as minimizing a loss function $$L(x,g(f(x)))$$Where L is a loss function penalizing $g(f(x))$ for being dissimilar from $x$, such as the mean squared error. 

When the decoder is linear and L is the mean squared error, an undercomplete autoencoder learns to span the same subspace as PCA. In this case, an autoencoder trained to perform the copying task has learned the principal subspace of the training data as a side eﬀect.

Autoencoders with nonlinear encoder functions f and nonlinear decoder functions g can thus learn a more powerful nonlinear generalization of PCA. Unfortunately, if the encoder and decoder are allowed too much capacity, the autoencoder can learn to perform the copying task without extracting useful information about the distribution of the data. 

### Regularized Autoencoders 
Regularized autoencoders provide the ability to  choose the code dimension and the capacity of the encoder and decoder based on the complexity of distribution to be modelled. Rather than limiting the model capacity by keeping the encoder and decoder shallow and the code size small, regularized autoencoders use a loss function that encourages the model to have other properties besides the ability to copy its input to its output.

These other properties include sparsity of the representation, smallness of the derivative of the representation, and robustness to noise or to missing inputs. A regularized autoencoder can be nonlinear and overcomplete but still learn something useful about the data distribution, even if the model capacity is great enough to learn a trivial identity function. 

#### Sparse Autoencoders
A  sparse autoencoder is simply an autoencoder whose training criterion involves a sparsity penalty $Ω(h)$ on the code layer h, in addition to the reconstruction error:$$L(x,g(f(x)))+\Omega(h)$$where $g(h)$ is the decoder output, and typically we have $h=f(x)$, the encoder output. 

This sparsity penalty is key to preventing the autoencoder from simply learning an identity function, where the output is just a copy of the input. Instead, it forces the autoencoder to discover and encode the most salient features of the dataset.

Let's look at the cost function of a sparse autoencoder to understand this better:
- $L(x, g(f(x)))$ represents the reconstruction error, which measures how well the decoder, denoted by $g$, reconstructs the original input $x$ from the encoded representation $h$.
- $Ω(h)$ is the sparsity penalty term, which encourages the hidden representation $h$ to be sparse.
- $f(x)$ is the encoder function, which maps the input $x$ to the hidden representation $h$.

One way to understand the role of the sparsity penalty is to think of it as a reflection of the model's prior assumptions about the distribution of the latent variables 'h'. From this viewpoint, the sparse autoencoder can be seen as an approximation of a generative model that has both visible variables 'x' and latent variables 'h'.

The objective of the generative model is to maximize the likelihood of the observed data `x` given the model's parameters. This involves considering all possible values of the latent variables 'h', which is computationally expensive. The sparse autoencoder simplifies this by focusing on a single, highly probable value of 'h' that is consistent with the observed data.

The selection of this 'h' is influenced by the model's prior distribution over the latent variables, represented by $p_{model}(h)$. If this prior distribution favors sparse values of 'h', then the sparsity penalty Ω(h) emerges naturally as a consequence of trying to maximize the likelihood of the data under this generative model.

For instance, choosing a Laplace prior for $p_{model}(h)$, which inherently favors sparse 'h' values, leads to an absolute value sparsity penalty:

$$
p_{model}(h_i) = (λ / 2) * \exp(-λ * |h_i|)
$$

which corresponds to:

$$
Ω(h) = λ ∑_i |h_i|
$$

where:

- λ is a hyperparameter controlling the strength of the sparsity preference.
- _hi_ represents the activation of the _i_-th neuron in the hidden layer.

Sparse autoencoders are frequently used for feature learning, aiming to discover meaningful and informative features from unlabeled data. The learned sparse representations can be more robust and generalizable compared to those learned by standard autoencoders. These features can then be utilized for various downstream tasks such as classification, where the learned representations can often lead to improved performance.

#### Denoising Autoencoders
Rather than adding a penalty Ω to the cost function, we can obtain an autoencoder that learns something useful by changing the reconstruction error term of the cost function.  Traditionally, autoencoders minimize some function $$L(x,g(f(x)))$$
Where L is a loss function penalizing $g(f(x))$ for being dissimilar from x. 
A denoising autoencoder (DAE) instead minimizes $$L(x,g(f(\tilde{x})))$$
where $\tilde{x}$ is a copy of x that has been corrupted by some form of noise. Denoising autoencoders must therefore undo this corruption rather than simply copying their input. 
#### Regularizing by Penalizing Derivatives

Another strategy for regularizing an autoencoder is to use a penalty Ω, as in sparse autoencoders,$$L(x,g(f(x)))+\Omega(h,x)$$
but with a diﬀerent form of Ω:$$\Omega(h,x)=\lambda\sum_i||\nabla_xh_i||^2$$
This forces the model to learn a function that does not change much when x changes slightly. Because this penalty is applied only at training examples, it forces the autoencoder to learn features that capture information about the training distribution. 
An autoencoder regularized in this way is called a contractive autoencoder, or CAE. This approach has theoretical connections to denoising autoencoders, manifold learning, and probabilistic modelling.


### Representational Power, Layer Size and Depth 

Autoencoders are often trained with only a single layer encoder and a single layer decoder. However, this is not a requirement. In fact, using deep encoders and decoders oﬀers many advantages.

One major advantage of nontrivial depth is that the universal approximator theorem guarantees that a feedforward neural network with at least one hidden layer can represent an approximation of any function to an arbitrary degree of accuracy, provided that it has enough hidden units. 

A deep autoencoder, with at least one additional hidden layer inside the encoder itself, can approximate any mapping from input to code arbitrarily well, given enough hidden units.

Depth can exponentially reduce the computational cost of representing some functions. Depth can also exponentially decrease the amount of training data needed to learn some functions
### Stochastic Encoders and Decoders
Instead of a fixed mapping from input to code, a stochastic encoder represents the encoding process as a conditional probability distribution, $p_{encoder}(h | x)$. Similarly, the decoder uses a conditional probability distribution, $p_{decoder}(x | h)$, to reconstruct the input given a code.

This shift towards probabilistic mappings injects noise into both the encoding and decoding steps. This means that for a given input 'x', the encoder can produce different codes 'h' with varying probabilities. Similarly, the decoder can generate different reconstructions 'x' for a given code 'h'.

Any latent variable model, denoted as $p_{model}(h, x)$, can be seen as defining both a stochastic encoder and decoder:
- Stochastic Encoder: $p_{encoder}(h | x) = p_{model}(h | x)$
- Stochastic Decoder: $p_{decoder}(x | h) = p_{model}(x | h)$ 

Importantly, the encoder and decoder distributions in a stochastic autoencoder do not necessarily have to come from a single, well-defined joint distribution $p_{model}(h, x)$. This flexibility opens up new possibilities for designing and training autoencoders.

Although the encoder and decoder distributions may initially be incompatible, training the autoencoder as a denoising autoencoder with sufficient capacity and training data can lead to asymptotic compatibility between them. This means that with enough training, the encoder and decoder will learn to represent a consistent probabilistic model of the data.

The use of stochastic encoders and decoders introduces a powerful mechanism for learning more robust and expressive representations of data. By incorporating noise and uncertainty into the encoding and decoding processes, these models can better capture the underlying structure of complex data distributions. This opens up new avenues for applying autoencoders in various domains, including generative modeling, representation learning, and anomaly detection.

### Denoising Autoencoders
The denoising autoencoder(DAE) is an autoencoder that receives a corrupted data point as input and is trained to predict the original, uncorrupted data point as its output. 
We introduce a corruption process $C(\tilde{x} | x)$, which represents a conditional distribution over corrupted samples $\tilde{x}$, given a data sample x.
![[Pasted image 20241020203725.png]]
The autoencoder then learns a reconstruction distribution $p_{reconstruct}(x |\tilde{x})$ estimated from training pairs $(x,\tilde{x})$ as follows:
- Sample a training example x from the training data.
- Sample a corrupted version $\tilde{x}$ from $C(\tilde{x} | x = x)$.
- Use $(x,\tilde{x})$ as a training example for estimating the autoencoder reconstruction distribution $p_{reconstruct}(x |\tilde{x}) =p_{decoder}(x | h)$ with h the output of encoder $f(\tilde{x})$ and $p_{decoder}$typically deﬁned by a decoder $g(h)$. 
Typically we can simply perform gradient-based approximate minimization on the negative log-likelihood $−log p_{decoder}(x | h)$. As long as the encoder is deterministic, the denoising autoencoder is a feedforward network and may be trained with exactly the same techniques as any other feedforward network. We can therefore view the DAE as performing stochastic gradient descent on the following expectation: $$-\mathbb{E}_{x \sim \hat p_{data}(x)}\mathbb{E}_{\tilde{x}\sim C(\tilde{x}|x)}\log~p_{decoder}(x|h=f(\tilde{x}))$$
#### Estimating the Score
Denoising autoencoders (DAEs) can be used to estimate the _score_ of a probability distribution, which is a vector field defined as the gradient of the log probability density function. During training, a DAE learns to map a corrupted data point back to its original, uncorrupted form. In doing so, it learns a vector field represented by the difference between the reconstructed output, `g(f(x))`, and the input, 'x'. This vector field approximates the score of the data distribution. It essentially points from the corrupted input towards the nearest point on the manifold of clean data.

This ability to estimate the score is a crucial aspect of how DAEs learn the underlying structure of data. The score acts as a guide, directing the model towards regions of higher probability density on the data manifold. By minimizing the distance between the reconstruction and the original data point, the DAE learns a representation that captures the key variations within the data. For continuous data with Gaussian noise, this score estimation capability holds true for a wide range of encoder and decoder architectures. 
![[Pasted image 20241020211035.png]]
### Learning Manifolds
Autoencoders, like many machine learning algorithms, are based on the idea that data tends to lie on or near a low-dimensional manifold. However, while other algorithms might only learn functions that behave well on the manifold, autoencoders go a step further and aim to actually learn the structure of the manifold itself.

To grasp how this works, it's important to understand some key characteristics of manifolds. A crucial concept is the **tangent plane**, which describes the local directions of variation allowed on the manifold at a particular point

Autoencoder training involves a trade-off between two competing forces:

1. **Reconstruction Accuracy:** Learning an encoding 'h' from the input 'x' that allows 'x' to be accurately reconstructed using a decoder. This force encourages the autoencoder to preserve as much information about the input as possible.

2. **Constraints or Regularization:** This could involve limiting the capacity of the autoencoder, adding sparsity penalties, or imposing constraints on the derivatives of the encoding or reconstruction functions. These constraints encourage the autoencoder to learn more general and robust representations.

The balance between these forces is what allows autoencoders to learn meaningful representations. The model can only afford to represent variations in the input that are necessary to reconstruct training examples. If the data lies on a low-dimensional manifold, the autoencoder learns a representation that captures the local coordinate system of the manifold around each training example.

Essentially, the encoder becomes sensitive to changes along the manifold directions but insensitive to changes orthogonal to the manifold. This is because variations orthogonal to the manifold are not needed to reconstruct the training data.


The contrast between autoencoders and traditional nonparametric manifold learning methods, which often rely on nearest neighbour graphs to capture the structure of the manifold. These methods can be effective when the manifolds are relatively smooth and well-sampled, but they struggle with complex, high-dimensional manifolds that require more than just local interpolation to be accurately represented.

Autoencoders, particularly deep autoencoders, are better equipped to handle such complex manifolds by leveraging their ability to learn distributed representations and hierarchical features. This advantage stems from their parametric nature and the ability to exploit parameter sharing across different regions of the input space.

An example of learning the manifold of image translations. Translating an image creates a path on a manifold through the high-dimensional space of all possible images. A successful autoencoder would learn to represent this translation manifold by capturing the underlying variations in the image caused by the translation. The encoder would be sensitive to changes in position but insensitive to other irrelevant features of the image.![[Pasted image 20241020211954.png]]
![[Pasted image 20241020212023.png]]
### Contractive Autoencoders
**Contractive autoencoders (CAEs)** as a type of regularized autoencoder that explicitly encourages the derivatives of the encoder function, `f(x)`, to be small. This is achieved by adding a penalty term, `Ω(h)`, to the reconstruction cost function. This penalty, based on the Frobenius norm of the Jacobian matrix of `f(x)`, promotes the contraction of the local neighborhood around each training point in the encoding space.

Here's the CAE penalty term:

```
Ω(h) =  λ || ∂f(x) / ∂x ||_F^2
```

where:

- `λ` is a hyperparameter controlling the strength of the penalty.
- `||.||_F` represents the Frobenius norm.
- `∂f(x) / ∂x` is the Jacobian matrix of the encoder function.

The intuition behind CAEs is that by minimizing the sensitivity of the encoder to small changes in the input, the model is forced to focus on the most salient variations in the data. This is particularly beneficial for learning the structure of manifolds, as discussed in section 14.6 of the sources.

Here's how CAEs are used to learn manifolds:

1. **Minimize Perturbations:** The CAE penalty encourages the model to map a neighborhood of input points to a smaller neighborhood in the encoding space. This contraction effect forces the encoder to be less sensitive to small, potentially irrelevant variations in the input.
2. **Focus on Tangent Directions:** The directions in the input space that correspond to larger values in the Jacobian matrix are those that cause the most significant changes in the encoding. These directions are likely to be aligned with the tangent planes of the manifold, as they represent the directions of greatest variation within the data.
3. **Learn Manifold Structure:** By focusing on these tangent directions and minimizing sensitivity to other variations, the CAE effectively learns a representation that captures the local coordinate system of the manifold.

The sources point out a key connection between CAEs and denoising autoencoders (DAEs): in the limit of small Gaussian input noise, the denoising reconstruction error becomes equivalent to a contractive penalty on the reconstruction function, `g(f(x))`.

This link highlights the shared goal of both methods: to learn robust representations that are insensitive to noise or minor perturbations in the input. While DAEs achieve this by explicitly reconstructing clean data from corrupted inputs, CAEs achieve it by directly penalizing the sensitivity of the encoder function to input variations.

Practical Considerations:
- **Deep CAEs:** When training deep CAEs, directly computing the Jacobian for the entire network can be computationally expensive. A common strategy is to train a stack of single-layer CAEs, each reconstructing the hidden layer of the previous one. This greedy layer-wise pretraining helps approximate the desired contractive property for the deep network.
    
- **Scaling Issues:** The contractive penalty can lead to degenerate solutions if the decoder is not properly scaled. For example, the encoder could learn to shrink the input, while the decoder simply expands it back, without learning anything useful. One way to prevent this is to tie the weights of the encoder and decoder, as suggested in the sources. This ensures that the encoder and decoder work together to learn a meaningful representation.
    

Benefits of Using CAEs:
- **Robust Representations:** By minimizing sensitivity to input perturbations, CAEs learn representations that are less susceptible to noise and small variations in the data.
- **Manifold Learning:** CAEs are effective in learning the structure of manifolds, allowing them to capture the underlying low-dimensional structure of complex, high-dimensional data.
- **Feature Extraction:** The learned representations can be used as features for downstream tasks like classification or information retrieval.

In essence, CAEs provide a powerful mechanism for learning representations that capture the most important variations within data while ignoring irrelevant noise. This makes them valuable tools for tasks requiring robust feature extraction and the understanding of complex data structures.

### Predictive Sparse Decomposition
## Predictive Sparse Decomposition: Bridging Sparse Coding and Autoencoders

**Predictive sparse decomposition (PSD)** combines elements of sparse coding and parametric autoencoders. The sources describe PSD as a hybrid model that utilizes a parametric encoder, denoted as `f(x)`, to predict the output of an iterative sparse coding inference process. This approach aims to learn features that are both sparse and predictable by a compact model.

- **Parametric Encoder (`f(x)`):** This component is a neural network that takes the input data `x` and produces a dense representation, which serves as an initial guess for the sparse code `h`.
- **Iterative Sparse Coding:** This step involves minimizing a cost function that encourages sparsity in the representation `h`, similar to traditional sparse coding methods. The cost function typically includes a reconstruction error term and a sparsity-inducing penalty, such as the L1 norm.
- **Parametric Decoder (`g(h)`):** This component takes the sparse code `h` and reconstructs the input data `x`. It's typically implemented as a neural network as well.

PSD training involves minimizing the following cost function:

```
|| x - g(h) ||^2 + λ|h|_1 + γ|| h - f(x) ||^2
```

where:

- `|| x - g(h) ||^2` is the reconstruction error, measuring the difference between the input `x` and the reconstruction `g(h)`.
- `λ|h|_1` is the sparsity penalty, encouraging the code `h` to have many zeros.
- `γ|| h - f(x) ||^2` is a prediction error term that encourages the parametric encoder `f(x)` to produce a good initial guess for the sparse code `h`.

The training procedure alternates between two steps:

1. **Sparse Coding Inference:** For a given input `x`, the code `h` is iteratively optimized to minimize the cost function. The parametric encoder `f(x)` provides a starting point for this optimization, making the process faster than traditional sparse coding.
2. **Parameter Updates:** The parameters of the encoder `f(x)` and decoder `g(h)` are updated to minimize the cost function, using gradient descent.

PSD offers several advantages over traditional sparse coding and basic autoencoders:

- **Fast Inference:** The parametric encoder provides a good initial guess for the sparse code, significantly speeding up the iterative sparse coding inference process.
- **Learned Approximate Inference:** PSD essentially learns a parametric approximation to the iterative sparse coding inference procedure. This learned approximation, represented by the encoder `f(x)`, can be used for fast feature extraction at test time.
- **Flexibility and Stackability:** Since the encoder and decoder are parametric functions (typically neural networks), PSD models can be easily stacked to create deeper architectures. This allows for hierarchical feature learning and can lead to more powerful representations.

PSD has been successfully applied to several tasks, including:

- **Object Recognition in Images and Video:** PSD has been used to learn features for object recognition tasks, demonstrating good performance on benchmark datasets.
- **Audio Processing:** PSD has also been utilized for audio feature learning, showing promise for tasks like speech recognition and music analysis.