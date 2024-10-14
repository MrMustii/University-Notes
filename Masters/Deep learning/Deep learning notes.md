NOTES FROM [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)  BOOK
# Chapter 1 Using neural nets to recognize handwritten digits 

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
# Chapter 2: How the backpropagation algorithm works
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


# Chapter 3: Improving the way neural networks learn
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



# Chapter 6 Deep learning 
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

## Recurrent Neural Networks 
![]({BB5D6B0F-7937-45FB-AF2E-9518C804B2A5}.png)
- One to one: vanilla neural networks
- One to many: e.g image captioning image > sequence of words  out put is variable in length 
- Many to one: e.g. sentiment classification. sequnce of words -> sentiment. Input is variable in length 
- Many to many: e.g. machine translation seq. of words > seq. of words. Both variable in length and the input and output do not have to be in the same length.
- Many to many: e.g. video classification on frame level, same length of input and output. they are  both variable. 

![]({A47F30CB-ACB6-465E-B3D0-240AA4B9C117}.png)
Take an input x, feed it to the RNN. The RNN has some internal hidden state. This state is updated every time the rnn reads a new input. which will be feed back into the model the text time it read an input. produce an output.
![]({6927937F-81A0-4776-ADE2-67549A2C869B}.png)
NOTICE: the same function $f_w$ and the same set of parameters are used at every time step.
 The simplest form 
![]({F04D1A92-FE6A-49B6-B9AF-AA7E764CE1B7}.png)

you can kind of think of recurrent neural networks in two ways.
- One is this concept of having a hidden state that feeds back at itself, recurrently
- Unrolling this computational graph for multiple time steps.  And this makes the data flow of the hidden states and the inputs and the outputs and the weights maybe a little bit more clear.
![]({61D5E856-0CDD-47E8-90FE-819C3CF14C0B}.png)
initialize the first hidden state to 0 for most contexts. you are reusing the weight matrix for all hidden function. Sometimes, you compute the loss of the output at each time step $L_t$ and the sum of the loss is L.  This is for many to many. 

![]({1190E85E-BC0E-44CF-8E82-DB6F4C85AFC0}.png)
for many to many where input and output are not the same. 
![]({12405AC1-73AC-434F-A972-044730A1538C}.png)
![]({97C4A4B9-9DEA-4A09-8668-5EFDAA70F5C1}.png)
![]({082A9F7D-B873-4530-828E-E84C75ECA997}.png)
