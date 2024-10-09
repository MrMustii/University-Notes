# Chapter 1 Neural Networks and Deep Learning
## Part 1 - **Sigmoid neurons simulating perceptrons**
**Suppose we take all the weights and biases in a network of perceptrons, and multiply them by a positive constant, $c>0$  Show that the behaviour of the network doesn't change**

The output of a perception is
$$\begin{equation*}
\text{output}=
\begin{cases}
0 &\text{if} & \sum_{j}  w_j \cdot x_j +b \leq 0\\
1 &\text{if} & \sum_{j}  w_j \cdot x_j +b \geq 0
\end{cases}
\end{equation*} 
$$
for both cases 
$$
\begin{flalign}
&= \sum_{j}c ( w_j \cdot x_j) +cb \leq 0\\
&= c \cdot\sum_{j}( w_j \cdot x_j) +b \leq 0\\ 
&=\sum_{j} ( w_j \cdot x_j) +b \leq \frac{0}{c}\\
&=\sum_{j} ( w_j \cdot x_j) +b \leq 0\\

\end{flalign}
$$
## Part 2 - **Sigmoid neurons simulating perceptrons** 
**Suppose we have the same setup as the last problem - a network of perceptrons. Suppose also that the overall input to the network of perceptrons has been chosen. We won't need the actual input value, we just need the input to have been fixed. Suppose the weights and biases are such that $w⋅x+b\neq0$ for the input $x$ to any particular perceptron in the network. Now replace all the perceptrons in the network by sigmoid neurons, and multiply the weights and biases by a positive constant $c>0$. Show that in the limit as $c→∞$ the behaviour of this network of sigmoid neurons is exactly the same as the network of perceptrons. How can this fail when $w⋅x+b=0$ for one of the perceptrons?**

The sigmoid function is 
$$
\sigma = \frac {1}{1+\exp(-\sum_jw_jx_j-b)}
$$
now if we add a constant $c$ to the exponent, we get
$$
\sigma = \frac {1}{1+\exp(-(c\cdot\sum_jw_jx_j-b))}
$$
as $c→∞$
$$
\begin{cases}
\sigma→1 & \text{if} &w_jx_j+b>0\\
\sigma→0 & \text{if} &w_jx_j+b<0\\
\sigma=\frac{1}{2} & \text{if} &w_jx_j+b=0
\end{cases}
$$
