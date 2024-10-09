# Neural networks - The feed forward model
![[Pasted image 20240902134844.jpg]]
let there be one hidden layer with M hidden units and 2 layers of adaptable parameters (weights) denoted by $w=w^{(1)},w^{(2)}$ sometimes $\theta$ is used instead of $w$

This is  [[Random notes#Learning Paradigms|Supervised learning]]  where we have the inputs (covariates) and associated outputs (labels, targets, response variables).  The network has D inputs: $\mathbf{x}=x_1,\ldots,x_D$ and $K$ outputs $\mathbf{y}=y_1,\ldots,y_K$ . Each layer first computes what is equivalent to the output $a_j$ of a linear statistical model and then applies a non-linear function to the linear model output. For the first layer, which takes the network input $x$ as input, these two steps look like this 

$$

\begin{align}

a^{(1)}_j & = \sum_{i=1}^D w^{(1)}_{ji} x_i + w^{(1)}_{j0} \\

z^{(1)}_j & = h_1(a^{(1)}_j ) \ ,

\end{align}

$$
where $h_1$ is the [[Random notes#Activation Functions|non-linear function]] in the first layer and $a_j^{(1)}$ is the computed weighted sum of the inputs at that node . We can get rid of writing the so-called [[Random notes#Bias Term in Neural Networks|bias]] $w^{(1)}_{j0}$ explicitly by adding an extra input $x_0$ that is always set to one and extending the sum to go from zero:
$$

\begin{align}

a^{(1)}_j & = \sum_{i=0}^D w^{(1)}_{ji} x_i \ .

\end{align}

$$
The second layer takes the output the of the first layer as input:

$$

a^{(2)}_j = \sum_{i}^M w^{(2)}_{ji} z^{(1)}_i \ .

$$

The second layer non-linear function is denoted by $h_2$ so the output of the network is

$$

y_j = h_2(a^{(2)}_j)

$$

This gives an example of how the neural network model input to output mapping can be specified.

## Exercises 
### Exercise A) 
Write $y_j$ directly as a function of $x$. That is, eliminate the $a$'s and $z$'s:
$$

y_j = h_2(\ldots)

$$
$$
\begin{equation}
y_j = h_2(\sum_{k=0}^M w_{jk}^{(2)} \cdot (h_1 (\sum_{i=0}^D w_{ki}^{(1)} \cdot x_i))) 
\end{equation}
$$
 

### Exercise B) 
Write the equation for a neural network with two hidden layers and three layers of weights $w=w^{(1)},w^{(2)},w^{(3)}$. Again, without using $a$'s and $z$'s.

$$

y_j = h_3(\ldots)

$$
Let N be the number of units in the second hidden layer

$$
\begin{equation}
y_j = 
h_3(\sum_{l=0}^N w_{jl}^{(3)} \cdot
h_2(\sum_{k=0}^M w_{lk}^{(2)} \cdot (
h_1 (\sum_{i=0}^D w_{ki}^{(1)} \cdot x_i)))) 
\end{equation}
$$


### Exercise C)
c) Write the equations for an FFNN with $L$ layers as recursion. Use $l$ as the index for the layer:

$$

\begin{align}

y_j & = h_L(a_{j}^{(L)})  & \\

z^{(l)}_j & = h_l(a_j^{(l)}) & l=1,\ldots,L-1 \\

a^{(l)}_j & = \sum_{i=0}^M w_{ji}^{(l)} z_j^{(l-1)}  & l=2,\ldots,L \\

a^{(1)}_j & =  \sum_{i=0}^D w^{(1)}_{ji} x_i \ . & \

\end{align}\\
$$
where $D=$ number of inputs and $M=$ the number of nodes in layer $l$

### Exercise D)– optional

**Do we really need the non-linearities? Show that if we remove the non-linear functions $h_l$ from the expressions above then the output becomes linear in $x$. This means that the model collapses back to the linear model and therefore cannot learn non-linear relations between $x$ and $y$.**

let $W_n$ be the matrix for layer n, $b_n$ be the biases and  $\boldsymbol x_{out} \text{ and } x_{in}$  be a vector of outputs and inputs of the previous layer $n-1$ respectively.   
$$
\begin{align}
y&=W_n \boldsymbol x_{out} +b_n\\
y&=W_n (W_{n-1} \boldsymbol x_{in} +b_{n-1}) +b_n\\
y&= (W_n W_{n-1})\boldsymbol x_{in} + (W_nb_{n-1}+b_{n})\\
y&=W'x_{in}+b'
\end{align}
$$
hence no matter how many layer there are x will only transform linearly once and the model depth will collapse

### Exercise E)
In this exercise you will show that with the two above assumptions, we can derive a loss function that contains $E(w)$ as a term. Two hints

  

1. With the used covariance we can write the Gaussian distribution as

$$

\mathcal{N}(\mathbf{t}_n|\mathbf{y}(\mathbf{x}_n),\sigma^2 \mathbf{I}) = \frac{1}{\sqrt{2\pi \sigma^2}^D}

\exp ( - || \mathbf{y}(\mathbf{x}_n) - \mathbf{t}_n||_2^2 /2\sigma^2 )

$$

2. In order to turn maximum likelihood into a loss/error function apply the (natural) logarithm to the likelihood objective and multiply by minus one.

  

Show that the loss we get is

$$

\frac{ND}{2} \log 2\pi \sigma^2 + \frac{1}{2\sigma^2} E(w) \ .

$$

Further, argue why applying the log and multiplying by minus one is the right thing to do in order to get a loss function. *Hint:* Will the optimum of the likelihood function change if we apply the logarithm?



**<span style="color:green">Solution</span>**
To find the loss function, we can try to find the maximum likelihood function and apply the natural logarithm and multiply by negative 1. Moreover, the maximum likelihood is as defined as  $p(\mathbf{t}_1,\ldots,\mathbf{t}_N|\mathbf{x}_1,\ldots,\mathbf{x}_N,w) \ .$  Furthermore, since the likelihood of a a single target given the input and model parameters is disrupted on a Gaussian distribution and with the given equation for the distribution ( see hint 1) we grt  

$$
\begin{align}
\text{from assumbtion 2}\\
\\
P(\mathbf{t}_n|\mathbf{x}_n,w) = \mathcal{N}(\mathbf{t}_n|\mathbf{y}(\mathbf{x}_n),\sigma^2 \mathbf{I})   =& \frac{1}{\sqrt{2\pi \sigma^2}^D}
\exp ( - || \mathbf{y}(\mathbf{x}_n) - \mathbf{t}_n||_2^2 /2\sigma^2 )\\


\\
\text{from assumbtion 1}\\
\\

p(\mathbf{t}_1,\ldots,\mathbf{t}_N|\mathbf{x}_1,\ldots,\mathbf{x}_N,w)  =& \prod_{n=1}^N \frac{1}{\sqrt{2\pi \sigma^2}^D}
\exp ( - || \mathbf{y}(\mathbf{x}_n) - \mathbf{t}_n||_2^2 /2\sigma^2 )\\

\\
\text{apply the log *-1}\\
\\
=&-1(\ln((\frac{1}{\sqrt{2\pi \sigma^2}^D})^N)+\ln(\prod_{n=1}^N \exp ( - || \mathbf{y}(\mathbf{x}_n) - \mathbf{t}_n||_2^2 /2\sigma^2 )))\\
=& \frac{1}{2}*N*D*ln(2\pi\sigma)+(-1\ln(\exp ( - || \mathbf{y}(\mathbf{x}_1) - \mathbf{t}_1||_2^2 /2\sigma^2 )
\cdot ... \cdot\
\\exp ( - || \mathbf{y}(\mathbf{x}_N) - \mathbf{t}_N||_2^2 /2\sigma^2 )) )\\
=& \frac{1}{2}*N*D*ln(2\pi\sigma)+ 
(-1((- || \mathbf{y}(\mathbf{x}_1) - \mathbf{t}_1||_2^2 /2\sigma^2 )
+...+
\\(- || \mathbf{y}(\mathbf{x}_N) - \mathbf{t}_N||_2^2 /2\sigma^2 )))\\
=& \frac{1}{2}*N*D*ln(2\pi\sigma)+ \frac{1}{2\sigma^2} \cdot \sum_{n=1}^N || \mathbf{y}(\mathbf{x}_n) - \mathbf{t}_n||_2^2 \ \\
p(\mathbf{t}_1,\ldots,\mathbf{t}_N|\mathbf{x}_1,\ldots,\mathbf{x}_N,w)
=& \frac{1}{2}*N*D*ln(2\pi\sigma)+ \frac{1}{2\sigma^2} \cdot E(w)

\end{align}

$$

The multiplication by negative 1 is there to change the problem from maximizing to minimizing, moreover, the natural log is applied to simplify the finding of the derivative because it easier to deal with the sum of function rather than the product.

### Exercise F) - optional undone 

Show that the optimum (= minimum of the loss) with respect to $w$ is not affected by the value of $\sigma^2$. Find the optimum for $\sigma^2$ as a function of $w$. 

This means that for the problem of finding $w$, sum of squares and maximum likelihood for the Gaussian likelihood with $\sigma^2 \mathbf{I}$ covariance are equivalent.  

### Exercise G) 
Show using the same procedure we used for regression that the loss function for classification is
$$
E(w) = - \sum_{n=1}^N \sum_{k=1}^K t_{nk} \log y_{k}(\mathbf{x}_n) \ .
$$
This is also known as the cross entropy loss. 
**<span style="color:green">Solution</span>**
Similarly to exercise E, assuming the data is not a series and it is iid, we can obtain the maximum likelihood function 
$$p(\mathbf{t}_1,\ldots,\mathbf{t}_N|\mathbf{x}_1,\ldots,\mathbf{x}_N,w) =\prod_{n=1}^N p(\mathbf{t}_n|\mathbf{x}_n,w) \ .$$ Furthermore, the probability function is given 
$$
p(\mathbf{t}_n|\mathbf{x}_n,w) = \prod_{k=1}^K \left[ y_k(\mathbf{x}_n)\right]^{t_{nk}} \
$$
hence:
$$
\begin{align}
p(\mathbf{t}_n|\mathbf{x}_n,w) =& \prod_{k=1}^K \left[ y_k(\mathbf{x}_n)\right]^{t_{nk}} \\

p(\mathbf{t}_1,\ldots,\mathbf{t}_N|\mathbf{x}_1,\ldots,\mathbf{x}_N,w) =& \prod_{n=1}^N \prod_{k=1}^K \left[ y_k(\mathbf{x}_n)\right]^{t_{nk}} \\


\text{Here we apply the natural log and multiply by -1}\\


E(w)
=
-1\bigg(\ln\bigg(\prod_{n=1}^N \prod_{k=1}^K \left[ y_k(\mathbf{x}_n)\right]^{t_{nk}}\bigg)\bigg)\\




E(w)
=
-1\bigg((\sum_{n=1}^N \sum_{k=1}^K \ln \big( \left[ y_k(\mathbf{x}_n)\right]^{t_{nk}}\big)\bigg)\\


E(w)
=
-1\bigg((\sum_{n=1}^N \sum_{k=1}^K \cdot t_{nk}\cdot \ln \big(  y_k(\mathbf{x}_n)\big)\bigg)\\

\end{align}
$$

### Exercise H) – optional
Prove that the gradient calculated on a random subset of the training set on average is proportional to the true gradient. 

### Exercise I) 
Calculate 
$$
\delta^{(L)}_j = \frac{\partial E(w)}{\partial a^{(L)}_{j}}
$$
for classification. 

Hint: It is much easier to find the derivative, if we write the loss function directly in terms of the logits $a^{(L)}_{j}$. So show first using the definition of the loss and the softmax that 
$$
E(w) = - \sum_{k=1}^K t_k a^{(L)}_{k} + \log \sum_{k=1}^K \exp( a^{(L)}_{k} ) \ . 
$$
Finally show that 
$$
\frac{\partial}{\partial a^{(L)}_{j}} \log \sum_{k=1}^K \exp( a^{(L)}_{k} ) = y_j
$$
to get the final result.
<span style="color:green">Solution</span>

As stated above we will assume that the training set consists of one  example, so we can drop the training point index $n$ hence 
$$
\begin{align}
& E(w)=-\sum_{k=1}^K t_k \ln (y_k)\\
& \text{here we will subtitue $y_k$ with the soft max function}\\
& E(w)=-\sum_{k=1}^K t_k 
\ln \bigg (\frac{ \exp ({a_k ^{(L)} })}{\sum_{i=1}^K \exp ({a_i ^{(L)} }) } \bigg)\\
& E(w)=-\sum_{k=1}^K t_k (a_k^{(L)}-\ln (\sum_{i=1}^K \exp ({a_i ^{(L)} })))\\
& E(w)=-\sum_{k=1}^K t_k a_k^{(L)}+
\sum_{k=1}^K t_k\ln (\sum_{i=1}^K \exp ({a_i ^{(L)} })))\\
& \text{in the second term, the sum of $t_k$ is one hot encoded then the sum must be exactly one}\\
& E(w)=-\sum_{k=1}^K t_k a_k^{(L)}+
\ln (\sum_{i=1}^K \exp ({a_i ^{(L)} })))\\



\end{align}
$$




### Exercise j) - the backpropagation rule for layer $l<L$

j) Use the above results to argue why the general backpropagation rule for $l<L$ is written as:
$$
\begin{align}
\frac{\partial E(w)}{\partial w^{(l)}_{ji}} & = \delta^{(l)}_j  z^{(l-1)}_i \\
\delta^{(l)}_j & = \sum_{k=1}^K \delta^{(l+1)}_k  w^{(l+1)}_{kj} h_{l}'(a^{(l)}_j) 
\end{align}
$$

### Exercise k) – optional
k) Derive the backpropagation rule for regression, that is, perform the calculation in exercise i) for the regression loss. Everything else stays the same. Actually it turns out that if you use $h_L(a) = a$ then you even get the same result as in i).

### Exercise k) – optional
k) Derive the backpropagation rule for regression, that is, perform the calculation in exercise i) for the regression loss. Everything else stays the same. Actually it turns out that if you use $h_L(a) = a$ then you even get the same result as in i).

-------
# **<span style="color:green">Solution</span>**
## Backpropagation 
 Backpropagation is nothing but stochastic gradient decent.
### cost function
- minus the log likelihood function
- The likelihood function is the probability of the data we have observed 
- in this case we are doing conditional modeling so we are not concerned about modeling the input features we are interested in molding the output given the inputs, such that the liklihood function becomes the the probability of y given x 

----
Considering only one sample, for classification problem, $$ E(w) = -\sum_{k=1}^K log{t_k}y_k\\ $$ $$ \delta_j^{(L)} = \frac{\partial E(w)}{\partial a_j^{(L)}}\\ = \frac{\partial E(w)}{y_k} \cdot \frac{\partial y_k}{\partial a_j^{(L)}}\\ = -\sum_{k=1}^K \frac{t_k}{y_k} \cdot \frac{\partial y_k}{\partial a_j^{(L)}} $$ Let's discuss ∂aj(L)​∂yk​​. $$ y_k = \frac{\exp ( a_k^{(L)} )}{\sum_i \exp ( a_i^{(L)} )} $$ $$ \frac{\partial y_k}{\partial a_j^{(L)}} = \frac{\frac{\partial exp(a_k^{(L)})}{\partial a_j^{(L)}}\cdot \sum_i exp(a_i^{(L)}) - exp(a_k^{(L)})\cdot exp(a_j^{(L)})}{(\sum_i exp(a_i^{(L)})^2} $$ There are two different cases.\ When k=j: $$ \frac{\partial y_k}{\partial a_j^{(L)}} = \frac{exp(a_k^{(L)})\cdot \sum_i exp(a_i^{(L)}) - exp(a_k^{(L)})^2}{(\sum_i exp(a_i^{(L)})^2}\\ = y_k -y_k^2\\ = y_k(1-y_k) $$ When k=j: $$ \frac{\partial y_k}{\partial a_j^{(L)}} = \frac{- exp(a_k^{(L)})\cdot exp(a_j^{(L)})}{(\sum_i exp(a_i^{(L)})^2}\\ = -y_k\cdot y_j $$ When taking ∂aj(L)​∂yk​​ into δj(L)​, there is: $$ \delta_j^{(L)} = \sum_{k=1}^K f_k $$ $$ f_k = t_k(y_j-1) \qquad when \qquad k=j\\ f_k = t_k\cdot y_j \qquad when \qquad k\neq j $$ Considering tk​ is in one-hot format, $$ \delta_j^{(L)} = y_j-1 \qquad when \qquad j = \textbf{k}\\ \delta_j^{(L)} = y_j \qquad when \qquad j \neq \textbf{k}  
$$ Here, k means that the point is in class k. Note that k is different from the indice counter k I used before.  
It's easy for us to notice that, δj(L)​=yj​−ts​, where ts​ is the supposed output of the j-th neutron.

---
$E(w)$ depends on aj(L)​ through yk​(xn​), as such we can utilize the chain rule to calculate the derivate of E(w) w.r.t. aj(L)​ as follows

$$ \begin{align} \frac{\partial E(w)}{\partial a_j^{(L)}} &= \frac{\partial E(w)}{\partial y_k } \frac{\partial y_k}{\partial a_j^{(L)}} \ \end{align} $$

As the training set consist of a single sample, the loss function can be reduced to

E(w)=−∑k=1K​tnk​logyk​(xn​)

Note. for brevity I will omit the depence of xn​ on yk​. Inserting E(w) in the above differentiation, yields

$$ \begin{align} \frac{\partial E(w)}{\partial y_k } \frac{\partial y_k}{\partial a_j^{(L)}} &= - \frac{\partial \sum_{k=1}^K t_{nk} \log y_{k}}{\partial y_k } \frac{\partial y_k}{\partial a_j^{(L)}} \\ &= - \sum_{k=1}^K \frac{\partial t_{nk} \log y_{k}}{\partial y_k } \frac{\partial y_k}{\partial a_j^{(L)}} \\ &= - \sum_{k=1}^K \frac{ t_{nk} }{ y_k } \frac{\partial y_k}{\partial a_j^{(L)}} \\ \end{align} $$

By applying the hint given above regarding the derivate of the softmax function yields

∂aj(L)​∂E(w)​={−∑k=1K​tnk​(1−yk​)ifj=k∑k=1K​tnk​yj​ifj=k​


Calculate $$ \delta^{(L)}_j = \frac{\partial E(w)}{\partial a^{(L)}_{j}} $$ for classification. Hint: Use that E(w) depends upon the model output y and that the derivative of the softmax function ∂aj​∂yk​​=yk​(1−yk​) for k=j and ∂aj​∂yk​​=−yk​yj​ when k=j.