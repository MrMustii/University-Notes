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
    

The iterative process discussed in this chapter, from simple models to more complex architectures, demonstrates how thoughtful modifications can significantly enhance performance, a lesson applicable in various deep learning applications.

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
