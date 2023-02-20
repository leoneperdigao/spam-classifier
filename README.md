# Spam Classifier using Neural Network

This is a spam classifier implementation using a neural network. The model contains a class called `SpamClassifier` 
that can be used for training and predicting.

## Methodology

The neural network consists of a configurable number of layers, each with a specified number of nodes and activation 
function. The default configuration is ((54, 'sigmoid'), (40, 'sigmoid'), (40, 'sigmoid'), (1, 'sigmoid')), which has 
three hidden layers with 54, 40, and 40 nodes respectively, and an output layer with a single node. The sigmoid function
is used as the activation function for all layers.

The Spam Classifier model supports three different activation functions: 
- sigmoid: The sigmoid activation function maps any input to a value between 0 and 1, making it suitable for binary classification problems where the output is a probability
  - Formula: `f(x) = 1 / (1 + exp(-x))`
- ReLU: The ReLU function, on the other hand, is best suited for problems with non-linear separable datasets. ReLU returns 0 for any negative input and the input value for any non-negative input, which can help prevent the vanishing gradient problem. 
  - Formula: `f(x) = max(0, x)`
- hyperbolic tangent (tanh): The tanh function maps any input to a value between -1 and 1, which can be useful for classification problems where the output may take on negative values. 
  - Formula: `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

The model is trained using backpropagation with L2 regularization. The training data is first normalized, and the 
weights are updated using the calculated gradients and the learning rate. The regularization parameter is used to 
penalize large weights and prevent overfitting.

## Hyperparameters

The performance of the spam classifier model can be greatly affected by the choice of hyperparameters. 

Here are the hyperparameters used in this model:

- **Layers configuration**: The number of layers, nodes per layer, and activation functions used in the model. 
Increasing the number of layers or nodes per layer can improve the model's accuracy but can also increase the risk of 
overfitting. Changing the activation functions can affect the speed of convergence and the final accuracy of the model.

- **Learning rate**: Determines the step size for updating the model weights during training. 
A high learning rate can cause the model to overshoot the optimal weights, while a low learning rate can result in slow 
convergence or getting stuck in a local minimum.

- **Epochs**: The number of training iterations. Too few epochs may result in an underfit model, while too many epochs 
can cause overfitting.

- **Regularization parameter**: Controls the degree of L2 regularization used to penalize large weights in the model. 
A high regularization parameter can result in a simpler model that is less prone to overfitting, but may also result in
lower accuracy.

## Optimizations

The model uses a randomized initialization for the weights, which helps avoid the problem of all weights being 
initialized to the same value. This can speed up convergence during training.

The activation functions used in the model (sigmoid, relu, and tanh) are vectorized using numpy ufuncs, which can 
significantly speed up computations.

## Further Improvements

One possible improvement to the model is to add more layers or nodes to the existing layers to improve its performance. 
The model can also be further optimized by using adaptive learning rate algorithms like Adam or RMSprop, which can reduce 
the sensitivity of the model to the choice of the learning rate.

Another possible improvement is to use a more advanced regularization technique like dropout or early stopping, which 
can help prevent overfitting and improve the generalization of the model.

Finally, the model can be trained on a larger and more diverse dataset to improve its accuracy on different types of 
spam messages.
