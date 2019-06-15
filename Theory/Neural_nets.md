# Deep Learning Using Pytorch

PyTorch takes these tensors and makes it simple to move them to GPUs 
for the faster processing needed when training neural networks.
It also provides a module that automatically calculates gradients (for backpropagation!) 
and another module specifically for building neural networks.


## Neural Networks

The networks are built from individual parts approximating neurons, typically called units or simply "neurons." 
Each unit has some number of weighted inputs. 
These weighted inputs are summed together (a linear combination) then passed through an activation function to get the unit's output. 

Mathematically this looks like: 

$$
\begin{align}
y &= f(w_1 x_1 + w_2 x_2 + b) \\
y &= f\left(\sum_i w_i x_i +b \right)
\end{align}
$$

With vectors this is the dot/inner product of two vectors:

$$
h = \begin{bmatrix}
x_1 \, x_2 \cdots  x_n
\end{bmatrix}
\cdot 
\begin{bmatrix}
           w_1 \\
           w_2 \\
           \vdots \\
           w_n
\end{bmatrix}
$$


## Tensors

Tensor is a n-dimensional generalisation of arrays which are used to create n-d Tensors of shape.


## Multiple Layer Neural Networks

Multi Layer Neural networks though more computing expensive but they more efficient to train our neural networks 
By defining the number of hidden layers we can implement our neural networks 
First layer is the Input layer which defines the Input or number of features of the data
Second Layer or other layers are the hidden layers which are the hyperparameters which decides the weights are trained
or not on the basis of them.
Last layer is the Output layer which derives the Output of the network

