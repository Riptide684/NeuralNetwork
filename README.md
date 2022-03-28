# NeuralNetwork

Standalone python neural network made from scratch. Created as a proof of concept and to enhance my own understanding of machine learning.

How it works:
Firstly the program creates a NeuralNetwork object, with layers and nodes, each with random weights and biases. The program feeds forward the training data via optimised matrix multiplication, and calculates the loss compared to the expected output. Depending on the loss function, the program calculates the mean gradient of the weight-biases to loss via back propogation and the multivariable chain rule. After each batch, it updates the weights aand biases according to this mean gradient, decreasing the average loss of the function. This repeats until the whole training set has been used, and then again for however many epochs. The trained neural network can then be compared against the test data to see how accurately it can predict the desired output. When trained on the MNIST database for hand-written digits it achieved an accuracy of around 95% from 3 epochs, compared to the 99% of Keras from TensorFlow. This is largely down to the not-yet-implemented Adam optimiser and different loss/activation functions.

How to use:
1. Create a data training set and test set.
2. Create a new NeuralNetwork() object and add layers via nn.add_layer, or (not yet implemented) import an existing neural network.
3. Parse the training data into a NumPy array and pass it to the train() function, along with the batch size and number of epochs.
4. (Not yet implemented) Choose an optomisation algorithm and loss funtion for the NN training.
5. Program trains NN and then tests itself against the test set after each iteration, plotting a graph of accuracy against number of epochs.
6. Terminate the program, or (not yet implemented) export the neural network.

Libraries required:
- NumPy
- Random
- MatPlotLib
- (Optional) Keras to test NN with MNIST dataset

Future updates:
- Make the train() function easier to interact with.
- Add different optimisation algorithms.
- Add different loss functions.
- Add different activation functions.
- Allow import & export of trained neural networks.
