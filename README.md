## Simple Neural Network Using CUDA

- The neural network grows with each part -> learning how this works too
- So something new is built on top of last part

### 1. Activation Layer
- Use sigmoid function for activation function
    - 1 / (1 + e^(-x))
- Assuming we have bunch of computed set values
    - Set values (z) are values of neurons before activation function -> weight * input + bias
- Compute activation values all at once using multiple GPU threads

### 2. Linear Layer
- Develop neural network further
- Instead of assuming set values, now calculate these and then activation
- All input neurons connected to each output neuron
- Set value for each neuron caluclating using weighted sum -> weight * input + bias
- Then use this to calculation activation values as in Activation Layer

### 3. Multiple Layers
- Added more hidden layers instead of just one and output
    - 4 in [8, 6, 4, 1] structure
- Same calculations but having to do over multiple layers
- Again using thread for each neuron