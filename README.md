## Simple Neural Network Using CUDA

### 1. Activation Layer
- Use sigmoid function for activation function
    - 1 / (1 + e^(-x))
- Assuming we have bunch of computed set values
    - Set values (z) are values of neurons before activation function -> weight * input + bias
- Compute activation values all at once using multiple GPU threads