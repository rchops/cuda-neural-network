#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <string>
#include <algorithm>

// New threading strategy
// Using block -> num of hidden neurons = blockDim.x, num of inputs = blockDim.y
 
// New storage strategy -> z values and activation different, weight and biases the same
// order of hidden layers the same, but now each layer holds indexes for multiple inputs

// z values
// first hidden layer -> instead of [0,5] (6 neurons), [0,11] (6 neurons x 2 for 2 inputs)
// second hidden layer -> [12,19] (4 neurons x 2 for 2 inputs)
// output layer -> [20,21] (1 neuron x 2 for 2 inputs)

// activation values
// input layer -> instead of [0,7] (8 neurons), [0,15] (8 neurons x 2 for 2 inputs)
// first hidden layer -> [16,27] (6 neurons x 2 for 2 inputs)
// second hidden layer -> [28,35] (4 neurons x 2 for 2 inputs)
// output layer -> [36,37] (2 neurons x 2 for 2 inputs)
