#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// computing weighted sum
// one thread for each neuron -> each thread computes weighted sum of input
// Wnp -> W = weight, n = neuron num in next layer, p = neuron num in prev layer
// e.g. formula for thread 1: z1 = W11 * X1 + W12 * X2 + W13 * X3 + W14 * X4 + b

// Inputs -> weight matrix, biases, x inputs, number of input neurons, number of output neurons
// Outputs -> set values (z values), activation values
// For every output neuron we have to loop over every input neuron -> look at formula
__global__ void linear_layer_and_activation(float *weight_mat, float *biases, float *x_inputs, float *z_val,
                                            float *activation_val, int num_output_neurons, int num_input_neurons){
    int idx = blockIdx.x * blockDim.x + threadIdx.x ;
    if(idx < num_output_neurons){
         // calculate weighted sum for each neuron
         // w * x
         // same rule as multiplying matrices -> weighted_mat[row * num_col + col] * x_input[i] (x_input is vector)
         for(int i = 0; i < num_input_neurons; ++i){
             z_val[idx] += weight_mat[(num_input_neurons) * idx + i] * x_inputs[i]; 
         }
     
         // w * x + b
         // gives final set values
         z_val[idx] += biases[idx];
     
         // activation function
         // sig(w * x + b)
         activation_val[idx] = 1.0 / (1.0 + exp(-z_val[idx]));
    }
}

int main(){

    const int INPUT_NEURONS = 4;
    const int OUTPUT_NEURONS = 3;

    // Initialise weights on host
    // every input neuron connected to every output neuron -> multiply together to find number of weights
    const int size_w = INPUT_NEURONS * OUTPUT_NEURONS;
    float *h_weights = new float [size_w] {0.80f, 0.87f, 0.16f, 0.96f, 0.89f, 0.87f, 0.31f, 0.08f, 0.09f, 0.69f, 0.03f, 0.42f};

    // Initialise biases on host
    const int size_b = OUTPUT_NEURONS;
    float *h_biases = new float [size_b] {0.68f, 0.83f, 0.01f};
    float h_input[INPUT_NEURONS] = {0.75f, 0.98f, 0.74f, 0.28f};

    // Initialise activations on host
    float *h_activation = new float [size_b] {0.0, 0.0, 0.0};

    // Initialise z matrix on host
    float *h_z = new float [size_b] {0.0, 0.0, 0.0};

    const size_t size_weights = size_w * sizeof(float);
    const size_t size_biases = size_b * sizeof(float);
    const size_t size_inputs = INPUT_NEURONS * sizeof(float);
    const size_t size_activation = size_b * sizeof(float);
    const size_t size_z = size_b * sizeof(float);

    float *d_weights, *d_biases, *d_input, *d_activation, *d_z;

    cudaMalloc(&d_weights, size_weights);
    cudaMalloc(&d_biases, size_biases);
    cudaMalloc(&d_input, size_inputs);
    cudaMalloc(&d_activation, size_activation);
    cudaMalloc(&d_z, size_z);

    cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases, size_biases, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, size_inputs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_activation, h_activation, size_activation, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, size_z, cudaMemcpyHostToDevice);

    /// <<<num_blocks, threads_per_block>>>
    // ___/256 + 1 ensures 1 block when output neurons < 256, and ensures more if > 256
    linear_layer_and_activation<<<OUTPUT_NEURONS / 256 + 1, OUTPUT_NEURONS>>>(d_weights, d_biases, d_input, d_z, 
                                                                            d_activation, OUTPUT_NEURONS, INPUT_NEURONS);

    // Back to host
    cudaMemcpy(h_activation, d_activation, size_activation, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, size_z, cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_input);
    cudaFree(d_activation);
    cudaFree(d_z);

    std::cout << "Z values: " << std::endl;
    for(int i = 0; i < OUTPUT_NEURONS; ++i){
        std::cout << h_z[i] << std::endl;
    }
    
    std::cout << "Activation values: " << std::endl;
    for(int i = 0; i < OUTPUT_NEURONS; ++i){
        std::cout << h_activation[i] << std::endl;
    }

    getchar();

    return 0;
}