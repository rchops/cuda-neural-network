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

// Inputs -> weight matrix, biases, x inputs, number of input neurons, number of output neurons
// Outputs -> set values (z values), activation values
// For every output neuron we have to loop over every input neuron -> look at formula
__global__ void multiple_inputs(float *weight_mat, float *biases, float *z_val,
                                float *activation_val, int *shape, int shape_length){
    int idx = threadIdx.x + (blockDim.x * threadIdx.y) + (threadIdx.z * blockDim.x * blockDim.y);
    
    // define offsets for current layer
    // help index into correct layer
    int layer_offset_weights = 0;
    int layer_offset_z = 0;
    int layer_offset_b = 0;
    int layer_offset_activations_input_layer = 0;
    int layer_offset_activations_current_layer = shape[0] * blockDim.y;

    // for multiple layers use variable to define shape of network -> [8,6,4,1]
    // for each layer we calculate the set values and activation values, so loop through every layer
    for(int shape_idx = 0; shape_idx < shape_length; ++shape_idx){
        
        // memory guard -> if(idx < output neurons (same as before)) -> shape[shape_idx + 1] because the number of output neurons
        // is number of neurons in next layer

        // New storage strategy -> z values and activation different, weight and biases the same
        // order of hidden layers the same, but now each layer holds indexes for multiple inputs

        // E.g.
        // z values
        // first hidden layer -> instead of [0,5] (6 neurons), [0,11] (6 neurons x 2 for 2 inputs)
        // second hidden layer -> [12,19] (4 neurons x 2 for 2 inputs)
        // output layer -> [20,21] (1 neuron x 2 for 2 inputs)

        // activation values
        // input layer -> instead of [0,7] (8 neurons), [0,15] (8 neurons x 2 for 2 inputs)
        // first hidden layer -> [16,27] (6 neurons x 2 for 2 inputs)
        // second hidden layer -> [28,35] (4 neurons x 2 for 2 inputs)
        // output layer -> [36,37] (2 neurons x 2 for 2 inputs)

        // 1 thread per input and neuron -> we are using 3 inputs, 6 neurons, so 18 threads
        
        if(idx < shape[shape_idx + 1]){
             
            // each neuron in the layer requires an input
            int num_input_in_layer = shape[shape_idx];
            int layer_size = shape[shape_idx + 1];

            for(int i = 0; i < num_input_in_layer; ++i){

                // weights work by starting at correct offset (layer_offset_weights)
                // then for each output neuron -> number of inputs * idx gives starting point
                // then + i for each input neuron for that output neuron

                // activation values simpler
                // just flat for each layer so as long as offset is correct can index into each neuron
                z_val[layer_offset_z_b + idx] += weight_mat[layer_offset_weights + (num_input_in_layer) * idx + i] * 
                                                activation_val[layer_offset_activations + i];
            }
        
            // w * x + b
            // gives final set values
            z_val[layer_offset_z_b + idx] += biases[layer_offset_z_b + idx];
        
            // activation function
            // sig(w * x + b)
            // + shape[shape_idx] -> write activation values for next layer so input values are not overwritten
            activation_val[layer_offset_activations + shape[shape_idx] + idx] = 1.0 / (1.0 + exp(-z_val[layer_offset_z_b + idx]));
        }
        
        // shift everthing forward as explained in memory layout earlier
        layer_offset_z_b += shape[shape_idx + 1];
        layer_offset_weights += shape[shape_idx] * shape[shape_idx + 1];
        layer_offset_activations += shape[shape_idx];

        // make sure all threads for previous layer complete before moving onto next
        __syncthreads();
    }
}

int main(){

    const int shape_length = 4;
    int shape[shape_length]  = {8, 6, 4, 1};
    const int num_inputs = 3;

    // initialise weights on host
    int num_weights = 0;
    for(int i = 0; i < shape_length - 1; ++i){
        num_weights += shape[i] * shape[i + 1];
    }

    float *h_weights = new float [num_weights] {1.62f, -0.61f, -0.53f, -1.07f, 0.87f, -2.30f, 1.74f, -0.76f, 0.32f, -0.25f, 1.46f, 
                                                -2.06f, -0.32f, -0.38f, 1.13f, -1.10f, -0.17f, -0.88f, 0.04f, 0.58f, -1.10f, 1.14f, 0.90f, 0.50f,
                                                0.90f, -0.68f, -0.12f, -0.94f, -0.27f, 0.53f, -0.69f, -0.40f, -0.69f, -0.85f, -0.67f, -0.01f,
                                                -1.12f, 0.23f, 1.66f, 0.74f, -0.19f, -0.89f, -0.75f, 1.69f, 0.05f, -0.64f, 0.19f, 2.10f, 0.12f,
                                                0.62f, 0.30f, -0.35f, -1.14f, -0.35f, -0.21f, 0.59f, 0.84f, 0.93f, 0.29f, 0.89f, -0.75f, 1.25f, 0.51f,
                                                -0.30f, 0.49f, -0.08f, 1.13f, 1.52f, 2.19f, -1.40f, -1.44f, -0.50f, 0.16f, 0.88f, 0.32f, -2.02f};

    // initialise biases on host
    int num_neurons = 0;
    int num_biases = 0;

    for(int i = 0; i < shape_length; ++i){
        num_neurons += shape[i];
    }

    num_biases = num_neurons - shape[0];

    float *h_biases = new float [num_biases] {-0.31f, 0.83f, 0.23f, 0.76f, -0.22f, -0.20f, 0.19f, 0.41f, 0.20f, 0.12f, -0.67f};

    // initialise activations on host
    // first 8 values are inputs (no need for separate input array anymore since we included it in activations)
    // rest are initialised to 0.0
    int num_activation = num_inputs * num_neurons;
    float *h_activation = new float [num_activation] {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f, 0.76f, -0.36f, -0.23f, -0.89f, 
                                                        -0.01f, -0.08f, -0.26f, -0.13f, -0.55f, -0.42f, -0.39f, -0.83f, 0.87f, 0.44f, -0.45f, -0.52f};

    // Initialise z matrix on host
    int num_z = num_biases * num_inputs;
    float *h_z = new float [num_z] {0.0f};

    const size_t size_weights = num_weights * sizeof(float);
    const size_t size_biases = num_biases * sizeof(float);
    const size_t size_activation = num_activation * sizeof(float);
    const size_t size_z = num_z * sizeof(float);
    const size_t size_shape = shape_length * sizeof(int);

    float *d_weights, *d_biases, *d_activation, *d_z;
    int *d_shape;

    cudaMalloc(&d_weights, size_weights);
    cudaMalloc(&d_biases, size_biases);
    cudaMalloc(&d_activation, size_activation);
    cudaMalloc(&d_z, size_z);
    cudaMalloc(&d_shape, size_shape);

    cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases, size_biases, cudaMemcpyHostToDevice);
    cudaMemcpy(d_activation, h_activation, size_activation, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, size_z, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, size_shape, cudaMemcpyHostToDevice);

    // <<<num_blocks, threads_per_block>>>
    // sets width of neural network (ours is 8)
    // finds the largest number of neurons in any layer -> the width
    // dim3 creates 2D block of threads -> (x,y) -> x = num neurons in layer, y = num of inputs being processed
    // e.g. num_threads_x_direction = 8, num_inputs = 2, creates 2D block of 16 threads
    // one thread for each neuron which computes 2 inputs for each neuron
    int num_threads_x_dimensions = *std::max_element(shape + 1, shape + shape_length);
    dim3 thread_block_dimensions(num_threads_x_dimensions, num_inputs);
    multiple_inputs<<<1, thread_block_dimensions>>>(d_weights, d_biases, d_z, d_activation, d_shape, shape_length);

    // Back to host
    cudaMemcpy(h_activation, d_activation, size_activation, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, size_z, cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_activation);
    cudaFree(d_z);
    cudaFree(d_shape);

    int z_offset = 0;
    for(int shape_idx = 1; shape_idx < shape_length; ++shape_idx){
        std::cout << "Z values: " << shape_idx << ". hidden layer" << std::endl;
        
        for(int i = 0; i < shape[shape_idx]; ++i){
            std::cout << "[";
            for(int j = 0; j < num_inputs; ++j){
                std::cout << h_z[z_offset + j * shape[shape_idx] + i] << ", ";
            }
            std::cout << "]" << std::endl;
        } 
        z_offset += shape[shape_idx] * num_inputs;
    }
    
    int activation_offset = shape[0]; // skip input values
    for(int shape_idx = 1; shape_idx < shape_length; ++shape_idx){
        std::cout << "Activations: " << shape_idx << ". hidden layer" << std::endl;
        for(int i = 0; i < shape[shape_idx]; ++i){
            std::cout << "[";
            for(int j = 0; j < num_inputs; ++j){
                std::cout << h_z[activation_offset + j * shape[shape_idx] + i] << ", ";
            }
            std::cout << "]" << std::endl;
        } 
        activation_offset += shape[shape_idx] * num_inputs;
    }

    getchar();

    return 0;
}