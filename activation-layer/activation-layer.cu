#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sigmoidActivation(float *z_mat, float *activation_mat){
    int thread_idx = threadIdx.x + (threadIdx.y * blockDim.x) + (threadIdx.z * blockDim.x * blockDim.y);

    activation_mat[thread_idx] = 1.0 / (1.0 + exp(-z_mat[thread_idx]));
}

int main(){

    const int arraySize = 5;

    // initialise host memory
    float h_z_val[arraySize] = {1., 2., 3., 4., 5.};
    float h_activation[arraySize] = {0};

    // calc sizes
    const size_t size_z_val = arraySize * sizeof(float);
    const size_t size_d_activation = arraySize * sizeof(float);
    
    float *d_z_val, *d_activation;

    // allocate dev memory
    cudaMalloc(&d_z_val, size_z_val);
    cudaMalloc(&d_activation, size_d_activation);

    // copy data
    cudaMemcpy(d_z_val, h_z_val, size_z_val, cudaMemcpyHostToDevice);
    cudaMemcpy(d_activation, h_activation, size_d_activation, cudaMemcpyHostToDevice);

    // calling function
    // 1 = num of blocks
    // arraySize = num of threads
    sigmoidActivation<<<1, arraySize>>>(d_z_val, d_activation);

    // copy back to host
    cudaMemcpy(h_activation, d_activation, size_d_activation, cudaMemcpyDeviceToHost);

    // print
    printf("sigmoid({1., 2., 3., 4., 5.}) = {%f, %f, %f, %f, %f}\n", 
        h_activation[0], h_activation[1], h_activation[2], h_activation[3], h_activation[4]);
    getchar();
        
    return 0;
}