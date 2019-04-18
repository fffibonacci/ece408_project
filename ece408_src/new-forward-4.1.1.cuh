
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
namespace mxnet
{
namespace op
{

#define KERNEL_WIDTH   5
#define TILE_WIDTH     20
#define CACHE_WIDTH    (KERNEL_WIDTH + TILE_WIDTH - 1) 
//__constant__ float deviceKernel[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH];


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0*W_out/TILE_WIDTH);
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.z % W_grid;
    int by = blockIdx.z / W_grid;

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h_o = by * TILE_WIDTH + ty; //very important!!  blockIdx.z
    int w_o = bx * TILE_WIDTH + tx;
    int r = KERNEL_WIDTH / 2;
    // int w_i = w_o + r ;
    // int h_i = h_o + r;
    
    __shared__ float Nds1[CACHE_WIDTH][CACHE_WIDTH]; //one channel
    __shared__ float Nds6 [6][CACHE_WIDTH][CACHE_WIDTH]; //one channel 

    float acc = 0.0;
    if (C == 6){  
        for(int c = 0; c < C; c++){
            //load tile to shared memory
            if(0<=h_o && h_o < H && 0<=w_o && w_o < W){
                Nds6[c][ty][tx] = x4d(b,c,h_o,w_o);
            }
            else{
                Nds6[c][ty][tx] = 0;
            }
        }
        __syncthreads();
        for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++){
                for(int q=0; q < K; q++){
                    if(ty<TILE_WIDTH && tx<TILE_WIDTH)
                        acc += Nds6[c][ty+p][tx+q] * k4d(m,c,p,q);
                }
            }
        }
    }
    else{
        //load tile to shared memory
        if(0<=h_o && h_o < H && 0<=w_o && w_o < W){
            Nds1[ty][tx] = x4d(b,0,h_o,w_o);
        }
        else{
            Nds1[ty][tx] = 0;
        }
        __syncthreads();
        for(int p = 0; p < K; p++){
            for(int q=0; q < K; q++){
                if(ty<TILE_WIDTH && tx<TILE_WIDTH)
                    acc += Nds1[ty+p][tx+q] * k4d(m,0,p,q);
            }
        }
    }

    if(h_o<H_out && w_o<W_out)
        y4d(b,m,h_o,w_o) = acc;

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];//featuremap
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0*W_out/TILE_WIDTH);
    const int H_grid = ceil(1.0*H_out/TILE_WIDTH);
    const int Z = H_grid * W_grid;
    //cout<<B<<" "<<M<<" "<<C<<" "<<H<<W<<" "<<K<<endl; 
    //10000,6,1,48,48,5
    //10000,16,6,22,22,5
    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);

    dim3 blockDim(CACHE_WIDTH, CACHE_WIDTH, 1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif