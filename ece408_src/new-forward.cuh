
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iostream>
namespace mxnet
{
namespace op
{
#define TILE_WIDTH 32
#define BLOCK_WIDTH 16
// this part is for shared memory convolution
#define KERNEL_WIDTH   5
//#define TILE_WIDTH     20
#define CACHE_WIDTH    (KERNEL_WIDTH + TILE_WIDTH - 1) 
// __constant__ float deviceKernel[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH];
__constant__ float kernel1[150];
__constant__ float kernel2[2400];



// Optimization 1: Shared Memory convolution
__global__ void forward_kernel_cov_shared(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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

// Optimization 2: constant memory for kernel matrix
__global__ void forward_kernel_origin(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0*W_out/BLOCK_WIDTH);
   // const int H_grid = H_out/TILE_WIDTH;
   // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * BLOCK_WIDTH + threadIdx.y; // we get the grid number of (blockIdx.z / W_grid) by this
                                                              // so we need to multiply by TILE_WIDTH
    int w = (blockIdx.z % W_grid) * BLOCK_WIDTH + threadIdx.x;
    float acc = 0.;
    for(int c = 0; c < C; c++){
        for(int p = 0; p < K; p++){
            for(int q=0; q < K; q++){
                if(h+p < H && w+q < W)
                    acc += x4d(b,c,h+p,w+q) * k4d(m,c,p,q);
            }
        }
    }
    if(h<H_out && w<W_out)
        y4d(b,m,h,w) = acc;

#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel_cons_kernel_1(float *__restrict__ y, const float *__restrict__ x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0*W_out/BLOCK_WIDTH);
   // const int H_grid = H_out/TILE_WIDTH;
   // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * BLOCK_WIDTH + threadIdx.y; // we get the grid number of (blockIdx.z / W_grid) by this
                                                              // so we need to multiply by TILE_WIDTH
    int w = (blockIdx.z % W_grid) * BLOCK_WIDTH + threadIdx.x;
    float acc1, acc2= 0.;
    for(int c = 0; c < C; c++){
        #pragma unroll 5
        for(int p = 0; p < K; p++){
            #pragma unroll 5
            for(int q=0; q < K; q++){
                if(h+p < H && w+q < W)
                    acc1 += x4d(b,c,h+p,w+q) * k4d(m,c,p,q);
                // if(h+p+1 < H && w+q < W)
                //     acc2 += x4d(b,c,h+p+1,w+q) * k4d(m,c,p,q);
                    // if( w+q+1 < W)
                    //     acc3 += x4d(b,c,h+p,w+q+1) * k4d(m,c,p,q);
                    // if(h+p+1 < H && w+q+1 < W)
                    //     acc4 += x4d(b,c,h+p+1,w+q+1) * k4d(m,c,p,q);
            }
        }
    }
    if(h<H_out && w<W_out) {
        y4d(b,m,h,w) = acc1;
        // if(h+1 < H_out)
        //     y4d(b,m,h+1,w) = acc2;
        // if(w+1 < W_out)
        //     y4d(b,m,h,w+1) = acc3;
        // if(h+1 < H_out && w+1 < W_out)
        //     y4d(b,m,h+1,w+1) = acc4;

    }


#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel_cons_kernel_2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(1.0*W_out/BLOCK_WIDTH);
   // const int H_grid = H_out/TILE_WIDTH;
   // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) kernel2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * BLOCK_WIDTH + threadIdx.y; // we get the grid number of (blockIdx.z / W_grid) by this
                                                              // so we need to multiply by TILE_WIDTH
    int w = (blockIdx.z % W_grid) * BLOCK_WIDTH + threadIdx.x;
    float acc = 0.;
    for(int c = 0; c < C; c++){
        for(int p = 0; p < K; p++){
            //#pragma unroll(3)
            for(int q=0; q < K; q++){
                if(h+p < H && w+q < W)
                    acc += x4d(b,c,h+p,w+q) * k4d(m,c,p,q);
            }
        }
    }
    if(h<H_out && w<W_out)
        y4d(b,m,h,w) = acc;

#undef y4d
#undef x4d
#undef k4d
}

#define NUM_THREADS 256

__constant__ float Mask[2400];

// Optimization 3: Unrolling + Matrix multiplication
__global__ void unroll(int C, int H, int W, int K, float *x, float *x_unroll)
{
    int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    int t = blockIdx.y * NUM_THREADS + threadIdx.y;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;
    int batch = blockIdx.x;
    //
    // #define x_unroll4d(i1, i0) x_unroll[i1 * W_unroll + i0]
    // #define x_4d(i2, i1, i0) x[i2 * (H * W) + i1 * (W) + i0]

    if (t < C * W_unroll) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        w_unroll = h_out * W_out + w_out; // s
        w_base = c * K * K;
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++) {
                h_unroll = w_base + p * K + q;
                    x_unroll[batch * ( C * K * K * W_unroll) + h_unroll * W_unroll + w_unroll] = x[batch*(H*W*C)+c*(H*W)+(h_out+p)*W + (w_out+q)];
            }
        }
    }
    // #undef x_4d
    // #undef x_unroll4d

}

// this is for unrolling + shared matmul
//#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *__restrict__ B, float *__restrict__ C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns, int CH, int H, int W) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  //__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int batch = blockIdx.x;
  int bx = blockIdx.y;
  int by = blockIdx.z;
  int tx = threadIdx.y;
  int ty = threadIdx.z;

    int W_out = W - 4;

  int Row = by*TILE_WIDTH + ty;
  int Col = bx*TILE_WIDTH + tx;

  float Pvalue = 0.0;
    #pragma unroll(8)
    for(int m = 0; m < (numAColumns-1)/TILE_WIDTH + 1; m++){

      // if( Row<numARows&&(m * TILE_WIDTH+tx) < numAColumns)
      //   subTileA[ty][tx] = A[Row*numAColumns + m * TILE_WIDTH+tx];
      // else
      //   subTileA[ty][tx] = 0.0;
            int ybase = Col / W_out;
            int xbase = Col % W_out;
            int channel = (m*TILE_WIDTH+ty)/25;
            int linidx = (m*TILE_WIDTH+ty)%25;
            int in_x = linidx % 5;
            int in_y = linidx / 5;

      if((m*TILE_WIDTH+ty)<numBRows&&Col<numBColumns)
        subTileB[ty][tx] = B[batch * (CH*H*W)+channel*(H*W)+(ybase+in_y)*W + xbase+in_x];//B[batch * (numBRows*numBColumns)+(m*TILE_WIDTH+ty)*numBColumns + Col];
      else
        subTileB[ty][tx] = 0.0;

      __syncthreads();

    if((Row<numCRows) && (Col<numCColumns)){
      #pragma unroll(32)
      for(int k = 0; k < TILE_WIDTH; ++k){
      //  Pvalue += subTileA[ty][k] * subTileB[k][tx];
            Pvalue += Mask[Row*numAColumns + m * TILE_WIDTH+k]* subTileB[k][tx];

      }
    }
       __syncthreads();
    }

  if((Row<numCRows) && (Col<numCColumns)){
    C[batch*(numCRows*numCColumns)+Row*numCColumns+Col] = Pvalue;

  }
}


__global__ void forward_kernel_unroll_matmul(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working
    int W_grid = ceil(1.0*W_out/TILE_WIDTH);

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int n, m, h, w, c, p, q;
    n = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
    float acc = 0.;
    for (c = 0; c < C; c++) {
    // sum over all input channels
      for (p = 0; p < K; p++){
      // loop over KxK filter
        for (q = 0; q < K; q++){
            acc += x4d(n, c, h + p, w + q) * k4d(m, c, p, q);
        }
      }
    }
    if(h<H_out && w<W_out)
      y4d(n, m, h, w) = acc;


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
    const int B = x.shape_[0]; // batch size
    const int M = y.shape_[1]; // # of output feature maps
    const int C = x.shape_[1]; // # of input channels
    const int H = x.shape_[2]; // input image width
    const int W = x.shape_[3]; // input image height
    const int K = w.shape_[3]; // kernel size
    //std::cout << C << std::endl;
    //std::cout << K << std::endl;
    //std::cout << W << std::endl;
    const int x_size = x.shape_[0] * x.shape_[1] * x.shape_[2] * x.shape_[3] / 2;
    const int y_size = y.shape_[0] * y.shape_[1] * y.shape_[2] * y.shape_[3] / 2;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = W_out*1.0/BLOCK_WIDTH;
    const int H_grid = H_out*1.0/(BLOCK_WIDTH);
    const int Z = H_grid * W_grid;
    dim3 gridDim(B, M, Z);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    // std::cout << x.shape_[0] << std::endl;
    // std::cout << x.shape_[1] << std::endl;
    // std::cout << x.shape_[2] << std::endl;
    // std::cout << x.shape_[3] << std::endl;
    // std::cout << y.shape_[0] << std::endl;
    // std::cout << y.shape_[1] << std::endl;
    // std::cout << y.shape_[2] << std::endl;
    // std::cout << y.shape_[3] << std::endl;
    //std::cout << "C:" << M << std::endl;
    //std::cout << ":" << M << std::endl;
    //std::cout << "M:" << M << std::endl;
    //std::cout << C*M*K*K << std::endl;
    // Set the kernel dimensions
    //cudaMemcpyToSymbol(deviceKernel, &w, sizeof(float) * M * C * K * K);

    // Launch Optimization 2: constant kernel matrix
    // Call the kernel
    if(C == 1) {
        cudaStream_t stream1;//,stream2;
        cudaStreamCreate(&stream1);
        //cudaStreamCreate(&stream2);
        forward_kernel_origin<<<gridDim,blockDim,0,stream1>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
        //forward_kernel_origin<<<gridDim,blockDim,0,stream2>>>(y.dptr_ + y_size,x.dptr_ + x_size, w.dptr_, B,M,C,H,W,K);
        cudaStreamDestroy(stream1);
        //cudaStreamDestroy(stream2);
        //cudaMemcpyToSymbol(kernel1, w.dptr_, sizeof(float) * C * M * K * K);
        //forward_kernel_cons_kernel_1<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    }
    else {
        //cudaMemcpyToSymbol(kernel2, w.dptr_, sizeof(float) * C * M * K * K);
        //forward_kernel_cons_kernel_2<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
        int W_unroll = C * K * K;
        int H_unroll = H_out * W_out;

        // //float *X_unrolled;
        // //cudaMalloc((void **) &X_unrolled, B * W_unroll * H_unroll * sizeof(float));

        cudaMemcpyToSymbol(Mask, w.dptr_, K*K*C*M*sizeof(float));

        dim3 mmGrid(B, ceil(1.0*H_unroll/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH));
        dim3 mmBlock(1, TILE_WIDTH, TILE_WIDTH);

        // //int num_threads = C * H_out * W_out;
        int num_blocks = ceil(1.0*(C * H_out * W_out) / (NUM_THREADS * 2));
        dim3 unrollGrid(B, num_blocks, 1);
        dim3 unrollBlock(1, NUM_THREADS, 1);


        // //unroll<<<unrollGrid, unrollBlock>>>(C, H, W, K, x.dptr_, X_unrolled);
        cudaStream_t stream1; // stream2;
        cudaStreamCreate(&stream1);
        //cudaStreamCreate(&stream2);
        matrixMultiplyShared<<<mmGrid, mmBlock,0,stream1>>>(x.dptr_, y.dptr_, M, W_unroll, W_unroll, H_unroll, M, H_unroll, C, H, W);
        //matrixMultiplyShared<<<mmGrid, mmBlock,0,stream2>>>(x.dptr_ + x_size, y.dptr_ + y_size, M, W_unroll, W_unroll, H_unroll, M, H_unroll, C, H, W);
        //forward_kernel_origin<<<gridDim,blockDim,0,stream1>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
        //forward_kernel_origin<<<gridDim,blockDim,0,stream2>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
        cudaStreamDestroy(stream1);
        //cudaStreamDestroy(stream2);
        //

    }

    //forward_kernel_origin<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    
    // Launch Optimization 1: shared memory convolution:
    //cout<<B<<" "<<M<<" "<<C<<" "<<H<<W<<" "<<K<<endl; 
    //10000,6,1,48,48,5
    //10000,16,6,22,22,5
    // Set the kernel dimensions
    // dim3 gridDim(B, M, Z);

    // dim3 blockDim(CACHE_WIDTH, CACHE_WIDTH, 1);

    // // Call the kernel
    // forward_kernel_cov_shared<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);


    // launch Optimization 3: unrolling + shared matrix multiply


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