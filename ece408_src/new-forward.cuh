
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


#define NUM_THREADS 256

__constant__ float Mask[2400];

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
		w_unroll = h_out * W_out + w_out;
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


#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *B, float *C,
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
    for(int m = 0; m < (numAColumns-1)/TILE_WIDTH + 1; ++m){

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
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];


    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // int W_grid = ceil(1.0*W_out/TILE_WIDTH);
    // int H_grid = ceil(1.0*H_out/TILE_WIDTH);
		//
    // int Z = H_grid * W_grid;
		//
		//
    // // Set the kernel dimensions
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(B, M, Z);
		//
		int W_unroll = C * K * K;
		int H_unroll = H_out * W_out;

		//float *X_unrolled;
		//cudaMalloc((void **) &X_unrolled, B * W_unroll * H_unroll * sizeof(float));

		cudaMemcpyToSymbol(Mask, w.dptr_, K*K*C*M*sizeof(float));

		dim3 mmGrid(B, ceil(1.0*H_unroll/TILE_WIDTH), ceil(1.0*M/TILE_WIDTH));
  	dim3 mmBlock(1, TILE_WIDTH, TILE_WIDTH);

		//int num_threads = C * H_out * W_out;
		int num_blocks = ceil(1.0*(C * H_out * W_out) / NUM_THREADS);
		dim3 unrollGrid(B, num_blocks, 1);
		dim3 unrollBlock(1, NUM_THREADS, 1);


		//unroll<<<unrollGrid, unrollBlock>>>(C, H, W, K, x.dptr_, X_unrolled);
		matrixMultiplyShared<<<mmGrid, mmBlock>>>(x.dptr_, y.dptr_, M, W_unroll, W_unroll, H_unroll, M, H_unroll, C, H, W);

		// for (int n=0; n < B; n++) {
		// 	//printf("batch: %d", n);
		//
		// 	matrixMultiplyShared<<<mmGrid, mmBlock>>>(X_unrolled + n*W_unroll * H_unroll, outptr, M, W_unroll, W_unroll, H_unroll, M, H_unroll);
		// 	//inptr += C*H*W;
		// 	outptr += M*H_out*W_out;
		// }
		//cudaFree(X_unrolled);

    // Call the kernel
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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
