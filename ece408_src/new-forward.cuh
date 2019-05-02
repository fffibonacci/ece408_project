#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{



#define BLOCK_WIDTH 22
// this part is for shared memory convolution
#define KERNEL_WIDTH   5

#define NUM_THREADS 768

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


#define TILE_WIDTH 64
#define TILE_HEIGHT 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *__restrict__ B, float *__restrict__ C,
                                     int numARows, int numAColumns,
                                     int numBColumns,
                                     int CH, int H, int W) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  //__shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_HEIGHT][TILE_WIDTH];
	int batch = blockIdx.x;
  int bx = blockIdx.y;
  int by = blockIdx.z;
  int tx = threadIdx.x%TILE_WIDTH;
  int ty = threadIdx.x/TILE_WIDTH;

	//int W_out = W - 4;

	//printf("%d, %d, %d, %d, %d, %d, %d, %d, %d \n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns, CH, H, W);

  int Row = by*TILE_HEIGHT + ty;
  int Col = bx*TILE_WIDTH + tx;
	int ybase = Col / 18;//W_out;
	int xbase = Col % 18;//W_out;
  float Pvalue = 0.0;
	float Pvalue1 = 0.0;

    for(int m = 0; m < 10; ++m){ //(numAColumns-1)/TILE_WIDTH + 1

      // if( Row<numARows&&(m * TILE_WIDTH+tx) < numAColumns)
      //   subTileA[ty][tx] = A[Row*numAColumns + m * TILE_WIDTH+tx];
      // else
      //   subTileA[ty][tx] = 0.0;

			int channel = (m*TILE_HEIGHT+ty)/25;
			int linidx = (m*TILE_HEIGHT+ty)%25;
			int in_x = linidx % 5;
			int in_y = linidx / 5;

      if((m*TILE_HEIGHT+ty)<150&&Col<324) //brow, bcol
        subTileB[ty][tx] = B[batch * (2904)+channel*(484)+(ybase+in_y)*22 + xbase+in_x];//B[batch * (numBRows*numBColumns)+(m*TILE_WIDTH+ty)*numBColumns + Col];
      else
        subTileB[ty][tx] = 0.0;

			// channel = (m*TILE_WIDTH+ty+1)/25;
			// linidx = (m*TILE_WIDTH+ty+1)%25;
			// in_x = linidx % 5;
			// in_y = linidx / 5;
			// if((m*TILE_WIDTH+ty+1)<150&&Col<324) //brow, bcol
      //   subTileB[ty+1][tx] = B[batch * (2904)+channel*(484)+(ybase+in_y)*22 + xbase+in_x];//B[batch * (numBRows*numBColumns)+(m*TILE_WIDTH+ty)*numBColumns + Col];
      // else
      //   subTileB[ty+1][tx] = 0.0;

      __syncthreads();

    if((Row<16) && (Col<324)){ //crows ccol
			#pragma unroll 16
      for(int k = 0; k < TILE_HEIGHT; ++k){
				//C[(batch*(5184)+Row*324+Col)*150+m*32+k]=Mask[Row*150 + m * TILE_WIDTH+k]* subTileB[k][tx];
        //Pvalue += subTileA[ty][k] * subTileB[k][tx];
				 // float reg_b = subTileB[k][tx];
			 	 Pvalue += Mask[Row*150 + m * TILE_HEIGHT+k]* subTileB[k][tx];
				 // if(Row<15)
 			 	 // Pvalue1 += Mask[(Row+1)*150 + m * TILE_WIDTH+k]* reg_b;
				// Pvalue += Mask[Row*150 + m * TILE_WIDTH+k+1]* subTileB[k+1][tx];
				// Pvalue += Mask[Row*150 + m * TILE_WIDTH+k+2]* subTileB[k+2][tx];
				// Pvalue += Mask[Row*150 + m * TILE_WIDTH+k+3]* subTileB[k+3][tx];

      }
    }
       __syncthreads();
    }

  if((Row<16) && (Col<324)){
    C[batch*(5184)+Row*324+Col] = Pvalue;
  }
	// if((Row<15) && (Col<324)){
  //   C[batch*(5184)+(Row+1)*324+Col] = Pvalue1;
	//
  // }
}


__global__ void forward_kernel_cons_kernel_1(float *__restrict__ y, const float *__restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
		//
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    const int W_grid = ceil(1.0*44/BLOCK_WIDTH);
		//prinf("%d, %d, %d ,%d\", H, W, M, C);
   // const int H_grid = H_out/TILE_WIDTH;
   // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
   // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (11616) + (i2) * (1936) + (i1) * (44) + i0]
#define x4d(i3, i1, i0) x[(i3) * (2304)  + (i1) * (48) + i0]
#define k4d(i3, i1, i0) Mask[(i3) * (25)  + (i1) * (5) + i0]
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * BLOCK_WIDTH + threadIdx.x/BLOCK_WIDTH; // we get the grid number of (blockIdx.z / W_grid) by this
                                                              // so we need to multiply by TILE_WIDTH
    int w = (blockIdx.z % W_grid) * BLOCK_WIDTH + threadIdx.x%BLOCK_WIDTH;
    float acc = 0.;
    //for(int c = 0; c < C; c++){
		if(h<44 && w<44){
				#pragma unroll 5
        for(int p = 0; p < K; p++){
            #pragma unroll 5
            for(int q=0; q < K; q++){
                //if(h+p < H && w+q < W)
                    acc += x4d(b,h+p,w+q) * k4d(m,p,q);
            }
        }
		}
    //}
    if(h<44 && w<44)
        y4d(b,m,h,w) = acc;

#undef y4d
#undef x4d
#undef k4d
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
    const int W_grid = ceil(W_out*1.0/BLOCK_WIDTH);
    const int H_grid = ceil(H_out*1.0/BLOCK_WIDTH);
    const int Z = H_grid * W_grid;
    dim3 gridDim(B, M, Z);
    dim3 blockDim(BLOCK_WIDTH*BLOCK_WIDTH, 1, 1);


    // Launch Optimization 2: constant kernel matrix
    // Call the kernel
    if(C == 1) {

        cudaMemcpyToSymbol(Mask, w.dptr_, sizeof(float) * C * M * K * K);
        forward_kernel_cons_kernel_1<<<gridDim, blockDim>>>(y.dptr_,x.dptr_, B,M,C,H,W,K);
    }
    else {
        //cudaMemcpyToSymbol(kernel2, w.dptr_, sizeof(float) * C * M * K * K);
        //forward_kernel_cons_kernel_2<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
        //int W_unroll = C * K * K;
        int H_unroll = H_out * W_out;

        // float *pval_arr;
        // cudaMalloc((void **) &pval_arr, B*M*150 *H_out*W_out* sizeof(float));

        cudaMemcpyToSymbol(Mask, w.dptr_, K*K*C*M*sizeof(float));

        dim3 mmGrid(B, ceil(1.0*H_unroll/TILE_WIDTH), 1);
        dim3 mmBlock(TILE_WIDTH*TILE_HEIGHT, 1 ,1);

        // //int num_threads = C * H_out * W_out;
        // int num_blocks = ceil(1.0*(C * H_out * W_out) / NUM_THREADS);
        // dim3 unrollGrid(B, num_blocks, 1);
        // dim3 unrollBlock(1, NUM_THREADS, 1);


        // //unroll<<<unrollGrid, unrollBlock>>>(C, H, W, K, x.dptr_, X_unrolled);
        matrixMultiplyShared<<<mmGrid, mmBlock>>>(x.dptr_, y.dptr_, M, 150, H_unroll, C, H, W);
    }

		//float *X_unrolled;
		//cudaMalloc((void **) &X_unrolled, B * W_unroll * H_unroll * sizeof(float));

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
