#ifndef _KDTREE_KERNEL_H_
#define _KDTREE_KERNEL_H_

/*-------------------------------------
  Include Files
-------------------------------------*/

#include <stdio.h>

#define SDATA( index)      cutilBankChecker(sdata, index)

/*-----------------------------------------------------------------------------
  Name:  vectorAdd
  Desc:  C = A + B as a vector
-----------------------------------------------------------------------------*/
__global__ void
vectorAdd
(
  float *A,		// IN - 1st vector to add
  float* B,		// IN - 2nd vector to add
  float* C, 	// OUT - result vector, C = A + B
  int    N		// IN - num of elements in vector
)
{
  // access thread id
  // access number of threads in this block
  //const unsigned int num_threads = blockDim.x;

  //int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  //int idx = row*width+col;
  int idx = col;

  C[idx] = A[idx] + B[idx];
}

#endif // #ifndef _KDTREE_KERNEL_H_
