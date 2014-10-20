//-----------------------------------------------------------------------------
//	CS 790-058 GPGPU
//	Final Project (Point Location using GPU)
//	
//	This file contains the GPU Kernels
//	
//	by Shawn Brown (shawndb@cs.unc.edu)
//-----------------------------------------------------------------------------

#ifndef _FP_GPU_H_
#define _FP_GPU_H_


//---------------------------------------------------------
//
//	Includes
//
//---------------------------------------------------------

#include <stdio.h>
//#include <float.h>
#include "KDTree_GPU.h"
#include "Main.h"


//---------------------------------------------------------
//
//	Function Definitions
//
//---------------------------------------------------------

//---------------------------------------------------------
//	Name:	PLQ_GPU_BF_DIST
//	Desc:	Computes distance for each point in
//			a vector against query point
//	Note:	
//			The optimal thread block size appears to be 
//			32 TPR x 1 RPB = 32 TPB
//			determined after extensive testing
//---------------------------------------------------------

__global__ void
PLQ_GPU_BF_DIST
( 
	float2* out,	// OUT:  Result of distance field calculations, 
	float4* in,		// IN:	 Distance Vector Field
	float4  qp,		// IN:	query point to compute distance for
	int     w,		// IN:  width of 2D vector field (# of columns)
	int     h		// IN:  height of 2D vector field (# of rows)
)
{
	// Grid Dimensions (of blocks within the grid)
		// bw * bh
	//const int blocksPerRow = gridDim.x;	// Columns (per Grid)
	//const int rowsPerGrid  = gridDim.y;	// Rows (per Grid)

	// 1D Block Dimensions (of threads within a thread block)
		// tw * th
	const int threadsPerRow  = blockDim.x;	// Columns (per block)
	const int rowsPerBlock   = blockDim.y;	// Rows (per block) 

	// Block index
    int bx = blockIdx.x;	// column in grid
	int by = blockIdx.y;	// row in grid

    // Thread index
    int tx = threadIdx.x;	// column in block
	int ty = threadIdx.y;	// row in block

	//
	//	1st Approach
	//		copy from device to local shared (fast) memory
	//		synchronize threads to avoid clobbering memory
	//

	// Shared Memory for storing results
    __shared__ float4 V[BFD_THREADS_PER_ROW][BFD_ROWS_PER_BLOCK];

	// Compute Starting position of curent thread
	int currCol = (bx * threadsPerRow) + tx;
	int currRow = (by * rowsPerBlock) + ty;
	int idx   = currRow * w + currCol;

	// Load distance vectors into Shared Memory (via each thread)
		// Note:  Copying the vector as a float4 is about twice
		//        as fast as copying individual components
	V[tx][ty] = in[idx];

	__syncthreads();

	// Compute distance from current point to query point
	// using standard 2-norm
	float4 diff;
	diff.x = V[tx][ty].x - qp.x;
	diff.y = V[tx][ty].y - qp.y;
	diff.z = V[tx][ty].z - qp.z;
	diff.w = V[tx][ty].w;		// Index of Point goes here

	float dist2 =   (diff.x * diff.x)
				  + (diff.y * diff.y)
				  + (diff.z * diff.z);
	float dist = sqrt( dist2 );

    // Write result
    out[idx].x = dist;		// Save Computed Distance to point
	out[idx].y = diff.w;	// Save Index of corresponding point

	__syncthreads();

	/*
	//
	//	Alternate Approach, Don't copy to local, don't synchronize
	//		appears to be approximately twice as slow as using local memory
	//		approach above
	//

	// Compute Starting position of curent thread
	int currCol = (bx * threadsPerRow) + tx;
	int currRow = (by * rowsPerBlock) + ty;
	int idx   = currRow * w + currCol;

	// Load distance vectors into Shared Memory (via each thread)
	float4 pnt = in[idx];

	// Compute distance of current point against query point
	float4 diff;
	diff.x = pnt.x - qp.x;
	diff.y = pnt.y - qp.y;
	diff.z = pnt.z - qp.z;
	diff.w = 0.0f;
	float dist2 = (diff.x * diff.x) + 
				  (diff.y * diff.y) + 
				  (diff.z * diff.z);
	float dist = sqrt( dist2 );


	out[idx] = dist;
	*/
}


//---------------------------------------------------------
//	Name:	PLQ_GPU_BF_MIN
//	Desc:	Reduces Each BLock
//	Note:	
//---------------------------------------------------------

__global__ void
PLQ_GPU_BF_MIN
( 
	float2*  distOut,	// OUT:  Reduced Vector Field
	float2*  distIn		// IN:	 Distance Vector Field
)
{
	// Grid Dimensions (of blocks within the grid)
		// bw * bh
	//const int blocksPerRow  = gridDim.x;	// Columns (per Grid)
	//const int rowsPerGrid   = gridDim.y;	// Rows (per Grid)
	//const int blocksPerGrid = gridDim.x * gridDim.y;

	// 1D Block Dimensions (of threads within a thread block)
		// tw * th
	//const int threadsPerRow   = blockDim.x;	// Columns (per block)
	//const int rowsPerBlock    = blockDim.y;	// Rows (per block) 
	const int threadsPerBlock = blockDim.x * blockDim.y; 

	// Block indices
    //int bx = blockIdx.x;	// column in grid
	//int by = blockIdx.y;	// row in grid
	//int bEven = 2 * bx;

	// Thread indices
    //int tx = threadIdx.x;	// column in block
    //int ty = threadIdx.y;	// row in block

    __shared__ float dists[BFMR_THREADS_PER_BLOCK2];
	__shared__ float idxs[BFMR_THREADS_PER_BLOCK2];

	// Block Index (relative to grid)
	int bidx = (blockIdx.y * gridDim.x) + (2 * blockIdx.x);	// Even Block
	int bout = (blockIdx.y * gridDim.x) + (blockIdx.x);		// out index
	int baseIdx = bidx * threadsPerBlock;					// block * threadsPerBlock

	// Thread Index (relative to current block)
	int tidx1 = (threadIdx.y * blockDim.x) + threadIdx.x;	// Even Block element
	int tidx2 = tidx1 + threadsPerBlock;					// Odd Block element

	// Get Starting Position
		// Even Block
	int inIdx1 = baseIdx + tidx1;
	int inIdx2 = inIdx1 + threadsPerBlock;

	// Load b[i],   even block of distances (& indices) into local (fast) memory
	// Load b[i+1], odd  block of distances (& indices) into local (fast) memory

	dists[tidx1] = distIn[inIdx1].x;
    dists[tidx2] = distIn[inIdx2].x;

	idxs[tidx1] = distIn[inIdx1].y;
    idxs[tidx2] = distIn[inIdx2].y;

	__syncthreads();


    // Perform reduction to find minimum element in these 2 blocks
    for (int stride = threadsPerBlock; stride > 0; stride /= 2)
    {
        if (tidx1 < stride)
        {
            float f0 = dists[tidx1];
            float f1 = dists[tidx1 + stride];
            //float i0 = idxs[tidx1];
            float i1 = idxs[tidx1 + stride];

			if (f1 < f0) 
			{
                dists[tidx1] = f1;
				idxs[tidx1] = i1;
            }
        }

		__syncthreads();
    }

    // Write final result to output
		// but only for 1st thread in thread block
	if (tidx1 == 0)
	{
		distOut[bout].x = dists[0];
		distOut[bout].y = idxs[0];
	}

	__syncthreads();
}

/*
__global__ void
PLQ_GPU_BF_MIN
( 
	float2*  distOut,	// OUT:  Reduced Vector Field
	float2*  distIn		// IN:	 Distance Vector Field
)
{

	// 1D Block Dimensions (of threads within a thread block)
		// tw * th
	const int threadsPerBlock = blockDim.x * blockDim.y; 

	// Block Index (relative to grid)
	int bIn     = (blockIdx.y * gridDim.x) + (2 * blockIdx.x);	// Current Even Block
	int bOut    = (blockIdx.y * gridDim.x) + blockIdx.x;		// Current Block
	int baseIn  = bIn * threadsPerBlock;						// even block * threadsPerBlock
	int baseOut = bOut * threadsPerBlock;						// block * threads per block

	// Thread Index (relative to current block)
	int tidx1 = (threadIdx.y * blockDim.x) + threadIdx.x;	// Even Block element
	//int tidx2 = tidx1 + threadsPerBlock;					// Odd Block element

	// Get Starting Position
		// Even Block
	int inIdx1 = baseIn + tidx1;
	int inIdx2 = inIdx1 + threadsPerBlock;
	int outIdx = baseOut + tidx1;

	// Compare elements in even block to elements in odd block
	// store minimum element in output
	float2 f0 = distIn[inIdx1];
    float2 f1 = distIn[inIdx2];

	if (f1.x < f0.x) 
	{
		distOut[outIdx] = f1;
	}
	else
	{
		distOut[outIdx] = f0;
	}
}

__global__ void
PLQ_GPU_BF_REDUCE_BLOCK
( 
	float2*  distOut,	// OUT:  Reduced Vector Field
	float2*  distIn		// IN:	 Distance Vector Field
)
{
    __shared__ float2 shared[BFMR_THREADS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Copy input.
    shared[tid] = distIn[tid];
    shared[tid + blockDim.x] = distIn[tid + blockDim.x];

    // Perform reduction to find minimum.
    for(int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float2 f0 = shared[tid];
            float2 f1 = shared[tid + d];
            
            if (f1.x < f0.x) {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) 
	{
		distOut[bid] = shared[0];
	}

    __syncthreads();
}
*/


#endif // #ifndef _FP_GPU_H_
