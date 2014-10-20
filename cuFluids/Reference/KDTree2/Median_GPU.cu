#ifndef _DIST_GPU_H_
#define _DIST_DPU_H_

/*-------------------------------------
  Include Files
-------------------------------------*/

#include <stdio.h>
#include "KD_API.h"


/*---------------------------------------------------------
  Name:  ComputeDist3D
  Desc:  Compute distance between each 3D point and a
         3D query point.
---------------------------------------------------------*/

__global__ void
ComputeDist3D_GPU
(
  float2* out,	// OUT: Result of 2D distance field calculations
  float4* in,	// IN:  Distance Vector Field
  float4  qp,	// IN:  Query point to compute distance for
  int     w,	// IN:  width of 2D vector field (# of columns)
  int     h		// IN:  height of 2D vector field (# of rows)
)
{
	// Grid Dimensions (of blocks within the grid)

	//
	//	1st Approach
	//		copy from device to local shared (fast) memory
	//		synchronize threads to avoid clobbering memory
	//

	// Shared Memory for storing results
	__shared__ float4 V[BFD_THREADS_PER_ROW][BFD_ROWS_PER_BLOCK];

	// Compute Starting position of current thread
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int currCol = (blockIdx.x * blockDim.x) + tx;
	int currRow = (blockIdx.y * blockDim.y) + ty;
	int currIdx = currRow * w + currCol;

	// Load distance vectors into Shared Memory (via each thread)
		// Note:  Copying the vector as a float4 is about twice
		//        as fast as copying individual components
	V[tx][ty] = in[currIdx];

	__syncthreads();

	// Compute distance from current point to query point
	// using standard 2-norm
	float4 diff;
	diff.x = V[tx][ty].x - qp.x;
	diff.y = V[tx][ty].y - qp.y;
	diff.z = V[tx][ty].z - qp.z;
	diff.w = V[tx][ty].w;		// Index of Point stored here

	float dist2 =   (diff.x * diff.x)
				  + (diff.y * diff.y)
				  + (diff.z * diff.z);
	float dist = sqrt( dist2 );

	// Write result
	out[currIdx].x = dist;		// Save Computed Distance to point
	out[currIdx].y = diff.w;	// Save Index of corresponding point

	__syncthreads();
}


/*---------------------------------------------------------
  Name:  Reduce_Min_GPU
  Desc:  Reduces each block to min answer
---------------------------------------------------------*/

__global__ void
Reduce_Min_GPU
(
	float2*  distOut,	// OUT:  Reduced Vector Field
	float2*  distIn		// IN:	 Distance Vector Field
)
{
	const int threadsPerBlock = blockDim.x * blockDim.y; 

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


#endif // #ifndef _DIST_GPU_H_
