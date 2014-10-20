/*-----------------------------------------------------------------------------
  CS 790-058 GPGPU
  Final Project (Point Location using GPU)

  This file contains the GPU Kernels

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

#ifndef _KD_GPU1_H_
#define _KD_GPU1_H_


/*---------------------------------------------------------
  Includes
---------------------------------------------------------*/

#include <stdio.h>
//#include <float.h>
#include "KDTree_GPU.h"
#include "KD_API.h"


/*---------------------------------------------------------
  Function Definitions
---------------------------------------------------------*/

/*---------------------------------------------------------
  Name: KDTREE_DIST_V1
  Desc: Finds Nearest Neighbor in KDTree
        for each query point
  Notes:  WORK IN PROGRESS (version)
---------------------------------------------------------*/

__global__ void KDTREE_DIST_V1
(
	QueryResult_GPU	  * qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float4   * qps,		// IN: query points to compute distance for (1D or 2D field)
	KDTreeNode_GPU_V1 * kdTree,		// IN: KD Tree (Nodes)
	         int        rootIdx,	// IN: index of root node in KD Tree
		 	 int        w,			// IN:  width of 2D query field (# of columns)
			 int        h			// IN:  height of 2D query field (# of rows)
)
{
	//static unsigned int c_Prev2D[3] = { 1, 0 };	// Previous Indices
	//static unsigned int c_Next2D[3] = { 1, 0 };	// Next Indices

	// Local Parameters
	QueryResult_GPU best;
	unsigned int currIdx, currAxis, currInOut, nextIdx, nextAxis, prevAxis;
	float q[2];
	float c[2];
	float d[2];
	float diff, diff2;
	float diffDist2;	//, bestDist2;
	float queryValue, splitValue;
	//KDSearch_GPU_Alt currSearch;
	unsigned int currFlags;
	//float currSplit;
	unsigned int stackTop = 0;

    __shared__ float4 queryPoints[KD_THREADS_PER_BLOCK];
	__shared__ KDTreeNode_GPU_V1 currNodes[KD_THREADS_PER_BLOCK];
	__shared__ KDSearch_GPU_V1 searchStack[KD_STACK_SIZE][KD_THREADS_PER_BLOCK];

	const int threadsPerRow  = blockDim.x;	// Columns (per block)
	const int rowsPerBlock   = blockDim.y;	// Rows (per block) 

	// Block index
    int bx = blockIdx.x;	// column in grid
	int by = blockIdx.y;	// row in grid

    // Thread index
    int tx = threadIdx.x;	// column in block
	int ty = threadIdx.y;	// row in block
	int tidx = (ty*threadsPerRow) + tx;

	// Compute Query Index
	int currCol = (bx * threadsPerRow) + tx;
	int currRow = (by * rowsPerBlock) + ty;
	int qidx   = currRow * w + currCol;

	// Load current Query Point into local (fast) memory
	queryPoints[tidx] = qps[qidx];

	// Get Root Node from KD Tree
	currIdx = rootIdx;
	currNodes[tidx] = kdTree[currIdx];

	// Query Position
	q[0] = queryPoints[tidx].x;
	q[1] = queryPoints[tidx].y;
	//q[2] = queryLocation.z;

	// Get Root Info
	c[0] = currNodes[tidx].pos[0];
	c[1] = currNodes[tidx].pos[1];
	//c[2] = currNodes[tidx].pos[2];
	currAxis = currNodes[tidx].SplitAxis;
	currInOut = 0;							// KD_IN
	splitValue = c[currAxis];

	// Setup Search Node (for root node)
	currFlags = (currIdx & 0x1FFFFFFF) | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);

	// Set Initial Guess equal to root node
	best.Id    = currIdx;
	best.ID     = currNodes[tidx].ID;
	best.cNodes = 0;

	d[0] = c[0] - q[0];
	d[1] = c[1] - q[1];
	//d[2] = c[2] - q[2];
	best.Dist = (d[0]*d[0]) + (d[1]*d[1]); // +d[2]*d[2];

	searchStack[stackTop][tidx].nodeFlags = currFlags;
	searchStack[stackTop][tidx].splitVal  = splitValue;
	stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		best.cNodes++;

		// Get Current Node from top of stack
		stackTop--;
		currFlags = searchStack[stackTop][tidx].nodeFlags;

		// Get Node Info
		currIdx   = (currFlags & 0x1FFFFFFFU);
		currAxis  = (currFlags & 0x60000000U) >> 29;
		currInOut = (currFlags & 0x80000000U) >> 31;
		
		nextAxis  = ((currAxis == 0) ? 1 : 0);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			prevAxis   = ((currAxis == 0) ? 1 : 0);
			queryValue = q[prevAxis];	
			splitValue = searchStack[stackTop][tidx].splitVal;	// Split Value of Parent Node
			diff  = splitValue - queryValue;		
			diff2 = diff*diff;
			if (diff2 >= best.Dist)
			{
				// We can do an early exit for this node
				continue;
			}
		}

		//
		// BUGBUG - should this go before or after early exit check ?!?...
		//
		// Load specified Current Node from KD Tree
		currNodes[tidx] = kdTree[currIdx];

		// Get Node position
		c[0] = currNodes[tidx].pos[0];
		c[1] = currNodes[tidx].pos[1];
		//c[2] = currNodes[tidx].pos[2];

		// Get Best Fit Dist for checking child ranges
		queryValue = q[currAxis];
		splitValue = c[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		d[0] = c[0] - q[0];
		d[1] = c[1] - q[1];
		//d[2] = c[2] - q[2];
		diffDist2 = (d[0]*d[0]) + d[1]*d[1]; // + d[2]*d[2];

		// Update closest point Idx
		if (diffDist2 < best.Dist)
		{
			best.Id   = currIdx;
			best.Dist = diffDist2;
			best.ID    = currNodes[tidx].ID;
		}

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < best.Dist)
			{
				nextIdx = currNodes[tidx].Right;
				if (0xFFFFFFFF != nextIdx)	// cInvalid
				{
					// Push Onto top of stack
					//currFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | ((1U << 31) & 0x80000000U);
					searchStack[stackTop][tidx].nodeFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | ((1U << 31) & 0x80000000U);
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			nextIdx = currNodes[tidx].Left;
			if (0xFFFFFFFF != nextIdx)
			{
				// Push Onto top of stack
				//currFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | ((0U << 31) & 0x80000000U);
				searchStack[stackTop][tidx].nodeFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | ((0U << 31) & 0x80000000U);
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}

		}
		else
		{
			// SV...[BD...QL...]		-> Include Right sub range only
			//		  or
			// [BD...SV...QL...]		-> Include Both Left and Right Sub Ranges

			// Check if we should add left sub-range to search path
			if (diff2 < best.Dist)
			{
				// Add to search stack
				nextIdx = currNodes[tidx].Left;
				if (0xFFFFFFFFU != nextIdx)
				{
					// Push Onto top of stack
					//currFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | ((1U << 31) & 0x80000000U);
					searchStack[stackTop][tidx].nodeFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | ((1U << 31) & 0x80000000U);
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			nextIdx = currNodes[tidx].Right;
			if (0xFFFFFFFFU != nextIdx)
			{
				// Push Onto top of stack
				//currFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | ((0U << 31) & 0x8000000U);
				searchStack[stackTop][tidx].nodeFlags = (nextIdx & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | ((0U << 31) & 0x8000000U);
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}
		}
	}

	// Turn Dist2 into true distance
	best.Dist = sqrt( best.Dist );

	// Store Result
	qrs[qidx] = best;
}

#endif // #ifndef _KD_GPU1_H_
