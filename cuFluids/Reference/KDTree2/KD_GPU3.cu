/*-----------------------------------------------------------------------------
  CS 790-058 GPGPU
  Final Project (Point Location using GPU)

  This file contains the GPU Kernels

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

#ifndef _KD_GPU2_H_
#define _KD_GPU2_H_


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
  Name:	KDTREE_DIST_V3
  Desc:	Finds Nearest Neighbor in KDTree
		for each query point
  Notes:  Improved Version
		  Fewer memory accesses
		  and less stack space required
		  resulting in more threads being
		  able to run in parrallel
---------------------------------------------------------*/

__global__ void
KDTREE_DIST_V3
(
	GPU_NN_Result	* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_2D_MED	* kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	         int          rootIdx,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Local Parameters
	GPU_NN_Result best;
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	float dx, dy;
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int stackTop = 0;

	__shared__ float4 queryPoints[KD_THREADS_PER_BLOCK];
	__shared__ GPUNode_2D_MED currNodes[KD_THREADS_PER_BLOCK];
	__shared__ GPU_Search searchStack[KD_STACK_SIZE][KD_THREADS_PER_BLOCK];

	unsigned int haveInfo;

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

	// Set Initial Guess equal to root node
	best.Id    = rootIdx;
	best.Dist   = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0;


	// Store root info
	haveInfo   = 1;

	currIdx    = rootIdx;
	currAxis   = 0;
	currInOut  = 0;			// Outside
	splitValue = 3.0e+38f;	// Use huge value to simulate infinity
	nextAxis   = 1;

	// Load root node into local fast node
	//currNodes[tidx] = kdTree[currIdx]

	// No longer add to stack top

	// Put root search info on stack
	//searchStack[stackTop][tidx].nodeFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
	//searchStack[stackTop][tidx].splitVal  = 3.0e+38F;
	//stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		//best.cNodes++;

		if (haveInfo == 0)
		{
			// Get current Search Node from top of stack
			stackTop--;

			// Get Node Info
			currIdx    = (searchStack[stackTop][tidx].nodeFlags & 0x1FFFFFFFU);
			currAxis   = (searchStack[stackTop][tidx].nodeFlags & 0x60000000U) >> 29;
			currInOut  = (searchStack[stackTop][tidx].nodeFlags & 0x80000000U) >> 31;
			splitValue = searchStack[stackTop][tidx].splitVal;  // Get Split Value of Parent Node
			nextAxis  = ((currAxis == 0) ? 1 : 0);

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				// Next Line is effectively queryValue = queryPoints[prevAxis];
				queryValue = ((currAxis == 0) ? queryPoints[tidx].y : queryPoints[tidx].x);
				//splitValue = searchStack[stackTop][tidx].splitVal;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}

			}
		}
		// else
		//{
		//   Already have info from root or traversing onside node
		//}

		// WARNING - It's much faster to load this node from global memory after the "Early Exit check"
		// Load specified current Node from KD Tree
		currNodes[tidx] = kdTree[currIdx];

		// Get Best Fit Dist for checking child ranges
		queryValue = ((currAxis == 0) ? queryPoints[tidx].x : queryPoints[tidx].y);
		splitValue = ((currAxis == 0) ? currNodes[tidx].pos[0] : currNodes[tidx].pos[1]);
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx].x;
		dy = currNodes[tidx].pos[1] - queryPoints[tidx].y;
		diffDist2 = (dx*dx) + (dy*dy);

		// Update closest point Idx
		//if (diffDist2 < best.Dist)
		//{
		//  best.Id   = currIdx;
		//	best.Dist = diffDist2;
		//}
		best.Id  = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
		best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < best.Dist)
			{
				//nextIdx = currNodes[tidx].Right;
				if (0xFFFFFFFF != currNodes[tidx].Right)	// cInvalid
				{
					// Push offside search node onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			//nextIdx = currNodes[tidx].Left;
			haveInfo = 0;
			if (0xFFFFFFFF != currNodes[tidx].Left)
			{
				// Push onside search node onto top of stack
				//searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
				//searchStack[stackTop][tidx].splitVal  = splitValue;
				//stackTop++;

				// Don't push node onto search stack, just update search info directly
				currIdx    = currNodes[tidx].Left;
				currAxis   = nextAxis;
				currInOut  = 0;	// KD_IN
				//splitValue = splitValue;

				haveInfo = 1;
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
				//nextIdx = currNodes[tidx].Left;
				if (0xFFFFFFFFU != currNodes[tidx].Left)
				{
					// Push offside node onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			//nextIdx = currNodes[tidx].Right;
			if (0xFFFFFFFFU != currNodes[tidx].Right)
			{
				// Push onside node top of stack
				//searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
				//searchStack[stackTop][tidx].splitVal  = splitValue;
				//stackTop++;

				// Don't push node onto search stack, just update the search info directly
				currIdx    = currNodes[tidx].Right;
				currAxis   = nextAxis;
				currInOut  = 0;	// KD_IN
				//splitValue = splitValue;

				haveInfo = 1;
			}
		}
	}

	// We now have the Best Index but we really need the best ID so grab it from ID list 
	best.Id = ids[best.Id];

	// Turn Dist2 into true distance
	best.Dist = sqrt( best.Dist );

	// Store Result
	qrs[qidx] = best;
}

#endif // #ifndef _KD_GPU2_H_
