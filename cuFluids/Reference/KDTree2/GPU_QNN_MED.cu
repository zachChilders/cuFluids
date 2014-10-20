/*-----------------------------------------------------------------------------
  Name:  GPU_QNN_2D_MED.cu
  Desc:  This file contains the QNN kd-tree GPU kernel

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

#ifndef _GPU_QNN_2D_MED_H_
#define _GPU_QNN_2D_MED_H_


/*---------------------------------------------------------
  Includes
---------------------------------------------------------*/

#include <stdio.h>
//#include <float.h>
#include "GPUTree_API.h"
//#include "CPUTree_API.h"


/*---------------------------------------------------------
  Function Definitions
---------------------------------------------------------*/

/*---------------------------------------------------------
  Name: GPU_QNN_2D_MED
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Note: S is represented by a 
        static balanced cyclical KDTree
        with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_QNN_2D_MED
(
	GPU_NN_Result	* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float2		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_2D_MED	* kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	         int          rootIdx,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ float2			queryPoints[QNN_THREADS_PER_BLOCK];					// query point
	__shared__ GPUNode_2D_MED	currNodes[QNN_THREADS_PER_BLOCK];					// current node
	__shared__ GPU_Search		searchStack[QNN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	float dx, dy;
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int stackTop = 0;

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
		// Read from slow RAM memory into faster shared memory
	queryPoints[tidx] = qps[qidx];

	// Set Initial Guess equal to root node
	best.Id    = rootIdx;
	best.Dist   = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0;

	// Put root search info on stack
	searchStack[stackTop][tidx].nodeFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
	searchStack[stackTop][tidx].splitVal  = 3.0e+38F;
	stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		//best.cNodes++;

		// Get Current Node from top of stack
		stackTop--;

		// Get Node Info
		currIdx   = (searchStack[stackTop][tidx].nodeFlags & 0x1FFFFFFFU);
		currAxis  = (searchStack[stackTop][tidx].nodeFlags & 0x60000000U) >> 29;
		currInOut = (searchStack[stackTop][tidx].nodeFlags & 0x80000000U) >> 31;
		
		nextAxis  = ((currAxis == 0) ? 1 : 0);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
				// Next Line is effectively queryValue = queryPoints[prevAxis];
			queryValue = ((currAxis == 0) ? queryPoints[tidx].y : queryPoints[tidx].x);
			splitValue = searchStack[stackTop][tidx].splitVal;	// Split Value of Parent Node
			diff  = splitValue - queryValue;
			diff2 = diff*diff;
			if (diff2 >= best.Dist)
			{
				// We can do an early exit for this node
				continue;
			}
		}

		// WARNING - This code is where it is because it is much faster this way...
		// IE, if the trim check throws this node away, why bother to read the node from memory ???

		// Load specified Current Node from KD Tree 
			// Read from slow RAM memory into faster shared memory
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
			// Old way
		//if (diffDist2 < best.Dist)
		//{
		//  best.Id   = currIdx;
		//  best.Dist = diffDist2;
		//}

			// New Way
			// Seems to be faster to do update this way instead
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
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			//nextIdx = currNodes[tidx].Left;
			if (0xFFFFFFFF != currNodes[tidx].Left)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
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
				//nextIdx = currNodes[tidx].Left;
				if (0xFFFFFFFFU != currNodes[tidx].Left)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			//nextIdx = currNodes[tidx].Right;
			if (0xFFFFFFFFU != currNodes[tidx].Right)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (currNodes[tidx].Right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}
		}
	}

	// We now have the Best Index but we really need the best ID so grab it from the ID list 
		// Read from slow memory (RAM)
	best.Id = ids[best.Id];

	// Turn Dist2 into a true distance measure
	best.Dist = sqrt( best.Dist );

	// Store Result
		// Write to slow memory (RAM)
	qrs[qidx] = best;
}

#endif // #ifndef _GPU_QNN_2D_MED_H_
