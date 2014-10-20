/*-----------------------------------------------------------------------------
  CS 790-058 GPGPU
  Final Project (Point Location using GPU)

  This file contains the GPU Kernels for KNN search using KDTrees

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

#ifndef _KNN_GPU_H_
#define _KNN_GPU_H_


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
  Name: KDTREE_KNN_V2
  Desc: Finds 'k' Nearest Neighbors in KDTree
  		for each query point
  Notes:  WORK IN PROGRESS (version)
			k can't exceed some small max number
			16 or 32 typically
---------------------------------------------------------*/

__global__ void
KDTREE_KNN_V2
(
	GPU_NN_Result	* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute k nearest neighbors for...
	GPUNode_2D_MED	* kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	         int          rootIdx,	// IN: index of root node in KD Tree
	    unsigned int      k			// IN: number of nearest neighbors to find
)
{
	// Local Parameters (shared memory)
	__shared__ GPUNode_2D_MED	currNodes[KD_THREADS_PER_BLOCK];
	__shared__ GPU_Search		searchStack[KD_STACK_SIZE][KD_THREADS_PER_BLOCK];	// Search Stack
	__shared__ GPU_NN_Result	knnHeap[KD_KNN_SIZE][KD_THREADS_PER_BLOCK];			// 'k' NN Heap
    __shared__ float4				queryPoints[KD_THREADS_PER_BLOCK];

	// Local Parameters (registers)
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	unsigned int stackTop, maxHeap, countHeap;
	float dx, dy, diff, diff2, diffDist2;
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	int tidx, width, currRow, currCol, qidx;

	// Compute Thread index
	tidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Compute Query Index
	width = gridDim.x * blockDim.x;
	currRow = (blockIdx.y * blockDim.y) + threadIdx.y;
	currCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	qidx = (currRow * width) + currCol;

	// Load current Query Point into local (fast) memory

	// BUGBUG - Had to copy componentwise to avoid kernel crash
	queryPoints[tidx].x = qps[qidx].x;
	queryPoints[tidx].y = qps[qidx].y;

	// Compute number of elements (in grid)
	int height = gridDim.y * blockDim.y;
	int nElems = height * width;

	// Search Stack Variables
	stackTop = 0;

	// 'k' NN Heap variables
	maxHeap   = k;			// Maximum # elements on knnHeap
	countHeap = 0;			// Current # elements on knnHeap
	dist2Heap = 0.0f;		// Max Dist of any element on heap
	bestDist2 = 3.0e38f;

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
			if (countHeap == maxHeap) // Is heap full yet ?!?
			{
				// Next Line is effectively queryValue = queryPoints[prevAxis];
				queryValue = ((currAxis == 0) ? queryPoints[tidx].y : queryPoints[tidx].x);
				splitValue = searchStack[stackTop][tidx].splitVal;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= dist2Heap)
				{
					// We can do an early exit for this node
					continue;
				}
			}
		}

		// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"

		// BUGBUG - Had to copy componentwise to avoid kernel crash
		currNodes[tidx].pos[0] = kdTree[currIdx].pos[0];
		currNodes[tidx].pos[1] = kdTree[currIdx].pos[1];
		currNodes[tidx].Left   = kdTree[currIdx].Left;
		currNodes[tidx].Right  = kdTree[currIdx].Right;

		// Get Best Fit Dist for checking child ranges
		queryValue = ((currAxis == 0) ? queryPoints[tidx].x : queryPoints[tidx].y);
		splitValue = ((currAxis == 0) ? currNodes[tidx].pos[0] : currNodes[tidx].pos[1]);
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx].x;
		dy = currNodes[tidx].pos[1] - queryPoints[tidx].y;
		diffDist2 = (dx*dx) + (dy*dy);

		// See if we should add this point to 'k' NN Heap
		if (countHeap < maxHeap)
		{
			//-------------------------------
			//	< 'k' elements on heap
			//	Do Simple Array Insertion
			//-------------------------------

			// Update Best Dist
			//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
			//bestDist2 = 3.0e38f;

			countHeap++;
			knnHeap[countHeap][tidx].Id  = currIdx;
			knnHeap[countHeap][tidx].Dist = diffDist2;

			// Do we need to Convert array into heap ?!?
			if (countHeap == maxHeap)
			{
				// Yes, turn array into a heap
				for (unsigned int z = countHeap/2; z >= 1; z--)
				{
					//
					// Demote each element in turn (to correct position in heap)
					//

					unsigned int parentHIdx = z;		// Start at specified element
					unsigned int childHIdx  = z << 1;	// left child of parent

					// Compare Parent to it's children
					while (childHIdx <= maxHeap)
					{
						// Update Distances
						float parentD2 = knnHeap[parentHIdx][tidx].Dist;
						float childD2  = knnHeap[childHIdx][tidx].Dist;

						// Find largest child 
						if (childHIdx < maxHeap)
						{
							float rightD2 = knnHeap[childHIdx+1][tidx].Dist;
							if (childD2 < rightD2)
							{
								// Use right child
								childHIdx++;	
								childD2 = rightD2;
							}
						}

						// Compare largest child to parent
						if (parentD2 >= childD2) 
						{
							// Parent is larger than both children, exit loop
							break;
						}

						// Demote parent by swapping with it's largest child
						GPU_NN_Result closeTemp = knnHeap[parentHIdx][tidx];
						knnHeap[parentHIdx][tidx]    = knnHeap[childHIdx][tidx];
						knnHeap[childHIdx][tidx]     = closeTemp;
						
						// Update indices
						parentHIdx = childHIdx;	
						childHIdx  = parentHIdx<<1;		// left child of parent
					}
				}

				// Update trim distances
				dist2Heap = knnHeap[1][tidx].Dist;
				bestDist2 = dist2Heap;
			}
		}
		else if (diffDist2 < dist2Heap)
		{
			//-------------------------------
			// >= k elements on heap
			// Do Heap Replacement
			//-------------------------------

			// Replace Root Element with new element
			knnHeap[1][tidx].Id  = currIdx;
			knnHeap[1][tidx].Dist = diffDist2;

			//
			// Demote new element (to correct position in heap)
			//
			unsigned int parentHIdx = 1;	// Start at Root
			unsigned int childHIdx  = 2;	// left child of parent

			// Compare current index to it's children
			while (childHIdx <= maxHeap)
			{
				// Update Distances
				float parentD2 = knnHeap[parentHIdx][tidx].Dist;
				float childD2  = knnHeap[childHIdx][tidx].Dist;

				// Find largest child 
				if (childHIdx < maxHeap)
				{
					float rightD2 = knnHeap[childHIdx+1][tidx].Dist;
					if (childD2 < rightD2)
					{
						// Use right child
						childHIdx++;	
						childD2 = rightD2;
					}
				}

				// Compare largest child to parent
				if (parentD2 >= childD2) 
				{
					// Parent node is larger than both children, exit
					break;
				}

				// Demote parent by swapping with it's largest child
				GPU_NN_Result closeTemp = knnHeap[parentHIdx][tidx];
				knnHeap[parentHIdx][tidx]    = knnHeap[childHIdx][tidx];
				knnHeap[childHIdx][tidx]     = closeTemp;
				
				// Update indices
				parentHIdx = childHIdx;	
				childHIdx  = parentHIdx<<1;		// left child of parent
			}

			// Update Trim distances
			dist2Heap = knnHeap[1][tidx].Dist;
			bestDist2 = dist2Heap;
		}

		// update bestDist2

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < bestDist2)
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
			if (diff2 < bestDist2)
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

	//
	//	Output Results
	//

	// We now have a heap of 'k' nearest neighbors
	// Write them to results array
	// Assume answers should be stored along z axis of 3 dimensional cube
	for (unsigned int i = 0; i < countHeap; i++)
	{
		unsigned int i1 = i+1;
		unsigned int offset = i * nElems;

		// Convert Nearest Neighbor Info to final format
		knnHeap[i1][tidx].Id  = ids[knnHeap[i1][tidx].Id];			// Really need ID's not indexs		
		knnHeap[i1][tidx].Dist = sqrtf( knnHeap[i1][tidx].Dist );		// Get True distance (not distance squared)

		// Store Result 

		// BUGBUG - Had to copy componentwise to avoid kernel crash
		qrs[qidx+offset].Id  = knnHeap[i1][tidx].Id;
		qrs[qidx+offset].Dist = knnHeap[i1][tidx].Dist;
	}

	// HACK HACK - Figure out strange bug !!!
	//qrs[qidx].Id  = 1;
	//qrs[qidx].Dist = 1.0f;
	//return;
}

#endif // #ifndef _KNN_GPU_H_
