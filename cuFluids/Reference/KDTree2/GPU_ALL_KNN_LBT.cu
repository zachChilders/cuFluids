/*-----------------------------------------------------------------------------
  Name:  GPU_ALL_KNN_LBT.cu
  Desc:  This file contains the ALL-KNN kd-tree GPU kernel
         in left-balanced array order

  Log:  Created by Shawn D. Brown (2/01/10)
-----------------------------------------------------------------------------*/

#ifndef _GPU_ALL_KNN_LBT_H_
#define _GPU_ALL_KNN_LBT_H_


/*---------------------------------------------------------
  Includes
---------------------------------------------------------*/

#include <stdio.h>
//#include <float.h>
#include "GPUTREE_API.h"


/*---------------------------------------------------------
  Function Definitions
---------------------------------------------------------*/

/*---------------------------------------------------------
  Name: GPU_ALL_KNN_2D_LBT
  Desc: Finds the 'k' Nearest Neighbors in 
		a search set 'S' for each query point in set 'Q'
  Notes:
	1.  The search set S and query set Q are the same
		for the All-KNN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_KNN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
)
{
	// Per Thread Local Parameters (shared memory)
	__shared__ GPUNode_2D_LBT   currNodes[ALL_KNN_THREADS_PER_BLOCK];					// Current kd-tree node
	__shared__ GPU_Search		searchStack[ALL_KNN_STACK_SIZE][ALL_KNN_THREADS_PER_BLOCK];	// Search Stack
	__shared__ GPU_NN_Result	knnHeap[KD_KNN_SIZE][ALL_KNN_THREADS_PER_BLOCK];		// 'k' NN Closest Heap
	__shared__ GPUNode_2D_LBT	queryPoints[ALL_KNN_THREADS_PER_BLOCK];					// Query Point

	// Per Thread Local Parameters (registers)
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int prevAxis, currAxis, nextAxis;
	unsigned int stackTop, maxHeap, countHeap;
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	float diff, diff2, diffDist2;
	float dx, dy;
	float * queryVals;
	int tidx, width, currRow, currCol, qidx;

	// Compute Thread index
	tidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Compute Query Index
	width = gridDim.x * blockDim.x;
	currRow = (blockIdx.y * blockDim.y) + threadIdx.y;
	currCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	qidx = (currRow * width) + currCol;

	// Load current Query Point into local (fast) memory
		// Slow read from RAM into shared memory
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)&(queryPoints[tidx]);

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
	searchStack[stackTop][tidx].nodeFlags = FLAGS_ROOT_START; 
	searchStack[stackTop][tidx].splitVal  = 3.0e+38F;
	stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		//best.cNodes++;

		// Get Current Node from top of stack
		stackTop--;

		// Get Node Info (decompress 3 fields from 1)
		currIdx   = (searchStack[stackTop][tidx].nodeFlags & NODE_INDEX_MASK);
		currAxis  = (searchStack[stackTop][tidx].nodeFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
		currInOut = (searchStack[stackTop][tidx].nodeFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1;
		rightIdx  = leftIdx + 1;
		
		nextAxis  = ((currAxis == 1u) ? 0u : 1u);
		prevAxis  = ((currAxis == 0u) ? 1u : 0u);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			if (countHeap == maxHeap) // Is heap full yet ?!?
			{
				queryValue = queryVals[prevAxis];
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

		// WARNING - It's Much faster to load this node from global memory after the "Early Exit check" !!!

		// Load current node
			// Slow read from RAM into shared memory
		currNodes[tidx] = kdTree[currIdx];

		// Get Best Fit Dist for checking child ranges
		queryValue = queryVals[currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx].pos[0];
		dy = currNodes[tidx].pos[1] - queryPoints[tidx].pos[1];
		diffDist2 = (dx*dx) + (dy*dy);

		// See if we should add this point to the 'k' NN Heap
		if (diffDist2 <= 0.0f)
		{
			// Do nothing, The query point found itself in the kd-tree
			// We don't want to add ourselves as a NN.
		}
		else if (countHeap < maxHeap)
		{
			//-------------------------------
			//	< 'k' elements on heap
			//	Do Simple Array append
			//-------------------------------

			countHeap++;
			knnHeap[countHeap][tidx].Id  = currIdx;
			knnHeap[countHeap][tidx].Dist = diffDist2;

			// Do we need to convert the array into a max distance heap ?!?
			if (countHeap == maxHeap)
			{
				// Yes, turn array into a heap, takes O(k) time
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
						GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
						knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
						knnHeap[childHIdx][tidx]  = closeTemp;
						
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
				GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
				knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
				knnHeap[childHIdx][tidx]  = closeTemp;
				
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
				if (rightIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			if (leftIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK)
					                                    | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														// | ONSIDE_VALUE; 
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
				if (leftIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			if (rightIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														//| ONSIDE_VALUE; 
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}
		}
	}

	/*-----------------------
	  Output Results
	-----------------------*/

	unsigned int i, offset, outIdx;

	// Remap query node idx to query point idx
		// Slow read from RAM memory
	outIdx = ids[qidx];

	// We now have a heap of the 'k' nearest neighbors
	// Write heap elements to the results array row by row	
	for (i = 1; i <= countHeap; i++)
	{
		offset = (i-1) * nElems;

		// REMAP:  Convert Nearest Neighbor Info to final format
			// Slow read from RAM memory
		knnHeap[i][tidx].Id   = ids[knnHeap[i][tidx].Id];			// Really need ID's not indexs		
		knnHeap[i][tidx].Dist = sqrtf( knnHeap[i][tidx].Dist );		// Get True distance (not distance squared)

		// Store Result 
			// Slow write to RAM memory
		qrs[outIdx+offset] = knnHeap[i][tidx];
	}
}


/*---------------------------------------------------------
  Name: GPU_ALL_KNN_3D_LBT
  Desc: Finds the 'k' Nearest Neighbors in 
		a search set 'S' for each query point in set 'Q'
  Notes:
	1.  The search set S and query set Q are the same
		for the All-KNN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_KNN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
)
{
	// Per Thread Local Parameters (shared memory)
	__shared__ GPUNode_3D_LBT	    currNodes[ALL_KNN_THREADS_PER_BLOCK];					// Current kd-tree node
	__shared__ GPU_Search			searchStack[ALL_KNN_STACK_SIZE][ALL_KNN_THREADS_PER_BLOCK];	// Search Stack
	__shared__ GPU_NN_Result		knnHeap[KD_KNN_SIZE][ALL_KNN_THREADS_PER_BLOCK];		// 'k' NN Closest Heap
	__shared__ GPUNode_3D_LBT		queryPoints[ALL_KNN_THREADS_PER_BLOCK];					// Query Point

	// Per Thread Local Parameters (registers)
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	unsigned int stackTop, maxHeap, countHeap;
	float dx, dy, dz;
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	float * queryVals;
	int tidx, width, currRow, currCol, qidx;

	// Compute Thread index
	tidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Compute Query Index
	width = gridDim.x * blockDim.x;
	currRow = (blockIdx.y * blockDim.y) + threadIdx.y;
	currCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	qidx = (currRow * width) + currCol;

	// Load current Query Point into local (fast) memory
		// Slow read from RAM into shared memory
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)&(queryPoints[tidx]);

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
	searchStack[stackTop][tidx].nodeFlags = FLAGS_ROOT_START; 
	searchStack[stackTop][tidx].splitVal  = 3.0e+38F;
	stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		//best.cNodes++;

		// Get Current Node from top of stack
		stackTop--;

		// Get Node Info
		currIdx   = (searchStack[stackTop][tidx].nodeFlags & NODE_INDEX_MASK);
		currAxis  = (searchStack[stackTop][tidx].nodeFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
		currInOut = (searchStack[stackTop][tidx].nodeFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1u;
		rightIdx  = leftIdx + 1u;
		
		nextAxis  = ((currAxis == 2u) ? 0u : currAxis+1u);
		prevAxis  = ((currAxis == 0u) ? 2u : currAxis-1u);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			if (countHeap == maxHeap) // Is heap full yet ?!?
			{
				queryValue = queryVals[prevAxis];
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

		// WARNING - It's Much faster to load this node from global memory after the "Early Exit check" !!!

		// Load current node
			// Slow read from RAM into shared memory
		currNodes[tidx] = kdTree[currIdx];

		// Get Best Fit Dist for checking child ranges
		queryValue = queryVals[currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryVals[0];
		dy = currNodes[tidx].pos[1] - queryVals[1];
		dz = currNodes[tidx].pos[2] - queryVals[2];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

		// See if we should add this point to the 'k' NN Heap
		if (diffDist2 <= 0.0f)
		{
			// Do nothing, The query point found itself in the kd-tree
			// We don't want to add ourselves as a NN.
		}
		else if (countHeap < maxHeap)
		{
			//-------------------------------
			//	< 'k' elements on heap
			//	Do Simple Array append
			//-------------------------------

			countHeap++;
			knnHeap[countHeap][tidx].Id  = currIdx;
			knnHeap[countHeap][tidx].Dist = diffDist2;

			// Do we need to convert the array into a max distance heap ?!?
			if (countHeap == maxHeap)
			{
				// Yes, turn array into a heap, takes O(k) time
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
						GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
						knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
						knnHeap[childHIdx][tidx]  = closeTemp;
						
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
				GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
				knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
				knnHeap[childHIdx][tidx]  = closeTemp;
				
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
				if (rightIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			if (leftIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														//| ONSIDE_VALUE; 
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
				if (leftIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			if (rightIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														//| ONSIDE_VALUE; 
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}
		}
	}

	/*-----------------------
	  Output Results
	-----------------------*/

	unsigned int i, offset, outIdx;

	// Remap query node idx to query point idx
		// Slow read from RAM memory
	outIdx = ids[qidx];

	// We now have a heap of the 'k' nearest neighbors
	// Write heap elements to the results array row by row	
	for (i = 1; i <= countHeap; i++)
	{
		offset = (i-1) * nElems;

		// REMAP:  Convert Nearest Neighbor Info to final format
			// Slow read from RAM memory
		knnHeap[i][tidx].Id   = ids[knnHeap[i][tidx].Id];			// Really need ID's not indexs		
		knnHeap[i][tidx].Dist = sqrtf( knnHeap[i][tidx].Dist );		// Get True distance (not distance squared)

		// Store Result 
			// Slow write to RAM memory
		qrs[outIdx+offset] = knnHeap[i][tidx];
	}
}


/*---------------------------------------------------------
  Name: GPU_ALL_KNN_4D_LBT
  Desc: Finds the 'k' Nearest Neighbors in 
		a search set 'S' for each query point in set 'Q'
  Notes:
	1.  The search set S and query set Q are the same
		for the All-KNN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_KNN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
)
{
	// Per Thread Local Parameters (shared memory)
	__shared__ GPUNode_4D_LBT	    currNodes[ALL_KNN_THREADS_PER_BLOCK];					// Current kd-tree node
	__shared__ GPU_Search			searchStack[ALL_KNN_STACK_SIZE][ALL_KNN_THREADS_PER_BLOCK];	// Search Stack
	__shared__ GPU_NN_Result		knnHeap[KD_KNN_SIZE][ALL_KNN_THREADS_PER_BLOCK];		// 'k' NN Closest Heap
	__shared__ GPUNode_4D_LBT		queryPoints[ALL_KNN_THREADS_PER_BLOCK];					// Query Point

	// Per Thread Local Parameters (registers)
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	unsigned int stackTop, maxHeap, countHeap;
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	float dx, dy, dz, dw;
	float diff, diff2, diffDist2;
	int tidx, width, currRow, currCol, qidx;
	float * queryVals;

	// Compute Thread index
	tidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Compute Query Index
	width = gridDim.x * blockDim.x;
	currRow = (blockIdx.y * blockDim.y) + threadIdx.y;
	currCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	qidx = (currRow * width) + currCol;

	// Load current Query Point into local (fast) memory
		// Slow read from RAM into shared memory
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)&(queryPoints[tidx]);

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
	searchStack[stackTop][tidx].nodeFlags = FLAGS_ROOT_START; 
	searchStack[stackTop][tidx].splitVal  = 3.0e+38F;
	stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		//best.cNodes++;

		// Get Current Node from top of stack
		stackTop--;

		// Get Node Info
		currIdx   = (searchStack[stackTop][tidx].nodeFlags & NODE_INDEX_MASK);
		currAxis  = (searchStack[stackTop][tidx].nodeFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
		currInOut = (searchStack[stackTop][tidx].nodeFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1u;
		rightIdx  = leftIdx + 1u;
		
		nextAxis  = ((currAxis == 3u) ? 0u : currAxis+1u);
		prevAxis  = ((currAxis == 0u) ? 3u : currAxis-1u);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			if (countHeap == maxHeap) // Is heap full yet ?!?
			{
				queryValue = queryVals[prevAxis];
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

		// WARNING - It's Much faster to load this node from global memory after the "Early Exit check" !!!

		// Load current node
			// Slow read from RAM into shared memory
		currNodes[tidx] = kdTree[currIdx];

		// Get Best Fit Dist for checking child ranges
		queryValue = queryVals[currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryVals[0];
		dy = currNodes[tidx].pos[1] - queryVals[1];
		dz = currNodes[tidx].pos[2] - queryVals[2];
		dw = currNodes[tidx].pos[3] - queryVals[3];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

		// See if we should add this point to the 'k' NN Heap
		if (diffDist2 <= 0.0f)
		{
			// Do nothing, The query point found itself in the kd-tree
			// We don't want to add ourselves as a NN.
		}
		else if (countHeap < maxHeap)
		{
			//-------------------------------
			//	< 'k' elements on heap
			//	Do Simple Array append
			//-------------------------------

			countHeap++;
			knnHeap[countHeap][tidx].Id  = currIdx;
			knnHeap[countHeap][tidx].Dist = diffDist2;

			// Do we need to convert the array into a max distance heap ?!?
			if (countHeap == maxHeap)
			{
				// Yes, turn array into a heap, takes O(k) time
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
						GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
						knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
						knnHeap[childHIdx][tidx]  = closeTemp;
						
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
				GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
				knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
				knnHeap[childHIdx][tidx]  = closeTemp;
				
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
				if (rightIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			if (leftIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														// | ONSIDE_VALUE; 
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
				if (leftIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			if (rightIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														// | ONSIDE_VALUE; 
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}
		}
	}

	/*-----------------------
	  Output Results
	-----------------------*/

	unsigned int i, offset, outIdx;

	// Remap query node idx to query point idx
		// Slow read from RAM memory
	outIdx = ids[qidx];

	// We now have a heap of the 'k' nearest neighbors
	// Write them to the results array
	// Assume answers should be stored along z axis of 3 dimensional cube

	// We now have a heap of the 'k' nearest neighbors
	// Write heap elements to the results array row by row	
	for (i = 1; i <= countHeap; i++)
	{
		offset = (i-1) * nElems;

		// REMAP:  Convert Nearest Neighbor Info to final format
			// Slow read from RAM memory
		knnHeap[i][tidx].Id   = ids[knnHeap[i][tidx].Id];			// Really need ID's not indexs		
		knnHeap[i][tidx].Dist = sqrtf( knnHeap[i][tidx].Dist );		// Get True distance (not distance squared)

		// Store Result 
			// Slow write to RAM memory
		qrs[outIdx+offset] = knnHeap[i][tidx];
	}
}


/*---------------------------------------------------------
  Name: GPU_ALL_KNN_6D_LBT
  Desc: Finds the 'k' Nearest Neighbors in 
		a search set 'S' for each query point in set 'Q'
  Notes:
	1.  The search set S and query set Q are the same
		for the All-KNN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_KNN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
)
{
	// Per Thread Local Parameters (shared memory)
	__shared__ GPUNode_6D_LBT	    currNodes[ALL_KNN_THREADS_PER_BLOCK];					// Current kd-tree node
	__shared__ GPU_Search			searchStack[ALL_KNN_STACK_SIZE][ALL_KNN_THREADS_PER_BLOCK];	// Search Stack
	__shared__ GPU_NN_Result		knnHeap[KD_KNN_SIZE][ALL_KNN_THREADS_PER_BLOCK];		// 'k' NN Closest Heap
	__shared__ GPUNode_6D_LBT		queryPoints[ALL_KNN_THREADS_PER_BLOCK];					// Query Point

	// Per Thread Local Parameters (registers)
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	unsigned int stackTop, maxHeap, countHeap;
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	float dx, dy, dz, dw, ds, dt;
	float diff, diff2, diffDist2;
	int tidx, width, currRow, currCol, qidx;
	float * queryVals;

	// Compute Thread index
	tidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Compute Query Index
	width = gridDim.x * blockDim.x;
	currRow = (blockIdx.y * blockDim.y) + threadIdx.y;
	currCol = (blockIdx.x * blockDim.x) + threadIdx.x;
	qidx = (currRow * width) + currCol;

	// Load current Query Point into local (fast) memory
		// Slow read from RAM into shared memory
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)&(queryPoints[tidx]);

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
	searchStack[stackTop][tidx].nodeFlags = FLAGS_ROOT_START; 
	searchStack[stackTop][tidx].splitVal  = 3.0e+38F;
	stackTop++;

	while (stackTop != 0)
	{
		// Statistics
		//best.cNodes++;

		// Get Current Node from top of stack
		stackTop--;

		// Get Node Info
		currIdx   = (searchStack[stackTop][tidx].nodeFlags & NODE_INDEX_MASK);
		currAxis  = (searchStack[stackTop][tidx].nodeFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
		currInOut = (searchStack[stackTop][tidx].nodeFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1u;
		rightIdx  = leftIdx + 1u;
		
		nextAxis  = ((currAxis == 5u) ? 0u : currAxis+1u);
		prevAxis  = ((currAxis == 0u) ? 5u : currAxis-1u);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			if (countHeap == maxHeap) // Is heap full yet ?!?
			{
				queryValue = queryVals[prevAxis];
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

		// WARNING - It's Much faster to load this node from global memory after the "Early Exit check" !!!

		// Load current node
			// Slow read from RAM into shared memory
		currNodes[tidx] = kdTree[currIdx];

		// Get Best Fit Dist for checking child ranges
		queryValue = queryVals[currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryVals[0];
		dy = currNodes[tidx].pos[1] - queryVals[1];
		dz = currNodes[tidx].pos[2] - queryVals[2];
		dw = currNodes[tidx].pos[3] - queryVals[3];
		ds = currNodes[tidx].pos[4] - queryVals[4];
		dt = currNodes[tidx].pos[5] - queryVals[5];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw) + (ds*ds) + (dt*dt);

		// See if we should add this point to the 'k' NN Heap
		if (diffDist2 <= 0.0f)
		{
			// Do nothing, The query point found itself in the kd-tree
			// We don't want to add ourselves as a NN.
		}
		else if (countHeap < maxHeap)
		{
			//-------------------------------
			//	< 'k' elements on heap
			//	Do Simple Array append
			//-------------------------------

			countHeap++;
			knnHeap[countHeap][tidx].Id  = currIdx;
			knnHeap[countHeap][tidx].Dist = diffDist2;

			// Do we need to convert the array into a max distance heap ?!?
			if (countHeap == maxHeap)
			{
				// Yes, turn array into a heap, takes O(k) time
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
						GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
						knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
						knnHeap[childHIdx][tidx]  = closeTemp;
						
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
				GPU_NN_Result closeTemp   = knnHeap[parentHIdx][tidx];
				knnHeap[parentHIdx][tidx] = knnHeap[childHIdx][tidx];
				knnHeap[childHIdx][tidx]  = closeTemp;
				
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
				if (rightIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}

			// Always Add Left Sub-range to search path
			if (leftIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														// | ONSIDE_VALUE; 
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
				if (leftIdx <= cNodes)
				{
					// Push Onto top of stack
					searchStack[stackTop][tidx].nodeFlags = (leftIdx & NODE_INDEX_MASK) 
															| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK)
															| OFFSIDE_VALUE; 
					searchStack[stackTop][tidx].splitVal  = splitValue;
					stackTop++;
				}
			}
				
			// Always Add Right Sub-range
			if (rightIdx <= cNodes)
			{
				// Push Onto top of stack
				searchStack[stackTop][tidx].nodeFlags = (rightIdx & NODE_INDEX_MASK) 
														| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
														// | ONSIDE_VALUE; 
				searchStack[stackTop][tidx].splitVal  = splitValue;
				stackTop++;
			}
		}
	}

	/*-----------------------
	  Output Results
	-----------------------*/

	unsigned int i, offset, outIdx;

	// Remap query node idx to query point idx
		// Slow read from RAM memory
	outIdx = ids[qidx];

	// We now have a heap of the 'k' nearest neighbors
	// Write them to the results array
	// Assume answers should be stored along z axis of 3 dimensional cube

	// We now have a heap of the 'k' nearest neighbors
	// Write heap elements to the results array row by row	
	for (i = 1; i <= countHeap; i++)
	{
		offset = (i-1) * nElems;

		// REMAP:  Convert Nearest Neighbor Info to final format
			// Slow read from RAM memory
		knnHeap[i][tidx].Id   = ids[knnHeap[i][tidx].Id];			// Really need ID's not indexs		
		knnHeap[i][tidx].Dist = sqrtf( knnHeap[i][tidx].Dist );		// Get True distance (not distance squared)

		// Store Result 
			// Slow write to RAM memory
		qrs[outIdx+offset] = knnHeap[i][tidx];
	}
}

#endif // _GPU_ALL_KNN_LBT_H_
