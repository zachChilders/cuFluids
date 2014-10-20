/*-----------------------------------------------------------------------------
  Name:  GPU_ALL_NN_LBT.cu
  Desc:  This file contains ALL_NN kd-tree GPU kernels
		 for use with GPU kd-nodes stored in a left balanced layout

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

#ifndef _GPU_ALL_NN_LBT_H_
#define _GPU_ALL_NN_LBT_H_


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
  Name: GPU_ALL_NN_2D
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Notes:
	1.  The search set S and query set Q are the same
		for the All-NN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_NN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ GPUNode_2D_LBT	queryPoints[QNN_THREADS_PER_BLOCK];						// query point
	__shared__ GPUNode_2D_LBT	currNodes[QNN_THREADS_PER_BLOCK];						// current node
	__shared__ GPU_Search		searchStack[ALL_NN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	unsigned int leftIdx, rightIdx;
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
	queryPoints[tidx] = kdTree[qidx];

	// Set Initial Guess equal to root node
	best.Id   = 1;
	best.Dist = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0;

	// Push root info onto search stack
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

		leftIdx   = currIdx << 1;
		rightIdx  = leftIdx + 1;
		
		nextAxis  = ((currAxis == 0) ? 1 : 0);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
				// Next Line is effectively queryValue = queryPoints[prevAxis];
			queryValue = ((currAxis == 0) ? queryPoints[tidx].pos[1] : queryPoints[tidx].pos[0]);
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
		queryValue = ((currAxis == 0) ? queryPoints[tidx].pos[0] : queryPoints[tidx].pos[1]);
		splitValue = ((currAxis == 0) ? currNodes[tidx].pos[0] : currNodes[tidx].pos[1]);
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx].pos[0];
		dy = currNodes[tidx].pos[1] - queryPoints[tidx].pos[1];
		diffDist2 = (dx*dx) + (dy*dy);

		// Update closest point Idx
			// Note: We need to exclude zero distance results
			//       otherwise each point will return itself
			//       as it's own nearest neighbor
		if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
		{
		  best.Id  = currIdx;
		  best.Dist = diffDist2;
		}

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < best.Dist)
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
			if (diff2 < best.Dist)
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
			//nextIdx = currNodes[tidx].Right;
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

	// We now have the Best Index but we really need the best ID so grab it from the ID list 
		// Read from slow memory (RAM)
	best.Id = ids[best.Id];

	// Turn Dist2 into a true distance measure
	best.Dist = sqrt( best.Dist );

	// Store Result
		// Remap query node idx to query point idx
	unsigned int outIdx = ids[qidx];
		// Write result to slow memory (RAM)
	qrs[outIdx] = best;
}


/*---------------------------------------------------------
  Name: GPU_ALL_NN_3D
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Notes:
	1.  The search set S and query set Q are the same
		for the All-NN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_NN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ GPUNode_3D_LBT	queryPoints[QNN_THREADS_PER_BLOCK];					// query point
	__shared__ GPUNode_3D_LBT	currNodes[QNN_THREADS_PER_BLOCK];					// current node
	__shared__ GPU_Search		searchStack[ALL_NN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	float dx, dy, dz;
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	float * queryVals;
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
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)(&queryPoints[tidx]);

	// Set Initial Guess equal to root node
	best.Id    = 1;
	best.Dist   = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0;

	// Push root info onto search stack
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

		leftIdx   = currIdx << 1;
		rightIdx  = leftIdx + 1;
		
		nextAxis  = ((currAxis == 2u) ? 0u : currAxis+1);
		prevAxis  = ((currAxis == 0u) ? 2u : currAxis-1);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			queryValue = queryVals[prevAxis];
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
		queryValue = queryVals[currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryVals[0];
		dy = currNodes[tidx].pos[1] - queryVals[1];
		dz = currNodes[tidx].pos[2] - queryVals[2];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

		// Update closest point Idx
			// Note: We need to exclude zero distance results
			//       otherwise each point will return itself
			//       as it's own nearest neighbor
		if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
		{
		    best.Id   = currIdx;
		    best.Dist = diffDist2;
		}

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < best.Dist)
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
			if (diff2 < best.Dist)
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
			//nextIdx = currNodes[tidx].Right;
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

	// We now have the Best Index but we really need the best ID so grab it from the ID list 
		// Read from slow memory (RAM)
	best.Id = ids[best.Id];

	// Turn Dist2 into a true distance measure
	best.Dist = sqrt( best.Dist );

	// Store Result
		// Remap query node idx to query point idx
	unsigned int outIdx = ids[qidx];
		// Write result to slow memory (RAM)
	qrs[outIdx] = best;
}


/*---------------------------------------------------------
  Name: GPU_ALL_NN_4D
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Notes:
	1.  The search set S and query set Q are the same
		for the All-NN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_NN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ GPUNode_4D_LBT	queryPoints[QNN_THREADS_PER_BLOCK];					// query point
	__shared__ GPUNode_4D_LBT	currNodes[QNN_THREADS_PER_BLOCK];					// current node
	__shared__ GPU_Search		searchStack[ALL_NN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw;
	float queryValue, splitValue;
	float * queryVals;
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
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)(&queryPoints[tidx]);

	// Set Initial Guess equal to root node
	best.Id    = 1;
	best.Dist   = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0;

	// Push root info onto search stack
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

		leftIdx   = currIdx << 1;
		rightIdx  = leftIdx + 1;
		
		nextAxis  = ((currAxis == 3u) ? 0u : currAxis+1);
		prevAxis  = ((currAxis == 0u) ? 3u : currAxis-1);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			queryValue = queryVals[prevAxis];
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

		// Update closest point Idx
			// Note: We need to exclude zero distance results
			//       otherwise each point will return itself
			//       as it's own nearest neighbor
		if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
		{
		    best.Id  = currIdx;
		    best.Dist = diffDist2;
		}

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < best.Dist)
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
			if (diff2 < best.Dist)
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
			//nextIdx = currNodes[tidx].Right;
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

	// We now have the Best Index but we really need the best ID so grab it from the ID list 
		// Read from slow memory (RAM)
	best.Id = ids[best.Id];

	// Turn Dist2 into a true distance measure
	best.Dist = sqrt( best.Dist );

	// Store Result
		// Remap query node idx to query point idx
	unsigned int outIdx = ids[qidx];
		// Write result to slow memory (RAM)
	qrs[outIdx] = best;
}


/*---------------------------------------------------------
  Name: GPU_ALL_NN_4D
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Notes:
	1.  The search set S and query set Q are the same
		for the All-NN search.
	2.  We need to exclude zero distance results
		Otherwise, each point will return itself as
		its own nearest neighbor
	3.  The search set S is represented by a 
		static balanced cyclical KDTree
		with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_ALL_NN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ GPUNode_6D_LBT	queryPoints[QNN_THREADS_PER_BLOCK];					// query point
	__shared__ GPUNode_6D_LBT	currNodes[QNN_THREADS_PER_BLOCK];					// current node
	__shared__ GPU_Search		searchStack[ALL_NN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw, ds, dt;
	float queryValue, splitValue;
	float * queryVals;
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
	queryPoints[tidx] = kdTree[qidx];
	queryVals = (float *)(&queryPoints[tidx]);

	// Set Initial Guess equal to root node
	best.Id    = 1;
	best.Dist   = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0;

	// Push root info onto search stack
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

		leftIdx   = currIdx << 1;
		rightIdx  = leftIdx + 1;
		
		nextAxis  = ((currAxis == 5u) ? 0u : currAxis+1);
		prevAxis  = ((currAxis == 0u) ? 5u : currAxis-1);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			queryValue = queryVals[prevAxis];
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

		// Update closest point Idx
			// Note: We need to exclude zero distance results
			//       otherwise each point will return itself
			//       as it's own nearest neighbor
		if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
		{
		    best.Id  = currIdx;
		    best.Dist = diffDist2;
		}

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < best.Dist)
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
			if (diff2 < best.Dist)
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
			//nextIdx = currNodes[tidx].Right;
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

	// We now have the Best Index but we really need the best ID so grab it from the ID list 
		// Read from slow memory (RAM)
	best.Id = ids[best.Id];

	// Turn Dist2 into a true distance measure
	best.Dist = sqrt( best.Dist );

	// Store Result
		// Remap query node idx to query point idx
	unsigned int outIdx = ids[qidx];
		// Write result to slow memory (RAM)
	qrs[outIdx] = best;
}


#endif // #ifndef _GPU_ALL_NN_LBT_H_
