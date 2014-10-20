/*-----------------------------------------------------------------------------
  Name:  GPU_QNN_LBT.cu
  Desc:  This file contains QNN kd-tree GPU kernels
		 for use with GPU kd-nodes stored in a left balanced layout

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

#ifndef _GPU_QNN_LBT_H_
#define _GPU_QNN_LBT_H_


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
  Name: GPU_QNN_2D_LBT
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Note: S is represented by a 
        static balanced cyclical KDTree
        with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_QNN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kd-tree Nearest Neighbor Algorithm
	         float2		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ float2           queryPoints[QNN_THREADS_PER_BLOCK];					// query point
	__shared__ GPUNode_2D_LBT   currNodes[QNN_THREADS_PER_BLOCK];					// current kd-node
	__shared__ GPU_Search       searchStack[QNN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

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
	int tx   = threadIdx.x;	// column in block
	int ty   = threadIdx.y;	// row in block
	int tidx = (ty*threadsPerRow) + tx;

	// Compute Query Index
	int currCol = (bx * threadsPerRow) + tx;
	int currRow = (by * rowsPerBlock) + ty;
	int qidx    = currRow * w + currCol;

	// Load current Query Point into local (fast) memory
		// Read from slow RAM memory into faster shared memory
	queryPoints[tidx] = qps[qidx];

	// Set Initial Guess equal to root node
	best.Id    = 1;
	best.Dist  = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
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

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1;
		rightIdx  = leftIdx + 1;
		
		nextAxis  = ((currAxis == 0) ? 1 : 0);

		// Early Exit Check
		if (currInOut == 1)	// Is 'Offside' Node ?
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
														//| ONSIDE_VALUE; 
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



/*---------------------------------------------------------
  Name: GPU_QNN_3D_LBT
  Desc: Finds the nearest neighbor in search set 'S'
        for each 3D query point in set 'Q'.
  Note: S is represented by a 
        static balanced cyclical KDTree
        with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_QNN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kd-tree Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute distance for (1D or 2D layout)
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ GPUNode_3D_LBT   currNodes[QNN_THREADS_PER_BLOCK];					// current kd-node
	__shared__ float            queryPoints[QNN_THREADS_PER_BLOCK][3];				// query point
	__shared__ GPU_Search		searchStack[QNN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int stackTop = 0u;
	unsigned int leftIdx, rightIdx;
	unsigned int currIdx, currInOut;
	unsigned int currAxis, nextAxis, prevAxis;
	float dx, dy, dz;
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	//float * queryVals;

	// Compute Thread index
	unsigned int tidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Compute Query Index (currRow * w + currCol)
	unsigned int qidx = ((blockIdx.y * blockDim.y) + threadIdx.y) * w + 
		                ((blockIdx.x * blockDim.x) + threadIdx.x);

	// Load current Query Point into local (fast) memory
		// Read from slow RAM memory into faster shared memory
	queryPoints[tidx][0] = qps[qidx].x;
	queryPoints[tidx][1] = qps[qidx].y;
	queryPoints[tidx][2] = qps[qidx].z;
	//queryVals = (float *)(&queryPoints[tidx]);

	// Set Initial Guess equal to root node
	best.Id    = 1u;
	best.Dist  = 3.0e+38f;					// Choose A huge Number to start with for Best Distance
	//best.cNodes = 0u;

	// Push root info onto search stack
	searchStack[stackTop][tidx].nodeFlags = FLAGS_ROOT_START; 
	searchStack[stackTop][tidx].splitVal  = 3.0e+38f;
	stackTop++;

	while (stackTop != 0u)
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

		// Get next and previous axis from current axis
		nextAxis  = ((currAxis == 2u) ? 0u : currAxis+1);
		prevAxis  = ((currAxis == 0u) ? 2u : currAxis-1);

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			//queryValue = queryVals[prevAxis];
			queryValue = queryPoints[tidx][prevAxis];
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
		//queryValue = queryVals[currAxis];
		queryValue = queryPoints[tidx][currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx][0];
		dy = currNodes[tidx].pos[1] - queryPoints[tidx][1];
		dz = currNodes[tidx].pos[2] - queryPoints[tidx][2];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

		// Update closest point Idx
			// Old way
		//if (diffDist2 < best.Dist)
		//{
		//  best.Id  = currIdx;
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
														//| ONSIDE_VALUE; 
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


/*---------------------------------------------------------
  Name: GPU_QNN_4D_LBT
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Note: S is represented by a 
        static balanced cyclical KDTree
        with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_QNN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kd-tree Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ float            queryPoints[QNN_THREADS_PER_BLOCK][4];				// query point
	__shared__ GPUNode_4D_LBT   currNodes[QNN_THREADS_PER_BLOCK];					// current kd-node
	__shared__ GPU_Search		searchStack[QNN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw;
	float queryValue, splitValue;
	unsigned int stackTop = 0;

	const int threadsPerRow  = blockDim.x;	// Columns (per block)
	const int rowsPerBlock   = blockDim.y;	// Rows (per block) 

	// Block index
	int bx = blockIdx.x;	// column in grid
	int by = blockIdx.y;	// row in grid

	// Thread index
	int tx   = threadIdx.x;	// column in block
	int ty   = threadIdx.y;	// row in block
	int tidx = (ty*threadsPerRow) + tx;

	// Compute Query Index
	int currCol = (bx * threadsPerRow) + tx;
	int currRow = (by * rowsPerBlock) + ty;
	int qidx    = currRow * w + currCol;

	// Load current Query Point into local (fast) memory
		// Read from slow RAM memory into faster shared memory
	queryPoints[tidx][0] = qps[qidx].x;
	queryPoints[tidx][1] = qps[qidx].y;
	queryPoints[tidx][2] = qps[qidx].z;
	queryPoints[tidx][3] = qps[qidx].w;

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

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1u;
		rightIdx  = leftIdx + 1u;
		
		nextAxis  = ((currAxis == 3u) ? 0u : (currAxis+1u));
		prevAxis  = ((currAxis == 0u) ? 3u : (currAxis-1u));

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			queryValue = queryPoints[tidx][prevAxis];
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
		queryValue = queryPoints[tidx][currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx][0];
		dy = currNodes[tidx].pos[1] - queryPoints[tidx][1];
		dz = currNodes[tidx].pos[2] - queryPoints[tidx][2];
		dw = currNodes[tidx].pos[3] - queryPoints[tidx][3];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

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
		// Write to slow memory (RAM)
	qrs[qidx] = best;
}


/*---------------------------------------------------------
  Name: GPU_QNN_6D_LBT
  Desc: Finds the nearest neighbor in search set 'S'
        for each query point in set 'Q'.
  Note: S is represented by a 
        static balanced cyclical KDTree
        with one search point stored per kd-tree node
---------------------------------------------------------*/

__global__ void
GPU_QNN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kd-tree Nearest Neighbor Algorithm
	GPU_Point6D			* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
)
{
	// Per thread Local Parameters (Shared Memory)
	__shared__ GPU_Point6D      queryPoints[QNN_THREADS_PER_BLOCK];					// query point
	__shared__ GPUNode_6D_LBT   currNodes[QNN_THREADS_PER_BLOCK];					// current kd-node
	__shared__ GPU_Search		searchStack[QNN_STACK_SIZE][QNN_THREADS_PER_BLOCK];	// search stack

	// Per thread Local Parameters (Registers)
	GPU_NN_Result best;
	unsigned int currIdx, currInOut;
	unsigned int leftIdx, rightIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw, ds, dt;
	float queryValue, splitValue;
	unsigned int stackTop = 0;

	const int threadsPerRow  = blockDim.x;	// Columns (per block)
	const int rowsPerBlock   = blockDim.y;	// Rows (per block) 

	// Block index
	int bx = blockIdx.x;	// column in grid
	int by = blockIdx.y;	// row in grid

	// Thread index
	int tx   = threadIdx.x;	// column in block
	int ty   = threadIdx.y;	// row in block
	int tidx = (ty*threadsPerRow) + tx;

	// Compute Query Index
	int currCol = (bx * threadsPerRow) + tx;
	int currRow = (by * rowsPerBlock) + ty;
	int qidx    = currRow * w + currCol;

	// Load current Query Point into local (fast) memory
		// Read from slow RAM memory into faster shared memory
	queryPoints[tidx] = qps[qidx];

	// Set Initial Guess equal to root node
	best.Id    = 1;
	best.Dist  = 3.0e+38F;					// Choose A huge Number to start with for Best Distance
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

		// Get left and right child indices from binary array layout
		leftIdx   = currIdx << 1u;
		rightIdx  = leftIdx + 1u;
		
		nextAxis  = ((currAxis == 6u) ? 0u : (currAxis+1u));
		prevAxis  = ((currAxis == 0u) ? 6u : (currAxis-1u));

		// Early Exit Check
		if (currInOut == 1)	// KD_OUT
		{
			queryValue = queryPoints[tidx].pos[prevAxis];
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
		queryValue = queryPoints[tidx].pos[currAxis];
		splitValue = currNodes[tidx].pos[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		dx = currNodes[tidx].pos[0] - queryPoints[tidx].pos[0];
		dy = currNodes[tidx].pos[1] - queryPoints[tidx].pos[1];
		dz = currNodes[tidx].pos[2] - queryPoints[tidx].pos[2];
		dw = currNodes[tidx].pos[3] - queryPoints[tidx].pos[3];
		ds = currNodes[tidx].pos[4] - queryPoints[tidx].pos[4];
		dt = currNodes[tidx].pos[5] - queryPoints[tidx].pos[5];
		diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw) + (ds*ds) + (dt*dt);

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
		// Write to slow memory (RAM)
	qrs[qidx] = best;
}


#endif // #ifndef _GPU_QNN_LBT_H_
