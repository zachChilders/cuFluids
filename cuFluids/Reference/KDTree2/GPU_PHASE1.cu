/*-----------------------------------------------------------------------------
  Name:  GPU_PHASE1.cu
  Desc:  This file contains GPU kernels for building a kd-tree
		 The kd-nodes are stored in a left balanced layout
  Notes: 
    Kd-tree attributes
		static		-- we need to know all points "a priori" before building the kd-tree
		balanced	-- Tree has maximum height of O( log<2> n )
	    Left-Balanced tree array layout
	        -- The kd-nodes in the kd-tree are stored in a left-balanced tree layout
			-- Given n points, We allocate n+1 nodes
			-- The kd-node at index zero is ignored (wasted space)
			-- The Root kd-node is always found at index 1
			-- Given any node at position 'i'
				-- The parent node is found at 'i/2'
				-- The left child node is found at '2*i'
				-- The right child node is found at '2*i+1'
		d-Dimensionality  -- 2D, 3D, 4D, ...
		cyclical	-- we follow a cyclical pattern in switching between axes
		               at each level of the tree, 
							for 2D <x,y,x,y,x,y,...>
							for 3D <x,y,z,x,y,z,...>
							for 4D <x,y,z,w,x,y,z,w,...>
							for 6D <x,y,z,w,s,t,x,y,z,w,s,t,...>
							etc.
		Point Storage -- 1 search point is stored at each internal or leaf node
		Minimal -- I have eliminated as many fields as possible
		           from the final kd-node data structures.
				   The only remaining field is the stored search point

	    During the build process, we need some temporary extra fields for tracking.

  by Shawn Brown (shawndb@cs.unc.edu)
-----------------------------------------------------------------------------*/

/*---------------------------------------------------------
  Includes
---------------------------------------------------------*/

#include <stdio.h>
//#include <float.h>
#include "GPUTREE_API.h"


/*---------------------------------------------------------
  Function Definitions
---------------------------------------------------------*/

/*-------------------------------------------------------------------------
  Name:	GPU_AxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

__device__ inline float GPU_NODE_2D_MED_AxisValue
( 
	const GPUNode_2D_MED * currNodes,	// IN:  IN node list
	unsigned int index,					// IN:  Index of node to retrieve value for
	unsigned int axis					// IN:  axis of value to retrieve
)
{
	return currNodes[index].pos[axis];
}

__device__ inline float GPU_NODE_2D_LBT_AxisValue
( 
	const GPUNode_2D_LBT * currNodes,	// IN:  IN node list
	unsigned int index,					// IN:  Index of node to retrieve value for
	unsigned int axis					// IN:  axis of value to retrieve
)
{
	return currNodes[index].pos[axis];
}


/*-------------------------------------------------------------------------
  Name:	GPU_Swap
  Desc:	Helper Method
-------------------------------------------------------------------------*/

__device__ inline void GPU_2D_NODE_MED_Swap
( 
	GPUNode_2D_MED * currNodes,		// IN: Median node list
	unsigned int idx1,				// IN: Index of 1st node to swap
	unsigned int idx2				// IN: Index of 2nd node to swap
)
{
	GPUNode_2D_MED temp = currNodes[idx1];	// slow read
	currNodes[idx1]		= currNodes[idx2];	// slow read and write
	currNodes[idx2]		= temp;				// slow write
}

__device__ inline void GPU_2D_NODE_LBT_Swap
( 
	GPUNode_2D_LBT * currNodes,		// IN: left-balanced node list
	unsigned int idx1,				// IN: Index of 1st node to swap
	unsigned int idx2				// IN: Index of 2nd node to swap
)
{
	GPUNode_2D_LBT temp = currNodes[idx1];	// slow read
	currNodes[idx1]     = currNodes[idx2];	// slow read and write
	currNodes[idx2]     = temp;				// slow write
}


/*-------------------------------------------------------------------------
  Name:	GPU_MedianOf3
  Desc:	Helper method,
		Implements Median of three variant 
		for Median Partitioning algorithm
		returns pivot value for partitioning algorithm
  Note:	finds middle element of left, mid, and right
		where mid = (left+right)/2
  
  enforces invariant that
		array[left].val <= array[mid].val <= array[right].val
-------------------------------------------------------------------------*/

__device__ inline unsigned int GPU_2D_NODE_MED_MedianOf3
(
	GPUNode_2D_MED * currNodes,	// IN - node list
	unsigned int leftIdx,		// IN - left index
	unsigned int rightIdx,		// IN - right index
	unsigned int axis			// IN - axis to compare
)
{
	// Compute Middle Index from left and right
	unsigned int middleIdx = (leftIdx+rightIdx)/2;	
	unsigned int temp;

	float leftVal   = GPU_NODE_2D_MED_AxisValue( currNodes, leftIdx,   axis );
	float rightVal  = GPU_NODE_2D_MED_AxisValue( currNodes, rightIdx,  axis );
	float middleVal = GPU_NODE_2D_MED_AxisValue( currNodes, middleIdx, axis );

	// Sort left, center, mid values into correct order
	if (leftVal > middleVal)
	{
		// Swap left and middle indices
		temp      = leftIdx;
		leftIdx   = middleIdx;
		middleIdx = temp;
	}

	if (leftVal > rightVal)
	{
		// Swap left and right indices
		temp      = leftIdx;
		leftIdx   = rightIdx;
		rightIdx  = temp;
	}

	if (middleVal > rightVal)
	{
		// Swap middle and right indices
		temp      = middleIdx;
		middleIdx = rightIdx;
		rightIdx  = temp;
	}

	// return middle as the pivot
	return middleIdx;
}


/*---------------------------------------------------------
  Name: GPU_PICK_PIVOT
  Desc: Count # of nodes in range[start,end]
        That are before, after, or equal to pivot value
---------------------------------------------------------*/

__global__ void
GPU_2D_NODE_MED_PICK_PIVOT
(
	unsigned int * pivot,		// OUT - pivot result
	GPUNode_2D_MED * currNodes,	// IN - node list
	unsigned int start,			// IN - range [start,end] to median select
	unsigned int end,			
	unsigned int axis			// IN - axis to compare
)
{
	// Block thread index (local)
	const int bidx = (threadIdx.y*blockDim.x) + threadIdx.x;
	if (0 == bidx)
	{
		pivot[0] = GPU_2D_NODE_MED_MedianOf3( currNodes, start, end, axis );
	}
}


/*---------------------------------------------------------
  Name: GPU_COUNTS
  Desc: Count # of nodes in range[start,end]
        That are before, after, or equal to pivot value
---------------------------------------------------------*/

__global__ void
GPU_2D_NODE_MED_COUNTS
(
	GPU_COUNTS_STARTS * counts,	// OUT: counts
	GPUNode_2D_MED * srcNodes,	// IN:  nodes are read & copied from this list (source)
	GPUNode_2D_MED * dstNodes,  // OUT: nodes are copied into this list (dest = scratch)
	unsigned int * pivot,		// IN: pivot location
	unsigned int nNodes,		// IN: number of nodes
	unsigned int start,			// IN: start of range to count
	unsigned int end,			// IN: end of range to count
	unsigned int axis			// IN: axis of dimension to work with
)
{
	__shared__ GPUNode_2D_MED	  currNode[BUILD_THREADS_PER_BLOCK];	// Current thread starts
	__shared__ GPU_COUNTS_STARTS  currCount[BUILD_THREADS_PER_BLOCK];	// Current thread starts

	// Local Variables (registers)
	float pivotVal, currVal;
	unsigned int countBefore = 0;	// # of elements less than pivot value (x < n[p])
	unsigned int countAfter  = 0;	// # of elements greater than pivot value (x > n[p]
	unsigned int countEqual  = 0;
	unsigned int startRow, endRow, currRow, pivotIdx;
	unsigned int startIdx, currIdx, leftOver;

	// Read in pivot value
		// Slow read from global memory (coalesced ???)
	pivotIdx = pivot[0];
	pivotVal = srcNodes[pivotIdx].pos[axis];

	/*-----------------------
	  Compute Thread Column
    -----------------------*/

	// Block thread index (local)
	const int bidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Grid thread idx (global)
	const int width  = (gridDim.x * blockDim.x);	// width of a grid row
	//const int height = (gridDim.y * blockDim.y);	// height of a grid column
	const int cRow  = (blockIdx.y * blockDim.y) + threadIdx.y;	// thread row in grid of blocks
	const int cCol  = (blockIdx.x * blockDim.x) + threadIdx.x;	// thread column in grid of blocks
	const int gidx  = (cRow * width) + cCol;

	// Compute Start & End Rows
	startRow = (start-1) / (unsigned int)(width);	// startRow = floor( start/width )
	endRow   = (end - 1) / (unsigned int)(width);	// endRow = ceil( end/width )
	leftOver = (end - 1) - (endRow*(unsigned int)(width));
	endRow   = (leftOver > 0) ? endRow+1 : endRow;

	/*-----------------------
	  count elements
    -----------------------*/

	startIdx = startRow * (unsigned int)(width) + (unsigned int)cCol;
	currIdx  = startIdx;

	for (currRow = startRow; currRow <= endRow; currRow++)
	{
		if ((currIdx < start) || (currIdx > end))
		{
			// Do nothing, the current element is outside of the range [start,end]
		}
		else
		{
			// Get current value
				// Slow read from global memory (coalesced ???)
			currNode[bidx] = srcNodes[currIdx];

			// Count # of values before and after pivot
			currVal = currNode[bidx].pos[axis];
			if (currVal < pivotVal)
			{
				countBefore++;
			}
			else if (currVal > pivotVal)
			{
				countAfter++;
			}
			else
			{
				countEqual++;
			}

			// Write node to scratch buffer
				// Slow write to external memory (coalesced ???)
			dstNodes[currIdx] = currNode[bidx];
		}

		// Move to next row
		currIdx += width;
	}
	
	// Store counts (shared memory)
	currCount[bidx].before = countBefore;
	currCount[bidx].after  = countAfter;
	currCount[bidx].equal  = countEqual;

	// Store counts (global memory)
		// Slow write to global memory (coalesced ???)
	counts[gidx] = currCount[bidx];
}


/*---------------------------------------------------------
  Name: GPU_PARTITION_2D
  Desc:	Partitions original data set {O}=[start,end] 
		with 'n' elements into 3 datasets <{l}, {m}, {r}}
        That are before, after, or equal to pivot value.
---------------------------------------------------------*/

__global__ void
GPU_2D_NODE_MED_PARTITION
(
	GPU_COUNTS_STARTS * starts,	// OUT: starts
	GPUNode_2D_MED * srcNodes,	// IN/OUT: Nodes are read from this list (source = scratch)
	GPUNode_2D_MED * dstNodes,	// IN/OUT: Nodes are partitioned into this list (dest)
	unsigned int nNodes,		// IN: number of nodes
	unsigned int start,			// IN: start of range to partition
	unsigned int end,			// IN: end of range to partition
	unsigned int axis,			// IN: axis of dimension to work with
	unsigned int * pivot		// IN: pivot index
)
{
	// Local Parameters (shared memory)
	__shared__ GPUNode_2D_MED		currNode[BUILD_THREADS_PER_BLOCK];	// Current thread starts
	__shared__ GPU_COUNTS_STARTS	currStart[BUILD_THREADS_PER_BLOCK];	// Current thread starts

	// BUGBUG: need to write
	// Local Variables (registers)
	float pivotVal, currVal;
	unsigned int startBefore, startAfter, startEqual, pivotIdx;
	unsigned int startRow, endRow, currRow, leftOver, outIdx;
	unsigned int startIdx, currIdx;

	// Read in pivot value
		// Slow read from global memory (coalesced ???)
	pivotIdx = pivot[0];
	pivotVal = srcNodes[pivotIdx].pos[axis];

	/*-----------------------
	  Compute Thread Column
    -----------------------*/

	// Block thread index (local)
	const int bidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Grid thread idx (global)
	const int width  = (gridDim.x * blockDim.x);	// width of a row
	//const int height = (gridDim.y * blockDim.y);	// # max threads in a column
	//const int maxElems = (gW * width);				// # max threads in grid of blocks
	const int cRow = (blockIdx.y * blockDim.y) + threadIdx.y;	// thread row in grid of blocks
	const int cCol = (blockIdx.x * blockDim.x) + threadIdx.x;	// thread column in grid of blocks
	const int gidx = (cRow * width) + cCol;

	// Read in starts 
		// Slow read from global memory (coalesced ???)
	currStart[bidx] = starts[gidx];
	startBefore = currStart[bidx].before;		// starting location of {L} set
	startAfter  = currStart[bidx].after;		// starting location of {M} set
	startEqual  = currStart[bidx].equal;		// starting location of {R} set

	// Compute Start & End Rows
	startRow = (start-1) / (unsigned int)(width);	// startRow = floor( start/width )
	endRow   = (end - 1) / (unsigned int)(width);	// endRow = ceil( end/width )
	leftOver = (end - 1) - (endRow*(unsigned int)(width));
	endRow   = (leftOver > 0) ? endRow+1 : endRow;


	/*-----------------------
	  Partition elements
    -----------------------*/

	startIdx = startRow * width + cCol;
	currIdx  = startIdx;

	for (currRow = startRow; currRow <= endRow; currRow++)
	{
		if ((currIdx < start) || (currIdx > end))
		{
			// Do nothing, the current element is outside of range [start,end]
		}
		else
		{
			// Read node from original location
				// Slow read from global memory (coalesced ???)
			currNode[bidx] = srcNodes[currIdx];

			// Partition node into appropriate location
			currVal = currNode[bidx].pos[axis];
			if (currVal < pivotVal)
			{
				outIdx = startBefore;
				startBefore++;
			}
			else if (currVal > pivotVal)
			{
				outIdx = startAfter;
				startAfter++;
			}
			else
			{
				outIdx = startEqual;
				startEqual++;
			}

			//__syncthreads();


			// Write node to new partitioned location
				// Slow write to external memory
			dstNodes[outIdx] = currNode[bidx];

			//__syncthreads();
		}

		// Move to next row
		currIdx += width;
	}

	__syncthreads();
}


/*---------------------------------------------------------
  Name: GPU_STORE_2D
  Desc:	Store left balanced median node in LBT node list
---------------------------------------------------------*/

__global__ void
GPU_2D_NODE_STORE
(
	GPUNode_2D_MED * medNodes,	// IN:  Median Nodes are read from this array
	GPUNode_2D_LBT * lbtNodes,	// OUT: LBT Nodes are stored in this array
	unsigned int *  pointIDS,	// OUT: point indices are stored in this array
	unsigned int medianIdx,		// IN: left balanced median index
	unsigned int targetIdx		// IN: Target index
)
{
	// Local Parameters (shared memory)
	__shared__ GPUNode_2D_MED	med;
	__shared__ GPUNode_2D_LBT	lbt;

	// Store current median node 
	// in left balanced list at target

		// Slow read from main memory
	med = medNodes[medianIdx];

#ifdef _DEVICEEMU
	//fprintf( stdout, "Store, Median=%u, Target=%u, x=%g, y=%g, PIDX=%u\n",
	//		 medianIdx, targetIdx, med.pos[0], med.pos[1], med.m_searchIdx );	    
#endif

	lbt.pos[0] = med.pos[0];
	lbt.pos[1] = med.pos[1];

		// Slow write to main memory
	lbtNodes[targetIdx] = lbt;
	pointIDS[targetIdx] = med.m_searchIdx;
}


/*---------------------------------------------------------
  Name: GPU_COUNTS_TO_STARTS
  Desc: Converts counts to starts 
        using modified prefix sum (scan) algorithm

  Notes: Based on the prefix sum (scan) algorithm 
         found in the book... 

  GPU GEMS 3, Chapter 39, on pages 851-875
  by Mark Harris, Shubhabrata Sengupta, and John Owens
---------------------------------------------------------*/

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
	    (((n) >> NUM_BANKS) + ((n) >> (2*LOG_NUM_BANKS)))

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

__global__ void 
GPU_2D_COUNTS_TO_STARTS
( 
	GPU_COUNTS_STARTS * starts,	// OUT - start list (store prefix sums here)
	GPU_COUNTS_STARTS * counts,	// IN  - Count list (to total)
	unsigned int nCounts,		// IN  - # of items in count list
	unsigned int currStart,		// IN  - range[start,end]
	unsigned int currEnd		//       ditto
)
{
	// Local Memory (Shared Memory)
	__shared__ GPU_COUNTS_STARTS sCounts[BUILD_CS_SCAN_MAX_ITEMS];

	// Local Memory (Registers)
	unsigned int ai, bi, aidx, bidx;
	unsigned int d, before, after, equal;
	unsigned int offset, bankOffsetA, bankOffsetB;
	unsigned int tid, n, n2;
	unsigned int baseBefore, baseAfter, baseEqual;
	unsigned int totalBefore, totalEqual;

	tid    = threadIdx.x;	// thread ID
	n      = blockDim.x;	// # of threads
	n2     = n << 1;		// 2*n 
	offset = 1;

	// load input into shared memory
	//temp[2*tid]   = g_idata[2*tid];	
	//temp[2*tid+1] = g_idata[2*tid+1];
		// Rewriten to avoid bank conflicts

	ai = tid;
	bi = tid + n;
	bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	// Read 1st value into shared memory
	if (ai < nCounts)
	{
		sCounts[ai + bankOffsetA] = counts[ai];
	}
	else
	{
		// Initialize counts to zero (additive identity)
		sCounts[ai + bankOffsetA].before = 0;
		sCounts[ai + bankOffsetA].after  = 0;
		sCounts[ai + bankOffsetA].equal  = 0;
	}

	// Read 2nd value into shared memory
	if (bi < nCounts)
	{
		sCounts[bi + bankOffsetB] = counts[bi];
	}
	else
	{
		// Initialize counts to zero (additive identity)
		sCounts[bi + bankOffsetB].before = 0;
		sCounts[bi + bankOffsetB].after  = 0;
		sCounts[bi + bankOffsetB].equal  = 0;
	}

#ifdef _DEVICEEMU
	/*
	__syncthreads();
	if (tid == 0)
	{
		fprintf( stdout, "Before Reduction\n" );
		unsigned int idx;
		for (idx = 0; idx < n2; idx++)
		{
			before = sCounts[idx].before;
			after  = sCounts[idx].after;
			equal  = sCounts[idx].equal;
			fprintf( stdout, "Counts[%u] = <B=%u, A=%u, E=%u>\n",
				     idx, before, after, equal );
		}
		fprintf( stdout, "\n" );
	}
	__syncthreads();
	*/
#endif

	// Reduce to Total Sum (Up-sweep)
		// by traversing the conceptual binary tree 
		// in a bottom-up in-place manner
	//for (d = n2 >> 1; d > 0; d >>= 1)
	for (d = n; d > 0; d >>= 1)
	{
		__syncthreads();	// Note:  We need this here to make sure all threads
							//        have updated the current level of the 
							//        conceptual binary tree before we move to 
							//		  the next level

		if (tid < d)
		{
			unsigned int aidx, bidx;
			aidx  = offset*(2*tid+1)-1;
			bidx  = offset*(2*tid+2)-1;
			aidx += CONFLICT_FREE_OFFSET( aidx );
			bidx += CONFLICT_FREE_OFFSET( bidx );

			sCounts[bidx].before += sCounts[aidx].before;
			sCounts[bidx].after  += sCounts[aidx].after;
			sCounts[bidx].equal  += sCounts[aidx].equal;
		}

		offset <<= 1;	// offset = offset * 2;
	}


	//---------------------------------
	// Compute totals and base offsets
	//---------------------------------

#ifdef _DEVICEEMU
	/*
	__syncthreads();
	if (tid == 0)
	{
		fprintf( stdout, "After Reduction\n" );
		unsigned int idx;
		for (idx = 0; idx < n2; idx++)
		{
			before = sCounts[idx].before;
			after  = sCounts[idx].after;
			equal  = sCounts[idx].equal;
			fprintf( stdout, "Counts[%u] = <B=%u, A=%u, E=%u>\n",
				     idx, before, after, equal );
		}
		fprintf( stdout, "\n" );
	}
	__syncthreads();
	*/
#endif

	__syncthreads();	// Note:  We need this here to make sure we have the
	                    //        correct total counts available to all threads

	// Have each thread grab the final total counts to create their bases
	aidx = n2-1 + CONFLICT_FREE_OFFSET(n2-1);	
	totalBefore = sCounts[aidx].before;
	//totalAfter  = sCounts[aidx].after;
	totalEqual  = sCounts[aidx].equal;

	baseBefore  = currStart;
	baseEqual   = currStart + totalBefore;
	baseAfter   = currStart + totalBefore + totalEqual;

	__syncthreads();	// Note: We need this here to avoid setting last element 
	                    //       to all zeros before all threads have successfully 
						//       grabbed their correct total counts

	if (tid == 0)
	{
		// Clear the last element
		sCounts[aidx].before = 0;
		sCounts[aidx].after  = 0;
		sCounts[aidx].equal  = 0;
	}


	// Build Prefix-sum (Down-sweep) 
		// by traversing the conceptual binary tree 
		// in a top-down in-place manner
	for (d = 1; d < n2; d <<= 1)
	{
		offset >>= 1;		// offset = offset / 2;
		
		__syncthreads();	// Note:  

		if (tid < d)
		{
			aidx = offset*(2*tid+1)-1;
			bidx = offset*(2*tid+2)-1;
			aidx += CONFLICT_FREE_OFFSET( aidx );
			bidx += CONFLICT_FREE_OFFSET( bidx );

			// Add in prefix sum
			before = sCounts[aidx].before;
			after  = sCounts[aidx].after;
			equal  = sCounts[aidx].equal;

			sCounts[aidx].before = sCounts[bidx].before;
			sCounts[aidx].after  = sCounts[bidx].after;
			sCounts[aidx].equal  = sCounts[bidx].equal;

			sCounts[bidx].before += before;
			sCounts[bidx].after  += after;
			sCounts[bidx].equal  += equal;
		}
	}

	__syncthreads();

#ifdef _DEVICEEMU
	/*
	__syncthreads();
	if (tid == 0)
	{
		fprintf( stdout, "After Scan\n" );
		unsigned int idx;
		for (idx = 0; idx < n2; idx++)
		{
			before = sCounts[idx].before; // + baseBefore;
			after  = sCounts[idx].after;  // + baseAfter;
			equal  = sCounts[idx].equal;  // + baseEqual;
			fprintf( stdout, "Counts[%u] = <B=%u, A=%u, E=%u>\n",
				     idx, before, after, equal );
		}
		fprintf( stdout, "\n" );
	}
	__syncthreads();
	*/
#endif

	// Store Results to output
	//g_odata[2*tid]   = temp[2*tid];
	//g_odata[2*tid+1] = temp[2*tid+1];

	// Add in currStart to each thread
	sCounts[ai + bankOffsetA].before += baseBefore;
	sCounts[ai + bankOffsetA].after  += baseAfter;
	sCounts[ai + bankOffsetA].equal  += baseEqual;

	if (ai < nCounts)
	{
		// Store result
		starts[ai] = sCounts[ai + bankOffsetA];
	}

	// Add in currStart to each thread
	sCounts[bi + bankOffsetB].before += baseBefore;
	sCounts[bi + bankOffsetB].after  += baseAfter;
	sCounts[bi + bankOffsetB].equal  += baseEqual;

	if (bi < nCounts)
	{
		// Store result
		starts[bi] = sCounts[bi + bankOffsetB];
	}

#ifdef _DEVICEEMU
	/*
	__syncthreads();
	if (tid == 127)
	{
		// Dump Results
		fprintf( stdout, "After Scan\n" );
		unsigned int idx;
		for (idx = 0; idx < n2; idx++)
		{
			before = sCounts[idx].before; // + baseBefore;
			after  = sCounts[idx].after;  // + baseAfter;
			equal  = sCounts[idx].equal;  // + baseEqual;
			fprintf( stdout, "Counts[%u] = <B=%u, A=%u, E=%u>\n",
				     idx, before, after, equal );
		}
		fprintf( stdout, "\n" );

		// Now Do it the slow but correct way
		unsigned int totalBefore, totalAfter, totalEqual;
		totalBefore = 0;
		totalAfter  = 0;
		totalEqual  = 0;

		// Compute Totals
		for (idx = 0; idx < nCounts; idx++)
		{
			totalBefore += counts[idx].before;
			totalAfter  += counts[idx].after;
			totalEqual  += counts[idx].equal;
		}

		// Double check totals are correct
		unsigned int totalCount = totalBefore + totalAfter + totalEqual;
		unsigned int nRange = currEnd - currStart + 1;
		if (totalCount != nRange)
		{
			// Error - we have a bug
			fprintf( stdout, "Count Totals(%d) != Range Size(%d)\n", totalCount, nRange );
			//exit( 0 );
		}

		// Initialize bases for first thread
		baseBefore = currStart;
		baseEqual  = baseBefore + totalBefore;
		baseAfter  = baseEqual + totalEqual;

		unsigned int startBefore = baseBefore;
		unsigned int startEqual  = baseEqual;
		unsigned int startAfter  = baseAfter;

		// Compute starts from counts and bases
		for (idx = 0; idx < nCounts; idx++)
		{
			// Set starts for current thread
			starts[idx].before = startBefore;
			starts[idx].after  = startAfter;
			starts[idx].equal  = startEqual;

			// Update running starts for next thread
			startBefore += counts[idx].before;
			startAfter  += counts[idx].after;
			startEqual  += counts[idx].equal;
		}

		// Validate Fast vs. Slow Starts
		unsigned int checkBefore, checkAfter, checkEqual;

		fprintf( stdout, "Slow but correct starts\n" );
 		for (idx = 0; idx < nCounts; idx++)
		{
			// Get result from fast scan starts
			checkBefore = sCounts[idx].before;
			checkAfter  = sCounts[idx].after;
			checkEqual  = sCounts[idx].equal;

			// Get result from slow
			before = starts[idx].before;
			after  = starts[idx].after;
			equal  = starts[idx].equal;

			if ((checkBefore != before) ||
				(checkAfter != after) ||
				(checkEqual != equal))
			{
				fprintf( stdout, "Fast Starts[%u] = <B=%u, A=%u, E=%u>\n",
					     idx, checkBefore, checkAfter, checkEqual );
				fprintf( stdout, "Slow Starts[%u] = <B=%u, A=%u, E=%u>\n",
					     idx, before, after, equal );
			}
			else
			{
				fprintf( stdout, "Match[%u] = <B=%u, A=%u, E=%u>\n",
					     idx, before, after, equal );
			}
		}
		fprintf( stdout, "\n" );
	}
	__syncthreads();
	*/
#endif
}


