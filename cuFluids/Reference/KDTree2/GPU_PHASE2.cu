/*-----------------------------------------------------------------------------
  Name:  GPU_PHASE2.cu
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

// Lookup "Median of 3" pivot index from 3 tests
__constant__ unsigned int g_m3Table[8] = 
{
			// M > R | L > R | L > M    Example:
			//-------|-------|-------
	1u,		//   0       0       0      <1 2 3>, <1 2 2>, or <2 2 2> => Median @ 2nd elem  
    0u,		//   0       0       1      <2 1 3> or <2 1 2>  => Median @ 1st elem
	0u,	    //   0       1       0      Invalid situation
	2u,		//   0       1       1      <3 1 2>  => Median @ 3rd elem
	2u,		//   1       0       0      <1 3 2>  => Median @ 3rd elem
	0u,	    //   1       0       1      Invalid situation
	0u,		//   1       1       0      <2 3 1> or <2 2 1> => Median @ 1st elem
	1u		//   1       1       1      <3 2 1>  => Median @ 2nd elem	
};


/*---------------------------------------------------------
  Name: GPU_BUILD_PHASE2
  Desc: Build sub-tree (sub-range) of kd-tree 
		starting from specified per thread 'build item'
---------------------------------------------------------*/

__global__ void
P2_2D_BUILD_LBT
(
	GPUNode_2D_LBT * lbtNodes,	// OUT: lbt node list
	unsigned int   * pointIDs,	// OUT: point indices are stored in this array
	GPUNode_2D_MED * medNodes,	// IN: median node list
	GPUNode_2D_MED * medScratch,// IN: scratch space for temporary copying
	GPU_BUILD_ITEM * buildQ,	// IN: build queue (per thread)
	unsigned int     nPoints	// IN: maximum # of points
)
{
	// Local variables (shared items)
	__shared__ GPU_BUILD_ITEM	currBuild[P2_BUILD_THREADS_PER_BLOCK][P2_BUILD_STACK_DEPTH];
	__shared__ GPUNode_2D_MED	currMED[P2_BUILD_THREADS_PER_BLOCK];
	__shared__ GPUNode_2D_LBT	currLBT[P2_BUILD_THREADS_PER_BLOCK];
	__shared__ float			m3Vals[3];

	// Local Variables (registers)
	float pivotVal, currVal;
	unsigned int currAxis, nextAxis;
	unsigned int h2, currMedian, validRoot;

	unsigned int currLeft, currRight;
	unsigned int currStart, currEnd, currItem;

	unsigned int currTarget, currFlags, outIdx, top;
	unsigned int origStart, origEnd, origN, bDone;

	unsigned int countBefore, countAfter, countEqual;
	unsigned int startBefore, startAfter, startEqual;
	unsigned int baseBefore, baseAfter, baseEqual;

	/*-----------------------
	  Compute Thread Column
    -----------------------*/

	// Block thread index (local)
	const int bidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Grid thread index (global)
	const int cRow = (blockIdx.y * blockDim.y) + threadIdx.y;	// thread row in grid of blocks
	const int cCol = (blockIdx.x * blockDim.x) + threadIdx.x;	// thread column in grid of blocks
	const int gidx = (cRow * (gridDim.x * blockDim.x)) + cCol;

	//----------------------------------
	// Push first build item onto stack
	//----------------------------------

	{
		unsigned int rootStart, rootEnd, rootFlags, rootAxis;
		unsigned int rootN, h;
	
		// Get root of sub-tree to process
		top = 0;
		currBuild[bidx][top] = buildQ[gidx];

		rootStart = currBuild[bidx][top].start & NODE_INDEX_MASK;
		rootEnd   = currBuild[bidx][top].end & NODE_INDEX_MASK;
		rootFlags = currBuild[bidx][top].flags;
		rootAxis  = ((rootFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT);

		// Compute initial 2^h value at root of sub-tree
		rootN = rootEnd - rootStart + 1;
		h     = (unsigned int)( floorf( log2f( (float)rootN ) ) );
		h2    = 1<<h;	 // 2^h

		// Reset flags at root (h2 + axis)
		currBuild[bidx][top].flags = (h2 & NODE_INDEX_MASK) 
								     | ((rootAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);

		// Is this a valid sub-tree for us to do work on ?
		validRoot = (rootEnd < rootStart) ? 0u : 1u;
		if (validRoot)
		{
			top++;			// Increment top of stack
		}
	}


	while (top > 0)
	{
		//--------------------------
		// Pop Build Item off stack
		//--------------------------

		top--;
		origStart  = (currBuild[bidx][top].start & NODE_INDEX_MASK);
		origEnd    = (currBuild[bidx][top].end & NODE_INDEX_MASK);
		currTarget = currBuild[bidx][top].targetID;
		currFlags  = currBuild[bidx][top].flags;

		currAxis   = ((currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT);
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);
		origN      = origEnd-origStart + 1;


		//----------------------------------
		// Compute Left-balanced Median
		//----------------------------------

		{
			unsigned int br, minVal, LBM;
			h2         = (currFlags & NODE_INDEX_MASK);
			br         = origN - (h2-1);		// n - (2^h-1)
			minVal     = min(2*br,h2);		
			LBM        = (h2 + minVal) >> 1;	// LBM = 2^h + min(2*br,2^h)
			currMedian = origStart + (LBM - 1);		// actual median
		}


		//---------------------------------------
		// Partition [start,end] range on Median
		//---------------------------------------

		currStart = origStart;
		currEnd   = origEnd;

		bDone = 0u;
		while (! bDone)
		{
			//-----------------------------
			// Compute Pivot
			//-----------------------------

			if (origN > 20)
			{
				//
				// Use median of 3 variant (by table look-up)
				//
				unsigned int m_idx, testIdx, pivotIdx;
				
				m_idx = (currStart+currEnd) >> 1;	// (l+r)/2
					// 3 slow reads from memory
				m3Vals[0] = medNodes[currStart].pos[currAxis];	// Left
				m3Vals[1] = medNodes[m_idx].pos[currAxis];		// Middle
				m3Vals[2] = medNodes[currEnd].pos[currAxis];	// Right

					// Compute pivot value via "Median of 3" table lookup
				testIdx =  ((m3Vals[1] > m3Vals[2]) << 2)		// M > R test
					       | ((m3Vals[0] > m3Vals[2]) << 1)		// L > R test
					       | (m3Vals[0] > m3Vals[1]);			// L > M test

			    pivotIdx = g_m3Table[testIdx];
				pivotVal = m3Vals[pivotIdx];
			}
			else
			{
				// Grab pivot from 1st element
					// Slow read from main memory
				pivotVal = medNodes[currStart].pos[currAxis];
			}


			//-------------------
			// Count Nodes
			//-------------------

			countBefore = 0;
			countEqual  = 0;
			countAfter  = 0;
			for (currItem = currStart; currItem <= currEnd; currItem++)
			{
				// Get current value
					// Slow read from global memory
				currMED[bidx] = medNodes[currItem];
				currVal = currMED[bidx].pos[currAxis];

				// Count # of values before and after pivot
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

					// Slow write to scratch buffer
				medScratch[currItem] = currMED[bidx];
			}


			//--------------------
			// Compute Starts
			//--------------------

			baseBefore = currStart;
			baseEqual  = currStart + countBefore;
			baseAfter  = currStart + countBefore + countEqual;

			startBefore = baseBefore;
			startEqual  = baseEqual;
			startAfter  = baseAfter;


			//-------------------
			// Partition Nodes
			//-------------------

			// partition nodes from scratch buffer
			// back into actual kd-node array
			for (currItem = currStart; currItem <= currEnd; currItem++)
			{
				// Read node from original location
					// Slow read from global memory
				currMED[bidx] = medScratch[currItem];

				// Partition node into appropriate location
				currVal = currMED[bidx].pos[currAxis];
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

				// Write node to new partitioned location
					// Slow write to external memory
				medNodes[outIdx] = currMED[bidx];
			}
	

			//-----------------------
			// Done partitioning ?!?
			//-----------------------

			if (currMedian < baseEqual)	
			{
				// Not done, iterate on {L} partition = [currStart, equalBase - 1]
				currEnd = baseEqual - 1;
			}
			else if (currMedian >= baseAfter)	// Median in after partition {R}
			{
				// Not done, iterate on {R} partition = range [afterBase, currEnd]
				currStart = baseAfter;
			}
			else // Median is in median partition {M}
			{
				// Done, the left-balanced median is where we want it
				bDone = 1u;
			}

		} // end while (!bDone)


		//---------------------------------------
		// Store Left-Balanced Median at target
		//---------------------------------------

			// Slow read from main memory
		currMED[bidx] = medNodes[currMedian];

		currLBT[bidx].pos[0] = currMED[bidx].pos[0];
		currLBT[bidx].pos[1] = currMED[bidx].pos[1];

			// Slow write to main memory
		lbtNodes[currTarget] = currLBT[bidx];
		pointIDs[currTarget] = currMED[bidx].m_searchIdx;


		//---------------------------------------
		// Compute Left and Right build items
		//---------------------------------------

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= nPoints)
		{
			unsigned int rStart = currMedian+1;
			unsigned int rEnd   = origEnd;

			// push right child onto stack
			currBuild[bidx][top].start    = (rStart & NODE_INDEX_MASK);
			currBuild[bidx][top].end      = (rEnd & NODE_INDEX_MASK);
			currBuild[bidx][top].targetID = (currRight & NODE_INDEX_MASK);
			//currBuild[bidx][top].flags    = ((currHalf >> 1) & NODE_INDEX_MASK) 
			//					            | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			currBuild[bidx][top].flags    = ((h2 >> 1) & NODE_INDEX_MASK) 
								            | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			top++;
		}

		if (currLeft <= nPoints)
		{
			unsigned int lStart = origStart;
			unsigned int lEnd   = currMedian-1;

			// push left child onto stack
			currBuild[bidx][top].start    = (lStart & NODE_INDEX_MASK);
			currBuild[bidx][top].end      = (lEnd & NODE_INDEX_MASK);
			currBuild[bidx][top].targetID = (currLeft & NODE_INDEX_MASK);
			//currBuild[bidx][top].flags    = ((currHalf >> 1) & NODE_INDEX_MASK) 
			//					            | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			currBuild[bidx][top].flags    = ((h2 >> 1) & NODE_INDEX_MASK) 
								            | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			top++;
		}
	}
}


__global__ void
P2_2D_BUILD_STATS
(
	GPUNode_2D_LBT * lbtNodes,	// OUT: lbt node list
	unsigned int   * pointIDs,	// OUT: point indices are stored in this array
	GPU_BUILD_STATS * statsQ,	// OUT: stats queue (per thread)
	GPUNode_2D_MED * medNodes,	// IN: median node list
	GPUNode_2D_MED * medScratch,// IN: scratch space for temporary copying
	GPU_BUILD_ITEM * buildQ,	// IN: build queue (per thread)
	unsigned int     nPoints	// IN: maximum # of points
)
{
	// Local variables (shared items)
	__shared__ GPU_BUILD_ITEM	currBuild[P2_BUILD_THREADS_PER_BLOCK][P2_BUILD_STACK_DEPTH];
	__shared__ GPUNode_2D_MED	currMED[P2_BUILD_THREADS_PER_BLOCK];
	__shared__ GPUNode_2D_LBT	currLBT[P2_BUILD_THREADS_PER_BLOCK];
#ifdef _BUILD_STATS
	__shared__ GPU_BUILD_STATS	currStats[P2_BUILD_THREADS_PER_BLOCK];
#endif
	__shared__ float			m3Vals[3];

	// Local Variables (registers)
	float pivotVal, currVal;
	unsigned int currAxis, nextAxis;
	unsigned int h2, currMedian, validRoot;

	unsigned int currLeft, currRight;
	unsigned int currStart, currEnd, currItem;

	unsigned int currTarget, currFlags, outIdx, top;
	unsigned int origStart, origEnd, origN, bDone;

	unsigned int countBefore, countAfter, countEqual;
	unsigned int startBefore, startAfter, startEqual;
	unsigned int baseBefore, baseAfter, baseEqual;

	/*-----------------------
	  Compute Thread Column
    -----------------------*/

	// Block thread index (local)
	const int bidx = (threadIdx.y*blockDim.x) + threadIdx.x;

	// Grid thread index (global)
	const int cRow  = (blockIdx.y * blockDim.y) + threadIdx.y;	// thread row in grid of blocks
	const int cCol  = (blockIdx.x * blockDim.x) + threadIdx.x;	// thread column in grid of blocks
	const int gidx = (cRow * (gridDim.x * blockDim.x)) + cCol;

#ifdef _BUILD_STATS
	// Initialize Stats
	currStats[bidx].cRootReads   = 0;
	currStats[bidx].cPivotReads  = 0;
	currStats[bidx].cCountReads  = 0;
	currStats[bidx].cCountWrites = 0;
	currStats[bidx].cPartReads   = 0;
	currStats[bidx].cPartWrites  = 0;
	currStats[bidx].cStoreReads	 = 0;
	currStats[bidx].cStoreWrites = 0;
	currStats[bidx].cNodeLoops   = 0;
	currStats[bidx].cPartLoops   = 0;
#endif

	//----------------------------------
	// Push first build item onto stack
	//----------------------------------

	{
		unsigned int rootStart, rootEnd, rootFlags, rootAxis;
		unsigned int rootN, h;
	
		// Get root of sub-tree to process
		top = 0;
			// Slow read from main memory
		currBuild[bidx][top] = buildQ[gidx];
#ifdef _BUILD_STATS
		currStats[bidx].cRootReads++;
#endif
		rootStart = currBuild[bidx][top].start & NODE_INDEX_MASK;
		rootEnd   = currBuild[bidx][top].end & NODE_INDEX_MASK;
		rootFlags = currBuild[bidx][top].flags;
		rootAxis  = ((rootFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT);

		// Compute initial 2^h value at root of sub-tree
		rootN     = rootEnd - rootStart + 1;
		h = (unsigned int)( floorf( log2f( (float)rootN ) ) );
		h2 = 1<<h;	 // 2^h

		// Reset flags at root (h2 + axis)
		currBuild[bidx][top].flags = (h2 & NODE_INDEX_MASK) 
								     | ((rootAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);

		// Is this a valid sub-tree for us to do work on ?
		validRoot = (rootEnd < rootStart) ? 0u : 1u;
		if (validRoot)
		{
			top++;			// Increment top of stack
		}
	}

	while (top > 0)
	{
		//--------------------------
		// Pop Build Item off stack
		//--------------------------

#ifdef _BUILD_STATS
		currStats[bidx].cNodeLoops++;
#endif

		top--;
		origStart  = (currBuild[bidx][top].start & NODE_INDEX_MASK);
		origEnd    = (currBuild[bidx][top].end & NODE_INDEX_MASK);
		currTarget = currBuild[bidx][top].targetID;
		currFlags  = currBuild[bidx][top].flags;

		currAxis   = ((currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT);
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);
		origN      = origEnd-origStart + 1;


		//----------------------------------
		// Compute Left-balanced Median
		//----------------------------------

		{
			unsigned int br, minVal, LBM;
			h2         = (currFlags & NODE_INDEX_MASK);
			br         = origN - (h2-1);		// n - (2^h-1)
			minVal     = min(2*br,h2);		
			LBM        = (h2 + minVal) >> 1;	// LBM = 2^h + min(2*br,2^h)
			currMedian = origStart + (LBM - 1);		// actual median
		}


		//---------------------------------------
		// Partition [start,end] range on Median
		//---------------------------------------

		currStart = origStart;
		currEnd   = origEnd;

		bDone = 0u;
		while (! bDone)
		{
#ifdef _BUILD_STATS
			currStats[bidx].cPartLoops++;
#endif

			//-----------------------------
			// Compute Pivot
			//-----------------------------

			if (origN > 20)
			{
				//
				// Use median of 3 variant (by table look-up)
				//
				unsigned int m_idx, testIdx, pivotIdx;
				
				m_idx = (currStart+currEnd) >> 1;	// (l+r)/2
					// 3 slow reads from memory
				m3Vals[0] = medNodes[currStart].pos[currAxis];	// Left
				m3Vals[1] = medNodes[m_idx].pos[currAxis];		// Middle
				m3Vals[2] = medNodes[currEnd].pos[currAxis];	// Right

#ifdef _BUILD_STATS
				currStats[bidx].cPivotReads += 3;
#endif

					// Compute pivot value via "Median of 3" table lookup
				testIdx =  ((m3Vals[1] > m3Vals[2]) << 2)		// M > R test
					       | ((m3Vals[0] > m3Vals[2]) << 1)		// L > R test
					       | (m3Vals[0] > m3Vals[1]);			// L > M test

			    pivotIdx = g_m3Table[testIdx];
				pivotVal = m3Vals[pivotIdx];
			}
			else
			{
				// Grab pivot from 1st element
					// Slow read from main memory
				pivotVal = medNodes[currStart].pos[currAxis];

#ifdef _BUILD_STATS
				currStats[bidx].cPivotReads++;
#endif
			}


			//-------------------
			// Count Nodes
			//-------------------

			countBefore = 0;
			countEqual  = 0;
			countAfter  = 0;
			for (currItem = currStart; currItem <= currEnd; currItem++)
			{
				// Get current value
					// Slow read from global memory
				currMED[bidx] = medNodes[currItem];
				currVal = currMED[bidx].pos[currAxis];

#ifdef _BUILD_STATS
				currStats[bidx].cCountReads++;
#endif

				// Count # of values before and after pivot
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

#ifdef _BUILD_STATS
				currStats[bidx].cCountWrites++;
#endif

					// Slow write to scratch buffer
				medScratch[currItem] = currMED[bidx];
			}


			//--------------------
			// Compute Starts
			//--------------------

			baseBefore = currStart;
			baseEqual  = currStart + countBefore;
			baseAfter  = currStart + countBefore + countEqual;

			startBefore = baseBefore;
			startEqual  = baseEqual;
			startAfter  = baseAfter;


			//-------------------
			// Partition Nodes
			//-------------------

			// partition nodes from scratch buffer
			// back into actual kd-node array
			for (currItem = currStart; currItem <= currEnd; currItem++)
			{
				// Read node from original location
					// Slow read from global memory
				currMED[bidx] = medScratch[currItem];

#ifdef _BUILD_STATS
				currStats[bidx].cPartReads++;
#endif

				// Partition node into appropriate location
				currVal = currMED[bidx].pos[currAxis];
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

#ifdef _BUILD_STATS
				currStats[bidx].cPartWrites++;
#endif

				// Write node to new partitioned location
					// Slow write to external memory
				medNodes[outIdx] = currMED[bidx];
			}
	

			//-----------------------
			// Done partitioning ?!?
			//-----------------------

			if (currMedian < baseEqual)	
			{
				// Not done, iterate on {L} partition = [currStart, equalBase - 1]
				currEnd = baseEqual - 1;
			}
			else if (currMedian >= baseAfter)	// Median in after partition {R}
			{
				// Not done, iterate on {R} partition = range [afterBase, currEnd]
				currStart = baseAfter;
			}
			else // Median is in median partition {M}
			{
				// Done, the left-balanced median is where we want it
				bDone = 1u;
			}

		} // end while (!bDone)


		//---------------------------------------
		// Store Left-Balanced Median at target
		//---------------------------------------

			// Slow read from main memory
		currMED[bidx] = medNodes[currMedian];

#ifdef _BUILD_STATS
		currStats[bidx].cStoreReads++;
#endif

		currLBT[bidx].pos[0] = currMED[bidx].pos[0];
		currLBT[bidx].pos[1] = currMED[bidx].pos[1];

			// Slow write to main memory
		lbtNodes[currTarget] = currLBT[bidx];
		pointIDs[currTarget] = currMED[bidx].m_searchIdx;

#ifdef _BUILD_STATS
		currStats[bidx].cStoreWrites +=2;
#endif


		//---------------------------------------
		// Compute Left and Right build items
		//---------------------------------------

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= nPoints)
		{
			unsigned int rStart = currMedian+1;
			unsigned int rEnd   = origEnd;

			// push right child onto stack
			currBuild[bidx][top].start    = (rStart & NODE_INDEX_MASK);
			currBuild[bidx][top].end      = (rEnd & NODE_INDEX_MASK);
			currBuild[bidx][top].targetID = (currRight & NODE_INDEX_MASK);
			currBuild[bidx][top].flags    = ((h2 >> 1) & NODE_INDEX_MASK) 
								            | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			top++;
		}

		if (currLeft <= nPoints)
		{
			unsigned int lStart = origStart;
			unsigned int lEnd   = currMedian-1;

			// push left child onto stack
			currBuild[bidx][top].start    = (lStart & NODE_INDEX_MASK);
			currBuild[bidx][top].end      = (lEnd & NODE_INDEX_MASK);
			currBuild[bidx][top].targetID = (currLeft & NODE_INDEX_MASK);
			currBuild[bidx][top].flags    = ((h2 >> 1) & NODE_INDEX_MASK) 
								            | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			top++;
		}
	}

#ifdef _BUILD_STATS
	// Store Stats to output array
	statsQ[gidx] = currStats[bidx];
#endif
}


