/*-----------------------------------------------------------------------------
  Name:  CPUTree_LBT.cpp
  Desc:  Implements a simple kd-tree on the CPU
         kd-nodes are stored in left-balanced array order

  Log:   Created by Shawn D. Brown (3/6/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

#ifndef _KD_FLAGS_H
	#include "KD_Flags.h"
#endif
#ifndef _CPUTREE_LBT_H
	#include "CPUTree_LBT.h"
#endif
#ifndef _CPUTREE_API_H
	#include "CPUTree_API.h"
#endif

#ifndef _DEQUE_
	#include <deque>	// Std::deque
#endif
#ifndef _STACK_
	#include <stack>	// Std::Stack
#endif
#include <intrin.h>		// Intrinsics (Intel Specific, Non-portable)

#include <iostream>
#include <algorithm>


/*-------------------------------------
  Helper Methods
-------------------------------------*/



/*--------------------------------------------------------------
  Name: KDTreeAxisValue
  Desc: Get axis value from 2D point stored in specified node
--------------------------------------------------------------*/

inline float KDTreeAxisValue
(
	const CPUNode_2D_LBT * currNodes,
	unsigned int index,
	unsigned int axis
)
{
	const CPUNode_2D_LBT & currNode = currNodes[index];
	return currNode[axis];
}


/*--------------------------------------------------------------
  Name: KDTreeSwap
  Desc: Swap specified nodes
--------------------------------------------------------------*/

inline void KDTreeSwap
(
	CPUNode_2D_LBT * currNodes,
	unsigned int idx1,
	unsigned int idx2
)
{
	CPUNode_2D_LBT & N1 = currNodes[idx1];
	CPUNode_2D_LBT & N2 = currNodes[idx2];
	CPUNode_2D_LBT temp;

	temp = N1;
	N1 = N2;
	N2 = temp;
}


/*--------------------------------------------------------------
  Name: KDTreeMedianOf3
  Desc: Sort median of 3 elements to last position
--------------------------------------------------------------*/

void KDTreeMedianOf3
( 
	CPUNode_2D_LBT * currNodes, 
	unsigned int axis,
	unsigned int first,
	unsigned int middle,
	unsigned int last
)
{	
	// Compare middle and first
	if (KDTreeAxisValue( currNodes, middle, axis ) < 
		KDTreeAxisValue( currNodes, first, axis ))
	{
		// Out of order, swap middle and first
		KDTreeSwap( currNodes, middle, first );
	}

	// Compare last and middle
	if (KDTreeAxisValue( currNodes, last, axis ) < 
		KDTreeAxisValue( currNodes, middle, axis ))
	{
		// Out of order, swap last and middle
		KDTreeSwap( currNodes, last, middle );
	}

	// Compare middle and first again
	if (KDTreeAxisValue( currNodes, middle, axis ) < 
		KDTreeAxisValue( currNodes, first, axis ))
	{
		// Out of order, swap middle and first
		KDTreeSwap( currNodes, middle, first );
	}
}

/*--------------------------------------------------------------
  Name: KDTreeInsertionSort
  Desc: Sort specified elements
  Note: adapted from Robert Sedgewick's book
        "Algorithms in C++", page 100
--------------------------------------------------------------*/

void KDTreeInsertionSort
( 
	CPUNode_2D_LBT * currNodes, 
	unsigned int axis,
	unsigned int first,
	unsigned int last
)
{	
	if (first != last)
	{
		unsigned int i,j;
		CPUNode_2D_LBT nodeVal;

		for (i = first+1; i <= last; i++)
		{
			nodeVal = currNodes[i];
			j = i;
			while (KDTreeAxisValue( currNodes, j-1, axis) > nodeVal[axis])
			{
				currNodes[j] = currNodes[j-1];
				j--;
			}
			currNodes[j] = nodeVal;
		}
	}
}


/*--------------------------------------------------------------
  Name: KDTreeNthElement
  Desc: Nth Element
--------------------------------------------------------------*/

bool KDTreeNthElement
( 
	CPUNode_2D_LBT * currNodes, 
	unsigned int axis, 
	unsigned int first, 
	unsigned int last,
	unsigned int nth
)
{
	// Check parameters
	if (NULL == currNodes) { return false; }
	if (last < first) 
	{
		unsigned int temp = first;
		first = last;
		last = temp;
	}
	if ((nth < first) || (nth > last)) 
	{ 
		return false; 
	}

	unsigned int midFirst, midLast;

	// Divide and conquer, ordering partition containing Nth element
		// If we have less than 32 element left, just sort the rest
	for (; 32 < last - first; )	
	{
		// Choose pivot (median of 3)
		unsigned int middle = (first + last)/2;
		KDTreeMedianOf3( currNodes, axis, last, middle, first );
		// Pivot now found at middle

		// Setup sentinels
		unsigned int pFirst = middle;
		unsigned int pLast  = middle+1;

		// Skip values equal to pivot value in left parition
		while ((first < pFirst) 
				&& !(KDTreeAxisValue( currNodes, pFirst-1, axis ) < 
				     KDTreeAxisValue( currNodes, pFirst, axis ))
			    && !(KDTreeAxisValue( currNodes, pFirst, axis ) <
				     KDTreeAxisValue( currNodes, pFirst-1, axis ))
			  )
		{
			--pFirst;
		}

		// Skip values equal to pivot value in right parition
		while ((pLast < last)
			   && !(KDTreeAxisValue( currNodes, pLast, axis ) < 
			        KDTreeAxisValue( currNodes, pFirst, axis ))
			   && !(KDTreeAxisValue( currNodes, pFirst, axis) <
			        KDTreeAxisValue( currNodes, pLast, axis ))
			  )
		{
			++pLast;
		}
				
		unsigned int gFirst = pLast;
		unsigned int gLast  = pFirst;

		// Partition 
		for (;;)
		{	
			for (; gFirst < last; ++gFirst) 
			{
				float pVal = KDTreeAxisValue( currNodes, pFirst, axis );
				float gVal = KDTreeAxisValue( currNodes, gFirst, axis );

				if (pVal < gVal)
				{
					// Do nothing
				}
				else if (gVal < pVal)
				{
					break;
				}
				else
				{
					// Swap pLast and gFirst 
					KDTreeAxisValue( currNodes, pLast++, gFirst );
				}
			}

			// 
			for (; first < gLast; --gLast)
			{
				float lastVal  = KDTreeAxisValue( currNodes, gLast - 1, axis );
				float firstVal = KDTreeAxisValue( currNodes, pFirst, axis );
				if (lastVal < firstVal)
				{
					// Do nothing
				}
				else if (firstVal < lastVal)
				{
					break;
				}
				else
				{
					KDTreeSwap( currNodes, --pFirst, gLast - 1 );
				}
			}
		
			if (gLast == first && gFirst == last)
			{
				// Done with partitioning
				midFirst = pFirst;
				midLast  = pLast;
				break;
			}

			if (gLast == first)
			{	
				// no room at bottom, rotate pivot upward
				if (pLast != gFirst)
				{
					KDTreeSwap( currNodes, pFirst, pLast );
				}
				++pLast;
				
				KDTreeSwap( currNodes, pFirst++, gFirst++ );
			}
			else if (gFirst == last)
			{	
				// no room at top, rotate pivot downward
				if (--gLast != --pFirst)
				{
					KDTreeSwap( currNodes, gLast, pFirst );
				}
				KDTreeSwap( currNodes, pFirst, --pLast );
			}
			else
			{
				KDTreeSwap( currNodes, gFirst++, --gLast );
			}
		}

		// Iterate on either the left or right range containing nth element
			// Or stop if we are inside a fat range
		if (midLast <= nth)
		{
			// Nth in right range
			first = midLast;
		}
		else if (midFirst <= nth)
		{
			// Nth inside fat pivot, we are done
			return true;	
		}
		else 
		{
			// Nth in left range
			last = midFirst;
		}
	}

	// Sort any small remainder to put nth element into correct position
	KDTreeInsertionSort( currNodes, axis, first, last );

	return true;
}



/*--------------------------------------------------------------
  Name: KDTreePartition
--------------------------------------------------------------*/

unsigned int KDTreePartition
( 
	CPUNode_2D_LBT * currNodes, 
	unsigned int axis, 
	unsigned int left, 
	unsigned int right,
	unsigned int pivot
)
{
	float pivotVal = KDTreeAxisValue( currNodes, pivot, axis );
	
	// Move pivot to end of list
	KDTreeSwap( currNodes, pivot, right );

	// Partition list
	unsigned int store = left;
	for (unsigned i = left; i < right; i++)
	{
		float currVal = KDTreeAxisValue( currNodes, i, axis );
		if (currVal < pivotVal)
		{
			KDTreeSwap( currNodes, store, i );
			store++;
		}
	}

	// Move pivot into its final position
	KDTreeSwap( currNodes, right, store );

	return store;
}

/*--------------------------------------------------------------
  Name: KDTreeSelect
--------------------------------------------------------------*/

void KDTreeSelect
(
	CPUNode_2D_LBT * currNodes,
	unsigned int axis,
	unsigned int left,
	unsigned int right,
	unsigned int nth
)
{
	for (;;)
	{
		// Get Pivot
		unsigned int median = (left+right)/2;
		KDTreeMedianOf3( currNodes, axis, left, median, right );
		
		// Partition on pivot
		unsigned int pivot = KDTreePartition( currNodes, axis, left, right, median );

		// Iterate on range containing nth element
		if (nth < pivot)
		{
			right = pivot - 1;
		}
		else if (nth > pivot)
		{
			left = pivot + 1;
		}
		else
		{
			// we have succesfully parititioned on the nth element.
			return;
		}
	}
}


/*-------------------------------------
  CPUTree_2D_LBT Methods Definitions
-------------------------------------*/

/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_LBT::GetNodeAxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

float CPUTree_2D_LBT::GetNodeAxisValue
( 
	const CPUNode_2D_LBT * currNodes,	// IN:  IN node list
	unsigned int index,				// IN:  Index of node to retrieve value for
	unsigned int axis				// IN:  axis of value to retrieve
) const
{
	const CPUNode_2D_LBT & currNode = currNodes[index];
	float axisValue = currNode[axis];
	return axisValue;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_LBT::SwapNodes
  Desc:	Helper Method
-------------------------------------------------------------------------*/

void CPUTree_2D_LBT::SwapNodes
( 
	CPUNode_2D_LBT * currNodes,	// IN: Node list
	unsigned int idx1,			// IN: Index of 1st node to swap
	unsigned int idx2			// IN: Index of 2nd node to swap
)
{
	CPUNode_2D_LBT & currNode1 = currNodes[idx1];
	CPUNode_2D_LBT & currNode2 = currNodes[idx2];
	CPUNode_2D_LBT temp;

	temp	  = currNode1;
	currNode1 = currNode2;
	currNode2 = temp;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_LBT::MedianOf3
  Desc:	Helper method,
		Implements Median of three variant 
		for Median Partitioning algorithm
		returns pivot value for partitioning algorithm
  Note:	enforces invariant
		array[left].val <= array[mid].val <= array[right].val
		where mid = (left+right)/2
-------------------------------------------------------------------------*/

void CPUTree_2D_LBT::MedianOf3
(
	CPUNode_2D_LBT * currNodes,	// IN - node list
	unsigned int leftIdx,		// IN - left index
	unsigned int rightIdx,		// IN - right index
	unsigned int axis			// IN - axis to compare
)
{
	// Compute Middle Index from left and right
	unsigned int midIdx = (leftIdx+rightIdx)/2;	

	float leftVal  = GetNodeAxisValue( currNodes, leftIdx, axis );
	float rightVal = GetNodeAxisValue( currNodes, rightIdx, axis );
	float midVal   = GetNodeAxisValue( currNodes, midIdx, axis );

#ifdef _BUILD_STATS
	m_cpuStats.cPivotReads += 3;	// 3 reads (left, middle, right)
#endif

	// Sort left, center, mid value into correct order
	if (leftVal > midVal)
	{
		SwapNodes( currNodes, leftIdx, midIdx );
#ifdef _BUILD_STATS
		m_cpuStats.cPivotSwaps++;		// 1 swap
		m_cpuStats.cPivotReads  +=2;	// 2 reads per swap
		m_cpuStats.cPivotWrites +=2;	// 2 writes per swap
#endif
	}
	if (leftVal > rightVal)
	{
		SwapNodes( currNodes, leftIdx, rightIdx );
#ifdef _BUILD_STATS
		m_cpuStats.cPivotSwaps++;		// 1 swap
		m_cpuStats.cPivotReads  +=2;	// 2 reads per swap
		m_cpuStats.cPivotWrites +=2;	// 2 writes per swap
#endif
	}
	if (midVal > rightVal)
	{
		SwapNodes( currNodes, midIdx, rightIdx );
#ifdef _BUILD_STATS
		m_cpuStats.cPivotSwaps++;		// 1 swap
		m_cpuStats.cPivotReads  +=2;	// 2 reads per swap
		m_cpuStats.cPivotWrites +=2;	// 2 writes per swap
#endif
	}

	// Deliberately move median value to end of array
	SwapNodes( currNodes, midIdx, rightIdx );

#ifdef _BUILD_STATS
	m_cpuStats.cPivotSwaps++;		// 1 swap
	m_cpuStats.cPivotReads  +=2;	// 2 reads per swap
	m_cpuStats.cPivotWrites +=2;	// 2 writes per swap
#endif

}


/*-------------------------------------------------------------------------
  Name: CPUTree_2D_LBT::MedianSortNodes
  Desc:	Partitions original data set {O} with 'n' elements
		into 3 datasets <{l}, {m}, {r}}

		where {l} contains approximately half the elements 
		and all elements are less than or equal to {m}
		where {m} contains only 1 element the "median value"
		and {r} contains approximately half the elements
		and all elements are greater than or equal to {m}

  Notes:
	Invariant:
		{l} = { all points with value less or equal to median }
		    count({l}) = (n - 1) / 2
		{m} = { median point }
		    count({m}) = 1
		{r} = { points with value greater than or equal to median point }
		    count({r}) = n - ((n-1)/2 + 1)
			Note: Either count({r}) = count({l}) or
			             count({r}) = count({l})+1
				  IE the count of sets {l} and {r} will be 
				     within one element of each other

	Performance:
		Should take O(N) time to process all points in input range

  Credit:	Based on the "Selection Sort" algorithm as
            presented by Robert Sedgewick in the book
            "Algorithms in C++"
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::MedianSortNodes
(
	CPUNode_2D_LBT * currNodes,	// IN - node list
	unsigned int start,     // IN - start of range
	unsigned int end,       // IN - end of range
	unsigned int & median,  // IN/OUT - approximate median number
	                        //          actual median number
	unsigned int axis       // IN - dimension(axis) to split along (x,y,z)
)
{
	if (start > end)
	{
		// Swap Start and End, if they are in the wrong order
		unsigned int temp = start;
		start = end;
		end = temp;
	}

	// Check Parameters
	unsigned int nNodes = (end - start) + 1;
	if (nNodes == 0) { return false; }
	if (nNodes == 1) { return true; }
	if (axis >= INVALID_AXIS) { return false; }
	if ((median < start) || (median > end)) { return false; }

	// Perform Median Sort
	int left   = static_cast<int>( start );
	int right  = static_cast<int>( end );
	int middle = static_cast<int>( median );
	int i,j;
	float pivotVal;

	while ( right > left ) 
	{
#ifdef _BUILD_STATS
		m_cpuStats.cPartLoops++;
#endif

		// Get Pivot value
			// Use Median of 3 variant
		MedianOf3( currNodes, left, right, axis );

		pivotVal = GetNodeAxisValue( currNodes, right, axis );
#ifdef _BUILD_STATS
		m_cpuStats.cPivotReads++;
#endif

		i = left - 1;
		j = right;

		// Partition into 3 sets
			// Left   = {start, pivot-1}	all values in Left <= median
			// Median = {pivot}				singleton containing pivot value
			// Right  = {pivot+1, end}		all values in right >= median
		for (;;) 
		{
			while ( GetNodeAxisValue( currNodes, ++i, axis ) < pivotVal )
			{
#ifdef _BUILD_STATS
				m_cpuStats.cPartReads++;
#endif
				// Deliberately do nothing
			}

			while ( (GetNodeAxisValue( currNodes, --j, axis ) > pivotVal) && 
				  (j > left) )
			{
#ifdef _BUILD_STATS
				m_cpuStats.cPartReads++;
#endif
				// Deliberately do nothing
			}
			
			if ( i >= j )
				break;

			SwapNodes( currNodes, i, j );
#ifdef _BUILD_STATS
			m_cpuStats.cPartSwaps++;		// 1 swap
			m_cpuStats.cPartReads  += 2;	// 2 reads per swap
			m_cpuStats.cPartWrites += 2;	// 2 writes per swap
#endif
		}

		// Put pivot value back into pivot position
		SwapNodes( currNodes, i, right );

#ifdef _BUILD_STATS
		m_cpuStats.cPartSwaps++;		// 1 swap
		m_cpuStats.cPartReads  += 2;	// 2 reads per swap
		m_cpuStats.cPartWrites += 2;	// 2 writes per swap
#endif

		// Iterate into left or right set until
		// we find the value at the true median
		if ( i >= middle )
			right = i-1;
		if ( i <= middle )
			left = i+1;
	}

	// return new median index
	median = middle;

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_LBT::ComputeBoundBox
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::ComputeBoundingBox
( 
	unsigned int start,		// IN - start index
	unsigned int end,		// IN - end index
	float        bounds[4]	// OUT - bounding box for all nodes in range
)
{
	// Check Parameters
	if (CPUNode_2D_LBT::c_Invalid == start) { return false; }
	if (CPUNode_2D_LBT::c_Invalid == end) { return false; }

	unsigned int s = start;
	unsigned int e = end;
	if (e < s) 
	{
		unsigned int temp = s;
		s = e;
		e = temp;
	}

	CPUNode_2D_LBT * currNode = NODE_PTR( s );
	if (NULL == currNode) { return false; }
	
	float x, y;

	x = currNode->X();
	y = currNode->Y();

	bounds[0] = x;
	bounds[1] = x;
	bounds[2] = y;
	bounds[3] = y;

	unsigned int i;
	for (i = s+1; i <= e; i++)
	{
		currNode = NODE_PTR( i );
		x = currNode->X();
		y = currNode->Y();

		// Update Min, Max for X,Y
		if (x < bounds[0]) { bounds[0] = x; }
		if (x > bounds[1]) { bounds[1] = x; }

		if (y < bounds[2]) { bounds[2] = y; }
		if (y > bounds[3]) { bounds[3] = y; }
	}

	// Success
	return true;
}


// Lookup tables for calculating left-balanced Median for small 'n'
static unsigned int g_leftMedianTable[32] = 
{ 
	0u,			// Wasted space (but necessary for 1-based indexing)
	1u,							// Level 1
	2u,2u,						// Level 2
	3u,4u,4u,4u,				// Level 3
	5u,6u,7u,8u,8u,8u,8u,8u,	// Level 4
	9u,10u,11u,12u,13u,14u,15u,16u,16u,16u,16u,16u,16u,16u,16u,16u // Level 5
};
static unsigned int g_halfTable[32] = 
{ 
	0u,			// Wasted space (but necessary for 1-based indexing)
	0u,							// Level 1
	1u,1u,						// Level 2
	2u,2u,2u,2u,				// Level 3
	4u,4u,4u,4u,4u,4u,4u,4u,	// Level 4
	8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u // Level 5
};


/*-------------------------------------------------------------------------
  Name:	KDTree::Build2D
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Build2D( unsigned int cPoints, const float2 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_2D_LBT * mNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_2D_LBT * lNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		currNode.X( x );
		currNode.Y( y );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_2D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_2D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_2D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[0];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29) & 0x3u);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;						// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_2D_LBT & medianNode = mNodes[currMedian];
		CPUNode_2D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		unsigned int currID = medianNode.SearchID();
		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build2DStats
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Build2DStats
( 
	unsigned int cPoints,		// IN:   Number of points in point list
	const float2 * pointList	// IN:	 point list to build into kd-tree
)
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

#ifdef _BUILD_STATS
	// Initialize CPU stats
	m_cpuStats.cNodeLoops   = 0;
	m_cpuStats.cPartLoops   = 0;
	m_cpuStats.cPartReads   = 0;
	m_cpuStats.cPartSwaps   = 0;
	m_cpuStats.cPartWrites  = 0;
	m_cpuStats.cPivotReads  = 0;
	m_cpuStats.cPivotSwaps  = 0;
	m_cpuStats.cPivotWrites = 0;
	m_cpuStats.cPivotSwaps  = 0;
	m_cpuStats.cStoreReads  = 0;
	m_cpuStats.cStoreWrites = 0;
#endif

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_2D_LBT * mNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_2D_LBT * lNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		currNode.X( x );
		currNode.Y( y );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_2D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_2D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_2D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[0];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
#ifdef _BUILD_STATS
		m_cpuStats.cNodeLoops++;
#endif
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29) & 0x3u);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;						// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target

		const CPUNode_2D_LBT & medianNode = mNodes[currMedian];
		CPUNode_2D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		unsigned int currID = medianNode.SearchID();

#ifdef _BUILD_STATS
		m_cpuStats.cStoreReads++;
#endif

		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

#ifdef _BUILD_STATS
		m_cpuStats.cStoreWrites++;
#endif

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

#ifdef _BUILD_STATS
	// Dump Build stats
	DumpBuildStats();
#endif

	// Success
	return true;
}


bool CPUTree_2D_LBT::Build2D( unsigned int cPoints, const float3 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_2D_LBT * mNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_2D_LBT * lNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		currNode.X( x );
		currNode.Y( y );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_2D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_2D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_2D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[0];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29) & 0x3u);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_2D_LBT & medianNode = mNodes[currMedian];
		CPUNode_2D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		unsigned int currID = medianNode.SearchID();
		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}


bool CPUTree_2D_LBT::Build2D( unsigned int cPoints, const float4 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_2D_LBT * mNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_2D_LBT * lNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		currNode.X( x );
		currNode.Y( y );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_2D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_2D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_2D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[0];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29) & 0x3u);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_2D_LBT & medianNode = mNodes[currMedian];
		CPUNode_2D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		unsigned int currID = medianNode.SearchID();
		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build2DStats
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Build2DStats
( 
	unsigned int cPoints,		// IN:   Number of points in point list
	const float4 * pointList	// IN:	 point list to build into kd-tree
)
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

#ifdef _BUILD_STATS
	// Initialize CPU stats
	m_cpuStats.cNodeLoops   = 0;
	m_cpuStats.cPartLoops   = 0;
	m_cpuStats.cPartReads   = 0;
	m_cpuStats.cPartSwaps   = 0;
	m_cpuStats.cPartWrites  = 0;
	m_cpuStats.cPivotReads  = 0;
	m_cpuStats.cPivotSwaps  = 0;
	m_cpuStats.cPivotWrites = 0;
	m_cpuStats.cPivotSwaps  = 0;
	m_cpuStats.cStoreReads  = 0;
	m_cpuStats.cStoreWrites = 0;
#endif

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_2D_LBT * mNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_2D_LBT * lNodes = new CPUNode_2D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		currNode.X( x );
		currNode.Y( y );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_2D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_2D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_2D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_2D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[0];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
#ifdef _BUILD_STATS
		m_cpuStats.cNodeLoops++;
#endif
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29) & 0x3u);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;						// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target

		const CPUNode_2D_LBT & medianNode = mNodes[currMedian];
		CPUNode_2D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		unsigned int currID = medianNode.SearchID();

#ifdef _BUILD_STATS
		m_cpuStats.cStoreReads++;
#endif

		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

#ifdef _BUILD_STATS
		m_cpuStats.cStoreWrites++;
#endif

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

#ifdef _BUILD_STATS
	// Dump Build stats
	DumpBuildStats();
#endif

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_LBT::BF_FindNN_2D()
  Desc:	Use Brute Force Algorithm to find closest Node (index)
			Takes O(N) time
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::BF_FindNN_2D
(
	const float4 & queryLocation,	// IN  - Location to sample
	unsigned int & nearestID,		// OUT - ID of nearest point
	float & nearestDistance			// OUT - distance to nearest point
)
{
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	// Get Query Point
	float qX, qY;
	qX = queryLocation.x;
	qY = queryLocation.y;

	// Get 1st Point
	unsigned int  bestIndex  = 1;
	CPUNode_2D_LBT * currNodePtr = NULL;
	CPUNode_2D_LBT * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID      = bestNodePtr->SearchID();

	float bX, bY;
	bX = bestNodePtr->X();
	bY = bestNodePtr->Y();

	// Calculate distance from query location
	float dX = bX - qX;
	float dY = bY - qY;
	float bestDist2 = dX*dX + dY*dY;
	float diffDist2;

	unsigned int i;
	for (i = 2; i <= nNodes; i++)
	{
		// Get Current Point
		CPUNode_2D_LBT & currNode = m_nodes[i];
		bX = currNode.X();
		bY = currNode.Y();

		// Calculate Distance from query location
		dX = bX - qX;
		dY = bY - qY;
		diffDist2 = dX*dX + dY*dY;

		// Update Best Point Index
		if (diffDist2 < bestDist2)
		{
			bestIndex = i;
			bestID    = currNode.SearchID();
			bestDist2 = diffDist2;
		}
	}

	//Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

	// Success (return results)
	nearestID       = bestID;
	nearestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_QNN_2D
  Desc:	Finds closest point in kd-tree for each query point
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Find_QNN_2D
( 
	CPU_NN_Result * queryResults,	// OUT: Results
	unsigned int      nQueries,		// IN: Number of Query points
	const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPU_NN_Result best;
	CPUNode_2D_LBT * currNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int currAxis, nextAxis, prevAxis;
	unsigned int left, right;
	float dx, dy;
	float queryVals[2];
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	unsigned int currFlags, currQuery;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Reset stack
		stackTop = 0;		

		// Load current Query Point into local (fast) memory
		queryVals[0] = queryPoints[currQuery].x;
		queryVals[1] = queryPoints[currQuery].y;

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & 0x1FFFFFFFu);
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			nextAxis  = (currAxis == 1u) ? 0u : 1u;
			prevAxis  = (currAxis == 0u) ? 1u : 0u;

			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal( currAxis );
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryVals[0];
			dy = currNode->Y() - queryVals[1];
			diffDist2 = (dx*dx) + (dy*dy);

			// Update closest point Idx
			if (diffDist2 < best.Dist)
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the Best node Index 
		//         but we really need the best point ID 
		//		   so grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfID = 0;
		float bfDist = 3.0e+38F;
		bTest = BF_FindNN_2D( queryPoint, bfID, bfDist );
		if ((! bTest) || (bfID != best.Id))
		{
			if (best.dist != bfDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN search returned a different result than Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
		queryResults[currQuery] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_NN_2D
  Desc:	Finds closest point in kd-tree for point in kd-tree
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Find_ALL_NN_2D
( 
	CPU_NN_Result * queryResults	// OUT: Results
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_2D_LBT * currNode  = NULL;
	CPUNode_2D_LBT * queryNode = NULL;
	CPU_NN_Result best;
	unsigned int currIdx, currInOut;
	unsigned int currAxis, nextAxis, prevAxis;
	unsigned int left, right;
	float dx, dy;
	float queryVals[2];
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int currFlags;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	unsigned int currQuery;
	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Reset top of stack for each query
		stackTop = 0;
		
		// Compute Query Index

		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->X();
		queryVals[1] = queryNode->Y();

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & 0x1FFFFFFFU);
			currAxis  = (currFlags & 0x60000000U) >> 29;
			currInOut = (currFlags & 0x80000000U) >> 31;
			
			nextAxis  = ((currAxis == 1u) ? 0u : 1u);
			prevAxis  = ((currAxis == 0u) ? 1u : 0u);

			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal( currAxis );
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryVals[0];
			dy = currNode->Y() - queryVals[1];
			diffDist2 = (dx*dx) + (dy*dy);

			// Update closest point Idx
			if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the Best node Index 
		//         but we really need the best point ID 
		//         so grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfID = 0;
		float bfDist = 3.0e+38F;
		bTest = BF_FindNN_2D( queryPoint, bfID, bfDist );
		if ((! bTest) || (bfID != best.Id))
		{
			if (best.dist != bfDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN search returned a different result than Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
			// Convert query node index into original query point index
		unsigned int outIdx = queryNode->SearchID();
			// Store result at query point index
		queryResults[outIdx] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_KNN_2D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Find_KNN_2D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nQueries,		// IN: Number of Query points
		unsigned int      nPadQueries,	// IN: Number of Padded queries
		const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_2D_LBT * currNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy;
	float queryVals[2];
	float queryValue, splitValue;
	unsigned int maxHeap, countHeap;
	float dist2Heap, bestDist2;

	KD_SEARCH_LBT		searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Get current Query Point
		queryVals[0] = queryPoints[currQuery].x;
		queryVals[1] = queryPoints[currQuery].y;

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & 0x1FFFFFFFu;
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 1u) ? 0u : 1u);
			prevAxis  = ((currAxis == 0u) ? 1u : 0u);

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal( currAxis );
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryVals[0];
			dy = currNode->Y() - queryVals[1];
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
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]     = knnHeap[childHIdx];
							knnHeap[childHIdx]      = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		= knnHeap[childHIdx];
					knnHeap[childHIdx]		= closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		// We now have a heap of the 'k' nearest neighbors
		// Write heap elements to the results array row by row	
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			unsigned int offset = (i-1) * nQueries;

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node index
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[currQuery+offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_KNN_2D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Find_ALL_KNN_2D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int    kVal,			// In: 'k' nearest neighbors to search for
		unsigned int    nSearch,		// IN: Number of search points (query points)
		unsigned int    nPadSearch		// IN: number of padded search points (query points)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx   = 1;
	unsigned int stackTop  = 0;
	CPUNode_2D_LBT * currNode  = NULL;
	CPUNode_2D_LBT * queryNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy;
	float queryVals[2];
	float queryValue, splitValue;
	unsigned int maxHeap, countHeap;
	float dist2Heap, bestDist2;

	KD_SEARCH_LBT		searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->X();
		queryVals[1] = queryNode->Y();

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & 0x1FFFFFFFu;
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 1u) ? 0u : 1u);
			prevAxis  = ((currAxis == 0u) ? 1u : 0u);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal( currAxis );
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryVals[0];
			dy = currNode->Y() - queryVals[1];
			diffDist2 = (dx*dx) + (dy*dy);

			// See if we should add this point to 'k' NN Heap
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
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]     = knnHeap[childHIdx];
							knnHeap[childHIdx]      = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		= knnHeap[childHIdx];
					knnHeap[childHIdx]		= closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		unsigned int offset, outIdx;

		// Store Query Result
			// Convert query node index into original query point index
		outIdx = queryNode->SearchID();

		// We now have a heap of 'k' nearest neighbors
		// Write them to the results array
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			offset = (i-1) * nSearch;			// -1 is to ignore zeroth element in heap which is not used

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node indices
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[outIdx + offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::DumpNodes
  Desc:	Dumps kd-tree in height order
-------------------------------------------------------------------------*/

void CPUTree_2D_LBT::DumpNodes() const
{
	printf( "\nKDTree_LBT, count = %d { \n\n", m_cNodes );

	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_2D_LBT & currNode = m_nodes[i];
		double x = (double)(currNode.X());
		double y = (double)(currNode.Y());
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//printf( "[%d] = <x=%3.6f, y=%3.6f, S=%d, N=%d>\n",
		//		i, x, y, sID, nID );
		
		printf( "%d, %3.6f, %3.6f, %d, %d\n", i, x, y, sID, nID );
	}

	printf( "\n} \n\n", m_cNodes );
}

/*-------------------------------------------------------------------------
  Name:	KDTree::DumpBuildStats
  Desc:	Dumps stats from building kd-tree
-------------------------------------------------------------------------*/

#ifdef _BUILD_STATS
void CPUTree_2D_LBT::DumpBuildStats() const
{
	printf( "\nKDTree_2D_LBT, Stats { \n\n" );
	printf( "\tNode Loops       = %u\n", m_cpuStats.cNodeLoops   );
	printf( "\tPartition Loops  = %u\n", m_cpuStats.cPartLoops   );
	printf( "\tStore Reads      = %u\n", m_cpuStats.cStoreReads  );
	printf( "\tStore Writes     = %u\n", m_cpuStats.cStoreWrites );
	printf( "\tPivot Reads      = %u\n", m_cpuStats.cPivotReads  );
	printf( "\tPivot Swaps      = %u\n", m_cpuStats.cPivotSwaps  );
	printf( "\tPivot Writes     = %u\n", m_cpuStats.cPivotWrites );
	printf( "\tPartition Reads  = %u\n", m_cpuStats.cPartReads  );
	printf( "\tPartition Swaps  = %u\n", m_cpuStats.cPartSwaps  );
	printf( "\tPartition Writes = %u\n", m_cpuStats.cPartWrites );
	printf( "\n} \n\n", m_cNodes );
}
#endif


/*-------------------------------------------------------------------------
  Name:	KDTree::Validate
  Desc:	Validate that the nodes in the kd-tree are actually
        in left-balanced array order
-------------------------------------------------------------------------*/

bool CPUTree_2D_LBT::Validate() const
{
	bool fResult = true;

	unsigned int i;
	float    currVals[2];
	float    leftVals[2];
	float    rightVals[2];
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_2D_LBT & currNode = m_nodes[i];
		currVals[0] = currNode.X();
		currVals[1] = currNode.Y();
		unsigned int currAxis = currNode.Axis();
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//unsigned int parent = nID >> 1;
		unsigned int left   = nID << 1;
		unsigned int right  = left + 1;		

		// Check left child
		if (left <= m_cNodes)
		{
			// Make sure left child is actually to the left of this current node
			leftVals[0] = m_nodes[left].X();
			leftVals[1] = m_nodes[left].Y();

			float fLeft = leftVals[currAxis];
			float fCurr = currVals[currAxis];

			if (fLeft > fCurr)
			{
				printf( "<%d,%3.6f> is not to left of it's parent <%d,%3.6f> !!!\n",
						left, (double)fLeft, nID, (double)fCurr );
				fResult = false;
			}
		}

		// Check right child
		if (right <= m_cNodes)
		{
			// Make sure left child is actually to the left of this current node
			rightVals[0] = m_nodes[right].X();
			rightVals[1] = m_nodes[right].Y();

			float fRight = rightVals[currAxis];
			float fCurr  = currVals[currAxis];

			if (fRight < fCurr)
			{
				printf( "<%d,%3.6f> is not to right of it's parent <%d,%3.6f> !!!\n",
						right, (double)fRight, nID, (double)fCurr );

				fResult = false;
			}
		}
	}

	return fResult;
}


/*-------------------------------------
  CPUTree_3D_LBT Methods Definitions
-------------------------------------*/

/*-------------------------------------------------------------------------
  Name:	CPUTree_3D_LBT::GetNodeAxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

float CPUTree_3D_LBT::GetNodeAxisValue
( 
	const CPUNode_3D_LBT * currNodes,	// IN:  IN node list
	unsigned int index,				// IN:  Index of node to retrieve value for
	unsigned int axis				// IN:  axis of value to retrieve
) const
{
	const CPUNode_3D_LBT & currNode = currNodes[index];
	float axisValue = currNode[axis];
	return axisValue;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_3D_LBT::SwapNodes
  Desc:	Helper Method
-------------------------------------------------------------------------*/

void CPUTree_3D_LBT::SwapNodes
( 
	CPUNode_3D_LBT * currNodes,	// IN: Node list
	unsigned int idx1,			// IN: Index of 1st node to swap
	unsigned int idx2			// IN: Index of 2nd node to swap
)
{
	CPUNode_3D_LBT & currNode1 = currNodes[idx1];
	CPUNode_3D_LBT & currNode2 = currNodes[idx2];
	CPUNode_3D_LBT temp;

	temp	  = currNode1;
	currNode1 = currNode2;
	currNode2 = temp;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_3D_LBT::MedianOf3
  Desc:	Helper method,
		Implements Median of three variant 
		for Median Partitioning algorithm
		returns pivot value for partitioning algorithm
  Note:	enforces invariant
		array[left].val <= array[mid].val <= array[right].val
		where mid = (left+right)/2
-------------------------------------------------------------------------*/

void CPUTree_3D_LBT::MedianOf3
(
	CPUNode_3D_LBT * currNodes,	// IN - node list
	unsigned int leftIdx,		// IN - left index
	unsigned int rightIdx,		// IN - right index
	unsigned int axis			// IN - axis to compare
)
{
	// Compute Middle Index from left and right
	unsigned int midIdx = (leftIdx+rightIdx)/2;	

	float leftVal  = GetNodeAxisValue( currNodes, leftIdx, axis );
	float rightVal = GetNodeAxisValue( currNodes, rightIdx, axis );
	float midVal   = GetNodeAxisValue( currNodes, midIdx, axis );

	// Sort left, center, mid value into correct order
	if (leftVal > midVal)
	{
		SwapNodes( currNodes, leftIdx, midIdx );
	}
	if (leftVal > rightVal)
	{
		SwapNodes( currNodes, leftIdx, rightIdx );
	}
	if (midVal > rightVal)
	{
		SwapNodes( currNodes, midIdx, rightIdx );
	}

	// Deliberately move median value to end of array
	SwapNodes( currNodes, midIdx, rightIdx );
}


/*-------------------------------------------------------------------------
  Name: CPUTree_3D_LBT::MedianSortNodes
  Desc:	Partitions original data set {O} with 'n' elements
		into 3 datasets <{l}, {m}, {r}}

		where {l} contains approximately half the elements 
		and all elements are less than or equal to {m}
		where {m} contains only 1 element the "median value"
		and {r} contains approximately half the elements
		and all elements are greater than or equal to {m}

  Notes:
	Invariant:
		{l} = { all points with value less or equal to median }
		    count({l}) = (n - 1) / 2
		{m} = { median point }
		    count({m}) = 1
		{r} = { points with value greater than or equal to median point }
		    count({r}) = n - ((n-1)/2 + 1)
			Note: Either count({r}) = count({l}) or
			             count({r}) = count({l})+1
				  IE the count of sets {l} and {r} will be 
				     within one element of each other

	Performance:
		Should take O(N) time to process all points in input range

  Credit:	Based on the "Selection Sort" algorithm as
            presented by Robert Sedgewick in the book
            "Algorithms in C++"
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::MedianSortNodes
(
	CPUNode_3D_LBT * currNodes,	// IN - node list
	unsigned int start,     // IN - start of range
	unsigned int end,       // IN - end of range
	unsigned int & median,  // IN/OUT - approximate median number
	                        //          actual median number
	unsigned int axis       // IN - dimension(axis) to split along (x,y,z)
)
{
	if (start > end)
	{
		// Swap Start and End, if they are in the wrong order
		unsigned int temp = start;
		start = end;
		end = temp;
	}

	// Check Parameters
	unsigned int nNodes = (end - start) + 1;
	if (nNodes == 0) { return false; }
	if (nNodes == 1) { return true; }
	if (axis >= INVALID_AXIS) { return false; }
	if ((median < start) || (median > end)) { return false; }

	// Perform Median Sort
	int left   = static_cast<int>( start );
	int right  = static_cast<int>( end );
	int middle = static_cast<int>( median );
	int i,j;
	float pivotVal;

	while ( right > left ) 
	{
		// Get Pivot value
			// Use Median of 3 variant
		MedianOf3( currNodes, left, right, axis );
		pivotVal = GetNodeAxisValue( currNodes, right, axis );

		i = left - 1;
		j = right;

		// Partition into 3 sets
			// Left   = {start, pivot-1}	all values in Left <= median
			// Median = {pivot}				singleton containing pivot value
			// Right  = {pivot+1, end}		all values in right >= median
		for (;;) 
		{
			while ( GetNodeAxisValue( currNodes, ++i, axis ) < pivotVal )
			{
				// Deliberately do nothing
			}

			while ( (GetNodeAxisValue( currNodes, --j, axis ) > pivotVal) && 
				  (j > left) )
			{
				// Deliberately do nothing
			}
			
			if ( i >= j )
				break;

			SwapNodes( currNodes, i, j );
		}

		// Put pivot value back into pivot position
		SwapNodes( currNodes, i, right );

		// Iterate into left or right set until
		// we find the value at the true median
		if ( i >= middle )
			right = i-1;
		if ( i <= middle )
			left = i+1;
	}

	// return new median index
	median = middle;

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_3D_LBT::ComputeBoundBox
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::ComputeBoundingBox
( 
	unsigned int start,		// IN - start index
	unsigned int end,		// IN - end index
	float        bounds[6]	// OUT - bounding box for all nodes in range
)
{
	// Check Parameters
	if (CPUNode_3D_LBT::c_Invalid == start) { return false; }
	if (CPUNode_3D_LBT::c_Invalid == end) { return false; }

	unsigned int s = start;
	unsigned int e = end;
	if (e < s) 
	{
		unsigned int temp = s;
		s = e;
		e = temp;
	}

	CPUNode_3D_LBT * currNode = NODE_PTR( s );
	if (NULL == currNode) { return false; }
	
	float x, y, z;

	x = currNode->X();
	y = currNode->Y();
	z = currNode->Z();

	bounds[0] = x;
	bounds[1] = x;
	bounds[2] = y;
	bounds[3] = y;
	bounds[4] = z;
	bounds[5] = z;

	unsigned int i;
	for (i = s+1; i <= e; i++)
	{
		currNode = NODE_PTR( i );
		x = currNode->X();
		y = currNode->Y();
		z = currNode->Z();

		// Update Min, Max for X,Y,Z
		if (x < bounds[0]) { bounds[0] = x; }
		if (x > bounds[1]) { bounds[1] = x; }
		
		if (y < bounds[2]) { bounds[2] = y; }
		if (y > bounds[3]) { bounds[3] = y; }

		if (z < bounds[4]) { bounds[4] = z; }
		if (z > bounds[5]) { bounds[5] = z; }
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build3D
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::Build3D( unsigned int cPoints, const float3 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_3D_LBT * mNodes = new CPUNode_3D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_3D_LBT * lNodes = new CPUNode_3D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );
	mNodes[0].Z( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_3D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		currNode.X( x );
		currNode.Y( y );
		currNode.Z( z );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_3D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );
	lNodes[0].Z( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_3D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_3D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_3D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[stackTop];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29u) & 0x3u);

		currN      = currEnd-currStart + 1u;
		nextAxis   = ((currAxis == 2u) ? 0u : (currAxis+1));

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_3D_LBT & medianNode = mNodes[currMedian];
		CPUNode_3D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		float currZ			= medianNode.Z();
		unsigned int currID = medianNode.SearchID();

		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.Z( currZ );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1u;
		currRight = currLeft + 1u;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1u;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1u) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1u;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1u) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}

bool CPUTree_3D_LBT::Build3D( unsigned int cPoints, const float4 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_3D_LBT * mNodes = new CPUNode_3D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_3D_LBT * lNodes = new CPUNode_3D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;

	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0 );	// Have it map back onto itself
	mNodes[0].NodeID( 0 );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );
	mNodes[0].Z( 0.0f );

		// Start at index 1
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		CPUNode_3D_LBT & currNode = mNodes[i+1];
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		currNode.X( x );
		currNode.Y( y );
		currNode.Z( z );
		currNode.SearchID( i );
		currNode.NodeID( CPUNode_3D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );
	lNodes[0].Z( 0.0f );

	for (i = 1; i <= cNodes; i++)
	{
		CPUNode_3D_LBT & currNode = lNodes[i];
		currNode.NodeID( i );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_3D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_3D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[stackTop];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29u) & 0x3u);

		currN      = currEnd-currStart + 1u;
		nextAxis   = ((currAxis == 2u) ? 0u : (currAxis+1));

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_3D_LBT & medianNode = mNodes[currMedian];
		CPUNode_3D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		float currZ			= medianNode.Z();
		unsigned int currID = medianNode.SearchID();

		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.Z( currZ );
		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1u;
		currRight = currLeft + 1u;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1u;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1u) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1u;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1u) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_3D_LBT::BF_FindNN_3D()
  Desc:	Use Brute Force Algorithm to find closest Node (index)
			Takes O(N) time
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::BF_FindNN_3D
(
	const float4 & queryLocation,	// IN  - Location to sample
	unsigned int & nearestID,		// OUT - ID of nearest point
	float & nearestDistance			// OUT - distance to nearest point
)
{
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	// Get Query Point
	float qX, qY, qZ;
	qX = queryLocation.x;
	qY = queryLocation.y;
	qZ = queryLocation.z;

	// Get 1st Point
	unsigned int  bestIndex  = 1;
	CPUNode_3D_LBT * currNodePtr = NULL;
	CPUNode_3D_LBT * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID      = bestNodePtr->SearchID();

	float bX, bY, bZ;
	bX = bestNodePtr->X();
	bY = bestNodePtr->Y();
	bZ = bestNodePtr->Z();

	// Calculate distance from query location
	float dX = bX - qX;
	float dY = bY - qY;
	float dZ = bZ - qZ;
	float bestDist2 = dX*dX + dY*dY + dZ*dZ;
	float diffDist2;

	unsigned int i;
	for (i = 2; i <= nNodes; i++)
	{
		// Get Current Point
		CPUNode_3D_LBT & currNode = m_nodes[i];
		bX = currNode.X();
		bY = currNode.Y();
		bZ = currNode.Z();

		// Calculate Distance from query location
		dX = bX - qX;
		dY = bY - qY;
		dZ = bZ - qZ;
		diffDist2 = dX*dX + dY*dY + dZ*dZ;

		// Update Best Point Index
		if (diffDist2 < bestDist2)
		{
			bestIndex = i;
			bestID    = currNode.SearchID();
			bestDist2 = diffDist2;
		}
	}

	//Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

	// Success (return results)
	nearestID       = bestID;
	nearestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_QNN_3D
  Desc:	Finds closest point in kd-tree for each query point
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::Find_QNN_3D
( 
	CPU_NN_Result * queryResults,	// OUT: Results
	unsigned int      nQueries,		// IN: Number of Query points
	const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }


	// Local Parameters
	CPU_NN_Result best;
	CPUNode_3D_LBT * currNode = NULL;
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float dx, dy, dz;
	float queryVals[3];
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	unsigned int currFlags, currQuery;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Reset stack
		stackTop = 0;		

		// Load current Query Point into local (fast) memory
		queryVals[0] = queryPoints[currQuery].x;
		queryVals[1] = queryPoints[currQuery].y;
		queryVals[2] = queryPoints[currQuery].z;

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = (rootIdx & 0x1FFFFFFFu); // | ((currAxis << 29u) & 0x60000000u); // | ((currInOut << 31u) & 0x8000000u);
		searchStack[stackTop].splitValue = 3.0e+38f;
		stackTop++;

		while (stackTop != 0u)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & 0x1FFFFFFFu);
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			// Get next & prev axis
			nextAxis  = ((currAxis == 2u) ? 0u : (currAxis + 1u));
			prevAxis  = ((currAxis == 0u) ? 2u : (currAxis - 1u));

			// Get left and right child positions
			//parent  = currIdx >> 1u;
			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryVals[0];
			dy = currNode->Y() - queryVals[1];
			dz = currNode->Z() - queryVals[2];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

			// Update closest point Idx
			if (diffDist2 < best.Dist)
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFu) | ((nextAxis << 29u) & 0x60000000u) | 0x80000000u;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFu) | ((nextAxis << 29u) & 0x60000000u); // | 0x80000000u;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFu) | ((nextAxis << 29u) & 0x60000000u) | 0x80000000u;;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFu) | ((nextAxis << 29u) & 0x60000000u); // | 0x8000000u
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP: We now have the Best node Index 
		//        but we really need the best point ID 
		//		  so grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfID = 0;
		float bfDist = 3.0e+38F;
		bTest = BF_FindNN_2D( queryPoint, bfID, bfDist );
		if ((! bTest) || (bfID != best.id))
		{
			if (best.dist != bfDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN search returned a different result than Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
		queryResults[currQuery] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_NN_3D
  Desc:	Finds closest point in kd-tree for point in kd-tree
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::Find_ALL_NN_3D
( 
	CPU_NN_Result * queryResults	// OUT: Results
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	// Local Parameters
	CPU_NN_Result best;
	CPUNode_3D_LBT * currNode  = NULL;
	CPUNode_3D_LBT * queryNode = NULL;
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float queryVals[3];
	float dx, dy, dz;
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	unsigned int currFlags, currQuery;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Reset top of stack for each query
		stackTop = 0;
		
		// Compute Query Index

		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->GetVal(0);
		queryVals[1] = queryNode->GetVal(1);
		queryVals[2] = queryNode->GetVal(2);

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & 0x1FFFFFFFU);
			currAxis  = (currFlags & 0x60000000U) >> 29;
			currInOut = (currFlags & 0x80000000U) >> 31;
			
			// Get next & prev axis
			nextAxis  = (currAxis == 2u) ? 0u : currAxis + 1;
			prevAxis  = (currAxis == 0u) ? 2u : currAxis - 1;

			// Get left and right child positions
			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

			// Update closest point Idx
			if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// We now have the Best Index but we really need the best ID so grab it from ID list
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfID = 0;
		float bfDist = 3.0e+38F;
		bTest = BF_FindNN_2D( queryPoint, bfID, bfDist );
		if ((! bTest) || (bfID != best.Id))
		{
			if (best.Dist != bfDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN search returned a different result than Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
			// Convert query node index into original query point index
		unsigned int outIdx = queryNode->SearchID();
			// Store results at query point index
		queryResults[outIdx] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_KNN_3D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::Find_KNN_3D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nQueries,		// IN: Number of Query points
		unsigned int      nPadQueries,	// IN: Number of Padded queries
		const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_3D_LBT * currNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz;
	float queryVals[3];
	float queryValue, splitValue;
	unsigned int maxHeap, countHeap;
	float dist2Heap, bestDist2;

	KD_SEARCH_LBT		searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Get current Query Point
		queryVals[0] = queryPoints[currQuery].x;
		queryVals[1] = queryPoints[currQuery].y;
		queryVals[2] = queryPoints[currQuery].z;

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & 0x1FFFFFFFu;
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;

			//parent = currIdx >> 1u;
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 2u) ? 0u : currAxis+1);
			prevAxis  = ((currAxis == 0u) ? 2u : currAxis-1);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

			// See if we should add this point to 'k' NN Heap
			if (countHeap < maxHeap)
			{
				//-------------------------------
				//	< 'k' elements on heap
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]       = knnHeap[childHIdx];
							knnHeap[childHIdx]        = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		  = knnHeap[childHIdx];
					knnHeap[childHIdx]		  = closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		// We now have a heap of the 'k' nearest neighbors
		// Write heap elements to the results array row by row	
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			unsigned int offset = (i-1) * nQueries;

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node index
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[currQuery+offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_KNN_3D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::Find_ALL_KNN_3D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: number of padded search points (query points)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_3D_LBT * currNode  = NULL;
	CPUNode_3D_LBT * queryNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, prevAxis, nextAxis;
	float dx, dy, dz;
	float queryVals[3];
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	unsigned int maxHeap, countHeap;
	float dist2Heap, bestDist2;
	KD_SEARCH_LBT		searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->X();
		queryVals[1] = queryNode->Y();
		queryVals[2] = queryNode->Z();

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & 0x1FFFFFFFu;
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 2u) ? 0u : currAxis+1u);
			prevAxis  = ((currAxis == 0u) ? 2u : currAxis-1u);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					// Next Line is effectively queryValue = queryPoints[prevAxis];
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal( 0u ) - queryVals[0u];
			dy = currNode->GetVal( 1u ) - queryVals[1u];
			dz = currNode->GetVal( 2u ) - queryVals[2u];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz);

			// See if we should add this point to 'k' NN Heap
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
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]       = knnHeap[childHIdx];
							knnHeap[childHIdx]        = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		  = knnHeap[childHIdx];
					knnHeap[childHIdx]		  = closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		unsigned int offset, outIdx;

		// Store Query Result
			// Convert query node index into original query point index
		outIdx = queryNode->SearchID();

		// We now have a heap of 'k' nearest neighbors
		// Write them to the results array
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			offset = (i-1) * nSearch;			// -1 is to ignore zeroth element in heap which is not used

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node indices
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[outIdx + offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::DumpNodes
  Desc:	Dumps kd-tree in height order
-------------------------------------------------------------------------*/

void CPUTree_3D_LBT::DumpNodes() const
{
	printf( "\nKDTree_LBT, count = %d { \n\n", m_cNodes );

	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_3D_LBT & currNode = m_nodes[i];
		double x = (double)(currNode.X());
		double y = (double)(currNode.Y());
		double z = (double)(currNode.Z());
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//printf( "[%d] = <x=%3.6f, y=%3.6f, S=%d, N=%d>\n",
		//		i, x, y, sID, nID );
		
		printf( "%d, %3.6f, %3.6f, %3.6f, %d, %d\n", i, x, y, z, sID, nID );
	}

	printf( "\n} \n\n", m_cNodes );
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Validate
  Desc:	Validate that the nodes in the kd-tree are actually
        in left-balanced array order
-------------------------------------------------------------------------*/

bool CPUTree_3D_LBT::Validate() const
{
	bool fResult = true;

	float    currVals[4];
	float    leftVals[4];
	float    rightVals[4];
	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_3D_LBT & currNode = m_nodes[i];
		currVals[0] = currNode.X();
		currVals[1] = currNode.Y();
		currVals[2] = currNode.Z();
		unsigned int currAxis = currNode.Axis();
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//unsigned int parent = nID >> 1;
		unsigned int left   = nID << 1;
		unsigned int right  = left + 1;		

		// Check left child
		if (left <= m_cNodes)
		{
			// Make sure left child is actually to the left of this current node
			leftVals[0] = m_nodes[left].X();
			leftVals[1] = m_nodes[left].Y();
			leftVals[2] = m_nodes[left].Z();

			float fLeft = leftVals[currAxis];
			float fCurr = currVals[currAxis];

			if (fLeft > fCurr)
			{
				printf( "<%d,%3.6f> is not to left of it's parent <%d,%3.6f> !!!\n",
						left, (double)fLeft, nID, (double)fCurr );
				fResult = false;
			}
		}

		// Check right child
		if (right <= m_cNodes)
		{
			// Make sure right child is actually to the right of this current node
			rightVals[0] = m_nodes[right].X();
			rightVals[1] = m_nodes[right].Y();
			rightVals[2] = m_nodes[right].Z();

			float fRight = rightVals[currAxis];
			float fCurr  = currVals[currAxis];

			if (fRight < fCurr)
			{
				printf( "<%d,%3.6f> is not to right of it's parent <%d,%3.6f> !!!\n",
						right, (double)fRight, nID, (double)fCurr );

				fResult = false;
			}
		}
	}

	return fResult;
}


/*-------------------------------------
  CPUTree_4D_LBT Methods Definitions
-------------------------------------*/

/*-------------------------------------------------------------------------
  Name:	CPUTree_4D_LBT::GetNodeAxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

float CPUTree_4D_LBT::GetNodeAxisValue
( 
	const CPUNode_4D_LBT * currNodes,	// IN:  IN node list
	unsigned int index,				// IN:  Index of node to retrieve value for
	unsigned int axis				// IN:  axis of value to retrieve
) const
{
	const CPUNode_4D_LBT & currNode = currNodes[index];
	float axisValue = currNode[axis];
	return axisValue;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_4D_LBT::SwapNodes
  Desc:	Helper Method
-------------------------------------------------------------------------*/

void CPUTree_4D_LBT::SwapNodes
( 
	CPUNode_4D_LBT * currNodes,	// IN: Node list
	unsigned int idx1,			// IN: Index of 1st node to swap
	unsigned int idx2			// IN: Index of 2nd node to swap
)
{
	CPUNode_4D_LBT & currNode1 = currNodes[idx1];
	CPUNode_4D_LBT & currNode2 = currNodes[idx2];
	CPUNode_4D_LBT temp;

	temp	  = currNode1;
	currNode1 = currNode2;
	currNode2 = temp;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_4D_LBT::MedianOf3
  Desc:	Helper method,
		Implements Median of three variant 
		for Median Partitioning algorithm
		returns pivot value for partitioning algorithm
  Note:	enforces invariant
		array[left].val <= array[mid].val <= array[right].val
		where mid = (left+right)/2
-------------------------------------------------------------------------*/

void CPUTree_4D_LBT::MedianOf3
(
	CPUNode_4D_LBT * currNodes,	// IN - node list
	unsigned int leftIdx,		// IN - left index
	unsigned int rightIdx,		// IN - right index
	unsigned int axis			// IN - axis to compare
)
{
	// Compute Middle Index from left and right
	unsigned int midIdx = (leftIdx+rightIdx)/2;	

	float leftVal  = GetNodeAxisValue( currNodes, leftIdx, axis );
	float rightVal = GetNodeAxisValue( currNodes, rightIdx, axis );
	float midVal   = GetNodeAxisValue( currNodes, midIdx, axis );

	// Sort left, center, mid value into correct order
	if (leftVal > midVal)
	{
		SwapNodes( currNodes, leftIdx, midIdx );
	}
	if (leftVal > rightVal)
	{
		SwapNodes( currNodes, leftIdx, rightIdx );
	}
	if (midVal > rightVal)
	{
		SwapNodes( currNodes, midIdx, rightIdx );
	}

	// Deliberately move median value to end of array
	SwapNodes( currNodes, midIdx, rightIdx );
}


/*-------------------------------------------------------------------------
  Name: CPUTree_4D_LBT::MedianSortNodes
  Desc:	Partitions original data set {O} with 'n' elements
		into 3 datasets <{l}, {m}, {r}}

		where {l} contains approximately half the elements 
		and all elements are less than or equal to {m}
		where {m} contains only 1 element the "median value"
		and {r} contains approximately half the elements
		and all elements are greater than or equal to {m}

  Notes:
	Invariant:
		{l} = { all points with value less or equal to median }
		    count({l}) = (n - 1) / 2
		{m} = { median point }
		    count({m}) = 1
		{r} = { points with value greater than or equal to median point }
		    count({r}) = n - ((n-1)/2 + 1)
			Note: Either count({r}) = count({l}) or
			             count({r}) = count({l})+1
				  IE the count of sets {l} and {r} will be 
				     within one element of each other

	Performance:
		Should take O(N) time to process all points in input range

  Credit:	Based on the "Selection Sort" algorithm as
            presented by Robert Sedgewick in the book
            "Algorithms in C++"
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::MedianSortNodes
(
	CPUNode_4D_LBT * currNodes,	// IN - node list
	unsigned int start,     // IN - start of range
	unsigned int end,       // IN - end of range
	unsigned int & median,  // IN/OUT - approximate median number
	                        //          actual median number
	unsigned int axis       // IN - dimension(axis) to split along (x,y,z)
)
{
	if (start > end)
	{
		// Swap Start and End, if they are in the wrong order
		unsigned int temp = start;
		start = end;
		end = temp;
	}

	// Check Parameters
	unsigned int nNodes = (end - start) + 1;
	if (nNodes == 0) { return false; }
	if (nNodes == 1) { return true; }
	if (axis >= INVALID_AXIS) { return false; }
	if ((median < start) || (median > end)) { return false; }

	// Perform Median Sort
	int left   = static_cast<int>( start );
	int right  = static_cast<int>( end );
	int middle = static_cast<int>( median );
	int i,j;
	float pivotVal;

	while ( right > left ) 
	{
		// Get Pivot value
			// Use Median of 3 variant
		MedianOf3( currNodes, left, right, axis );
		pivotVal = GetNodeAxisValue( currNodes, right, axis );

		i = left - 1;
		j = right;

		// Partition into 3 sets
			// Left   = {start, pivot-1}	all values in Left <= median
			// Median = {pivot}				singleton containing pivot value
			// Right  = {pivot+1, end}		all values in right >= median
		for (;;) 
		{
			while ( GetNodeAxisValue( currNodes, ++i, axis ) < pivotVal )
			{
				// Deliberately do nothing
			}

			while ( (GetNodeAxisValue( currNodes, --j, axis ) > pivotVal) && 
				  (j > left) )
			{
				// Deliberately do nothing
			}
			
			if ( i >= j )
				break;

			SwapNodes( currNodes, i, j );
		}

		// Put pivot value back into pivot position
		SwapNodes( currNodes, i, right );

		// Iterate into left or right set until
		// we find the value at the true median
		if ( i >= middle )
			right = i-1;
		if ( i <= middle )
			left = i+1;
	}

	// return new median index
	median = middle;

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_4D_LBT::ComputeBoundBox
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::ComputeBoundingBox
( 
	unsigned int start,		// IN - start index
	unsigned int end,		// IN - end index
	float        bounds[8]	// OUT - bounding box for all nodes in range
)
{
	// Check Parameters
	if (CPUNode_4D_LBT::c_Invalid == start) { return false; }
	if (CPUNode_4D_LBT::c_Invalid == end) { return false; }

	unsigned int s = start;
	unsigned int e = end;
	if (e < s) 
	{
		unsigned int temp = s;
		s = e;
		e = temp;
	}

	CPUNode_4D_LBT * currNode = NODE_PTR( s );
	if (NULL == currNode) { return false; }
	
	float x, y, z, w;

	x = currNode->X();
	y = currNode->Y();
	z = currNode->Z();
	w = currNode->W();

	bounds[0] = x;
	bounds[1] = x;
	bounds[2] = y;
	bounds[3] = y;
	bounds[4] = z;
	bounds[5] = z;
	bounds[6] = w;
	bounds[7] = w;

	unsigned int i;
	for (i = s+1; i <= e; i++)
	{
		currNode = NODE_PTR( i );
		x = currNode->X();
		y = currNode->Y();
		z = currNode->Z();
		w = currNode->W();

		// Update Min, Max for X,Y,Z,W
		if (x < bounds[0]) { bounds[0] = x; }
		if (x > bounds[1]) { bounds[1] = x; }
		
		if (y < bounds[2]) { bounds[2] = y; }
		if (y > bounds[3]) { bounds[3] = y; }
		
		if (z < bounds[4]) { bounds[4] = z; }
		if (z > bounds[5]) { bounds[5] = z; }

		if (w < bounds[6]) { bounds[6] = w; }
		if (w > bounds[7]) { bounds[7] = w; }
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build4D
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::Build4D( unsigned int cPoints, const float4 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_4D_LBT * mNodes = new CPUNode_4D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_4D_LBT * lNodes = new CPUNode_4D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;


	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	mNodes[0].SearchID( 0u );	// Have it map back onto itself
	mNodes[0].NodeID( 0u );
	mNodes[0].X( 0.0f );
	mNodes[0].Y( 0.0f );
	mNodes[0].Z( 0.0f );
	mNodes[0].W( 0.0f );

	// Copy points into nodes
		// Point indices are in range [0,n-1]
		// Node Indices are in range [1, n]
	unsigned int pntIdx, nodeIdx;
	float x,y,z,w;
	for (pntIdx = 0; pntIdx < cNodes; pntIdx++)
	{
		nodeIdx = pntIdx + 1;
		CPUNode_4D_LBT & currNode = mNodes[nodeIdx];

		x = pointList[pntIdx].x;
		y = pointList[pntIdx].y;
		z = pointList[pntIdx].z;
		w = pointList[pntIdx].w;

		currNode.X( x );
		currNode.Y( y );
		currNode.Z( z );
		currNode.W( w );
		currNode.SearchID( pntIdx );
		currNode.NodeID( CPUNode_4D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/

	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );
	lNodes[0].X( 0.0f );
	lNodes[0].Y( 0.0f );
	lNodes[0].Z( 0.0f );
	lNodes[0].W( 0.0f );

	for (nodeIdx = 1; nodeIdx <= cNodes; nodeIdx++)
	{
		CPUNode_4D_LBT & currNode = lNodes[nodeIdx];
		currNode.NodeID( nodeIdx );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_4D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_4D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[stackTop];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & 0x1FFFFFFFu) | ((currAxis & 0x03u) << 29u));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags >> 29) & 0x3u);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 3u) ? 0u : (currAxis+1));

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & 0x1FFFFFFFu); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_4D_LBT & medianNode = mNodes[currMedian];
		CPUNode_4D_LBT & targetNode = lNodes[currTarget];

		float currX         = medianNode.X();
		float currY         = medianNode.Y();
		float currZ			= medianNode.Z();
		float currW         = medianNode.W();
		unsigned int currID = medianNode.SearchID();

		targetNode.X( currX );
		targetNode.Y( currY );
		targetNode.Z( currZ );
		targetNode.W( currW );

		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & 0x1FFFFFFFu) | ((nextAxis & 0x3u) << 29u);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_4D_LBT::BF_FindNN_4D()
  Desc:	Use Brute Force Algorithm to find closest Node (index)
			Takes O(N) time
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::BF_FindNN_4D
(
	const float4 & queryLocation,	// IN  - Location to sample
	unsigned int & nearestID,		// OUT - ID of nearest point
	float & nearestDistance			// OUT - distance to nearest point
)
{
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	// Get Query Point
	float qX, qY, qZ, qW;
	qX = queryLocation.x;
	qY = queryLocation.y;
	qZ = queryLocation.z;
	qW = queryLocation.w;

	// Get 1st Point
	unsigned int  bestIndex  = 1;
	CPUNode_4D_LBT * currNodePtr = NULL;
	CPUNode_4D_LBT * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID      = bestNodePtr->SearchID();

	float bX, bY, bZ, bW;
	bX = bestNodePtr->X();
	bY = bestNodePtr->Y();
	bZ = bestNodePtr->Z();
	bW = bestNodePtr->W();

	// Calculate distance from query location
	float dX = bX - qX;
	float dY = bY - qY;
	float dZ = bZ - qZ;
	float dW = bW - qW;
	float bestDist2 = dX*dX + dY*dY + dZ*dZ + dW*dW;
	float diffDist2;

	unsigned int i;
	for (i = 2; i <= nNodes; i++)
	{
		// Get Current Point
		CPUNode_4D_LBT & currNode = m_nodes[i];
		bX = currNode.X();
		bY = currNode.Y();
		bZ = currNode.Z();
		bW = currNode.W();

		// Calculate Distance from query location
		dX = bX - qX;
		dY = bY - qY;
		dZ = bZ - qZ;
		dW = bW - qW;
		diffDist2 = dX*dX + dY*dY + dZ*dZ + dW*dW;

		// Update Best Point Index
		if (diffDist2 < bestDist2)
		{
			bestIndex = i;
			bestID    = currNode.SearchID();
			bestDist2 = diffDist2;
		}
	}

	//Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

	// Success (return results)
	nearestID       = bestID;
	nearestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_QNN_4D
  Desc:	Finds closest point in kd-tree for each query point
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::Find_QNN_4D
( 
	CPU_NN_Result * queryResults,	// OUT: Results
	unsigned int      nQueries,		// IN: Number of Query points
	const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	// Local Parameters
	CPU_NN_Result best;
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw;
	float queryVals[4];
	unsigned int currFlags, currQuery;
	float queryValue, splitValue;
	CPUNode_4D_LBT * currNode = NULL;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Reset stack
		stackTop = 0;		

		// Load current Query Point into local (fast) memory
		queryVals[0] = queryPoints[currQuery].x;
		queryVals[1] = queryPoints[currQuery].y;
		queryVals[2] = queryPoints[currQuery].z;
		queryVals[3] = queryPoints[currQuery].w;

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & 0x1FFFFFFFu);
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			// Get next & prev axis
			nextAxis  = ((currAxis == 3u) ? 0u : (currAxis + 1));
			prevAxis  = ((currAxis == 0u) ? 3u : (currAxis - 1));

			// Get left and right child positions
			//parent  = currIdx >> 1u;
			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryVals[0];
			dy = currNode->Y() - queryVals[1];
			dz = currNode->Z() - queryVals[2];
			dw = currNode->W() - queryVals[3];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

			// Update closest point Idx
			if (diffDist2 < best.Dist)
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the Best Node Index
		//		   But we really need the Best Point ID 
		//		   So grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfNearID = 0;
		float bfNearDist = 3.0e+38F;
		const float4 & queryPoint = queryPoints[currQuery];
		bTest = BF_FindNN_4D( queryPoint, bfNearID, bfNearDist );

		if ((! bTest) || (bfNearID != best.Id))
		{
			if (best.Dist != bfNearDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN 4D kd-tree search returned a different result than 4D Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
		queryResults[currQuery] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_NN_4D
  Desc:	Finds closest point in kd-tree for point in kd-tree
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::Find_ALL_NN_4D
( 
	CPU_NN_Result * queryResults	// OUT: Results
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	// Local Parameters
	CPU_NN_Result best;
	CPUNode_4D_LBT * currNode  = NULL;
	CPUNode_4D_LBT * queryNode = NULL;
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float queryVals[4];
	float dx, dy, dz, dw;
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	unsigned int currFlags, currQuery;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Reset top of stack for each query
		stackTop = 0;
		
		// Compute Query Index

		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->GetVal(0);
		queryVals[1] = queryNode->GetVal(1);
		queryVals[2] = queryNode->GetVal(2);
		queryVals[3] = queryNode->GetVal(3);

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & 0x1FFFFFFFU);
			currAxis  = (currFlags & 0x60000000U) >> 29;
			currInOut = (currFlags & 0x80000000U) >> 31;
			
			// Get next & prev axis
			nextAxis  = (currAxis == 3u) ? 0u : currAxis + 1;
			prevAxis  = (currAxis == 0u) ? 3u : currAxis - 1;

			// Get left and right child positions
			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			dw = currNode->GetVal(3) - queryVals[3];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

			// Update closest point Idx
			if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the Best node Index 
		//         but we really need the best point ID 
		//         so grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfID = 0;
		float bfDist = 3.0e+38F;
		bTest = BF_FindNN_2D( queryPoint, bfID, bfDist );
		if ((! bTest) || (bfID != best.Id))
		{
			if (best.Dist != bfDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN search returned a different result than Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
			// Convert query node index into original query point index
		unsigned int outIdx = queryNode->SearchID();
			// Store results at query point index
		queryResults[outIdx] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_KNN_4D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::Find_KNN_4D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nQueries,		// IN: Number of Query points
		unsigned int      nPadQueries,	// IN: Number of Padded queries
		const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_4D_LBT * currNode = NULL;
	unsigned int left, right;
	unsigned int currIdx, currInOut;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw;
	float queryVals[4];
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	unsigned int maxHeap, countHeap;

	KD_SEARCH_LBT		searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Get current Query Point
		queryVals[0] = queryPoints[currQuery].x;
		queryVals[1] = queryPoints[currQuery].y;
		queryVals[2] = queryPoints[currQuery].z;
		queryVals[3] = queryPoints[currQuery].w;

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & 0x1FFFFFFFu;
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;

			//parent = currIdx >> 1u;
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 3u) ? 0u  : currAxis+1);
			prevAxis  = ((currAxis == 0u) ? 3u : currAxis-1);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			dw = currNode->GetVal(3) - queryVals[3];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

			// See if we should add this point to 'k' NN Heap
			if (countHeap < maxHeap)
			{
				//-------------------------------
				//	< 'k' elements on heap
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]       = knnHeap[childHIdx];
							knnHeap[childHIdx]        = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		  = knnHeap[childHIdx];
					knnHeap[childHIdx]		  = closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		// We now have a heap of the 'k' nearest neighbors
		// Write heap elements to the results array row by row	
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			unsigned int offset = (i-1) * nQueries;

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node index
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[currQuery+offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_KNN_4D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::Find_ALL_KNN_4D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: number of padded search points (query points)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_4D_LBT * currNode  = NULL;
	CPUNode_4D_LBT * queryNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, prevAxis, nextAxis;
	float dx, dy, dz, dw;
	float queryVals[4];
	float queryValue, splitValue;
	float diff, diff2, diffDist2;
	unsigned int maxHeap, countHeap, currQuery;
	float dist2Heap, bestDist2;

	KD_SEARCH_LBT		searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->X();
		queryVals[1] = queryNode->Y();
		queryVals[2] = queryNode->Z();
		queryVals[3] = queryNode->W();

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & 0x1FFFFFFFu;
			currAxis  = (currFlags & 0x60000000u) >> 29u;
			currInOut = (currFlags & 0x80000000u) >> 31u;
			
			//parentIdx = currIdx >> 1u;
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 3u) ? 0u : currAxis+1u);
			prevAxis  = ((currAxis == 0u) ? 3u : currAxis-1u);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					// Next Line is effectively queryValue = queryPoints[prevAxis];
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			dw = currNode->GetVal(3) - queryVals[3];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw);

			// See if we should add this point to 'k' NN Heap
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
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]       = knnHeap[childHIdx];
							knnHeap[childHIdx]        = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		  = knnHeap[childHIdx];
					knnHeap[childHIdx]		  = closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		unsigned int offset, outIdx;

		// Store Query Result
			// Convert query node index into original query point index
		outIdx = queryNode->SearchID();

		// We now have a heap of 'k' nearest neighbors
		// Write them to the results array
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			offset = (i-1) * nSearch;			// -1 is to ignore zeroth element in heap which is not used

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node indices
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[outIdx + offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::DumpNodes
  Desc:	Dumps kd-tree in height order
-------------------------------------------------------------------------*/

void CPUTree_4D_LBT::DumpNodes() const
{
	printf( "\nKDTree_LBT, count = %d { \n\n", m_cNodes );

	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_4D_LBT & currNode = m_nodes[i];
		double x = (double)(currNode.X());
		double y = (double)(currNode.Y());
		double z = (double)(currNode.Z());
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//printf( "[%d] = <x=%3.6f, y=%3.6f, S=%d, N=%d>\n",
		//		i, x, y, sID, nID );
		
		printf( "%d, %3.6f, %3.6f, %3.6f, %d, %d\n", i, x, y, z, sID, nID );
	}

	printf( "\n} \n\n", m_cNodes );
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Validate
  Desc:	Validate that the nodes in the kd-tree are actually
        in left-balanced array order
-------------------------------------------------------------------------*/

bool CPUTree_4D_LBT::Validate() const
{
	bool fResult = true;

	float    currVals[4];
	float    leftVals[4];
	float    rightVals[4];
	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_4D_LBT & currNode = m_nodes[i];
		currVals[0] = currNode.X();
		currVals[1] = currNode.Y();
		currVals[2] = currNode.Z();
		currVals[3] = currNode.W();
		unsigned int currAxis = currNode.Axis();
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//unsigned int parent = nID >> 1;
		unsigned int left   = nID << 1;
		unsigned int right  = left + 1;		

		// Check left child
		if (left <= m_cNodes)
		{
			// Make sure left child is actually to the left of this current node
			leftVals[0] = m_nodes[left].X();
			leftVals[1] = m_nodes[left].Y();
			leftVals[2] = m_nodes[left].Z();
			leftVals[3] = m_nodes[left].W();

			float fLeft = leftVals[currAxis];
			float fCurr = currVals[currAxis];

			if (fLeft > fCurr)
			{
				printf( "<%d,%3.6f> is not to left of it's parent <%d,%3.6f> !!!\n",
						left, (double)fLeft, nID, (double)fCurr );
				fResult = false;
			}
		}

		// Check right child
		if (right <= m_cNodes)
		{
			// Make sure right child is actually to the right of this current node
			rightVals[0] = m_nodes[right].X();
			rightVals[1] = m_nodes[right].Y();
			rightVals[2] = m_nodes[right].Z();
			rightVals[3] = m_nodes[right].W();

			float fRight = rightVals[currAxis];
			float fCurr  = currVals[currAxis];

			if (fRight < fCurr)
			{
				printf( "<%d,%3.6f> is not to right of it's parent <%d,%3.6f> !!!\n",
						right, (double)fRight, nID, (double)fCurr );

				fResult = false;
			}
		}
	}

	return fResult;
}


/*-------------------------------------
  CPUTree_6D_LBT Methods Definitions
-------------------------------------*/

/*-------------------------------------------------------------------------
  Name:	CPUTree_6D_LBT::GetNodeAxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

float CPUTree_6D_LBT::GetNodeAxisValue
( 
	const CPUNode_6D_LBT * currNodes,	// IN:  IN node list
	unsigned int index,				// IN:  Index of node to retrieve value for
	unsigned int axis				// IN:  axis of value to retrieve
) const
{
	const CPUNode_6D_LBT & currNode = currNodes[index];
	float axisValue = currNode[axis];
	return axisValue;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_6D_LBT::SwapNodes
  Desc:	Helper Method
-------------------------------------------------------------------------*/

void CPUTree_6D_LBT::SwapNodes
( 
	CPUNode_6D_LBT * currNodes,	// IN: Node list
	unsigned int idx1,			// IN: Index of 1st node to swap
	unsigned int idx2			// IN: Index of 2nd node to swap
)
{
	CPUNode_6D_LBT & currNode1 = currNodes[idx1];
	CPUNode_6D_LBT & currNode2 = currNodes[idx2];
	CPUNode_6D_LBT temp;

	temp	  = currNode1;
	currNode1 = currNode2;
	currNode2 = temp;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_6D_LBT::MedianOf3
  Desc:	Helper method,
		Implements Median of three variant 
		for Median Partitioning algorithm
		returns pivot value for partitioning algorithm
  Note:	enforces invariant
		array[left].val <= array[mid].val <= array[right].val
		where mid = (left+right)/2
-------------------------------------------------------------------------*/

void CPUTree_6D_LBT::MedianOf3
(
	CPUNode_6D_LBT * currNodes,	// IN - node list
	unsigned int leftIdx,		// IN - left index
	unsigned int rightIdx,		// IN - right index
	unsigned int axis			// IN - axis to compare
)
{
	// Compute Middle Index from left and right
	unsigned int midIdx = (leftIdx+rightIdx)/2;	

	float leftVal  = GetNodeAxisValue( currNodes, leftIdx, axis );
	float rightVal = GetNodeAxisValue( currNodes, rightIdx, axis );
	float midVal   = GetNodeAxisValue( currNodes, midIdx, axis );

	// Sort left, center, mid value into correct order
	if (leftVal > midVal)
	{
		SwapNodes( currNodes, leftIdx, midIdx );
	}
	if (leftVal > rightVal)
	{
		SwapNodes( currNodes, leftIdx, rightIdx );
	}
	if (midVal > rightVal)
	{
		SwapNodes( currNodes, midIdx, rightIdx );
	}

	// Deliberately move median value to end of array
	SwapNodes( currNodes, midIdx, rightIdx );
}


/*-------------------------------------------------------------------------
  Name: CPUTree_6D_LBT::MedianSortNodes
  Desc:	Partitions original data set {O} with 'n' elements
		into 3 datasets <{l}, {m}, {r}}

		where {l} contains approximately half the elements 
		and all elements are less than or equal to {m}
		where {m} contains only 1 element the "median value"
		and {r} contains approximately half the elements
		and all elements are greater than or equal to {m}

  Notes:
	Invariant:
		{l} = { all points with value less or equal to median }
		    count({l}) = (n - 1) / 2
		{m} = { median point }
		    count({m}) = 1
		{r} = { points with value greater than or equal to median point }
		    count({r}) = n - ((n-1)/2 + 1)
			Note: Either count({r}) = count({l}) or
			             count({r}) = count({l})+1
				  IE the count of sets {l} and {r} will be 
				     within one element of each other

	Performance:
		Should take O(N) time to process all points in input range

  Credit:	Based on the "Selection Sort" algorithm as
            presented by Robert Sedgewick in the book
            "Algorithms in C++"
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::MedianSortNodes
(
	CPUNode_6D_LBT * currNodes,	// IN - node list
	unsigned int start,     // IN - start of range
	unsigned int end,       // IN - end of range
	unsigned int & median,  // IN/OUT - approximate median number
	                        //          actual median number
	unsigned int axis       // IN - dimension(axis) to split along (x,y,z)
)
{
	if (start > end)
	{
		// Swap Start and End, if they are in the wrong order
		unsigned int temp = start;
		start = end;
		end = temp;
	}

	// Check Parameters
	unsigned int nNodes = (end - start) + 1;
	if (nNodes == 0) { return false; }
	if (nNodes == 1) { return true; }
	if (axis >= INVALID_AXIS) { return false; }
	if ((median < start) || (median > end)) { return false; }

	// Perform Median Sort
	int left   = static_cast<int>( start );
	int right  = static_cast<int>( end );
	int middle = static_cast<int>( median );
	int i,j;
	float pivotVal;

	while ( right > left ) 
	{
		// Get Pivot value
			// Use Median of 3 variant
		MedianOf3( currNodes, left, right, axis );
		pivotVal = GetNodeAxisValue( currNodes, right, axis );

		i = left - 1;
		j = right;

		// Partition into 3 sets
			// Left   = {start, pivot-1}	all values in Left <= median
			// Median = {pivot}				singleton containing pivot value
			// Right  = {pivot+1, end}		all values in right >= median
		for (;;) 
		{
			while ( GetNodeAxisValue( currNodes, ++i, axis ) < pivotVal )
			{
				// Deliberately do nothing
			}

			while ( (GetNodeAxisValue( currNodes, --j, axis ) > pivotVal) && 
				  (j > left) )
			{
				// Deliberately do nothing
			}
			
			if ( i >= j )
				break;

			SwapNodes( currNodes, i, j );
		}

		// Put pivot value back into pivot position
		SwapNodes( currNodes, i, right );

		// Iterate into left or right set until
		// we find the value at the true median
		if ( i >= middle )
			right = i-1;
		if ( i <= middle )
			left = i+1;
	}

	// return new median index
	median = middle;

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_6D_LBT::ComputeBoundBox
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::ComputeBoundingBox
( 
	unsigned int start,		// IN - start index
	unsigned int end,		// IN - end index
	float        bounds[12]	// OUT - bounding box for all nodes in range
)
{
	// Check Parameters
	if (CPUNode_6D_LBT::c_Invalid == start) { return false; }
	if (CPUNode_6D_LBT::c_Invalid == end) { return false; }

	unsigned int sI = start;
	unsigned int eI = end;
	if (eI < sI) 
	{
		unsigned int temp = sI;
		sI = eI;
		eI = temp;
	}

	CPUNode_6D_LBT * currNode = NODE_PTR( sI );
	if (NULL == currNode) { return false; }
	
	float x, y, z, w, s, t;

	x = currNode->X();
	y = currNode->Y();
	z = currNode->Z();
	w = currNode->W();
	s = currNode->S();
	t = currNode->T();

	bounds[0] = x;
	bounds[1] = x;
	bounds[2] = y;
	bounds[3] = y;
	bounds[4] = z;
	bounds[5] = z;
	bounds[6] = w;
	bounds[7] = w;
	bounds[8] = s;
	bounds[9] = s;
	bounds[10] = t;
	bounds[11] = t;

	unsigned int i;
	for (i = sI+1; i <= eI; i++)
	{
		currNode = NODE_PTR( i );
		x = currNode->X();
		y = currNode->Y();
		z = currNode->Z();
		w = currNode->W();
		s = currNode->S();
		t = currNode->T();

		// Update Min, Max for X,Y,Z,W
		if (x < bounds[0]) { bounds[0] = x; }
		if (x > bounds[1]) { bounds[1] = x; }
		
		if (y < bounds[2]) { bounds[2] = y; }
		if (y > bounds[3]) { bounds[3] = y; }
		
		if (z < bounds[4]) { bounds[4] = z; }
		if (z > bounds[5]) { bounds[5] = z; }

		if (w < bounds[6]) { bounds[6] = w; }
		if (w > bounds[7]) { bounds[7] = w; }

		if (s < bounds[8]) { bounds[8] = s; }
		if (s > bounds[9]) { bounds[9] = s; }

		if (t < bounds[10]) { bounds[10] = t; }
		if (t > bounds[11]) { bounds[11] = t; }
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build6D
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::Build6D( unsigned int cPoints, const CPU_Point6D * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for kd-node Lists

		// Node list in median order
	unsigned int cNodes = cPoints;
	CPUNode_6D_LBT * mNodes = new CPUNode_6D_LBT[cPoints+1];
	if (NULL == mNodes) { return false; }

		// Node list in left-balanced order
	CPUNode_6D_LBT * lNodes = new CPUNode_6D_LBT[cPoints+1];
	if (NULL == lNodes) { return false; }

	m_cNodes = cNodes;


	/*---------------------------------
	  Initialize Median Node List
	---------------------------------*/

	float * pntDst = NULL;

	// First Median node is wasted space (for 1-based indexing)
	const float * pntSrc = NULL;

	mNodes[0].SearchID( 0u );	// Have it map back onto itself
	mNodes[0].NodeID( 0u );

	pntDst = mNodes[0].BASE_PNT();
	pntDst[0] = 0.0f;
	pntDst[1] = 0.0f;
	pntDst[2] = 0.0f;
	pntDst[3] = 0.0f;
	pntDst[4] = 0.0f;
	pntDst[5] = 0.0f;

	// Copy points into nodes
		// Point indices are in range [0,n-1]
		// Node Indices are in range [1, n]
	unsigned int pntIdx, nodeIdx;
	//float pos[6];
	for (pntIdx = 0; pntIdx < cNodes; pntIdx++)
	{
		nodeIdx = pntIdx + 1;
		CPUNode_6D_LBT & currNode = mNodes[nodeIdx];

		pntDst = currNode.BASE_PNT();
		pntSrc = &(pointList[pntIdx].pos[0]);
		pntDst[0] =  pntSrc[0];
		pntDst[1] =  pntSrc[1];
		pntDst[2] =  pntSrc[2];
		pntDst[3] =  pntSrc[3];
		pntDst[4] =  pntSrc[4];
		pntDst[5] =  pntSrc[5];

		currNode.SearchID( pntIdx );
		currNode.NodeID( CPUNode_6D_LBT::c_Invalid );
	}


	/*--------------------------------------
	  Initialize Left Balanced Node List
	--------------------------------------*/


	// First Median node is wasted space (for 1-based indexing)
	lNodes[0].SearchID( 0 );		// map back onto itself
	lNodes[0].NodeID( 0 );

	pntDst = lNodes[0].BASE_PNT();
	pntDst[0] = 0.0f;
	pntDst[1] = 0.0f;
	pntDst[2] = 0.0f;
	pntDst[3] = 0.0f;
	pntDst[4] = 0.0f;
	pntDst[5] = 0.0f;

	for (nodeIdx = 1; nodeIdx <= cNodes; nodeIdx++)
	{
		CPUNode_6D_LBT & currNode = lNodes[nodeIdx];
		currNode.NodeID( nodeIdx );
	}


	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart   = 1;
	unsigned int currEnd     = cNodes;
	unsigned int currN       = (currEnd - currStart) + 1;
	unsigned int currHalf;
	unsigned int currLBM;		
	unsigned int currMedian;
	unsigned int currTarget  = CPUNode_6D_LBT::c_Invalid;
	unsigned int currFlags;

	// Get Left-balanced median position for root
	KD_LBM_CPU( currN, currLBM, currHalf );
	currMedian = currStart + currLBM - 1;

	CPUNode_6D_LBT * currNodePtr   = NULL;

	// Root goes into position 1
	m_rootIdx   = 1;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLeft, currRight;
	unsigned int lastRow;

	// Build Stack
	unsigned int stackTop = 0;
	KD_BUILD_LBT buildStack[CPU_SEARCH_STACK_SIZE];

	// Push root info onto build stack
	KD_BUILD_LBT & rootInfo = buildStack[stackTop];
	rootInfo.start     = currStart;
	rootInfo.end       = currEnd;
	rootInfo.targetID  = m_rootIdx;
	rootInfo.flags     = ((currHalf & NODE_INDEX_MASK) | 
		                 ((currAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK));
	++stackTop;

	// Process child ranges until we reach 1 node per range (leaf nodes)
	while (stackTop != 0)
	{
		// Get Build Info from top of stack
		--stackTop;
		const KD_BUILD_LBT & currBuild = buildStack[stackTop];

		currStart  = currBuild.start;
		currEnd    = currBuild.end;
		currTarget = currBuild.targetID;
		currFlags  = currBuild.flags;

		currAxis   = ((currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT);

		currN      = currEnd-currStart + 1;
		nextAxis   = ((currAxis == 5u) ? 0u : (currAxis+1));

		// No need to do median sort if only one element is in range (IE a leaf node)
		if (currN > 1)
		{
			if (currN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTable[currN];
				currLBM  = g_leftMedianTable[currN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & NODE_INDEX_MASK); 
				lastRow    = KD_Min( currHalf, currN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = currStart + (currLBM - 1);			// Get actual median 

			// Sort nodes into 2 buckets (on axis plane)
			  // Uses Sedgewick Partition Algorithm with Median of 3 variant
			bool bResult = MedianSortNodes( mNodes, currStart, currEnd, currMedian, currAxis );
			if (false == bResult) 
			{ 
				return false; 
			}
		}
		else
		{
			currMedian = currStart;
		}

		// Store current median node 
		// in left balanced list at target
		const CPUNode_6D_LBT & medianNode = mNodes[currMedian];
		CPUNode_6D_LBT & targetNode = lNodes[currTarget];
		unsigned int currID = medianNode.SearchID();

		const float * medBase = medianNode.BASE_PNT();
		targetNode.X( medBase[0] );
		targetNode.Y( medBase[1] );
		targetNode.Z( medBase[2] );
		targetNode.W( medBase[3] );
		targetNode.S( medBase[4] );
		targetNode.T( medBase[5] );

		targetNode.SearchID( currID );
		targetNode.NodeID( currTarget );

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currRight <= cPoints)
		{
			// Add right child to build stack
			KD_BUILD_LBT & rightBuild = buildStack[stackTop];
			rightBuild.start     = currMedian + 1;
			rightBuild.end       = currEnd;
			rightBuild.targetID  = currRight;
			rightBuild.flags     = ((currHalf >> 1) & NODE_INDEX_MASK) | 
								   ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			++stackTop;
		}

		if (currLeft <= cPoints)
		{
			// Add left child to build stack
			KD_BUILD_LBT & leftBuild = buildStack[stackTop];
			leftBuild.start     = currStart;
			leftBuild.end       = currMedian - 1;
			leftBuild.targetID  = currLeft;
			leftBuild.flags     = ((currHalf >> 1) & NODE_INDEX_MASK) | 
				                   ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			++stackTop;
		}

	}

	// Set
	m_cNodes = cNodes;
	m_nodes  = lNodes;

	/*
	// Dump Results of build for debugging
	Dump();

	// Validate that left balanced layout is correct
	bool bTest;
	bTest = Validate();
	if (! bTest)
	{
		printf( "Invalid layout\n" );
	}
	*/

	// Cleanup
	if (NULL != mNodes)
	{
		delete [] mNodes;
		mNodes = NULL;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_6D_LBT::BF_FindNN_6D()
  Desc:	Use Brute Force Algorithm to find closest Node (index)
			Takes O(N) time
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::BF_FindNN_6D
(
	const CPU_Point6D & queryLocation,	// IN  - Location to sample
	unsigned int & nearestID,		// OUT - ID of nearest point
	float & nearestDistance			// OUT - distance to nearest point
)
{
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	// Get Query Point
	float qX, qY, qZ, qW, qS, qT;
	qX = queryLocation.pos[0];
	qY = queryLocation.pos[1];
	qZ = queryLocation.pos[2];
	qW = queryLocation.pos[3];
	qS = queryLocation.pos[4];
	qT = queryLocation.pos[5];

	// Get 1st Point
	unsigned int  bestIndex  = 1;
	CPUNode_6D_LBT * currNodePtr = NULL;
	CPUNode_6D_LBT * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID      = bestNodePtr->SearchID();

	float bX, bY, bZ, bW, bS, bT;
	bX = bestNodePtr->X();
	bY = bestNodePtr->Y();
	bZ = bestNodePtr->Z();
	bW = bestNodePtr->W();
	bS = bestNodePtr->S();
	bT = bestNodePtr->T();

	// Calculate distance from query location
	float dX = bX - qX;
	float dY = bY - qY;
	float dZ = bZ - qZ;
	float dW = bW - qW;
	float dS = bS - qS;
	float dT = bT - qT;
	float bestDist2 = dX*dX + dY*dY + dZ*dZ + dW*dW + dS*dS + dT*dT;
	float diffDist2;

	unsigned int i;
	for (i = 2; i <= nNodes; i++)
	{
		// Get Current Point
		CPUNode_6D_LBT & currNode = m_nodes[i];
		bX = currNode.X();
		bY = currNode.Y();
		bZ = currNode.Z();
		bW = currNode.W();
		bS = currNode.S();
		bT = currNode.T();

		// Calculate Distance from query location
		dX = bX - qX;
		dY = bY - qY;
		dZ = bZ - qZ;
		dW = bW - qW;
		dS = bS - qS;
		dT = bT - qT;
		diffDist2 = dX*dX + dY*dY + dZ*dZ + dW*dW + dS*dS + dT*dT;

		// Update Best Point Index
		if (diffDist2 < bestDist2)
		{
			bestIndex = i;
			bestID    = currNode.SearchID();
			bestDist2 = diffDist2;
		}
	}

	//Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

	// Success (return results)
	nearestID       = bestID;
	nearestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_QNN_6D
  Desc:	Finds closest point in kd-tree for each query point
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::Find_QNN_6D
( 
	CPU_NN_Result * queryResults,	// OUT: Results
	unsigned int      nQueries,		// IN: Number of Query points
	const CPU_Point6D	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	// Local Parameters
	CPU_NN_Result best;
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw, ds, dt;
	float queryVals[6];
	unsigned int currFlags, currQuery;
	float queryValue, splitValue;
	CPUNode_6D_LBT * currNode = NULL;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	const float * pntSrc  = NULL;
	      float * pntDest = NULL;

	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Reset stack
		stackTop = 0;		

		// Load current Query Point into local (fast) memory
		pntSrc = &(queryPoints[currQuery].pos[0]);
		queryVals[0] = pntSrc[0];
		queryVals[1] = pntSrc[1];
		queryVals[2] = pntSrc[2];
		queryVals[3] = pntSrc[3];
		queryVals[4] = pntSrc[4];
		queryVals[5] = pntSrc[5];

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = FLAGS_ROOT_START; 
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & NODE_INDEX_MASK);
			currAxis  = (currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
			currInOut = (currFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;
			
			// Get next & prev axis
			nextAxis  = ((currAxis == 5u) ? 0u : (currAxis + 1));
			prevAxis  = ((currAxis == 0u) ? 5u : (currAxis - 1));

			// Get left and right child positions
			//parent  = currIdx >> 1u;
			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			pntSrc = currNode->BASE_PNT();
			dx = pntSrc[0] - queryVals[0];
			dy = pntSrc[1] - queryVals[1];
			dz = pntSrc[2] - queryVals[2];
			dw = pntSrc[3] - queryVals[3];
			ds = pntSrc[4] - queryVals[4];
			dt = pntSrc[5] - queryVals[5];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw) + (ds*ds) + (dt*dt);

			// Update closest point Idx
			if (diffDist2 < best.Dist)
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & NODE_INDEX_MASK) 
							         | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									 | OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & NODE_INDEX_MASK) 
						        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK); 
								// | ONSIDE_VALUE;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & NODE_INDEX_MASK) 
									| ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									| OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & NODE_INDEX_MASK) 
						        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK); 
								// | ONSIDE_VALUE
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the Best Node Index
		//		   But we really need the Best Point ID 
		//		   So grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfNearID = 0;
		float bfNearDist = 3.0e+38F;
		const float4 & queryPoint = queryPoints[currQuery];
		bTest = BF_FindNN_4D( queryPoint, bfNearID, bfNearDist );

		if ((! bTest) || (bfNearID != best.Id))
		{
			if (best.Dist != bfNearDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN 4D kd-tree search returned a different result than 4D Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
		queryResults[currQuery] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_NN_6D
  Desc:	Finds closest point in kd-tree for point in kd-tree
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::Find_ALL_NN_6D
( 
	CPU_NN_Result * queryResults	// OUT: Results
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	// Local Parameters
	CPU_NN_Result best;
	CPUNode_6D_LBT * currNode  = NULL;
	CPUNode_6D_LBT * queryNode = NULL;
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, nextAxis, prevAxis;
	float queryVals[6];
	float dx, dy, dz, dw, ds, dt;
	float diff, diff2, diffDist2;
	float queryValue, splitValue;
	unsigned int currFlags, currQuery;

	KD_SEARCH_LBT searchStack[CPU_SEARCH_STACK_SIZE];

	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Reset top of stack for each query
		stackTop = 0;
		
		// Compute Query Index

		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->GetVal(0);
		queryVals[1] = queryNode->GetVal(1);
		queryVals[2] = queryNode->GetVal(2);
		queryVals[3] = queryNode->GetVal(3);
		queryVals[4] = queryNode->GetVal(4);
		queryVals[5] = queryNode->GetVal(5);

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cVisited = 0;

		// Put root search info on stack
		searchStack[stackTop].flags		 = FLAGS_ROOT_START; 
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = (currFlags & NODE_INDEX_MASK);
			currAxis  = (currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
			currInOut = (currFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;
			
			// Get next & prev axis
			nextAxis  = (currAxis == 5u) ? 0u : currAxis + 1;
			prevAxis  = (currAxis == 0u) ? 5u : currAxis - 1;

			// Get left and right child positions
			left      = currIdx << 1u;
			right     = left + 1u;

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				queryValue = queryVals[prevAxis];
				splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 >= best.Dist)
				{
					// We can do an early exit for this node
					continue;
				}
			}

			// WARNING - It's Much faster to load this node from global memory after the "Early Exit check"
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			dw = currNode->GetVal(3) - queryVals[3];
			ds = currNode->GetVal(4) - queryVals[4];
			dt = currNode->GetVal(5) - queryVals[5];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw) + (ds*ds) + (dt*dt);

			// Update closest point Idx
			if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.Id   = ((diffDist2 < best.Dist) ? currIdx   : best.Id);
			//best.Dist = ((diffDist2 < best.Dist) ? diffDist2 : best.Dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (right <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (right & NODE_INDEX_MASK) 
							         | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									 | OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & NODE_INDEX_MASK) 
						         | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
								 // | ONSIDE_VALUE;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & NODE_INDEX_MASK) 
							         | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									 | OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & NODE_INDEX_MASK) 
						         | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
								 //| ONSIDE_VALUE;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the Best node Index 
		//         but we really need the best point ID 
		//         so grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->SearchID();

		/*
		// Debugging, Do a Brute Force search to validate
		bool bTest;
		unsigned int bfID = 0;
		float bfDist = 3.0e+38F;
		bTest = BF_FindNN_2D( queryPoint, bfID, bfDist );
		if ((! bTest) || (bfID != best.Id))
		{
			if (best.Dist != bfDist)
			{
				// Error, we don't have a match
				printf( "Error - QNN search returned a different result than Brute force search" );
			}
		}
		*/

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
			// Convert query node index into original query point index
		unsigned int outIdx = queryNode->SearchID();
			// Store results at query point index
		queryResults[outIdx] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_KNN_6D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::Find_KNN_6D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nQueries,		// IN: Number of Query points
		unsigned int      nPadQueries,	// IN: Number of Padded queries
		const CPU_Point6D * queryPoints	// IN: query points to compute distance for (1D or 2D field)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }
	if (NULL == queryPoints)  { return false; }
	if (nQueries == 0) { return true; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_6D_LBT * currNode = NULL;
	unsigned int left, right;
	unsigned int currIdx, currInOut;
	unsigned int currAxis, nextAxis, prevAxis;
	float diff, diff2, diffDist2;
	float dx, dy, dz, dw, ds, dt;
	float queryVals[6];
	float queryValue, splitValue;
	float dist2Heap, bestDist2;
	unsigned int maxHeap, countHeap;

	const float * pntSrc = NULL;
	float * pntDest = NULL;

	KD_SEARCH_LBT	searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		pntSrc = &(queryPoints[currQuery].pos[0]);

		// Get current Query Point
		queryVals[0] = pntSrc[0];
		queryVals[1] = pntSrc[1];
		queryVals[2] = pntSrc[2];
		queryVals[3] = pntSrc[3];
		queryVals[4] = pntSrc[4];
		queryVals[5] = pntSrc[5];

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = FLAGS_ROOT_START;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & NODE_INDEX_MASK;
			currAxis  = (currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
			currInOut = (currFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;

			//parent = currIdx >> 1u;
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 5u) ? 0u  : currAxis+1);
			prevAxis  = ((currAxis == 0u) ? 5u : currAxis-1);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			pntSrc = currNode->BASE_PNT();
			dx = pntSrc[0] - queryVals[0];
			dy = pntSrc[1] - queryVals[1];
			dz = pntSrc[2] - queryVals[2];
			dw = pntSrc[3] - queryVals[3];
			ds = pntSrc[4] - queryVals[4];
			dt = pntSrc[5] - queryVals[5];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw) + (ds*ds) + (dt*dt);

			// See if we should add this point to 'k' NN Heap
			if (countHeap < maxHeap)
			{
				//-------------------------------
				//	< 'k' elements on heap
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]       = knnHeap[childHIdx];
							knnHeap[childHIdx]        = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		  = knnHeap[childHIdx];
					knnHeap[childHIdx]		  = closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & NODE_INDEX_MASK) 
							        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									| OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & NODE_INDEX_MASK) 
						        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK); 
								// | ONSIDE_VALUE;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & NODE_INDEX_MASK) 
							        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									| OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & NODE_INDEX_MASK) 
						        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK); 
								// | ONSIDE_VALUE;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		// We now have a heap of the 'k' nearest neighbors
		// Write heap elements to the results array row by row	
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			unsigned int offset = (i-1) * nQueries;

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node index
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[currQuery+offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_KNN_6D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::Find_ALL_KNN_6D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: number of padded search points (query points)
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}

	// Local Parameters
	unsigned int rootIdx  = 1;
	unsigned int stackTop = 0;
	CPUNode_6D_LBT * currNode  = NULL;
	CPUNode_6D_LBT * queryNode = NULL;
	unsigned int currIdx, currInOut;
	unsigned int left, right;
	unsigned int currAxis, prevAxis, nextAxis;
	float dx, dy, dz, dw, ds, dt;
	float queryVals[6];
	float queryValue, splitValue;
	float diff, diff2, diffDist2;
	unsigned int maxHeap, countHeap, currQuery;
	float dist2Heap, bestDist2;

	KD_SEARCH_LBT	searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	for (currQuery = 1; currQuery <= m_cNodes; currQuery++)
	{
		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryVals[0] = queryNode->X();
		queryVals[1] = queryNode->Y();
		queryVals[2] = queryNode->Z();
		queryVals[3] = queryNode->W();
		queryVals[4] = queryNode->S();
		queryVals[5] = queryNode->T();

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e+38F;

		// Put root search info on stack
		unsigned int currFlags = FLAGS_ROOT_START;
		searchStack[stackTop].flags       = currFlags; 
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cVisited++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currFlags = searchStack[stackTop].flags;
			currIdx   = currFlags & NODE_INDEX_MASK;
			currAxis  = (currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT;
			currInOut = (currFlags & ON_OFF_MASK) >> ON_OFF_SHIFT;
			
			//parentIdx = currIdx >> 1u;
			left   = currIdx << 1u;
			right  = left + 1u;

			nextAxis  = ((currAxis == 5u) ? 0u : currAxis+1u);
			prevAxis  = ((currAxis == 0u) ? 5u : currAxis-1u);

			// Early Exit Check
			if (currInOut == 1u)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					// Next Line is effectively queryValue = queryPoints[prevAxis];
					queryValue = queryVals[prevAxis];
					splitValue = searchStack[stackTop].splitValue;	// Split Value of Parent Node
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
			// Load specified Current Node from KD Tree
			currNode = NODE_PTR( currIdx );

			// Get Best Fit Dist for checking child ranges
			queryValue = queryVals[currAxis];
			splitValue = currNode->GetVal(currAxis);
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->GetVal(0) - queryVals[0];
			dy = currNode->GetVal(1) - queryVals[1];
			dz = currNode->GetVal(2) - queryVals[2];
			dw = currNode->GetVal(3) - queryVals[3];
			ds = currNode->GetVal(4) - queryVals[4];
			dt = currNode->GetVal(5) - queryVals[5];
			diffDist2 = (dx*dx) + (dy*dy) + (dz*dz) + (dw*dw) + (ds*ds) + (dt*dt);

			// See if we should add this point to 'k' NN Heap
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
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e+38F;

				countHeap++;
				knnHeap[countHeap].Id   = currIdx;
				knnHeap[countHeap].Dist = diffDist2;

				// Do we need to Convert array into heap ?!?
				if (countHeap == maxHeap)
				{
					// Yes, turn array into a heap
					for (unsigned int m = countHeap/2; m >= 1; m--)
					{
						//
						// Demote each element in turn (to correct position in heap)
						//

						unsigned int parentHIdx = m;		// Start at specified element
						unsigned int childHIdx  = m << 1;	// left child of parent

						// Compare Parent to it's children
						while (childHIdx <= maxHeap)
						{
							// Update Distances
							float parentD2 = knnHeap[parentHIdx].Dist;
							float childD2  = knnHeap[childHIdx].Dist;

							// Find largest child 
							if (childHIdx < maxHeap)
							{
								float rightD2 = knnHeap[childHIdx+1].Dist;
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
							CPU_NN_Result closeTemp = knnHeap[parentHIdx];
							knnHeap[parentHIdx]       = knnHeap[childHIdx];
							knnHeap[childHIdx]        = closeTemp;
							
							// Update indices
							parentHIdx = childHIdx;	
							childHIdx  = parentHIdx<<1;		// left child of parent
						}
					}

					// Update trim distances
					dist2Heap = knnHeap[1].Dist;
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
				knnHeap[1].Id   = currIdx;
				knnHeap[1].Dist = diffDist2;

				//
				// Demote new element (to correct position in heap)
				//
				unsigned int parentHIdx = 1;	// Start at Root
				unsigned int childHIdx  = 2;	// left child of parent

				// Compare current index to it's children
				while (childHIdx <= maxHeap)
				{
					// Update Distances
					float parentD2 = knnHeap[parentHIdx].Dist;
					float childD2  = knnHeap[childHIdx].Dist;

					// Find largest child 
					if (childHIdx < maxHeap)
					{
						float rightD2 = knnHeap[childHIdx+1].Dist;
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
					CPU_NN_Result closeTemp = knnHeap[parentHIdx];
					knnHeap[parentHIdx]		  = knnHeap[childHIdx];
					knnHeap[childHIdx]		  = closeTemp;
					
					// Update indices
					parentHIdx = childHIdx;	
					childHIdx  = parentHIdx<<1;		// left child of parent
				}

				// Update Trim distances
				dist2Heap = knnHeap[1].Dist;
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
					if (right <= m_cNodes)	// cInvalid
					{
						// Push Onto top of stack
						currFlags = (right & NODE_INDEX_MASK) 
							        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									| OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (left <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (left & NODE_INDEX_MASK) 
						        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK); 
								// | ONSIDE_VALUE;
					searchStack[stackTop].flags      = currFlags;
					searchStack[stackTop].splitValue = splitValue;
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
					if (left <= m_cNodes)
					{
						// Push Onto top of stack
						currFlags = (left & NODE_INDEX_MASK) 
							        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK) 
									| OFFSIDE_VALUE;
						searchStack[stackTop].flags      = currFlags;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (right <= m_cNodes)
				{
					// Push Onto top of stack
					currFlags = (right & NODE_INDEX_MASK) 
						        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK); 
								// | ONSIDE_VALUE;
					searchStack[stackTop].flags		 = currFlags;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		/*-------------------
		  Output Results
		-------------------*/

		unsigned int offset, outIdx;

		// Store Query Result
			// Convert query node index into original query point index
		outIdx = queryNode->SearchID();

		// We now have a heap of 'k' nearest neighbors
		// Write them to the results array
		for (unsigned int i = 1; i <= countHeap; i++)
		{
			offset = (i-1) * nSearch;			// -1 is to ignore zeroth element in heap which is not used

			currNode = NODE_PTR( knnHeap[i].Id );
			knnHeap[i].Id   = currNode->SearchID();			// Really need Search ID's not Node indices
			knnHeap[i].Dist = sqrtf( knnHeap[i].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[outIdx + offset] = knnHeap[i];
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::DumpNodes
  Desc:	Dumps kd-tree in height order
-------------------------------------------------------------------------*/

void CPUTree_6D_LBT::DumpNodes() const
{
	printf( "\nKDTree_LBT, count = %d { \n\n", m_cNodes );

	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_6D_LBT & currNode = m_nodes[i];
		double x = (double)(currNode.X());
		double y = (double)(currNode.Y());
		double z = (double)(currNode.Z());
		double w = (double)(currNode.W());
		double s = (double)(currNode.S());
		double t = (double)(currNode.T());
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//printf( "[%d] = <x=%3.6f, y=%3.6f, S=%d, N=%d>\n",
		//		i, x, y, sID, nID );
		
		printf( "%d, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %d, %d\n", 
				i, x, y, z, w, s, t, sID, nID );
	}

	printf( "\n} \n\n", m_cNodes );
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Validate
  Desc:	Validate that the nodes in the kd-tree are actually
        in left-balanced array order
-------------------------------------------------------------------------*/

bool CPUTree_6D_LBT::Validate() const
{
	bool fResult = true;

	const float * pntCurr;
	const float * pntLeft;
	const float * pntRight;
	float fCurr, fLeft, fRight;

	unsigned int i;
	for (i = 1; i <= m_cNodes; i++)
	{
		const CPUNode_6D_LBT & currNode = m_nodes[i];
		pntCurr = currNode.BASE_PNT();
		unsigned int currAxis = currNode.Axis();
		unsigned int sID = currNode.SearchID();
		unsigned int nID = currNode.NodeID();

		//unsigned int parent = nID >> 1;
		unsigned int left   = nID << 1;
		unsigned int right  = left + 1;		

		// Check left child
		if (left <= m_cNodes)
		{
			// Make sure left child is actually to the left of this current node
			pntLeft = m_nodes[left].BASE_PNT();
			fLeft = pntLeft[currAxis];
			fCurr = pntCurr[currAxis];

			if (fLeft > fCurr)
			{
				printf( "<%d,%3.6f> is not to left of it's parent <%d,%3.6f> !!!\n",
						left, (double)fLeft, nID, (double)fCurr );
				fResult = false;
			}
		}

		// Check right child
		if (right <= m_cNodes)
		{
			// Make sure right child is actually to the right of this current node
			pntRight = m_nodes[right].BASE_PNT();
			fRight = pntRight[currAxis];
			fCurr  = pntCurr[currAxis];

			if (fRight < fCurr)
			{
				printf( "<%d,%3.6f> is not to right of it's parent <%d,%3.6f> !!!\n",
						right, (double)fRight, nID, (double)fCurr );

				fResult = false;
			}
		}
	}

	return fResult;
}
