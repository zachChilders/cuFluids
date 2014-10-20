/*-----------------------------------------------------------------------------
  Name:  CPUTree_MED.cpp
  Desc:  Implements Simple kd-tree on CPU in median array layout

  Log:   Created by Shawn D. Brown (4/15/07)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

#ifndef _CPUTREE_MED_H
	#include "CPUTree_MED.h"
#endif
#ifndef _DEQUE_
	#include <deque>	// Std::deque
#endif
#ifndef _STACK_
	#include <stack>	// Std::Stack
#endif

#include <iostream>

/*-------------------------------------
  Methods Definitions
-------------------------------------*/

/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::GetNodeAxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

float CPUTree_2D_MED::GetNodeAxisValue
( 
	unsigned int index, 
	unsigned int axis
) const
{
	const CPUNode_2D_MED & currNode = m_nodes[index];
	float axisValue = currNode[axis];
	return axisValue;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::GetNodeAxisValue
  Desc:	Helper Method
-------------------------------------------------------------------------*/

void CPUTree_2D_MED::SwapNodes
( 
	unsigned int idx1, 
	unsigned int idx2
)
{
	CPUNode_2D_MED & currNode1 = m_nodes[idx1];
	CPUNode_2D_MED & currNode2 = m_nodes[idx2];
	CPUNode_2D_MED temp;

	temp = currNode1;
	currNode1 = currNode2;
	currNode2 = temp;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::Median1
  Desc: returns median of 1 elements
-------------------------------------------------------------------------*/

unsigned int CPUTree_2D_MED::Median1
(
	unsigned int start,	// IN - index of 1st element in set
	unsigned int shift,	// IN - shift factor to get subsequent elements
	unsigned int axis	// IN - axis to compare on
)
{
	// Nothing to do
	return start;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::Median2
  Desc: returns median of 2 elements
  Note: arbitrarily choose the min value as the median
-------------------------------------------------------------------------*/

unsigned int CPUTree_2D_MED::Median2
(
	unsigned int * v,
	unsigned int start,	// IN - index of 1st element in set
	unsigned int shift,	// IN - shift factor to get subsequent elements
	unsigned int axis	// IN - axis to compare on
)
{
	unsigned int n[2];
	//float v[2];

	n[0] = start;
	n[1] = start+shift;

	//v[0] = GetNodeAxisValue( n[0], axis );
	//v[1] = GetNodeAxisValue( n[1], axis );

	// Compare 1st and 2nd elements
	// Return the minimum of the 2 as our median
	if (v[1] < v[0])
		return n[1];
	else
		return n[0];
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::Median3
  Desc: returns median of 3 elements in 3 comparisions
  Note: 1. Basically sorts 3 element list in 3 comparisions
        2. Extracts middle element to get median
-------------------------------------------------------------------------*/

unsigned int CPUTree_2D_MED::Median3
(
	unsigned int * v,
	unsigned int start,	// IN - index of 1st element in set
	unsigned int shift,	// IN - shift factor to get subsequent elements
	unsigned int axis	// IN - axis to compare on
)
{
	unsigned int i[3];
	unsigned int n[3];
	//float v[3];
	unsigned int tmp;

	// Get Indices
	i[0] = 0;
	i[1] = 1;
	i[2] = 2;

	n[0] = start;
	n[1] = start+shift;
	n[2] = start+2*shift;

	// Get values from indices
	//v[0] = GetNodeAxisValue( n[0], axis );
	//v[1] = GetNodeAxisValue( n[1], axis );
	//v[2] = GetNodeAxisValue( n[2], axis );

	// Compare 1st & 2nd elements
	if (v[i[1]] < v[i[0]])
	{
		// Swap 1st & 2nd elements
		tmp  = i[0];
		i[0] = i[1];
		i[1] = tmp;
	}

	// Compare 1st & 3rd elements
	if (v[i[2]] < v[i[0]])
	{
		// Swap 1st & 3rd elements
		tmp  = i[0];
		i[0] = i[2];
		i[2] = tmp;
	}

	// Compare 2nd & 3rd elements
	if (v[i[2]] < v[i[1]])
	{
		// Swap 2nd & 3rd elements
		tmp  = i[1];
		i[1] = i[2];
		i[2] = tmp;
	}

	// We have successfully sorted such that 1 <= 2 <= 3

	// Return the middle element as our median
	return n[i[1]];
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::Median4
  Desc: returns median of 4 elements in 4 comparisons
  Note: 1. Start merge sort in 2 comparisions
        2. Use 1D interval analysis to find median in 2 more comparisons
-------------------------------------------------------------------------*/

unsigned int CPUTree_2D_MED::Median4
(
	unsigned int * v,
	unsigned int start,	// IN - index of 1st element in set
	unsigned int shift,	// IN - shift factor to get subsequent elements
	unsigned int axis	// IN - axis to compare on
)
{
	unsigned int i[4];
	unsigned int n[4];
	//float v[4];
	unsigned int tmp, median;

	// Get Indices
	i[0] = 0;
	i[1] = 1;
	i[2] = 2;
	i[3] = 3;

	n[0] = start;
	n[1] = start+shift;
	n[2] = start+2*shift;
	n[3] = start+3*shift;

	// Get values from indices
	//v[0] = GetNodeAxisValue( n[0], axis );
	//v[1] = GetNodeAxisValue( n[1], axis );
	//v[2] = GetNodeAxisValue( n[2], axis );
	//v[3] = GetNodeAxisValue( n[3], axis );

	// Start merge sorting in pairs (2 comparisons)

	// Compare 1st & 2nd elements
	if (v[i[1]] < v[i[0]])
	{
		// Swap 1st & 2nd elements
		tmp  = i[0];
		i[0] = i[1];
		i[1] = tmp;
	}

	// Compare 3rd & 4th elements
	if (v[i[3]] < v[i[2]])
	{
		// Swap 3rd & 4th elements
		tmp  = i[2];
		i[2] = i[3];
		i[3] = tmp;
	}

	//
	// We now know that 1 <= 2 and 3 <= 4
	//
	// Looking at these 2 intervals on the 1D number line
	// They can overlap in 6 ways
	//
	//  [1  2][3  4]  interval [1 2] precedes interval [3 4]
	//  [3  4][1  2]  interval [3 4] precedes interval [1 2]
	//  [1 [3  4] 2]  interval [1 2] contains interval [3 4]
	//  [3 [1  2] 4]  interval [3 4] contains interval [1 2]
	//  [1 [3  2] 4]  intervals partially overlap
	//  [3 [1  4] 2]  intervals partially overlap
	// 
	// If we look at the 2nd column where we will eventually
	// grab our median from, we see that 1's and 3's
	// are twice as likely as 2's and 4's giving us a
	// target to go after first.
	//

	// Compare 1st and 3rd elements
	if (v[i[0]] < v[i[2]])
	{
		// Intervals left are ...
		// [1  2][3  4]
		// [1 [3  4] 2]
		// [1 [3  2] 4]
		// Note: we can now safely ignore 1 & 4 as medians.
		// So we are just trying to decide between 2 and 3

		// Choose between 2 and 3 as median
		if (v[i[1]] < v[i[2]])
		{
			// [1  2][3  4]
			// median is at 2nd element
			median = n[i[1]];
		}
		else
		{
			// Two possibilities left
			// [1 [3  4] 2]
			// [1 [3  2] 4]
			// median is at 3rd element
			median = n[i[2]];
		}
	}
	else
	{
		// Intervals are ...
		// [3  4][1  2]
		// [3 [1  2] 4]
		// [3 [1  4] 2]
		// Note: we can now safely ignore 3 & 2 as medians.
		// So we are just trying to decide between 1 and 4

		// Compare 4th and 1st elements
		if (v[i[3]] < v[i[0]])
		{
			// median is 4th element
			median = n[i[3]];
		}
		else
		{
			// median is 1st element
			median = n[i[0]];
		}
	}

	// Return the 2nd element as our median
	return median;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::Median5
  Desc:	Implements Median of 5 elements in 6 comparisions
  Note: 1. Partial Merge sort of 4 elements in 3 comparisions
		   Allows us to safely exclude the lowest element
		   and replace with the 5th element.
		2. One more compare to restore merge sort order on 
		   the updated pair 
		3. Similar to median of 4 above, we can do interval
		   analysis to find the median in 2 more compares
  Note: It takes 7 comparisions to fully sort this 5 element list
-------------------------------------------------------------------------*/

unsigned int CPUTree_2D_MED::Median5
(
	unsigned int * v,
	unsigned int start,	// IN - starting element of group of 5
	unsigned int shift,	// IN - shift increment to get to next element
	unsigned int axis	// IN - axis of value
)
{
	unsigned int i[5];
	unsigned int n[5];
	//float v[5];
	unsigned int tmp, median;

	// Get raw indices
	i[0] = 0;
	i[1] = 1;
	i[2] = 2;
	i[3] = 3;
	i[4] = 4;

	// Get indices of 5 elements (use shift factor)
	n[0] = start;
	n[1] = start+shift;
	n[2] = start+2*shift;
	n[3] = start+3*shift;
	n[4] = start+4*shift;

	// Get 5 values from 5 indices
	//v[0] = GetNodeAxisValue( n[0], axis );
	//v[1] = GetNodeAxisValue( n[1], axis );
	//v[2] = GetNodeAxisValue( n[2], axis );
	//v[3] = GetNodeAxisValue( n[3], axis );
	//v[4] = GetNodeAxisValue( n[4], axis );

	// Compare 1st and 2nd pair of elements (#1)
	if (v[i[1]] < v[i[0]])
	{
		// Swap 1st and 2nd elements
		tmp  = i[0];
		i[0] = i[1];
		i[1] = tmp;
	}

	// Compare 3rd and 4th pair of elements (#2)
	if (v[i[3]] < v[i[2]])
	{
		// Swap 3rd and 4th elements
		tmp  = i[2];
		i[2] = i[3];
		i[3] = tmp;
	}

	// Compare 1st and 3rd elements (#3)
	// The minimum of 4 elements can't possibly be the median of 5 elements
	if (v[i[2]] < v[i[0]])
	{
		// 3rd element is the minimum and we can safely replace it with the 5th element
		i[2] = i[4];

		// Restore pair order between 3rd and 4th element (#4)
		if (v[i[3]] < v[i[2]])
		{
			tmp  = i[2];
			i[2] = i[3];
			i[3] = tmp;
		}
	}
	else
	{
		// 1st element is the minimum and we can safely replace it with the 5th element
		i[0] = i[4];

		// Restore pair order between 1st and 2nd element (#4)
		if (v[i[1]] < v[i[0]])
		{
			tmp  = i[0];
			i[0] = i[1];
			i[1] = tmp;
		}
	}

	// Use interval analysis to find median 
	// of remaining 4 elements in 2 more comparisions
	// the Median of these 4 elements also happens
	// to be the median of the original 5 elements
	if (v[i[0]] < v[i[2]]) // #5
	{
		// Intervals left are ...
		// [1  2][3  4]
		// [1 [3  4] 2]
		// [1 [3  2] 4]
		// Note: we can now safely ignore 1 & 4 as medians.
		// So we are just trying to decide between 2 and 3

		// Choose between 2 and 3 as median
		if (v[i[1]] < v[i[2]]) // #6
		{
			// [1  2][3  4]
			// median is at 2nd element
			median = n[i[1]];
		}
		else
		{
			// Two possibilities left
			// [1 [3  4] 2]
			// [1 [3  2] 4]
			// median is at 3rd element
			median = n[i[2]];
		}
	}
	else
	{
		// Intervals are ...
		// [3  4][1  2]
		// [3 [1  2] 4]
		// [3 [1  4] 2]
		// Note: we can now safely ignore 3 & 2 as medians.
		// So we are just trying to decide between 1 and 4

		// Compare 4th and 1st elements
		if (v[i[3]] < v[i[0]]) // #6
		{
			// median is 4th element
			median = n[i[3]];
		}
		else
		{
			// median is 1st element
			median = n[i[0]];
		}
	}

	// Return our median value
	return median;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::Median5Sort
  Desc:	Implements Median of 5 elements in 10 comparisions and
        table lookup
-------------------------------------------------------------------------*/

unsigned int CPUTree_2D_MED::Median5Sort
(
	unsigned int start,	// IN - starting element of group of 5
	unsigned int shift,	// IN - shift increment to get to next element
	unsigned int axis	// IN - axis of value
)
{
	unsigned int i[5];
	unsigned int n[5];
	float v[5];
	unsigned int tmp, median;

	// Get raw indices
	i[0] = 0;
	i[1] = 1;
	i[2] = 2;
	i[3] = 3;
	i[4] = 4;

	// Get indices of 5 elements (use shift factor)
	n[0] = start;
	n[1] = start+shift;
	n[2] = start+2*shift;
	n[3] = start+3*shift;
	n[4] = start+4*shift;

	// Get 5 values from 5 indices
	v[0] = GetNodeAxisValue( n[0], axis );
	v[1] = GetNodeAxisValue( n[1], axis );
	v[2] = GetNodeAxisValue( n[2], axis );
	v[3] = GetNodeAxisValue( n[3], axis );
	v[4] = GetNodeAxisValue( n[4], axis );

	// Compare 1st and 2nd elements (#1)
	if (v[i[1]] < v[i[0]])
	{
		// Swap 1st and 2nd elements
		tmp  = i[0];
		i[0] = i[1];
		i[1] = tmp;
	}

	// Compare 3rd and 4th elements (#2)
	if (v[i[3]] < v[i[2]])
	{
		// Swap 3rd and 4th elements
		tmp  = i[2];
		i[2] = i[3];
		i[3] = tmp;
	}

	// Find minimum of first 4 elements by comparing 1st and 3rd elements
	// This lowest value of the first 4 values cannot possibly be the median
	// It would end up occupying the 1st or 2nd slot in a fully sorted array
	// So, we can safely ignore this 1st element from now on
	if (v[i[2]] < v[i[0]]) // (#3)
	{
		// Swap 1st & 3rd elements
		tmp  = i[0];
		i[0] = i[2];
		i[2] = tmp;
	}

	// Compare the 2nd and 5th elements (#4)
	if (v[i[1]] < v[i[4]] )
	{
		// Compare the 2nd and 3rd elements (#5)
		if (v[i[1]] < v[i[2]])
		{
			// Compare the 3rd and 5th elements (#6)
			if (v[i[2]] < v[i[4]])
			{
				// Median is 3rd element
				median = n[i[2]];
			}
			else
			{
				// Median is 5th element
				median = n[i[4]];
			}
		}
		else
		{
			// Compare the 2nd and 5th elements (#6)
			if (v[i[1]] < v[i[4]])
			{
				// Median is 2nd element
				median = n[i[1]];
			}
			else
			{
				// Median is 5th element
				median = n[i[4]];
			}
		}
	}
	else
	{
		// Compare the 3rd and 5th elements (#5)
		if (v[i[2]] < v[i[4]])
		{
			// Compare the 4th and 5th elements (#6)
			if (v[i[3]] < v[i[4]])
			{
				// Median is the 3rd element
				median = n[i[2]];
			}
			else
			{
				// Median is the 5th element
				median = n[i[4]];
			}
		}
		else
		{
			// Compare the 2nd and 3rd elements (#6)
			if (v[i[1]] < v[i[2]])
			{
				// Median is the 2nd element
				median = n[i[1]];
			}
			else
			{
				// Median is the 3rd element
				median = n[i[2]];
			}
		}
	}

	// Return our median value
	return median;
}



/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::MedianOf3
  Desc:	Helper method,
		Implements Median of three variant 
		for Median Partitioning algorithm
		returns pivot value for partitioning algorithm
  Note:	enforces invariant
		array[left].val <= array[mid].val <= array[right].val
		where mid = (left+right)/2
-------------------------------------------------------------------------*/

void CPUTree_2D_MED::MedianOf3
(
	unsigned int leftIdx,		// IN - left index
	unsigned int rightIdx,		// IN - right index
	unsigned int axis			// IN - axis to compare
)
{
	// Compute Middle Index from left and right
	unsigned int midIdx = (leftIdx+rightIdx)/2;	

	float leftVal  = GetNodeAxisValue( leftIdx, axis );
	float rightVal = GetNodeAxisValue( rightIdx, axis );
	float midVal   = GetNodeAxisValue( midIdx, axis );

	// Sort left, center, mid value into correct order
	if (leftVal > midVal)
	{
		SwapNodes( leftIdx, midIdx );
	}
	if (leftVal > rightVal)
	{
		SwapNodes( leftIdx, rightIdx );
	}
	if (midVal > rightVal)
	{
		SwapNodes( midIdx, rightIdx );
	}

	// Deliberately move median value to end of array
	SwapNodes( midIdx, rightIdx );
}


/*-------------------------------------------------------------------------
  Name: CPUTree_2D_MED::MedianSortNodes
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

bool CPUTree_2D_MED::MedianSortNodes
(
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
	if (m_cNodes == 0) { return false; }
	if (axis >= INVALID_AXIS) { return false; }
	if (start >= m_cNodes) { return false; }
	if (end >= m_cNodes) { return false; }
	if ((median >= m_cNodes) || 
		(median < start) || 
		(median > end))
	{
		median = (start + end)/2;
	}

	// Perform Median Sort
	int left   = static_cast<int>( start );
	int right  = static_cast<int>( end );
	int middle = static_cast<int>( median );
	int i,j;
	float pivotVal;

	while ( right > left ) 
	{
		// Use Median of 3 variant
		MedianOf3( left, right, axis );
		pivotVal = GetNodeAxisValue( right, axis );

		i = left - 1;
		j = right;

		for (;;) 
		{
			while ( GetNodeAxisValue( ++i, axis ) < pivotVal )
			{
				// Deliberately do nothing
			}

			while ( (GetNodeAxisValue( --j, axis ) > pivotVal) && 
				  (j > left) )
			{
				// Deliberately do nothing
			}
			
			if ( i >= j )
				break;

			SwapNodes( i, j );
		}

		SwapNodes( i, right );

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


/*---------------------------------------------------------
  Name:	KDTree::FindMedianIndex
  Desc:	Find the index of the Median of the elements
		of an array that occurs at every "shift" positions.
---------------------------------------------------------*/

unsigned int CPUTree_2D_MED::FindMedianIndex
(
	unsigned int left,		// IN - Left range to search
	unsigned int right,		// IN - Right range to search
	unsigned int shift,		// IN - Amount to shift
	unsigned int axis		// IN - axis <x,y,z,...> to work on
)
{
	unsigned int i, j, k;
	unsigned int minIndex, groups;
	float minValue, currValue;
	
	groups = (right - left)/shift + 1;
	k = left + groups/2*shift;

	// Insertion Sort on group elements
	for (i = left; i <= k; i += shift)
	{
		minIndex = i;
		minValue = GetNodeAxisValue( minIndex, axis );

		for (j = i; j <= right; j += shift)
		{
			currValue = GetNodeAxisValue( j, axis );
			if (currValue < minValue)
			{
				minIndex = j;
				minValue = currValue;
			}
		}
		SwapNodes( i, minIndex );
	}

	return k;
}


/*---------------------------------------------------------
  Name:	KDTree::FindMedianOfMedians
  Desc:	Computes the median of each group of 5 elements 
		and stores it as the first element of the group. 
		This is done recursively until there is only one 
		group and hence only one Median.
---------------------------------------------------------*/

unsigned int CPUTree_2D_MED::FindMedianOfMedians
(
	unsigned int left,		// IN - left of range to search
	unsigned int right,		// IN - right of range to search
	unsigned int axis		// IN - axis <x,y,z,...> to work on
)
{
	if (left == right)
	{
		return left;
	}

	unsigned int i;
	unsigned int shift = 1;
	while( shift <= (right - left))
	{
		for (i = left; i <= right; i += shift*5)
		{
			unsigned int endIndex = ((i + shift*5 - 1 < right) ? (i + shift*5 - 1) : (right));
			unsigned int medianIndex = FindMedianIndex( i, endIndex, shift, axis );

			SwapNodes( i, medianIndex );
		}
		shift *= 5;
	}

	return left;
}


/*-----------------------------------------------
  Name:	KDTree::MedianOfMediansSortNodes
  Desc:	Sorts nodes between [start,end] into 3
		buckets, 
		nodes with points below the median value,
		the node corresponding to the median point 
		(constains only 1 element)
		and nodes with points above the median value
  Notes:	
	1.  Should take O(N) time to process 
		all points in input range [start,end]
  Notes:
	This approach should be easy to parrallize
	for GPGPU style programming...
	IE we have 'M' GPU's each do 'M' parrallel
	median of median on 5 entries at a time. 
	Which continues to parralize until we have fewer
	than 5*M values left to compute the median for
	at which point we switch back to a single GPU
	approach for the remaining elements.
	For example if we had 32 GPU's and 10,000 elements
	First level, 10,000 elements = 2000 runs of 5 elements 
	  each GPU would do approximately 63 runs each
	Second level, 2,000 elements = 400 runs of 5 elements each
	  each GPU would do approximately 13 runs each
	Third level, 400 elements = 80 runs of 5 elements each
	  each GPU would do approximately 3 runs each
	Fourth Level, 80 elements is less than 5*M (160)
	  do on a single GPU, takes 16 runs
	Fifth Level, 16 elements is less than 5*M (160)
	  do on a single GPU, takes 4 runs
	Sixth Level,  4 elements is less than 5*M (160)
	  do on a single GPU, takes 1 run

	How many runs to compute Median?
	Total GPU Runs:  GPU =   63 +  13 +  3 + (16+4+1) =  180 runs
	Total CPU Runs:  CPU = 2000 + 400 + 80 + (16+4+1) = 2501 runs
	Estimated Speed-Up:  13.89 times faster in this example

	Once we get below 5*M (160), there is no advantage of GPU over CPU

  Credits: 
       Based on the Selection Sort Pivoting 
	   algorithm using the "Median of Medians" 
	   technique from Wikipedia under the
	   "Quick Sort" entry
	   http://en.wikipedia.org/wiki/Quicksort
-----------------------------------------------*/

bool CPUTree_2D_MED::MedianOfMediansSortNodes
(
	unsigned int start,     // IN - start of range
	unsigned int end,       // IN - end of range
	unsigned int & median,  // IN/OUT - approximate median number
	                        //          actual median number
	unsigned int axis       // IN - dimension(axis) to split along (x,y,z)
)
{
	unsigned int pivotIndex, index, idx;
	float pivotValue, currValue;

	// Get the pivot from the median of medians
	pivotIndex = FindMedianOfMedians( start, end, axis );
	pivotValue = GetNodeAxisValue( pivotIndex, axis );
	index = start;

	// Move pivot value to end of array
	SwapNodes( pivotIndex, end );

	// Enforce partition Invariant: 
	//	Partitions original data set {O} with 'n' elements
	//	into 3 datasets <{l}, {m}, {r}>
		// where {l} contains approximately half the elements 
		// and all elements in {l} are less than or equal to {m}
		// {m} contains only 1 element the "median value"
		// and {r} contains approximately half the elements
		// and all elements in {r} are greater than or equal to {m}
	for (idx = start; idx < end; idx++)
	{
		currValue = GetNodeAxisValue( idx, axis );
		if (currValue < pivotValue)
		{
			SwapNodes( idx, index );
			index += 1;
		}
	}

	// Move Median back into correct position in array
	SwapNodes( end, index );

	// Set median index
	median = index;
	return true;
}



/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::ComputeBoundBox
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::ComputeBoundingBox
( 
	unsigned int start,		// IN - start index
	unsigned int end,		// IN - end index
	float        bounds[4]	// OUT - bounding box for all nodes in range
)
{
	// Check Parameters
	if (CPUNode_2D_MED::c_Invalid == start) { return false; }
	if (CPUNode_2D_MED::c_Invalid == end) { return false; }

	unsigned int s = start;
	unsigned int e = end;
	if (e < s) 
	{
		unsigned int temp = s;
		s = e;
		e = temp;
	}

	CPUNode_2D_MED * currNode = NODE_PTR( s );
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




/*-------------------------------------------------------------------------
  Name:	KDTree::Build2D
  Desc:	Build median layout KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Build2D( unsigned int cPoints, const float4 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for Node List
	unsigned int cNodes = cPoints;
	m_nodes = new CPUNode_2D_MED[cPoints];
	if (NULL == m_nodes) { return false; }
	m_cNodes = cNodes;

	// Initialize Node List
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		m_nodes[i].ID( i );
		m_nodes[i].X( x );
		m_nodes[i].Y( y );
		m_nodes[i].Z( z );

		// Bounds Box
		//m_nodes[i].MINX( 0.0f );
		//m_nodes[i].MAXX( 0.0f );
		//m_nodes[i].MINY( 0.0f );
		//m_nodes[i].MAXX( 0.0f );
	}

	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart  = 0;
	unsigned int currEnd    = cNodes - 1;
	unsigned int median     = (currStart + currEnd)/2;
	unsigned int currParentIdx = CPUNode_2D_MED::c_Invalid;
	unsigned int currNodeIdx   = CPUNode_2D_MED::c_Invalid;

	CPUNode_2D_MED * currParentPtr = NULL;
	CPUNode_2D_MED * currNodePtr   = NULL;

	bool bGrabRoot = true;
	m_rootIdx   = median;
	m_startAxis = X_AXIS;

	unsigned int currAxis = m_startAxis;
	unsigned int nextAxis = X_AXIS;
	unsigned int currLR   = KD_NEITHER;

	// Add Root Info to Build Queue
	CPU_BUILD_MED currBuild;
	currBuild.start     = currStart;
	currBuild.end       = currEnd;
	currBuild.parent    = currParentIdx;
	currBuild.leftRight = static_cast<unsigned short>( currLR );
	currBuild.axis      = static_cast<unsigned short>( currAxis );

	// Stack Version
	std::stack<CPU_BUILD_MED> buildStack;
	buildStack.push( currBuild );

	// Process child ranges until we reach 1 node per range (leaf nodes)
	bool bDone = false;
	while (! bDone)
	{
		// Is Build Stack empty ?
		if (buildStack.empty())
		{
			bDone = true;
		}
		else
		{
			// Get Build Info from top of stack
			currBuild = buildStack.top();
			buildStack.pop();

			currStart  = currBuild.start;
			currEnd    = currBuild.end;
			median     = (currStart + currEnd) / 2;
			currNodeIdx   = median;
			currParentIdx = currBuild.parent;
			currLR     = static_cast<unsigned int>( currBuild.leftRight );
			currAxis   = static_cast<unsigned int>( currBuild.axis );
			nextAxis   = NextAxis2D( currAxis );

			// No need to do median sort if only one element is in range (IE a leaf node)
			if (currEnd > currStart)
			{
				// Sort nodes into 2 buckets (on axis plane)
				  // Uses Sedgwedick Partition Algorithm with Median of 3 variant
				bool bResult = MedianSortNodes( currStart, currEnd, median, currAxis );
				if (false == bResult) { return false; }
				currNodeIdx = median;

				if (bGrabRoot)
				{
					// Update root to correct starting value
					m_rootIdx = median;
					bGrabRoot = false;
				}
			}

			// Update Current Node to correct values
			currNodePtr = &(m_nodes[currNodeIdx]);

			currNodePtr->Parent( currParentIdx );
			currNodePtr->Left( CPUNode_2D_MED::c_Invalid );
			currNodePtr->Right( CPUNode_2D_MED::c_Invalid );
			currNodePtr->Axis( currAxis );

			// Compute Bounding Box for this Node
			//float bounds[4];
			//ComputeBoundingBox( currStart, currEnd, bounds );
			//currNodePtr->BOUNDS( bounds );

			// Update Parent Node to point to current node as
			// either it's left or right child
			if (CPUNode_2D_MED::c_Invalid != currParentIdx)
			{
				currParentPtr = &(m_nodes[currParentIdx]);

				switch (currLR)
				{
				case KD_LEFT:
					currParentPtr->Left( currNodeIdx );
					break;
				case KD_RIGHT:
					currParentPtr->Right( currNodeIdx );
					break;
				default:
					break;
				}
			}

			if (currStart < median)
			{
				// Add left child range to build stack
				currBuild.start     = currStart;
				currBuild.end       = median - 1;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_LEFT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );
			
				buildStack.push( currBuild );
			}

			if (median < currEnd)
			{
				// Add right child range to build stack
				currBuild.start     = median + 1;
				currBuild.end       = currEnd;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_RIGHT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );

				buildStack.push( currBuild );
			}
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build2D
  Desc:	Build KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Build2D( const std::vector<float4> & pointList )
{
	// Check Parameters
	unsigned int cPoints = static_cast<unsigned int>( pointList.size() );
	if (0 == cPoints) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for Node List
	unsigned int cNodes = cPoints;
	m_nodes = new CPUNode_2D_MED[cPoints];
	if (NULL == m_nodes) { return false; }
	m_cNodes = cNodes;

	// Initialize Node List
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		m_nodes[i].ID( i );
		m_nodes[i].X( x );
		m_nodes[i].Y( y );
		m_nodes[i].Z( z );
	}

	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart  = 0;
	unsigned int currEnd    = cNodes - 1;
	unsigned int median     = (currStart + currEnd)/2;
	unsigned int currParentIdx = CPUNode_2D_MED::c_Invalid;
	unsigned int currNodeIdx   = CPUNode_2D_MED::c_Invalid;

	CPUNode_2D_MED * currParentPtr = NULL;
	CPUNode_2D_MED * currNodePtr   = NULL;

	bool bGrabRoot = true;
	m_rootIdx   = median;
	m_startAxis = X_AXIS;

	unsigned int currAxis   = m_startAxis;
	unsigned int nextAxis   = X_AXIS;
	unsigned int currLR     = KD_NEITHER;

	// Add Root Info to Build Queue
	CPU_BUILD_MED currBuild;
	currBuild.start     = currStart;
	currBuild.end       = currEnd;
	currBuild.parent    = currParentIdx;
	currBuild.leftRight = static_cast<unsigned short>( currLR );
	currBuild.axis      = static_cast<unsigned short>( currAxis );

	// Deque Version
	std::deque<CPU_BUILD_MED> buildQueue;
	buildQueue.push_back( currBuild );

	// Process child ranges until we reach 1 node per range (leaf nodes)
	bool bDone = false;
	while (! bDone)
	{
		// Is Build queue empty ?
		if (buildQueue.empty())
		{
			bDone = true;
		}
		else
		{
			// Get Build Info from front of queue
			currBuild = buildQueue.front();

			currStart  = currBuild.start;
			currEnd    = currBuild.end;
			median     = (currStart + currEnd) / 2;
			currNodeIdx   = median;
			currParentIdx = currBuild.parent;
			currLR     = static_cast<unsigned int>( currBuild.leftRight );
			currAxis   = static_cast<unsigned int>( currBuild.axis );
			nextAxis   = NextAxis2D( currAxis );

			// No need to do median sort if only one element is in range (IE a leaf node)
			if (currEnd > currStart)
			{
				// Sort nodes into 2 buckets (on axis plane)
				bool bResult = MedianSortNodes( currStart, currEnd, median, currAxis );
				if (false == bResult) { return false; }
				currNodeIdx = median;

				if (bGrabRoot)
				{
					m_rootIdx = median;
					bGrabRoot = false;
				}
			}

			// Update Current Node to correct values
			currNodePtr = &(m_nodes[currNodeIdx]);

			currNodePtr->Parent( currParentIdx );
			currNodePtr->Left( NULL );
			currNodePtr->Right( NULL );
			currNodePtr->Axis( currAxis );

			// Update Parent Node to point to current node as
			// either it's left or right child
			if (CPUNode_2D_MED::c_Invalid != currParentIdx)
			{
				currParentPtr = &(m_nodes[currParentIdx]);

				switch (currLR)
				{
				case KD_LEFT:
					currParentPtr->Left( currNodeIdx );
					break;
				case KD_RIGHT:
					currParentPtr->Right( currNodeIdx );
					break;
				default:
					break;
				}
			}

			if (currStart < median)
			{
				// Add left child range to build queue
				currBuild.start     = currStart;
				currBuild.end       = median - 1;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_LEFT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );
			
				buildQueue.push_back( currBuild );
			}

			if (median < currEnd)
			{
				// Add right child range to build queue
				currBuild.start     = median + 1;
				currBuild.end       = currEnd;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_RIGHT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );

				buildQueue.push_back( currBuild );
			}

			// Pop front element from build queue
			buildQueue.pop_front();
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build3D
  Desc:	Build KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Build3D( unsigned int cPoints, const float4 * pointList )
{
	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for Node List
	unsigned int cNodes = cPoints;
	m_nodes = new CPUNode_2D_MED[cPoints];
	if (NULL == m_nodes) { return false; }
	m_cNodes = cNodes;

	// Initialize Node List
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		m_nodes[i].ID( i );
		m_nodes[i].X( x );
		m_nodes[i].Y( y );
		m_nodes[i].Z( z );
	}

	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart  = 0;
	unsigned int currEnd    = cNodes - 1;
	unsigned int median     = (currStart + currEnd)/2;
	unsigned int currParentIdx = CPUNode_2D_MED::c_Invalid;
	unsigned int currNodeIdx   = CPUNode_2D_MED::c_Invalid;

	CPUNode_2D_MED * currParentPtr = NULL;
	CPUNode_2D_MED * currNodePtr   = NULL;

	bool bGrabRoot = true;
	m_rootIdx   = median;
	m_startAxis = X_AXIS;

	unsigned int currAxis   = m_startAxis;
	unsigned int nextAxis   = X_AXIS;
	unsigned int currLR     = KD_NEITHER;

	// Add Root Info to Build Queue
	CPU_BUILD_MED currBuild;
	currBuild.start     = currStart;
	currBuild.end       = currEnd;
	currBuild.parent    = currParentIdx;
	currBuild.leftRight = static_cast<unsigned short>( currLR );
	currBuild.axis      = static_cast<unsigned short>( currAxis );

	// Deque Version
	std::deque<CPU_BUILD_MED> buildQueue;
	buildQueue.push_back( currBuild );

	// Process child ranges until we reach 1 node per range (leaf nodes)
	bool bDone = false;
	while (! bDone)
	{
		// Is Build queue empty ?
		if (buildQueue.empty())
		{
			bDone = true;
		}
		else
		{
			// Get Build Info from front of queue
			currBuild = buildQueue.front();

			currStart  = currBuild.start;
			currEnd    = currBuild.end;
			median     = (currStart + currEnd) / 2;
			currNodeIdx   = median;
			currParentIdx = currBuild.parent;
			currLR     = static_cast<unsigned int>( currBuild.leftRight );
			currAxis   = static_cast<unsigned int>( currBuild.axis );
			nextAxis   = NextAxis3D( currAxis );

			// No need to do median sort if only one element is in range (IE a leaf node)
			if (currEnd > currStart)
			{
				// Sort nodes into 2 buckets (on axis plane)
				bool bResult = MedianSortNodes( currStart, currEnd, median, currAxis );
				if (false == bResult) { return false; }
				currNodeIdx = median;

				if (bGrabRoot)
				{
					m_rootIdx = median;
					bGrabRoot = false;
				}
			}

			// Update Current Node to correct values
			currNodePtr = &(m_nodes[currNodeIdx]);

			currNodePtr->Parent( currParentIdx );
			currNodePtr->Left( NULL );
			currNodePtr->Right( NULL );
			currNodePtr->Axis( currAxis );

			// Update Parent Node to point to current node as
			// either it's left or right child
			if (CPUNode_2D_MED::c_Invalid != currParentIdx)
			{
				currParentPtr = &(m_nodes[currParentIdx]);

				switch (currLR)
				{
				case KD_LEFT:
					currParentPtr->Left( currNodeIdx );
					break;
				case KD_RIGHT:
					currParentPtr->Right( currNodeIdx );
					break;
				default:
					break;
				}
			}

			if (currStart < median)
			{
				// Add left child range to build queue
				currBuild.start     = currStart;
				currBuild.end       = median - 1;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_LEFT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );
			
				buildQueue.push_back( currBuild );
			}

			if (median < currEnd)
			{
				// Add right child range to build queue
				currBuild.start     = median + 1;
				currBuild.end       = currEnd;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_RIGHT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );

				buildQueue.push_back( currBuild );
			}

			// Pop front element from build queue
			buildQueue.pop_front();
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Build3D
  Desc:	Build KDTree from point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Build3D( const std::vector<float4> & pointList )
{
	// Check Parameters
	unsigned int cPoints = static_cast<unsigned int>( pointList.size() );
	if (0 == cPoints) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for Node List
	unsigned int cNodes = cPoints;
	m_nodes = new CPUNode_2D_MED[cPoints];
	if (NULL == m_nodes) { return false; }
	m_cNodes = cNodes;

	// Initialize Node List
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		m_nodes[i].ID( i );
		m_nodes[i].X( x );
		m_nodes[i].Y( y );
		m_nodes[i].Z( z );
	}

	/*---------------------------------
	  Add Root Info to Build Queue
	---------------------------------*/
	unsigned int currStart  = 0;
	unsigned int currEnd    = cNodes - 1;
	unsigned int median     = (currStart + currEnd)/2;
	unsigned int currParentIdx = CPUNode_2D_MED::c_Invalid;
	unsigned int currNodeIdx   = CPUNode_2D_MED::c_Invalid;

	CPUNode_2D_MED * currParentPtr = NULL;
	CPUNode_2D_MED * currNodePtr   = NULL;

	bool bGrabRoot = true;
	m_rootIdx   = median;
	m_startAxis = X_AXIS;

	unsigned int currAxis   = m_startAxis;
	unsigned int nextAxis   = X_AXIS;
	unsigned int currLR     = KD_NEITHER;

	// Add Root Info to Build Queue
	CPU_BUILD_MED currBuild;
	currBuild.start     = currStart;
	currBuild.end       = currEnd;
	currBuild.parent    = currParentIdx;
	currBuild.leftRight = static_cast<unsigned short>( currLR );
	currBuild.axis      = static_cast<unsigned short>( currAxis );

	// Deque Version
	std::deque<CPU_BUILD_MED> buildQueue;
	buildQueue.push_back( currBuild );

	// Process child ranges until we reach 1 node per range (leaf nodes)
	bool bDone = false;
	while (! bDone)
	{
		// Is Build queue empty ?
		if (buildQueue.empty())
		{
			bDone = true;
		}
		else
		{
			// Get Build Info from front of queue
			currBuild = buildQueue.front();

			currStart  = currBuild.start;
			currEnd    = currBuild.end;
			median     = (currStart + currEnd) / 2;
			currNodeIdx   = median;
			currParentIdx = currBuild.parent;
			currLR     = static_cast<unsigned int>( currBuild.leftRight );
			currAxis   = static_cast<unsigned int>( currBuild.axis );
			nextAxis   = NextAxis3D( currAxis );

			// No need to do median sort if only one element is in range (IE a leaf node)
			if (currEnd > currStart)
			{
				// Sort nodes into 2 buckets (on axis plane)
				bool bResult = MedianSortNodes( currStart, currEnd, median, currAxis );
				if (false == bResult) { return false; }
				currNodeIdx = median;

				if (bGrabRoot)
				{
					m_rootIdx = median;
					bGrabRoot = false;
				}
			}

			// Update Current Node to correct values
			currNodePtr = &(m_nodes[currNodeIdx]);

			currNodePtr->Parent( currParentIdx );
			currNodePtr->Left( NULL );
			currNodePtr->Right( NULL );
			currNodePtr->Axis( currAxis );

			// Update Parent Node to point to current node as
			// either it's left or right child
			if (CPUNode_2D_MED::c_Invalid != currParentIdx)
			{
				currParentPtr = &(m_nodes[currParentIdx]);

				switch (currLR)
				{
				case KD_LEFT:
					currParentPtr->Left( currNodeIdx );
					break;
				case KD_RIGHT:
					currParentPtr->Right( currNodeIdx );
					break;
				default:
					break;
				}
			}

			if (currStart < median)
			{
				// Add left child range to build queue
				currBuild.start     = currStart;
				currBuild.end       = median - 1;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_LEFT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );
			
				buildQueue.push_back( currBuild );
			}

			if (median < currEnd)
			{
				// Add right child range to build queue
				currBuild.start     = median + 1;
				currBuild.end       = currEnd;
				currBuild.parent    = currNodeIdx;
				currBuild.leftRight = static_cast<unsigned short>( KD_RIGHT );
				currBuild.axis      = static_cast<unsigned short>( nextAxis );

				buildQueue.push_back( currBuild );
			}

			// Pop front element from build queue
			buildQueue.pop_front();
		}
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::BruteForceFindClosestPoint2D()
  Desc:	Use Brute Force Algorithm to find closest Node (index)
			Takes O(N) time
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::BruteForceFindClosestPoint2D
(
	const float4 & queryLocation,		// IN  - Location to sample
	unsigned int & closestPointIndex,	// OUT - Index of Closest Point
	unsigned int & closestPointID,		// OUT - ID of Closest Point
	float & bestDistance				// OUT - closest distance
)
{
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	// Get Query Point
	float qX, qY;
	qX = queryLocation.x;
	qY = queryLocation.y;

	// Get 1st Point
	unsigned int  bestIndex = 0;
	CPUNode_2D_MED * currNodePtr = NULL;
	CPUNode_2D_MED * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID = bestNodePtr->ID();

	float bX, bY, bZ;
	bX = bestNodePtr->X();
	bY = bestNodePtr->Y();

	// Calculate distance from query location
	float dX = bX - qX;
	float dY = bY - qY;
	float bestDist2 = dX*dX + dY*dY;
	float diffDist2;

	unsigned int i;
	for (i = 1; i < nNodes; i++)
	{
		// Get Current Point
		CPUNode_2D_MED * currNodePtr = NODE_PTR( i );
		bX = currNodePtr->X();
		bY = currNodePtr->Y();
		bZ = currNodePtr->Z();

		// Calculate Distance from query location
		dX = bX - qX;
		dY = bY - qY;
		diffDist2 = dX*dX + dY*dY;

		// Update Best Point Index
		if (diffDist2 < bestDist2)
		{
			bestIndex = i;
			bestID    = currNodePtr->ID();
			bestDist2 = diffDist2;
		}
	}

	//Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

	// Success (return results)
	closestPointIndex = bestIndex;
	closestPointID = bestID;
	bestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::BruteForceFindClosestPoint3D()
  Desc:	Use Brute Force Algorithm to find closest Node (index)
		Takes O(N) time
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::BruteForceFindClosestPoint3D
(
	const float4 & queryLocation,		// IN  - Location to sample
	unsigned int & closestPointIndex,	// OUT - Index of Closest Point
	unsigned int & closestPointID,		// OUT - ID of Closest Point
	float & bestDistance				// OUT - closest distance
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
	unsigned int  bestIndex = 0;
	CPUNode_2D_MED * currNodePtr = NULL;
	CPUNode_2D_MED * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID = bestNodePtr->ID();

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
	for (i = 1; i < nNodes; i++)
	{
		// Get Current Point
		CPUNode_2D_MED * currNodePtr = NODE_PTR( i );
		bX = currNodePtr->X();
		bY = currNodePtr->Y();
		bZ = currNodePtr->Z();

		// Calculate Distance from query location
		dX = bX - qX;
		dY = bY - qY;
		dZ = bZ - qZ;
		diffDist2 = dX*dX + dY*dY + dZ*dZ;

		// Update Best Point Index
		if (diffDist2 < bestDist2)
		{
			bestIndex = i;
			bestID    = currNodePtr->ID();
			bestDist2 = diffDist2;
		}
	}

	//Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

	// Success (return results)
	closestPointIndex = bestIndex;
	closestPointID    = bestID;
	bestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}



/*-------------------------------------------------------------------------
  Name:	KDTree::FindClosestPoint2D
  Desc:	Use a stack instead of a queue
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::FindClosestPoint2D
(
	const float4 & queryLocation,	// IN  - Query Location
	unsigned int & closestIndex,	// OUT - closest point index to sample location
	unsigned int & closestID,		// OUT - ID of closest point
	       float & bestDistance		// OUT - best distance
) const
{
	// Make sure we have something to search
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	unsigned int start, end, median, currNodeIdx, parentIdx;
	unsigned int currAxis, nextAxis, prevAxis;
	float qX, qY;
	float cX, cY;
	float dX, dY;
	float diffDist2, bestDist2;
	float queryValue, splitValue, parentSplit, parentQV;
	float diff, diff2, pDiff, pDiff2;
	const CPUNode_2D_MED * currNodePtr = NULL;
	const CPUNode_2D_MED * parentPtr = NULL;
	const CPUNode_2D_MED * bestNodePtr = NULL;
	unsigned int bestIndex, bestID;

	qX = queryLocation.x;
	qY = queryLocation.y;
	//qZ = queryLocation.z;

	// Setup Search Queue
	CPU_SEARCH_MED currSearch;
	currSearch.start     = 0;
	currSearch.end       = nNodes - 1;
	median               = (currSearch.start + currSearch.end)/2;
	currSearch.InOut     = static_cast<unsigned short>( KD_UNKNOWN );
	currSearch.axis      = static_cast<unsigned short>( m_startAxis );

	median = (currSearch.start + currSearch.end)/2;

	bestIndex   = median;
	bestNodePtr = NODE_PTR( bestIndex );
	bestID = bestNodePtr->ID();

	cX = bestNodePtr->X();
	cY = bestNodePtr->Y();
	dX = cX - qX;
	dY = cY - qY;
	bestDist2 = dX*dX + dY*dY;

	std::stack<CPU_SEARCH_MED> searchStack;		
	searchStack.push( currSearch );

	int nNodesProcessed = 0;

	while (! searchStack.empty())
	{
		nNodesProcessed++;

		// Get Current Node from top of stack
		currSearch = searchStack.top();

		// Pop node from top of stack
		searchStack.pop();

		// Get Median Node
		start    = currSearch.start;
		end      = currSearch.end;
		// Assert( start <= end );
		median   = (start+end)/2;		// Root Index (Split Index) for this range
		currAxis = static_cast<unsigned int>( currSearch.axis );
		nextAxis = NextAxis2D( currAxis );

		currNodeIdx = median;
		currNodePtr = NODE_PTR( currNodeIdx );

		// Early Exit Check
		if (currSearch.InOut == KD_OUT)
		{
			parentIdx = currNodePtr->Parent();
			if (CPUNode_2D_MED::c_Invalid != parentIdx)
			{
				parentPtr = NODE_PTR( parentIdx );
				prevAxis = PrevAxis2D( currAxis );
				parentQV = AxisValue( queryLocation, prevAxis );
				parentSplit = (*parentPtr)[prevAxis];
				pDiff  = parentSplit - parentQV;
				pDiff2 = pDiff*pDiff;
				if (pDiff2 >= bestDist2)
				{
					// We can do an early exit for this node
					continue;
				}
			}
		}

		// Get Best Fit Dist for checking child ranges
		queryValue = AxisValue( queryLocation, currAxis );
		splitValue = (*currNodePtr)[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		cX = currNodePtr->X();
		cY = currNodePtr->Y();
		dX = cX - qX;
		dY = cY - qY;
		diffDist2 = dX*dX + dY*dY;

		// Update best point (so far)
		if (diffDist2 < bestDist2)
		{
			bestIndex = currNodeIdx;
			bestDist2 = diffDist2;
			bestID    = currNodePtr->ID();
		}

		if (start < end)
		{
			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges

				// Check if we should add Right Sub-range to stack
				if (diff2 < bestDist2)
				{
					// Add to Search Stack
					if (median < end)
					{
						// Push onto top of stack
						currSearch.start = median+1;
						currSearch.end   = end;
						currSearch.axis  = static_cast<unsigned short>( nextAxis );
						currSearch.InOut = static_cast<unsigned short>( KD_OUT );	// Query Point outside this 1D interval
						searchStack.push( currSearch );
					}
				}

				// Always Add Left Sub-range to search path
				if (start < median)
				{
					// Push onto top of stack
					currSearch.start = start;
					currSearch.end   = median-1;
					currSearch.axis  = static_cast<unsigned short>( nextAxis );
					currSearch.InOut = static_cast<unsigned short>( KD_IN );	// Query Point inside this 1D interval
					searchStack.push( currSearch );
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
					if (start < median)
					{
						// Push onto top of stack
						currSearch.start = start;
						currSearch.end   = median-1;
						currSearch.axis  = static_cast<unsigned short>( nextAxis );
						currSearch.InOut = static_cast<unsigned short>( KD_OUT );	// Query Point outside this 1D interval
						searchStack.push( currSearch );
					}
				}
					
				// Always Add Right Sub-range
				if (median < end)
				{
					// Push onto top of stack
					currSearch.start = median+1;
					currSearch.end   = end;
					currSearch.axis  = static_cast<unsigned short>( nextAxis );
					currSearch.InOut = static_cast<unsigned short>( KD_IN );	// Query Point inside this 1D interval
					searchStack.push( currSearch );
				}
			}

		}

		//unsigned int stackSize = static_cast<unsigned int>( searchStack.size() );
		//printf( "Stack Size = %d\n", stackSize );
	}

	printf( "Num Nodes Processed = %d\n", nNodesProcessed );

	// Successful (return results)
	closestIndex = bestIndex;
	closestID    = bestID;
	bestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::FindClosestPoint2DAlt
  Desc:	Use a stack instead of a queue
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::FindClosestPoint2DAlt
(
	const float4 & queryLocation,	// IN  - Query Location
	unsigned int & closestIndex,	// OUT - closest point index to sample location
	unsigned int & closestID,		// OUT - ID of closest point
	       float & bestDistance		// OUT - best distance
) const
{
	static unsigned int c_Prev2D[3] = { 1, 0 };	// Previous Indices
	static unsigned int c_Next2D[3] = { 1, 0 };	// Next Indices

	//static unsigned int c_Prev3D[3] = { 2, 0, 1 };	// Previous Indices
	//static unsigned int c_Next3D[3] = { 1, 2, 0 };	// Next Indices

	// Make sure we have something to search
	if (m_cNodes <= 0) { return false; }

	const CPUNode_2D_MED * currPtr = NULL;
	unsigned int currIdx, currAxis, currInOut, nextIdx, nextAxis;
	unsigned int bestIdx, bestID;
	float q[3];
	float c[3];
	float d[3];
	float diff, diff2;
	float diffDist2, bestDist2;
	float queryValue, splitValue;
	unsigned int nNodesProcessed = 0;

	// Query Position
	q[0] = queryLocation.x;
	q[1] = queryLocation.y;
	//q[2] = queryLocation.z;

	// Compute Root Index
	currIdx = (m_cNodes-1)>>1;
	currPtr = NODE_PTR( currIdx );

	// Curr Node Position
	c[0] = currPtr->X();
	c[1] = currPtr->Y();
	//c[2] = currPtr->Z();

	// Setup Search Node (for root node)
	CPU_SEARCH_ALT_MED currSearch;
	currSearch.nodeFlags  = (currIdx & 0x1FFFFFFF) | ((m_startAxis << 29) & 0x60000000) | ((KD_IN << 31) & 0x8000000);		                                 
	currSearch.splitValue = c[m_startAxis];

	// Set Initial Guess equal to root node
	bestIdx = currIdx;
	bestID  = currPtr->ID();

	d[0] = c[0] - q[0];
	d[1] = c[1] - q[1];
	//d[2] = c[2] - q[2];
	bestDist2 = (d[0]*d[0]) + (d[1]*d[1]); // +d[2]*d[2];

	std::stack<CPU_SEARCH_ALT_MED> searchStack;
	searchStack.push( currSearch );

	while (! searchStack.empty())
	{
		// Statistics
		nNodesProcessed++;

		// Get Current Node from top of stack
		currSearch = searchStack.top();

		// Pop node from top of stack
		searchStack.pop();

		// Get Node Info
		currIdx   = (currSearch.nodeFlags & 0x1FFFFFFF);
		currAxis  = (currSearch.nodeFlags & 0x60000000) >> 29;
		currInOut = (currSearch.nodeFlags & 0x80000000) >> 31;
		
		nextAxis  = c_Next2D[currAxis];

		// Early Exit Check
		if (currInOut == KD_OUT)
		{
			queryValue = q[c_Prev2D[currAxis]];	
			splitValue = currSearch.splitValue;		// Split Value of Parent Node
			diff  = splitValue - queryValue;		
			diff2 = diff*diff;
			if (diff2 >= bestDist2)
			{
				// We can do an early exit for this node
				continue;
			}
		}

		currPtr = NODE_PTR( currIdx );

		// Get Best Fit Dist for checking child ranges
		queryValue = q[currAxis];
		splitValue = (*currPtr)[currAxis];
		diff  = splitValue - queryValue;
		diff2 = diff*diff;

		// Calc Dist from Median Node to queryLocation
		c[0] = currPtr->X();
		c[1] = currPtr->Y();
		//c[2] = currPtr->Z();
		d[0] = c[0] - q[0];
		d[1] = c[1] - q[1];
		//d[2] = c[2] - q[2];
		diffDist2 = (d[0]*d[0]) + d[1]*d[1]; // + d[2]*d[2];

		// Update closest point Idx
		if (diffDist2 < bestDist2)
		{
			bestIdx   = currIdx;
			bestDist2 = diffDist2;
			bestID    = currPtr->ID();
		}

		if (queryValue <= splitValue)
		{
			// [...QL...BD]...SV		-> Include Left range only
			//		or
			// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
			
			// Check if we should add Right Sub-range to stack
			if (diff2 < bestDist2)
			{
				nextIdx = currPtr->Right();
				if (CPUNode_2D_MED::c_Invalid != nextIdx)
				{
					currSearch.nodeFlags = (nextIdx & 0x1FFFFFFF) | ((nextAxis << 29) & 0x60000000) | ((KD_OUT << 31) & 0x80000000);
					currSearch.splitValue = splitValue;
					searchStack.push( currSearch );
				}
			}

			// Always Add Left Sub-range to search path
			nextIdx = currPtr->Left();
			if (CPUNode_2D_MED::c_Invalid != nextIdx)
			{
				currSearch.nodeFlags = (nextIdx & 0x1FFFFFFF) | ((nextAxis << 29) & 0x60000000) | ((KD_IN << 31) & 0x80000000);
				currSearch.splitValue = splitValue;
				searchStack.push( currSearch );
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
				nextIdx = currPtr->Left();
				if (CPUNode_2D_MED::c_Invalid != nextIdx)
				{
					currSearch.nodeFlags = (nextIdx & 0x1FFFFFFF) | ((nextAxis << 29) & 0x60000000) | ((KD_OUT << 31) & 0x80000000);
					currSearch.splitValue = splitValue;
					searchStack.push( currSearch );
				}
			}
				
			// Always Add Right Sub-range
			nextIdx = currPtr->Right();
			if (CPUNode_2D_MED::c_Invalid != nextIdx)
			{
				currSearch.nodeFlags = (nextIdx & 0x1FFFFFFF) | ((nextAxis << 29) & 0x60000000) | ((KD_IN << 31) & 0x8000000);
				currSearch.splitValue = splitValue;
				searchStack.push( currSearch );
			}
		}

		//unsigned int stackSize = static_cast<unsigned int>( searchStack.size() );
		//printf( "Stack Size = %d\n", stackSize );
	}

	//printf( "Num Nodes Processed = %d\n", nNodesProcessed );

	// Successful (return results)
	closestIndex = bestIdx;
	closestID    = bestID;
	bestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}



/*-------------------------------------------------------------------------
  Name:	KDTree::Find_QNN_2D
  Desc:	Finds closest point in kd-tree for each query point
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Find_QNN_2D
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

	unsigned int rootIdx = (m_cNodes-1)>>1;
	CPUNode_2D_MED * currNode = NULL;

	CPU_SEARCH_ALT_MED searchStack[KD_STACK_SIZE_CPU];

	// Local Parameters
	CPU_NN_Result best;
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	float dx, dy;
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int stackTop = 0;

	unsigned int currQuery;
	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Reset stack for each search
		stackTop = 0;
		
		// Compute Query Index

		// Load current Query Point into local (fast) memory
		const float4 & queryPoint = queryPoints[currQuery];

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cNodes = 0;

		// Put root search info on stack
		searchStack[stackTop].nodeFlags  = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cNodes++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currIdx   = (searchStack[stackTop].nodeFlags & 0x1FFFFFFFU);
			currAxis  = (searchStack[stackTop].nodeFlags & 0x60000000U) >> 29;
			currInOut = (searchStack[stackTop].nodeFlags & 0x80000000U) >> 31;
			
			nextAxis  = ((currAxis == 0) ? 1 : 0);

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				// Next Line is effectively queryValue = queryPoints[prevAxis];
				queryValue = ((currAxis == 0) ? queryPoint.y : queryPoint.x);
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
			queryValue = ((currAxis == 0) ? queryPoint.x : queryPoint.y);
			splitValue = ((currAxis == 0) ? currNode->X() : currNode->Y());
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryPoint.x;
			dy = currNode->Y() - queryPoint.y;
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
					if (0xFFFFFFFF != currNode->Right())	// cInvalid
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (0xFFFFFFFF != currNode->Left())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
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
					if (0xFFFFFFFFU != currNode->Left())
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (0xFFFFFFFFU != currNode->Right())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// We now have the Best Index but we really need the best ID so grab it from ID list
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->ID();

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
  Desc:	Find All Nearest neighbors
  Note: The query points and search points are the same
        in Find_All_NN_2D
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Find_ALL_NN_2D
( 
	CPU_NN_Result * queryResults	// OUT: Results
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	unsigned int rootIdx = (m_cNodes-1)>>1;
	CPUNode_2D_MED * currNode = NULL;
	CPUNode_2D_MED * queryNode = NULL;
	float2 queryPoint;

	// Local Parameters
	CPU_NN_Result best;
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	float dx, dy;
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int stackTop = 0;
	CPU_SEARCH_ALT_MED searchStack[KD_STACK_SIZE_CPU];

	unsigned int currQuery;
	for (currQuery = 0; currQuery < m_cNodes; currQuery++)
	{
		// Reset stack for each query
		stackTop = 0;
		
		// Compute Query Index

		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryPoint.x = queryNode->X();
		queryPoint.y = queryNode->Y();

		// Set Initial Guess equal to root node
		best.Id    = rootIdx;
		best.Dist  = 3.0e+38F;	// Choose A huge Number to start with for Best Distance
		//best.cNodes = 0;

		// Put root search info on stack
		searchStack[stackTop].nodeFlags  = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cNodes++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currIdx   = (searchStack[stackTop].nodeFlags & 0x1FFFFFFFU);
			currAxis  = (searchStack[stackTop].nodeFlags & 0x60000000U) >> 29;
			currInOut = (searchStack[stackTop].nodeFlags & 0x80000000U) >> 31;
			
			nextAxis  = ((currAxis == 0) ? 1 : 0);

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				// Next Line is effectively queryValue = queryPoints[prevAxis];
				queryValue = ((currAxis == 0) ? queryPoint.y : queryPoint.x);
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
			queryValue = ((currAxis == 0) ? queryPoint.x : queryPoint.y);
			splitValue = ((currAxis == 0) ? currNode->X() : currNode->Y());
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryPoint.x;
			dy = currNode->Y() - queryPoint.y;
			diffDist2 = (dx*dx) + (dy*dy);

			// Update closest point Idx
			if ((diffDist2 < best.Dist) && (diffDist2 > 0.0f))
			{
				best.Id   = currIdx;
				best.Dist = diffDist2;
			}
			//best.id  = ((diffDist2 < best.dist) ? currIdx   : best.id);
			//best.dist = ((diffDist2 < best.dist) ? diffDist2 : best.dist);

			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges
				
				// Check if we should add Right Sub-range to stack
				if (diff2 < best.Dist)
				{
					if (0xFFFFFFFF != currNode->Right())	// cInvalid
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				if (0xFFFFFFFF != currNode->Left())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
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
					if (0xFFFFFFFFU != currNode->Left())
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				if (0xFFFFFFFFU != currNode->Right())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].splitValue = splitValue;
					stackTop++;
				}
			}
		}

		// REMAP:  We now have the best node index 
		//         but we really need the best point ID so grab it.
		currNode = NODE_PTR( best.Id );
		best.Id = currNode->ID();

		// Turn Dist2 into true distance
		best.Dist = sqrt( best.Dist );

		// Store Query Result
		queryResults[currQuery] = best;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	KDTree::Find_KNN_2D
  Desc:	Finds 'k' closest points in the kd-tree to each query point
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Find_KNN_2D
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

	unsigned int rootIdx = (m_cNodes-1)>>1;		// Root found at median of [0,n-1] = (0+n-1)/2
	CPUNode_2D_MED * currNode = NULL;

	if (kVal > 100)
	{
		kVal = 100;
	}

	// Local Parameters
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	float dx, dy;
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int stackTop, maxHeap, countHeap;
	float dist2Heap, bestDist2;
	CPU_SEARCH_ALT_MED	searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	unsigned int currQuery;
	for (currQuery = 0; currQuery < nQueries; currQuery++)
	{
		// Get current Query Point
		const float4 & qryPoint = queryPoints[currQuery];

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e38f;

		// Put root search info on stack
		searchStack[stackTop].nodeFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cNodes++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currIdx   = (searchStack[stackTop].nodeFlags & 0x1FFFFFFFU);
			currAxis  = (searchStack[stackTop].nodeFlags & 0x60000000U) >> 29;
			currInOut = (searchStack[stackTop].nodeFlags & 0x80000000U) >> 31;
			
			nextAxis  = ((currAxis == 0) ? 1 : 0);

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					// Next Line is effectively queryValue = queryPoints[prevAxis];
					queryValue = ((currAxis == 0) ? qryPoint.y : qryPoint.x);
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
			queryValue = ((currAxis == 0) ? qryPoint.x : qryPoint.y);
			splitValue = ((currAxis == 0) ? currNode->X() : currNode->Y());
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - qryPoint.x;
			dy = currNode->Y() - qryPoint.y;
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
					if (0xFFFFFFFF != currNode->Right())	// cInvalid
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				//nextIdx = currNodes[tidx].Left;
				if (0xFFFFFFFF != currNode->Left())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
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
					//nextIdx = currNodes[tidx].Left;
					if (0xFFFFFFFFU != currNode->Left())
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				//nextIdx = currNodes[tidx].Right;
				if (0xFFFFFFFFU != currNode->Right())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].splitValue = splitValue;
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
			unsigned int offset = i * nPadQueries;

			currNode = NODE_PTR( knnHeap[i1].Id );
			knnHeap[i1].Id   = currNode->ID();					// Really need ID's not indices
			knnHeap[i1].Dist = sqrtf( knnHeap[i1].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[currQuery+offset] = knnHeap[i1];
		}
	}

	// Success
	return true;
}



/*-------------------------------------------------------------------------
  Name:	KDTree::Find_ALL_KNN_2D
  Desc:	Find All 'k' Nearest neighbors
  Note: The query points and search points are the same
        in Find_All_KNN_2D
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Find_ALL_KNN_2D
( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nPadSearch	// In: number of padded search points
)
{
	// Check Parameters
	if (NULL == queryResults) { return false; }

	unsigned int rootIdx = (m_cNodes-1)>>1;
	CPUNode_2D_MED * currNode = NULL;
	CPUNode_2D_MED * queryNode = NULL;

	if (kVal > 100)
	{
		kVal = 100;
	}

	// Local Parameters
	unsigned int currIdx, currAxis, currInOut, nextAxis;
	float dx, dy;
	float diff, diff2;
	float diffDist2;
	float queryValue, splitValue;
	unsigned int stackTop, maxHeap, countHeap;
	float dist2Heap, bestDist2;
	float2 queryPoint;
	CPU_SEARCH_ALT_MED	searchStack[32];	// Search Stack
	CPU_NN_Result 	knnHeap[100];		// 'k' NN Heap

	// Loop over search points in kd-tree as query points
	unsigned int currQuery;
	for (currQuery = 0; currQuery < m_cNodes; currQuery++)	
	{
		// Load current Query Point (search point) into local (fast) memory
		queryNode = NODE_PTR( currQuery );
		queryPoint.x = queryNode->X();
		queryPoint.y = queryNode->Y();

		// Search Stack Variables
		currIdx = rootIdx;
		stackTop = 0;

		// 'k' NN Heap variables
		maxHeap   = kVal;		// Maximum # elements on knnHeap
		countHeap = 0;			// Current # elements on knnHeap
		dist2Heap = 0.0f;		// Max Dist of any element on heap
		bestDist2 = 3.0e38f;

		// Put root search info on stack
		searchStack[stackTop].nodeFlags = (rootIdx & 0x1FFFFFFF); // | ((currAxis << 29) & 0x60000000); // | ((currInOut << 31) & 0x8000000);;
		searchStack[stackTop].splitValue  = 3.0e+38F;
		stackTop++;

		while (stackTop != 0)
		{
			// Statistics
			//best.cNodes++;

			// Get Current Node from top of stack
			stackTop--;

			// Get Node Info
			currIdx   = (searchStack[stackTop].nodeFlags & 0x1FFFFFFFU);
			currAxis  = (searchStack[stackTop].nodeFlags & 0x60000000U) >> 29;
			currInOut = (searchStack[stackTop].nodeFlags & 0x80000000U) >> 31;
			
			nextAxis  = ((currAxis == 0) ? 1 : 0);

			// Early Exit Check
			if (currInOut == 1)	// KD_OUT
			{
				if (countHeap == maxHeap) // Is heap full yet ?!?
				{
					// Next Line is effectively queryValue = queryPoints[prevAxis];
					queryValue = ((currAxis == 0) ? queryPoint.y : queryPoint.x);
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
			queryValue = ((currAxis == 0) ? queryPoint.x : queryPoint.y);
			splitValue = ((currAxis == 0) ? currNode->X() : currNode->Y());
			diff  = splitValue - queryValue;
			diff2 = diff*diff;

			// Calc Dist from Median Node to queryLocation
			dx = currNode->X() - queryPoint.x;
			dy = currNode->Y() - queryPoint.y;
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
				//	Do Simple Array Insertion
				//-------------------------------

				// Update Best Dist
				//dist2Heap = ((countHeap == 0) ? diffDist2 : ((diffDist2 > dist2Heap) ? diff2Dist2 : dist2Heap);
				//bestDist2 = 3.0e38f;

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
					if (0xFFFFFFFF != currNode->Right())	// cInvalid
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}

				// Always Add Left Sub-range to search path
				//nextIdx = currNodes[tidx].Left;
				if (0xFFFFFFFF != currNode->Left())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x80000000U;
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
					//nextIdx = currNodes[tidx].Left;
					if (0xFFFFFFFFU != currNode->Left())
					{
						// Push Onto top of stack
						searchStack[stackTop].nodeFlags  = (currNode->Left() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U) | 0x80000000U;
						searchStack[stackTop].splitValue = splitValue;
						stackTop++;
					}
				}
					
				// Always Add Right Sub-range
				//nextIdx = currNodes[tidx].Right;
				if (0xFFFFFFFFU != currNode->Right())
				{
					// Push Onto top of stack
					searchStack[stackTop].nodeFlags  = (currNode->Right() & 0x1FFFFFFFU) | ((nextAxis << 29) & 0x60000000U); // | 0x8000000U;
					searchStack[stackTop].splitValue = splitValue;
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
			unsigned int offset = i * nPadSearch;

			currNode = NODE_PTR( knnHeap[i1].Id );
			knnHeap[i1].Id   = currNode->ID();					// Really need ID's not indices
			knnHeap[i1].Dist = sqrtf( knnHeap[i1].Dist );		// Get True distance (not distance squared)

			// Store Result 
			queryResults[currQuery+offset] = knnHeap[i1];
		}
	}

	// Success
	return true;
}



/*-------------------------------------------------------------------------
  Name:	KDTree::Find_QNN_3D
  Desc:	Find closest point index to query location in KD Tree
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::Find_QNN_3D
(
	const float4 & queryLocation,	// IN  - Query Location
	unsigned int & closestIndex,	// OUT - closest point index to sample location
	unsigned int & closestID,		// OUT - ID of closest point
	       float & bestDistance		// OUT - best distance
) const
{
	// Make sure we have something to search
	unsigned int nNodes = m_cNodes;
	if (nNodes <= 0) { return false; }

	unsigned int start, end, median, currNodeIdx;
	unsigned int currAxis, nextAxis;
	float qX, qY, qZ;
	float cX, cY, cZ;
	float dX, dY, dZ;
	float diffDist2, bestDist2;
	float queryValue, splitValue;
	float diff, diff2;
	const CPUNode_2D_MED * currNodePtr = NULL;

	qX = queryLocation.x;
	qY = queryLocation.y;
	qZ = queryLocation.z;

	// Setup Search Queue
	CPU_SEARCH_MED currSearch;
	currSearch.start     = 0;
	currSearch.end       = nNodes - 1;
	//currSearch.parent    = CPUNode_2D_MED::c_Invalid;
	//currSearch.leftRight = static_cast<unsigned short>( KD_NEITHER );
	currSearch.axis      = static_cast<unsigned short>( m_startAxis );

	median = (currSearch.start + currSearch.end)/2;

	unsigned int bestIndex		 = median;
	const CPUNode_2D_MED * bestNodePtr = NODE_PTR( bestIndex );
	unsigned int bestID = bestNodePtr->ID();
	cX = bestNodePtr->X();
	cY = bestNodePtr->Y();
	cZ = bestNodePtr->Z();

	dX = cX - qX;
	dY = cY - qY;
	dZ = cZ - qZ;
	bestDist2 = dX*dX + dY*dY + dZ*dZ;

	std::deque<CPU_SEARCH_MED> searchQueue;		
	searchQueue.push_back( currSearch );

	//int nNodesProcessed = 0;

	// Start searching for closest points
	while (! searchQueue.empty())
	{
		//nNodesProcessed++;

		// Get Current Node from front of queue
		currSearch  = searchQueue.front();

		// Get Median Node
		start    = currSearch.start;
		end      = currSearch.end;
		// Assert( start <= end );
		median   = (start+end)/2;		// Root Index (Split Index) for this range
		currAxis = static_cast<unsigned int>( currSearch.axis );
		nextAxis = NextAxis2D( currAxis );

		// Calc Dist from Median Node to queryLocation
		currNodeIdx = median;
		currNodePtr = NODE_PTR( currNodeIdx );
		cX = currNodePtr->X();
		cY = currNodePtr->Y();
		cZ = currNodePtr->Z();
		dX = cX - qX;
		dY = cY - qY;
		dZ = cZ - qZ;
		diffDist2 = dX*dX + dY*dY + dZ*dZ;

		// Update closest point Idx
		if (diffDist2 < bestDist2)
		{
			bestIndex = currNodeIdx;
			bestID    = currNodePtr->ID();
			bestDist2 = diffDist2;
		}

		// Get Best Fit Dist for checking child ranges
		switch (currAxis)
		{
		case X_AXIS:
			queryValue = qX;
			break;
		case Y_AXIS:
			queryValue = qY;
			break;
		case Z_AXIS:
			queryValue = qZ;
			break;
		}
		splitValue = (*currNodePtr)[currAxis];

		if (start < end)
		{
			if (queryValue <= splitValue)
			{
				// [...QL...BD]...SV		-> Include Left range only
				//		or
				// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges

				// Always Add Left Sub-range to search path
				if (start < median)
				{
					currSearch.start = start;
					currSearch.end   = median-1;
					currSearch.axis  = static_cast<unsigned short>( nextAxis );
					//currSearch.parent    = ???;
					//currSearch.leftRight = static_cast<unsigned short>( KD_LEFT );
					searchQueue.push_back( currSearch );
				}

				// Check if we should add Right Sub-range to search path
				diff  = splitValue - queryValue;
				diff2 = diff*diff;
				if (diff2 < bestDist2)
				{
					// Add to Search Queue
					if (median < end)
					{
						currSearch.start = median+1;
						currSearch.end   = end;
						currSearch.axis  = static_cast<unsigned short>( nextAxis );
						//currSearch.parent    = ???;
						//currSearch.leftRight = static_cast<unsigned short>( KD_RIGHT );
						searchQueue.push_back( currSearch );
					}
				}
			}
			else
			{
				// SV...[BD...QL...]		-> Include Right sub range only
				//		  or
				// [BD...SV...QL...]		-> Include Both Left and Right Sub Ranges

				// Check if we should add left sub-range to search path
				diff = queryValue - splitValue;
				diff2 = diff*diff;
				if (diff2 < bestDist2)
				{
					// Add to search queue
					if (start < median)
					{
						currSearch.start = start;
						currSearch.end   = median-1;
						currSearch.axis  = static_cast<unsigned short>( nextAxis );
						//currSearch.parent    = ???;
						//currSearch.leftRight = static_cast<unsigned short>( KD_LEFT );
						searchQueue.push_back( currSearch );
					}
				}
					
				// Always Add Right Sub-range
				if (median < end)
				{
					currSearch.start = median+1;
					currSearch.end   = end;
					currSearch.axis  = static_cast<unsigned short>( nextAxis );
					//currSearch.parent    = ???;
					//currSearch.leftRight = static_cast<unsigned short>( KD_RIGHT );
					searchQueue.push_back( currSearch );
				}
			}
		}

		// Finished processing this node, get rid of it
		searchQueue.pop_front();
	}

	//Dumpf( TEXT( "\r\n Find Closest Point Index - Num Nodes Processed = %d\r\n\r\n" ), nNodesProcessed );

	// Successful (return results)
	closestIndex = bestIndex;
	closestID    = bestID;
	bestDistance = static_cast<float>( sqrt( bestDist2 ) );
	return true;
}


/*-------------------------------------------------------------------------
  Name:	FindClosestPoint2D
  Desc:	Find Closest Point
-------------------------------------------------------------------------*/

bool FindClosestPoint2D
( 
	const std::vector<float4> & searchList,	// IN - Points to put in Search List
	const std::vector<float4> & queryList,	// IN - Points to query against search list
	std::vector<CPU_NN_Result> & queryResults // OUT - Results of queries
)
{
	unsigned int cSearch = static_cast<unsigned int>( searchList.size() );
	if (cSearch <= 0) { return false; }

	unsigned int cQuery = static_cast<unsigned int>( queryList.size() );
	if (cQuery <= 0) { return false; }

	// Make Sure Results list is large enough to hold results
	unsigned int cResults = static_cast<unsigned int>( queryResults.size() );
	if (cResults < cQuery) 
	{
		queryResults.resize( cQuery );
		cResults = cQuery;
	}

	// Build KD Tree
	CPUTree_2D_MED myKDTree;
	myKDTree.Build2D( searchList );

	unsigned int idxVal, idVal;
	float distVal;
	bool bResult = true;

	// Search All Query Points for closest points 
	unsigned int i;
	for (i = 0; i < cQuery; i++)
	{
		// Get Current Query Point
		const float4 & currQuery = queryList[i];

		// Search KDTree for closest point
		bResult = myKDTree.FindClosestPoint2D( currQuery, idxVal, idVal, distVal );
		if (! bResult)
		{
			// Error
			return false;
		}

		// Store Results in Result List
		queryResults[i].Id   = idVal;
		queryResults[i].Dist = distVal;
	}

	// Success
	return true;
}


/*-------------------------------------------------------------------------
  Name:	FindClosestPoint3D
  Desc:	Find Closest Point 3D
-------------------------------------------------------------------------*/

bool FindClosestPoint3D
( 
	const std::vector<float4> & searchList,	// IN - Points to put in Search List
	const std::vector<float4> & queryList,	// IN - Points to query against search list
	std::vector<CPU_NN_Result> & queryResults // OUT - Results of queries
)
{
	unsigned int cSearch = static_cast<unsigned int>( searchList.size() );
	if (cSearch <= 0) { return false; }

	unsigned int cQuery = static_cast<unsigned int>( queryList.size() );
	if (cQuery <= 0) { return false; }

	// Make Sure Results list is large enough to hold results
	unsigned int cResults = static_cast<unsigned int>( queryResults.size() );
	if (cResults < cQuery) 
	{
		queryResults.resize( cQuery );
		cResults = cQuery;
	}

	// Build KD Tree
	CPUTree_2D_MED myKDTree;
	myKDTree.Build3D( searchList );

	unsigned int idxVal, idVal;
	float distVal;
	bool bResult = true;

	// Search All Query Points for closest points 
	unsigned int i;
	for (i = 0; i < cQuery; i++)
	{
		// Get Current Query Point
		const float4 & currQuery = queryList[i];

		// Search KDTree for closest point
		bResult = myKDTree.Find_QNN_3D( currQuery, idxVal, idVal, distVal );
		if (! bResult)
		{
			// Error
			return false;
		}

		// Store Results in Result List
		queryResults[i].Id   = idVal;
		queryResults[i].Dist = distVal;
	}

	// Success
	return true;
}


/*-----------------------------------------------
  Name:  CPUTree_2D_MED::ChoosePivot()
  Desc:  Choose Pivot according to 
         Median of Medians algorithm
  Note:  
	1. Divide vector into n/5 subsequences
       of 5 consecutive elements each
    2. Find median of 5 in each sequence
	   Move to front of sequence
    3. Recursively find the median of medians
	   This will be our pivot value
    Need special case logic to handle last
	subsequence which can be 0-4 elements long
-----------------------------------------------*/

unsigned int CPUTree_2D_MED::ChoosePivot
(
	unsigned int start,	// IN - Starting index in range
	unsigned int end,	// IN - Ending index in range
	unsigned int axis	// IN - Axis to pivot on
)
{
	unsigned int nElems, nGroups, leftOver, currElems;
	unsigned int idx, shift, currStart;
	unsigned int groupMedian;

	shift = 1;	// Start with shift size of 1

	// assert( end >= start );
	
	nElems = end - start + 1;
	while (shift <= nElems)
	{
		currElems = nElems / shift;
		nGroups = currElems / 5;
		leftOver = currElems - (nGroups*5);

		// Handle all full groups of 5 elements
		currStart = start;
		for (idx = 0; idx < nGroups; idx++)
		{
			// Find median for this group of 5 elements
			groupMedian = Median5( NULL, currStart, shift, axis );

			// Swap the median element to front of group
			if (groupMedian != currStart)
			{
				SwapNodes( currStart, groupMedian );
			}

			// Move to next group
			currStart += 5 * shift;
		}

		// Handle last partially full group
		if (leftOver > 0)
		{
			switch (leftOver)
			{
			case 1: // Compute median of 1 element (trivial)
				groupMedian = currStart;
				break;

			case 2: // Compute median of 2 elements
				groupMedian = Median2( NULL, currStart, shift, axis );
				break;

			case 3: // Compute median of 3 elements
				groupMedian = Median3( NULL, currStart, shift, axis );
				break;

			case 4: // Compute median of 4 elements
				// Get median of 4
				groupMedian = Median4( NULL, currStart, shift, axis );
				break;

			default: // Error, should never get here
				break;
			}

			// Swap the median element to front of group
			if (groupMedian != currStart)
			{
				SwapNodes( currStart, groupMedian );
			}
		}

		// Increase shift size by factor of 5
		shift *= 5;
	}

	// The computed Median of Medians partitioning pivot 
	// should now be stored in the 1st element of our array range.
	return start;
}


/*-----------------------------------------------
  Name:  CPUTree_2D_MED::Partition
  Desc:  paritition array A into 3 sets 
	{<L>, <M>, <R>} for specified pivot index.

	m = value at specified pivot index
	<L> = Left Set {i: a(i) <= m}
	<M> = {m}, IE singleton value
	<R> = Right set {j: a(j) >= m)
-----------------------------------------------*/

unsigned int CPUTree_2D_MED::Partition
(
	unsigned int start,	// IN - start of array
	unsigned int end,	// IN - end of array
	unsigned int pivot,	// IN - index of pivot value
	unsigned int axis	// IN - axis to do 'partition' on
)
{
	unsigned int i, j;
	float pivotVal, currVal;

	pivotVal = GetNodeAxisValue( pivot, axis );

	i = start;
	j = end;

	while (1)
	{
		currVal = GetNodeAxisValue( i, axis );
		while (currVal >= pivotVal)
		{
			i = i + 1;
			currVal = GetNodeAxisValue( i, axis );
		}

		currVal = GetNodeAxisValue( j, axis );
		while (currVal <= pivotVal)
		{
			j = j + 1;
			currVal = GetNodeAxisValue( j, axis );
		}

		if (i < j)
		{
			SwapNodes( i, j );
		}
		else
		{
			// Successfully partitioned
			return  i;
		}

	}
}



/*-----------------------------------------------
  Name:  Selection
  Desc:  modify array A such that the nth element
         parition's A into 3 sets {<L>,<M>,<R>}
		 <L> = {i: a(i) < m}
		 <M> = {m = a(n) }
		 <R> = {i: a(i) >= m}
-----------------------------------------------*/

unsigned int CPUTree_2D_MED::Select
(
	unsigned int start,		// IN - start of array
	unsigned int end,		// IN - end of array
	unsigned int nth,		// IN - nth element from start of array to select
	unsigned int axis		// IN - axis to do 'select' on
)
{
	unsigned int pivotIdx, partIdx, temp;
	unsigned int s, e, k, nElems;

	// Make sure start <= end
	if (end < start)
	{
		temp = start;
		start = end;
		end = temp;
	}

	// Make sure nth is in range [start,end]
	nElems = end - start + 1;
	if (nth >= nElems )
	{
		// Error - nth element is out of range
		return (unsigned int)-1;
	}

	s = start;
	e = end;
	k = nth;
	partIdx = e+1;

	while (partIdx != k)
	{
		// Find the Pivot value <M>
		pivotIdx = ChoosePivot( s, e, axis );

		// Partition on pivot into 3 sets {<L>, <M>, <R>}
		// return index of partition set <M>
		partIdx  = Partition( s, e, pivotIdx, axis );

		if (partIdx > nth)
		{
			// Iterate on left partition to find nth element
			//s = s;
			e = partIdx;
			// k = k
		}
		else
		{
			// Iterate on right partition to find nth element
			s = partIdx+1;
			e = e;
			k = k - partIdx;
		}
	}

	// return index of nth element
	return nth;
}



void swap( char a[], int i, int j) 
{
	char tmp;
	tmp  = a[i];
	a[i] = a[j];
	a[j] = tmp;
}

void enumerate( char a[], int n) 
{
	int i;
	if (0 == n)
	{
		printf("%s\n", a);
	}
	else
	{
		for (i = 0; i < n; i++)
		{
			swap(a, i, n-1);
			enumerate(a, n-1);
			swap(a, n-1, i);
		}
	}
}

/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::TestMedian
  Desc:	Test Median of 5 behavior
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::TestMedian()
{
	// All 2 permutations of 2 values
	unsigned int test2Vals[2][2] = 
	{
		{1,2},
		{2,1}
	};

	// All 6 permutations of 3 values
	unsigned int test3Vals[6][3] = 
	{
		{1,2,3},
		{1,3,2},
		{2,1,3},
		{2,3,1},
		{3,1,2},
		{3,2,1}
	};

	// All 24 permutations of 4 values
	unsigned int test4Vals[24][4] = {
		{1,2,3,4},
		{1,2,4,3},
		{1,3,2,4},
		{1,3,4,2},
		{1,4,2,3},
		{1,4,3,2},
		{2,1,3,4},
		{2,1,4,3},
		{2,3,1,4},
		{2,3,4,1},
		{2,4,1,3},
		{2,4,3,1},
		{3,1,2,4},
		{3,1,4,2},
		{3,2,1,4},
		{3,2,4,1},
		{3,4,1,2},
		{3,4,2,1},
		{4,1,2,3},
		{4,1,3,2},
		{4,2,1,3},
		{4,2,3,1},
		{4,3,1,2},
		{4,3,2,1}
	};

	// All 120 permutations of 5 values
	unsigned int test5Vals[120][5] = 
	{
		{1,2,3,4,5},
		{1,2,3,5,4},
		{1,2,4,3,5},
		{1,2,4,5,3},
		{1,2,5,3,4},
		{1,2,5,4,3},
		{1,3,2,4,5},
		{1,3,2,5,4},
		{1,3,4,2,5},
		{1,3,4,5,2},
		{1,3,5,2,4},
		{1,3,5,4,2},
		{1,4,2,3,5},
		{1,4,2,5,3},
		{1,4,3,5,2},
		{1,4,3,2,5},
		{1,4,5,2,3},
		{1,4,5,3,2},
		{1,5,2,3,4},
		{1,5,2,4,3},
		{1,5,3,2,4},
		{1,5,3,4,2},
		{1,5,4,2,3},
		{1,5,4,3,2},

		{2,1,3,4,5},
		{2,1,3,5,4},
		{2,1,4,3,5},
		{2,1,4,5,3},
		{2,1,5,3,4},
		{2,1,5,4,3},
		{2,3,1,4,5},
		{2,3,1,5,4},
		{2,3,4,1,5},
		{2,3,4,5,1},
		{2,3,5,1,4},
		{2,3,5,4,1},
		{2,4,1,3,5},
		{2,4,1,5,3},
		{2,4,3,1,5},
		{2,4,3,5,1},
		{2,4,5,1,3},
		{2,4,5,3,1},
		{2,5,1,3,4},
		{2,5,1,4,3},
		{2,5,3,1,4},
		{2,5,3,4,1},
		{2,5,4,1,3},
		{2,5,4,3,1},

		{3,1,2,4,5},
		{3,1,2,5,4},
		{3,1,4,2,5},
		{3,1,4,5,2},
		{3,1,5,2,4},
		{3,1,5,4,2},
		{3,2,1,4,5},
		{3,2,1,5,4},
		{3,2,4,1,5},
		{3,2,4,5,1},
		{3,2,5,1,4},
		{3,2,5,4,1},
		{3,4,1,2,5},
		{3,4,1,5,2},
		{3,4,2,1,5},
		{3,4,2,5,1},
		{3,4,5,1,2},
		{3,4,5,2,1},
		{3,5,1,2,4},
		{3,5,1,4,2},
		{3,5,2,1,4},
		{3,5,2,4,1},
		{3,5,4,1,2},
		{3,5,4,2,1},

		{4,1,2,3,5},
		{4,1,2,5,3},
		{4,1,3,2,5},
		{4,1,3,5,2},
		{4,1,5,2,3},
		{4,1,5,3,2},
		{4,2,1,3,5},
		{4,2,1,5,3},
		{4,2,3,1,5},
		{4,2,3,5,1},
		{4,2,5,1,3},
		{4,2,5,3,1},
		{4,3,1,2,5},
		{4,3,1,5,2},
		{4,3,2,1,5},
		{4,3,2,5,1},
		{4,3,5,1,2},
		{4,3,5,2,1},
		{4,5,1,2,3},
		{4,5,1,3,2},
		{4,5,2,1,3},
		{4,5,2,3,1},
		{4,5,3,1,2},
		{4,5,3,2,1},

		{5,1,2,3,4},
		{5,1,2,4,3},
		{5,1,3,2,4},
		{5,1,3,4,2},
		{5,1,4,2,3},
		{5,1,4,3,2},
		{5,2,1,3,4},
		{5,2,1,4,3},
		{5,2,3,1,4},
		{5,2,3,4,1},
		{5,2,4,1,3},
		{5,2,4,3,1},
		{5,3,1,2,4},
		{5,3,1,4,2},
		{5,3,2,1,4},
		{5,3,2,4,1},
		{5,3,4,1,2},
		{5,3,4,2,1},
		{5,4,1,2,3},
		{5,4,1,3,2},
		{5,4,2,1,3},
		{5,4,2,3,1},
		{5,4,3,1,2},
		{5,4,3,2,1},
	};


	// Test Median of 2 functionality
	std::cout << "Median of 2 test" << std::endl;

	unsigned int i, j;
	for (i = 0; i < 2; i++)
	{
		unsigned int * currVals = &(test2Vals[i][0]);
		unsigned int currIndex = Median2( currVals, 0, 1, 0 );
		if (currVals[currIndex] != 1 )
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 2; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Fail !!! " << std::endl; 
		}
		else
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 2; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Success !!! " << std::endl; 
		}
	}
	std::cout << std::endl;



	// Test Median of 3 functionality
	std::cout << "Median of 3 test" << std::endl;
	for (i = 0; i < 6; i++)
	{

		unsigned int * currVals = &(test3Vals[i][0]);
		unsigned int currIndex = Median3( currVals, 0, 1, 0 );
		if (currVals[currIndex] != 2 )
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 3; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Fail !!! " << std::endl; 
		}
		else
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 3; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Success !!! " << std::endl; 
		}
	}
	std::cout << std::endl;


	// Test Median of 4 functionality
	std::cout << "Median of 4 test" << std::endl;
	for (i = 0; i < 24; i++)
	{
		unsigned int * currVals = &(test4Vals[i][0]);
		unsigned int currIndex = Median4( currVals, 0, 1, 0 );
		if (currVals[currIndex] != 2 )
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 4; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Fail !!! " << std::endl; 
		}
		else
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 4; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Success !!! " << std::endl; 
		}
	}
	std::cout << std::endl;


	// Test Median of 5 functionality
	std::cout << "Median of 5 test" << std::endl;
	for (i = 0; i < 120; i++)
	{

		unsigned int * currVals = &(test5Vals[i][0]);
		unsigned int currIndex = Median5( currVals, 0, 1, 0 );
		if (currVals[currIndex] != 3 )
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 5; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Fail !!! " << std::endl; 
		}
		else
		{
			std::cout << "[" << i << "] - ";
			for (j = 0; j < 5; j++)
			{
				std::cout << currVals[j] << " ";
			}
			std::cout << " - Success !!! " << std::endl; 
		}
	}
	std::cout << std::endl;

	return false;
}



/*-------------------------------------------------------------------------
  Name:	CPUTree_2D_MED::TestSelect
  Desc:	Test Select behavior on specified point list
-------------------------------------------------------------------------*/

bool CPUTree_2D_MED::TestSelect
(
	unsigned int cPoints,		// Number of points in list
	const float4 * pointList,	// Raw points
	unsigned int kth,			// Find kth element with select
	unsigned int axis			// Axis to select on
)
{
	TestMedian();

	// Check Parameters
	if (0 == cPoints) { return false; }
	if (NULL == pointList) { return false; }

	// Release current resources for KD Tree
	FiniNodes();

	// Allocate Space for Node List
	unsigned int cNodes = cPoints;
	m_nodes = new CPUNode_2D_MED[cPoints];
	if (NULL == m_nodes) { return false; }
	m_cNodes = cNodes;

	// Initialize Node List
	unsigned int i;
	float x,y,z;
	for (i = 0; i < cNodes; i++)
	{
		x = pointList[i].x;
		y = pointList[i].y;
		z = pointList[i].z;
		m_nodes[i].ID( i );
		m_nodes[i].X( x );
		m_nodes[i].Y( y );
		m_nodes[i].Z( z );

		// Bounds Box
		//m_nodes[i].MINX( 0.0f );
		//m_nodes[i].MAXX( 0.0f );
		//m_nodes[i].MINY( 0.0f );
		//m_nodes[i].MAXX( 0.0f );
	}

	// Test Select behavior
	unsigned int start = 0;
	unsigned int end = cPoints - 1;
	unsigned int rIdx;
	rIdx = Select( start, end, kth, axis );

	// Check result
	bool bResult = false;
	if (rIdx != kth)
		bResult = true;
	else
		bResult = false;
	return bResult;
}
