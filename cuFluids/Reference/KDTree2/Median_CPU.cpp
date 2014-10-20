/*-----------------------------------------------------------------------------
  File:  Median_CPU.cpp
  Desc:
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// Standard Includes
#ifndef _INC_STDLIB
	#include <stdlib.h>
#endif
#ifndef _INC_STDIO
	#include <stdio.h>
#endif
#ifndef _STRING_
	#include <string.h>
#endif
#ifndef _INC_MATH
	#include <math.h>
#endif

// Cuda Includes
#include <cutil.h>
#include <vector_types.h>	

// App Includes
#include "QueryResult.h"
#include "KD_API.h"
#include "KDTree_CPU.h"
#include "KDTree_GPU.h"


/*-------------------------------------
  Function Definitions
-------------------------------------*/

/*---------------------------------------------------------
  Name: ValueAt
  Desc: gets specified value at idx, axis
  Note: assumes pointValues is an array of points
		where each point has 4 values {x,y,z,w}
---------------------------------------------------------*/

inline float ValueAt
(
	const float* pointValues,	// IN - 'points' vector
	I32 pntIdx,					// IN - index of point to retrieve
	I32 axis					// IN - axis of point value to retrieve
)
{
	I32 currIdx = 4*pntIdx+axis;
	float currVal = pointValues[currIdx];
	return currVal;
}


/*---------------------------------------------------------
  Name: SwapNodes
  Desc: swaps 2 points at specified indices
  Note: assumes pointValues is an array of points
		where each point has 4 values {x,y,z,w}
---------------------------------------------------------*/

inline I32 SwapNodes
(
	float* pointValues,	// IN/OUT - 'points' vector
	I32 iIdx,			// IN - first index of point to swap
	I32 jIdx			// IN - second index of other point to swap
)
{
/*
	I32 currIdx1 = 4*iIdx;
	I32 currIdx2 = 4*minIndex;
	float4 tempPoint;
	tempPoint = *((float4 *)(pointValues[currIdx1]);
	*((float4 *)(pointValues[currIdx1]) = *((float4 *)(pointValues[currIdx2]);
	*((float4 *)(pointValues[currIdx2]) = tempPoint;
*/
}


/*---------------------------------------------------------
  Name: FindMedianIndex
  Desc: Find the index of the Median of the elements
		of an array that occur at every "shift" positions.
  Note: assumes pointValues is an array of points
		where each point has 4 values {x,y,z,w}
---------------------------------------------------------*/

inline I32 FindMedianIndex
(
	float* pointValues,	// IN/OUT - 'points' vector
	I32 iLeft,			// IN - Left range to search
	I32 iRight,			// IN - Right range to search
	I32 iShift,			// IN - Amount to shift
	I32 axis		// IN - axis <x,y,z,...> to work on
)
{
	I32 iIdx, minIndex, jIdx, currIdx, swapIdx;
	I32 iGroups = (iRight - iLeft)/iShift + 1;
	I32 kIdx = iLeft + iGroups/2*iShift;
	float4 currPoint;
	float minValue, currValue;

	for (iIdx = iLeft; iIdx <= kIdx; iIdx += iShift)
	{
		minIndex = iIdx;
		minValue = ValueAt( pointValues, minIndex, axis );

		for (jIdx = iIdx; jIdx <= iRight; jIdx += iShift)
		{
			currValue = ValueAt( pointValues, jIdx, axis );
			if (currValue < minValue)
			{
				minIndex = jIdx;
				minValue = currValue;
			}
		}

		// Swap 2 points at specified indices
		SwapNodes( pointValues, iIdx, minIndex );
	}

	return kIdx;
}


/*---------------------------------------------------------
  Name: FindMedianOfMedians
  Desc: Computes the median of each group of 5 elements 
		and stores it as the first element of the group. 
		Recursively does this till there is only one group 
		and hence only one Median
  Note: assumes point values is an array of points
		where each point has 4 values {x,y,z,w}
---------------------------------------------------------*/

inline I32 FindMedianOfMedians
(
	float* pointValues,	// IN/OUT - 'points' vector
	I32 iLeft,			// IN - left of range to search
	I32 iRight,			// IN - right of range to search
	I32 axis			// IN - axis <x,y,z,...> to work on
)
{
	if (iLeft == iRight)
	{
		return iLeft;
	}

	I32 iIdx;
	I32 iShift = 1;
	while( iShift <= (iRight - iLeft))
	{
		for (iIdx = iLeft; iIdx <= iRight; iIdx += iShift*5)
		{
			I32 endIndex = ((iIdx + iShift*5 - 1 < iRight) ? (iIdx + iShift*5 - 1) : (iRight));
			I32 medianIndex = FindMedianIndex( iIdx, endIndex, iShift, axis );

			// Swap 2 points at specified indices
			SwapNodes( pointValues, iIdx, medianIndex );
		}
		iShift *= 5;
	}

	return iLeft;
}


/*-----------------------------------------------
  Name:	MedianSort_CPU
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

I32 MedianSort_CPU
(
	float* pointValues,	// IN/OUT - 'points' vector
	I32 iLeft,			// IN - left or range to partition
	I32 iRight,			// IN - right of range to partition
	I32 axis			// IN - axis value <x,y,z,...> to work on
)
{
	// Makes the leftmost element a good pivot,
	// specifically the median of medians
	I32 pivotIndex = FindMedianOfMedians( iLeft, iRight, axis );
	float pivotValue = ValueAt( pointValues, pivotIndex, axis );
	float currValue;
	I32 index = iLeft;
	I32 idx;

	// Move pivot to end of array
	SwapNodes( pointValues, pivotIndex, iRight );

	for (idx = iLeft; idx < iRight; idx++)
	{
		currValue = ValueAt( pointValues, idx, axis );
		if (currValue < pivotValue)
		{
			SwapNodes( pointValues, idx, index );
			index += 1;
		}
	}

	// Move Median back into correct position
	SwapNodes( pointValues, iRight, index );

	// Return median index
	return index;
}

