/*-----------------------------------------------------------------------------
  File: KDTree_main.cu
  Desc: Main entry point of program

  Log:   Created by Shawn D. Brown (4/15/07)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, CUDA
#include <cutil_inline.h>

// includes, project
#include "CPUTREE_API.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

AppGlobals g_app;


/*-------------------------------------
  Local Function Declarations
-------------------------------------*/

bool MaxHeapify
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
);

bool MaxHeapifyReverse
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
);

bool MinHeapify
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
);

bool MinHeapifyReverse
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
);

void TestDualHeap();


/*-------------------------------------
  Function Definitions
-------------------------------------*/


/*---------------------------------------------------------
  Name:   main()
  Desc:   main entry point of program
---------------------------------------------------------*/

int
main( int argc, const char** argv) 
{
	bool bResult;
	int rc = EXIT_FAILURE;

	//TestDualHeap();

	// Initialize Global Variables to Default settings
	bResult = InitGlobals( g_app );
	if (false == bResult)
	{
		goto lblCleanup;
	}

	// Get Command Line Parameters
		// Which overrides some of the global settings
	printf( "Get Command Line Params...\n" );
	bResult = GetCommandLineParams( argc, argv, g_app );
	if (false == bResult)
	{
		goto lblCleanup;
	}
	printf( "Done getting Command Line Params...\n\n" );

	// Initialize CUDA display device
	printf( "Initializing Device...\n" );
	bResult = InitCUDA( g_app );
	if (false == bResult)
	{
		goto lblCleanup;
	}
	printf( "Done Initializing Device...\n\n" );

	//if (false == bResult)
	//{
	//	goto lblCleanup;
	//}

	bResult = Test_NN_API();
#if (TEST_TYPE == TEST_BRUTE_FORCE)
	// brute force test
	//bResult = BruteForce3DTest();
#elif (TEST_TYPE == TEST_MEDIAN_KDTREE)
	// kd-tree test (Median layout)
	//bResult = CPUTest_2D_MED( g_app );
#elif (TEST_TYPE == TEST_LBT_KDTREE)
	// kd-tree test (Left-balanced layout)
	//bResult = CPUTest_2D_LBT( g_app );
	//bResult = CPUTest_3D_LBT( g_app );
	//bResult = CPUTest_4D_LBT( g_app );
#elif (TEST_TYPE == TEST_MEDIAN_SELECT)
	// Not yet implemented ...
#else
#endif

	if (false == bResult)
	{
		goto lblCleanup;
	}
	
	// Success
	rc = EXIT_SUCCESS;

lblCleanup:
	// Cleanup
	FiniCUDA();

	// Prompt User before exiting ?!?
	if (!g_app.nopromptOnExit)
	{
		printf("\nPress ENTER to exit...\n");
		getchar();
	}

	// Exit Application
	exit( rc );
}


/*-----------------------------------------------
  Name:  MaxHeapify
  Desc:  convert an array of elements into 
         a max heap
  Notes: in normal order from start to stop

  Pre-invariants:  <None>

Properties:
  Given index 'idx' in range [1,n]

  Parent(i) = floor(i/2)  or  i >> 1;
  Left   = (idx*2);		  or  i << 1;
  Right  = (idx*2) + 1	  or  (i << 1) + 1

  the root is at position 1.
  the root has no parent

  A node is a leaf node if
	left > n && right > n

Post-Invariants:  
  1.) Binary array represents a complete binary tree
  2.) Heap Invariant:  
	  a[Parent(i)] >= a[i] for all i in [1,n]
-----------------------------------------------*/

bool MaxHeapify
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
)
{
	// Check params
	if (start > stop)
	{
		int temp = start;
		start    = stop;
		stop     = temp;
	}
	int nElems = stop - start + 1;

	if ((nElems <= 0) || (elems == NULL))
	{
		// Error
		return false;
	}

	int * zeroElems = &(elems[start]);	// Array with zero-based indexing
	int * oneElems = zeroElems - 1;	    // Array with one-based indexing
	I32 k, currIdx, childIdx, rightIdx;
	int origVal, currVal, childVal, rightVal;
	for (k = nElems/2; k >= 1; k--)
	{
		currIdx  = k;		// Get index of value to demote	
		childIdx = k << 1;	// Get left child of current index

		// Save original value creating first hole for extended swap
		origVal = oneElems[currIdx];
		currVal = origVal;

		// Compare current index to it's children
		while (childIdx <= nElems)
		{
			// Find largest child 
			childVal = oneElems[childIdx];		// Left child
			rightIdx = childIdx+1;
			if (rightIdx <= nElems)
			{
				rightVal = oneElems[rightIdx];	// Right child
				if (childVal < rightVal)
				{
					// Right child is largest
					childIdx = rightIdx;
					childVal = rightVal;
				}
			}

			// Compare largest child priority to current priority
			if (currVal >= childVal)
			{
				// Current is larger than both children, exit loop
				break;
			}
			
			// Fill hole with child value
				// continuing extended swap
			oneElems[currIdx] = oneElems[childIdx];

			// Update indices
			currIdx  = childIdx;	
			childIdx = currIdx << 1; 
		}

		// Put original value back into final hole (ending extended swap)
		oneElems[currIdx] = origVal;
	}

	// Success, done heapifying
	return true;
}


/*-----------------------------------------------
  Name:  MaxHeapifyReverse
  Desc:  convert an array of elements into 
         a max heap
  Notes: in reverse order from stop to start

  Pre-invariants:  <None>

Properties:
  Given index 'idx' in range [1,n]

  Parent(i) = floor(i/2)  or  i >> 1;
  Left   = (idx*2);		  or  i << 1;
  Right  = (idx*2) + 1	  or  (i << 1) + 1

  the root is at position n
  the root has no parent

  A node is a leaf node if
	left > n && right > n

Post-Invariants:  
  1.) Binary array represents a complete binary tree
  2.) Heap Invariant:  
	  a[Parent(i)] >= a[i] for all i in [1,n]

-----------------------------------------------*/

bool MaxHeapifyReverse
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
)
{
	// Check params
	if (start > stop)
	{
		int temp = start;
		start    = stop;
		stop     = temp;
	}
	int nElems = stop - start + 1;

	if ((nElems <= 0) || (elems == NULL))
	{
		// Error
		return false;
	}

	// Start from end of array
		// Use negative indices to work backwards
	int * zeroElems = &(elems[stop]);	// Array with zero-based indexing
	int * oneElems = zeroElems + 1;	    // Array with one-based indexing
	I32 k, currIdx, childIdx, rightIdx;
	int origVal, currVal, childVal, rightVal;
	for (k = nElems/2; k >= 1; k--)
	{
		currIdx  = k;		// Get index of value to demote	
		childIdx = k << 1;	// Get left child of current index

		// Save original value creating first hole for extended swap
		origVal = oneElems[-currIdx];
		currVal = origVal;

		// Compare current index to it's children
		while (childIdx <= nElems)
		{
			// Find largest child 
			childVal = oneElems[-childIdx];		// Left child
			rightIdx = childIdx+1;
			if (rightIdx <= nElems)
			{
				rightVal = oneElems[-rightIdx];	// Right child
				if (childVal < rightVal)
				{
					// Right child is largest
					childIdx = rightIdx;
					childVal = rightVal;
				}
			}

			// Compare largest child priority to current priority
			if (currVal >= childVal)
			{
				// Current is larger than both children, exit loop
				break;
			}
			
			// Fill hole with child value
				// continuing extended swap
			oneElems[-currIdx] = oneElems[-childIdx];

			// Update indices
			currIdx  = childIdx;	
			childIdx = currIdx << 1; 
		}

		// Put original value back into final hole (ending extended swap)
		oneElems[-currIdx] = origVal;
	}

	// Success, done heapifying
	return true;
}


/*-----------------------------------------------
  Name:  MinHeapify
  Desc:  convert an array of elements into 
         a min heap
  Notes: In normal order from start to stop
-----------------------------------------------*/

bool MinHeapify
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
)
{
	// Check params
	if (start > stop)
	{
		int temp = start;
		start    = stop;
		stop     = temp;
	}
	int nElems = stop - start + 1;

	if ((nElems <= 0) || (elems == NULL))
	{
		// Error
		return false;
	}

	int * zeroElems = &(elems[start]);	// Array with zero-based indexing
	int * oneElems = zeroElems - 1;	    // Array with one-based indexing
	I32 k, currIdx, childIdx, rightIdx;
	int origVal, currVal, childVal, rightVal;
	for (k = nElems/2; k >= 1; k--)
	{
		currIdx  = k;		// Get index of value to demote	
		childIdx = k << 1;	// Get left child of current index

		// Save original value creating first hole for extended swap
		origVal = oneElems[currIdx];
		currVal = origVal;

		// Compare current index to it's children
		while (childIdx <= nElems)
		{
			// Find smallest child 
			childVal = oneElems[childIdx];		// Left child
			rightIdx = childIdx+1;
			if (rightIdx <= nElems)
			{
				rightVal = oneElems[rightIdx];	// Right child
				if (childVal > rightVal)
				{
					// Right child is smallest
					childIdx = rightIdx;
					childVal = rightVal;
				}
			}

			// Compare smallest child priority to current priority
			if (currVal <= childVal)
			{
				// Current is smaller than both children, exit loop
				break;
			}
			
			// Fill hole with child value
				// continuing extended swap
			oneElems[currIdx] = oneElems[childIdx];

			// Update indices
			currIdx  = childIdx;	
			childIdx = currIdx << 1; 
		}

		// Put original value back into final hole (ending extended swap)
		oneElems[currIdx] = origVal;
	}

	// Success, done heapifying
	return true;
}


/*-----------------------------------------------
  Name:  MinHeapifyReverse
  Desc:  convert an array of elements into 
         a min heap
  Notes: In reverse order from stop to start
-----------------------------------------------*/

bool MinHeapifyReverse
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int start,		// IN - range [start,stop] to turn into a heap
	int stop		// IN - ditto
)
{
	// Check params
	if (start > stop)
	{
		int temp = start;
		start    = stop;
		stop     = temp;
	}
	int nElems = stop - start + 1;

	if ((nElems <= 0) || (elems == NULL))
	{
		// Error
		return false;
	}

	int * zeroElems = &(elems[stop]);	// Array with zero-based indexing
	int * oneElems = zeroElems + 1;	    // Array with one-based indexing
	I32 k, currIdx, childIdx, rightIdx;
	int origVal, currVal, childVal, rightVal;
	for (k = nElems/2; k >= 1; k--)
	{
		currIdx  = k;		// Get index of value to demote	
		childIdx = k << 1;	// Get left child of current index

		// Save original value creating first hole for extended swap
		origVal = oneElems[-currIdx];
		currVal = origVal;

		// Compare current index to it's children
		while (childIdx <= nElems)
		{
			// Find smallest child 
			childVal = oneElems[-childIdx];		// Left child
			rightIdx = childIdx+1;
			if (rightIdx <= nElems)
			{
				rightVal = oneElems[-rightIdx];	// Right child
				if (childVal > rightVal)
				{
					// Right child is smallest
					childIdx = rightIdx;
					childVal = rightVal;
				}
			}

			// Compare smallest child priority to current priority
			if (currVal <= childVal)
			{
				// Current is smaller than both children, exit loop
				break;
			}
			
			// Fill hole with child value
				// continuing extended swap
			oneElems[-currIdx] = oneElems[-childIdx];

			// Update indices
			currIdx  = childIdx;	
			childIdx = currIdx << 1; 
		}

		// Put original value back into final hole (ending extended swap)
		oneElems[-currIdx] = origVal;
	}

	// Success, done heapifying
	return true;
}


#define CHECK_STATE 1
#define	SWAP_STATE  2

typedef struct _heapStackNode
{
	int idx;
	int state;
} HeapStackNode;

/*-----------------------------------------------
  Name:  HeapTreeSwap
  Notes: Assumes one-based indexing
-----------------------------------------------*/

bool HeapTreeSwap
( 
	int * elems,	// IN/OUT - elements to turn into a heap
	int   start,	// IN - start range
	int   stop,		// IN - end range
	int   kth		// IN - kth element
)
{
	// Check parameters
	if (start > stop)
	{
		int temp = start;
		start = stop;
		stop = temp;
	}
	int nElems = stop - start + 1;
	if (((nElems <= 0) || (NULL == elems)) ||
		((kth < 0) || (kth > nElems)))
	{
		// Error, invalid parametrs
		return false;
	}

	// Get min of [1,k] and [k,n] ranges
	int smallSize = kth;
	int largeSize = nElems - kth;
	int maxSize = smallSize;
	if (largeSize < maxSize)
	{
		maxSize = largeSize;
	}

	int * startPtr = &(elems[start]);

	int * oneSmall = &(startPtr[kth]);
	int * oneLarge = oneSmall - 1;

	int stackTop = 0;
	HeapStackNode swapStack[64];	// Stack for swapping

	swapStack[stackTop].idx   = 1;
	swapStack[stackTop].state = CHECK_STATE;
	stackTop++;

	int smallVal, largeVal;
	int left, right;
	int currIdx, currState, curr, child, rightIdx;
	int origVal, currVal, childVal, rightVal;

	while (stackTop > 0)
	{
		// Pop HeapNode off of stack
		stackTop--;
		currIdx   = swapStack[stackTop].idx;
		currState = swapStack[stackTop].state;

		switch (currState)
		{
		case CHECK_STATE:
			smallVal = oneSmall[-currIdx];	// small heap is in reverse order
			largeVal = oneLarge[currIdx];	// large heap is in normal order
			if (smallVal > largeVal)
			{
				// These two nodes in the tree are out of sync with respect to each other
				// Need to swap and process these 2 nodes 
				// But only after we have checked and processed both children nodes

				// Change state to 'SWAP' and leave this node on stack (first on, last off)
				swapStack[stackTop].state = SWAP_STATE;
				stackTop++;

				left  = currIdx << 1;
				right = left + 1;

				// Add Right Child to stack to check in future (second on, second off)
				if (right <= maxSize)
				{
					swapStack[stackTop].idx   = right;
					swapStack[stackTop].state = CHECK_STATE;
				}

				// Add Left Child to stack to check in future (last on, first off)
				if (left <= maxSize)
				{
					swapStack[stackTop].idx = left;
					swapStack[stackTop].state = CHECK_STATE;
				}
			}
			else
			{
				// Safe to do nothing, both sub-trees anchored at these positions are fine
			}
			break;

		case SWAP_STATE:
		{
			// Swap the two out of place heap values between the two heaps
			int tempVal = oneSmall[-currIdx];
			oneSmall[-currIdx] = oneLarge[currIdx];
			oneLarge[currIdx] = tempVal;

			/*-------------------------------------------
			  Demote value in small reversed max-heap 
			  [1,k] to correct position
		    -------------------------------------------*/

			curr  = currIdx;		// Get index of value to demote	
			child = currIdx << 1;	// Get left child of current index

			// Save original value creating first hole for extended swap
			origVal = oneSmall[-currIdx];
			currVal = origVal;

			// Compare current index to it's children
			while (child <= smallSize)
			{
				// Find largest child 
				childVal = oneSmall[-child];		// Left child
				rightIdx = child+1;
				if (rightIdx <= smallSize)
				{
					rightVal = oneSmall[-rightIdx];	// Right child
					if (childVal < rightVal)
					{
						// Right child is largest
						child    = rightIdx;
						childVal = rightVal;
					}
				}

				// Compare largest child priority to current priority
				if (currVal >= childVal)
				{
					// Current is larger than both children, exit loop
					break;
				}
			
				// Fill hole with child value
					// continuing extended swap
				oneSmall[-curr] = oneSmall[-child];

				// Update indices
				curr  = child;	
			    child = curr << 1; 
			}

			// Put original value back into final hole (ending extended swap)
			oneSmall[-curr] = origVal;


			/*-------------------------------------------
			  Demote value in large min-heap 
			  [k,n] to correct position
		    -------------------------------------------*/

			curr  = currIdx;		// Get index of value to demote	
			child = currIdx << 1;	// Get left child of current index

			// Save original value creating first hole for extended swap
			origVal = oneLarge[currIdx];
			currVal = origVal;

			// Compare current index to it's children
			while (child <= largeSize)
			{
				// Find smallest child 
				childVal = oneLarge[child];		// Left child
				rightIdx = child+1;
				if (rightIdx <= largeSize)
				{
					rightVal = oneLarge[rightIdx];	// Right child
					if (childVal > rightVal)
					{
						// Right child is smallest
						child    = rightIdx;
						childVal = rightVal;
					}
				}

				// Compare smallest child value to current value
				if (currVal <= childVal)
				{
					// Current value is smaller than both children, exit loop
					break;
				}
			
				// Fill hole with child value
					// continuing extended swap
				oneLarge[curr] = oneLarge[child];

				// Update indices
				curr  = child;	
			    child = curr << 1; 
			}

			// Put original value back into final hole (ending extended swap)
			oneLarge[curr] = origVal;
		}
			break;
		} // end switch(currState)
	}

	// Success
	return true;
}


void TestDualHeap()
{
	int n = 100;
	int * origVals = new int[n];
	int * heapVals = new int[n];

	int kth_val = 33;

	// Set seed for random number generator
	RandomInit( 2010 );

	// 
	//  Fill arrays with random values
	//
	int idx;
	for (idx = 0; idx < n; idx++)
	{
		double dVal = RandomF64();
		origVals[idx] = (int)(dVal * 99.0) + 1;
		heapVals[idx] = origVals[idx];
	}

	// Dump Values
	printf( "Original Points\n" );
	for (idx = 0; idx < n; idx++)
	{
		printf( "Orig Val[%u] = %d\n", idx+1, origVals[idx] );
	}
	printf( "\n\n" );


	/*-----------------------
	  Construction phase
	-----------------------*/

	// Make min Heap [1,n]
	MinHeapify( heapVals, 0, n - 1 );

	// Dump Values
	printf( "After MaxHeap[1,n] before MinHeap[1,k]\n" );
	for (idx = 0; idx < n; idx++)
	{
		printf( "Heap Val[%u] = %d\n", idx+1, heapVals[idx] );
	}
	printf( "\n\n" );

	// Make reverse max heap [1, k]
	MaxHeapifyReverse( heapVals, 0, kth_val - 1 );

	// Make min heap [k+1, n]
	MinHeapify( heapVals, kth_val, n-1 );


	// Dump Values
	printf( "After construction, [1, k]\n" );
	for (idx = 0; idx < kth_val; idx++)
	{
		printf( "Heap Val[%u] = %d\n", idx+1, heapVals[idx] );
	}
	printf( "\n\n" );

	printf( "After construction [k+1, n-1]\n" );
	for (idx = kth_val; idx < n; idx++)
	{
		printf( "Heap Val[%u] = %d\n", idx+1, heapVals[idx] );
	}
	printf( "\n\n" );

	/*-----------------------
	  Swap Phase
	-----------------------*/

	// Dump Values again
	printf( "\n\nAfter Swap Phase\n" );
	for (idx = 0; idx < n; idx++)
	{
		printf( "Orig Val[%u] = %d\n", idx, origVals[idx] );
		printf( "Heap Val[%u] = %d\n", idx, heapVals[idx] );
	}
	printf( "\n\nAfter Construction Phase, before Swap Phase\n" );

}
