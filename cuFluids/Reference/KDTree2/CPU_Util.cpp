/*-----------------------------------------------------------------------------
  File:  CPU_Util.cpp
  Desc:  Various CPU helper functions

  Log:   Created by Shawn D. Brown (4/15/07)
		 Modified by Shawn D. Brown (3/22/10)
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
#include "CPUTree_API.h"	// 
#include "GPUTree_API.h"

//#include "CPUTree_MED.h"	// CPU kd-tree (median layout)
#include "CPUTree_LBT.h"	// CPU kd-tree (left-balanced layout)


/*-------------------------------------
  Global Variables 
-------------------------------------*/

// Lookup tables for calculating left-balanced Median for small 'n'
static unsigned int g_leftTableCPU[32] = 
{ 
	0u,			// Wasted space (but necessary for 1-based indexing)
	1u,							// Level 1
	2u,2u,						// Level 2
	3u,4u,4u,4u,				// Level 3
	5u,6u,7u,8u,8u,8u,8u,8u,	// Level 4
	9u,10u,11u,12u,13u,14u,15u,16u,16u,16u,16u,16u,16u,16u,16u,16u // Level 5
};

static unsigned int g_halfTableCPU[32] = 
{ 
	0u,			// Wasted space (but necessary for 1-based indexing)
	0u,							// Level 1
	1u,1u,						// Level 2
	2u,2u,2u,2u,				// Level 3
	4u,4u,4u,4u,4u,4u,4u,4u,	// Level 4
	8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u,8u // Level 5
};


/*-------------------------------------
  Function Definitions
-------------------------------------*/

/*---------------------------------------------------------
  Name:	 KD_LBM_CPU()
  Desc:  Find the left balanced median of 'n' elements
  Note:  Also returns the 'half' value for possible use 
		 in kd-tree build algorithm for faster performance
		 
		 half = root(1 element) + 
		        size of complete left sub-tree (minus last row)
				(2^h-2)-1 elements
			  = (2^h-2)
---------------------------------------------------------*/

void KD_LBM_CPU
( 
	unsigned int n,				// IN:   Number to find left balanced median for
	unsigned int & median,		// OUT:  Left balanced median for 'n'
	unsigned int & half			// OUT:  Size of 1/2 tree including root minus last row
)
{
	// Return answer via lookup table for small 'n'
		// Also solves problem of small trees with height <= 2
	if (n <= 31u) 
	{
		half   = g_halfTableCPU[n];
		median = g_leftTableCPU[n];
		return;
	}

	// Compute height of tree
#if 1
	// Find position of highest set bit
		// Non-portable solution (Intel Intrinsic)
	unsigned long bitPos;
	_BitScanReverse( &bitPos, (unsigned long)(n+1) );
	int h       = (int)(bitPos+1);	
#else
	// Binary search for log2(n)
		// Portable solution
	unsigned int height  = KDTreeIntLog2( n+1 );	// Get height of tree
	unsigned int h = height+1;
#endif

	unsigned int lastRow;

	// Compute Left-balanced median
	half    = 1 << (h-2);						// 2^(h-2), Get size of left sub-tree (minus last row)
	lastRow = KD_Min( half, n-2*half+1 );	// Get # of elements to include from last row
	median  = half + lastRow;					// Return left-balanced median
	return;
}


/*---------------------------------------------------------
  Name:	 KD_IntLog2_CPU
  Desc:  Find the log base 2 for a 32-bit unsigned integer
  Note:  Does a binary search to find log2(val)
	     Takes O( log n ) time where n is input value
---------------------------------------------------------*/

unsigned int KD_IntLog2_CPU( unsigned int inVal )
{
	// Binary search for log2(n)
	unsigned int logVal = 0u;

	// Note: we assume unsigned integers are 32-bit
		// if unsigned integers are actually 64-bit 
		// then uncomment the following line
//  if (inVal >= 1u << 32u) { inVal >>= 32u; logVal |= 32u; }

	if (inVal >= (1u << 16u)) { inVal >>= 16u; logVal |= 16u; }
	if (inVal >= (1u <<  8u)) { inVal >>=  8u; logVal |=  8u; }
	if (inVal >= (1u <<  4u)) { inVal >>=  4u; logVal |=  4u; }
	if (inVal >= (1u <<  2u)) { inVal >>=  2u; logVal |=  2u; }
	if (inVal >= (1u <<  1u)) { logVal |= 1u; }

	return logVal;
}


/*-------------------------------------------------------------------------
  Name:	AxisToString
  Desc:	Get Human Readable string for Axis
-------------------------------------------------------------------------*/

const char * AxisToString( unsigned int currAxis )
{
	char * sz = "?";
	switch (currAxis)
	{
	case X_AXIS:
		sz = "X";
		break;
	case Y_AXIS:
		sz =  "Y";
		break;
	case Z_AXIS:
		sz =  "Z";
		break;
	case W_AXIS:
		sz =  "W";
		break;
	case S_AXIS:
		sz =  "S";
		break;
	case T_AXIS:
		sz =  "T";
		break;
	case U_AXIS:
		sz =  "U";
		break;
	case V_AXIS:
		sz =  "V";
		break;
	default:
		break;
	}
	return sz;
}


/*---------------------------------------------------------
  Name:	CPU_QNN_2D_MEDIAN
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

#if 0
bool CPU_QNN_2D_MEDIAN
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_MED * cpuKDTree = static_cast<CPUTree_2D_MED *>( kdTree );
	if (NULL == cpuKDTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuKDTree->Find_QNN_2D( resultList, cQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	/*
	// Make Sure Results list is large enough to hold results
	unsigned int cResult = cQuery;

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
		bResult = cpuKDTree->FindClosestPoint2DAlt( currQuery, idxVal, idVal, distVal );
		if (! bResult)
		{
			// Error
			return false;
		}

		// Store Results in Result List
		resultList[i].id   = idVal;
		resultList[i].dist = distVal;
	}
	*/

	// Success
	return true;
}
#endif

/*---------------------------------------------------------
  Name:	CPU_QNN_2D_LBT
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_QNN_2D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_LBT * cpuTree = static_cast<CPUTree_2D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_QNN_2D( resultList, cQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_QNN_3D_LBT
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_QNN_3D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_3D_LBT * cpuTree = static_cast<CPUTree_3D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_QNN_3D( resultList, cQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_QNN_4D_LBT
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_QNN_4D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_4D_LBT * cpuTree = static_cast<CPUTree_4D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_QNN_4D( resultList, cQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_ALL_NN_2D_MEDIAN
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

#if 0
bool CPU_ALL_NN_2D_MEDIAN
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_MED * cpuKDTree = static_cast<CPUTree_2D_MED *>( kdTree );
	if (NULL == cpuKDTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuKDTree->Find_ALL_NN_2D( resultList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}


	// Success
	return true;
}
#endif

/*---------------------------------------------------------
  Name:	CPU_ALL_NN_2D_LBT
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_ALL_NN_2D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_LBT * cpuTree = static_cast<CPUTree_2D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_ALL_NN_2D( resultList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}


	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_ALL_NN_3D_LBT
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_ALL_NN_3D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_3D_LBT * cpuTree = static_cast<CPUTree_3D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_ALL_NN_3D( resultList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}


	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_ALL_NN_4D_LBT
  Desc: Finds nearest neighbor in search kd-tree for
        each query point
  Note:	Algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_ALL_NN_4D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_4D_LBT * cpuTree = static_cast<CPUTree_4D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_ALL_NN_4D( resultList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}


	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_KNN_2D
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/
#if 0
bool CPU_KNN_2D_MEDIAN
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_MED * cpuKDTree = static_cast<CPUTree_2D_MED *>( kdTree );
	if (NULL == cpuKDTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuKDTree->Find_KNN_2D( resultList, kVal, cQuery, cPadQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}
#endif

/*---------------------------------------------------------
  Name:	CPU_KNN_2D_LBT
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_KNN_2D_LBT
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_LBT * cpuTree = static_cast<CPUTree_2D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_KNN_2D( resultList, kVal, cQuery, cPadQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_KNN_3D_LBT
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_KNN_3D_LBT
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_3D_LBT * cpuTree = static_cast<CPUTree_3D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_KNN_3D( resultList, kVal, cQuery, cPadQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_KNN_4D_LBT
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_KNN_4D_LBT
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_4D_LBT * cpuTree = static_cast<CPUTree_4D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_KNN_4D( resultList, kVal, cQuery, cPadQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_ALL_KNN_2D
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/
#if 0
bool CPU_ALL_KNN_2D_MEDIAN
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_MED * cpuKDTree = static_cast<CPUTree_2D_MED *>( kdTree );
	if (NULL == cpuKDTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuKDTree->Find_ALL_KNN_2D( resultList, kVal, cPadSearch );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}
#endif

/*---------------------------------------------------------
  Name:	CPU_ALL_KNN_2D_LBT
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_ALL_KNN_2D_LBT
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_2D_LBT * cpuTree = static_cast<CPUTree_2D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_ALL_KNN_2D( resultList, kVal, cSearch, cPadSearch );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_ALL_KNN_3D_LBT
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_ALL_KNN_3D_LBT
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_3D_LBT * cpuTree = static_cast<CPUTree_3D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_ALL_KNN_3D( resultList, kVal, cSearch, cPadSearch );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CPU_ALL_KNN_4D_LBT
  Desc:	Finds 'k' nearest points to each query point
  Note:	algorithm done on CPU
        as check on GPU algorithm
---------------------------------------------------------*/

bool CPU_ALL_KNN_4D_LBT
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	CPUTree_4D_LBT * cpuTree = static_cast<CPUTree_4D_LBT *>( kdTree );
	if (NULL == cpuTree) { return false; }
	if (cSearch <= 0) { return false; }
	//if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuTree->Find_ALL_KNN_4D( resultList, kVal, cSearch, cPadSearch );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	BUILD_CPU_2D_MEDIAN
  Desc:	Builds KD Tree on CPU in median array layout
---------------------------------------------------------*/
#if 0
bool BUILD_CPU_2D_MEDIAN
(
	void        ** kdTree,			// IN/OUT - KDTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
)
{
	*kdTree = NULL;
	CPUTree_2D_MED * cpuKDTree = new CPUTree_2D_MED();
	if (NULL == cpuKDTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = cpuKDTree->Build2D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete cpuKDTree;
	}

	// Success
	*kdTree = static_cast<void *>( cpuKDTree );
	return true;
}
#endif


/*---------------------------------------------------------
  Name:	BUILD_CPU_2D_LBT
  Desc:	Builds left-balanced kd-tree on CPU
---------------------------------------------------------*/

bool BUILD_CPU_2D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float2 * search_CPU		// IN -  CPU Search Point List
)
{
	*cpuTree = NULL;
	CPUTree_2D_LBT * myTree = new CPUTree_2D_LBT();
	if (NULL == myTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = myTree->Build2D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete myTree;
	}

	// Success
	*cpuTree = static_cast<void *>( myTree );
	return true;
}

bool BUILD_CPU_2D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float3 * search_CPU		// IN -  CPU Search Point List
)
{
	*cpuTree = NULL;
	CPUTree_2D_LBT * myTree = new CPUTree_2D_LBT();
	if (NULL == myTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = myTree->Build2D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete myTree;
	}

	// Success
	*cpuTree = static_cast<void *>( myTree );
	return true;
}

bool BUILD_CPU_2D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
)
{
	*cpuTree = NULL;
	CPUTree_2D_LBT * myTree = new CPUTree_2D_LBT();
	if (NULL == myTree) { return false; }

	// Build KD Tree
	bool bResult;

#ifdef _BUILD_STATS
	bResult = myTree->Build2DStats( nSearch, search_CPU );
#else
	bResult = myTree->Build2D( nSearch, search_CPU );
#endif
	if (false == bResult)
	{
		delete myTree;
	}

	// Success
	*cpuTree = static_cast<void *>( myTree );
	return true;
}


/*---------------------------------------------------------
  Name:	BUILD_CPU_3D_LBT
  Desc:	Builds left-balanced kd-tree on CPU
---------------------------------------------------------*/

bool BUILD_CPU_3D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float3 * search_CPU		// IN -  CPU Search Point List
)
{
	*cpuTree = NULL;
	CPUTree_3D_LBT * myTree = new CPUTree_3D_LBT();
	if (NULL == myTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = myTree->Build3D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete myTree;
	}

	// Success
	*cpuTree = static_cast<void *>( myTree );
	return true;
}

bool BUILD_CPU_3D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
)
{
	*cpuTree = NULL;
	CPUTree_3D_LBT * myTree = new CPUTree_3D_LBT();
	if (NULL == myTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = myTree->Build3D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete myTree;
	}

	// Success
	*cpuTree = static_cast<void *>( myTree );
	return true;
}


/*---------------------------------------------------------
  Name:	BUILD_CPU_4D_LBT
  Desc:	Builds left-balanced kd-tree on CPU
---------------------------------------------------------*/

bool BUILD_CPU_4D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
)
{
	*cpuTree = NULL;
	CPUTree_4D_LBT * myTree = new CPUTree_4D_LBT();
	if (NULL == myTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = myTree->Build4D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete myTree;
	}

	// Success
	*cpuTree = static_cast<void *>( myTree );
	return true;
}


/*---------------------------------------------------------
  Name:	FINI_CPU_2D_MEDIAN
  Desc:	Cleanup KDTree on CPU
---------------------------------------------------------*/
#if 0
bool FINI_CPU_2D_MEDIAN
(
	void ** kdTree		// IN/OUT - KD Tree pointer
)
{
	CPUTree_2D_MED * cpuKDTree = static_cast<CPUTree_2D_MED *>( *kdTree );
	if (NULL != cpuKDTree)
	{
		delete cpuKDTree;
	}
	*kdTree = NULL;

	// Success
	return true;
}
#endif

/*---------------------------------------------------------
  Name:	FINI_CPU_2D_LBT
  Desc:	Cleanup KDTree on CPU
---------------------------------------------------------*/

bool FINI_CPU_2D_LBT
(
	void ** kdTree		// IN/OUT - KD Tree pointer
)
{
	CPUTree_2D_LBT * cpuTree = static_cast<CPUTree_2D_LBT *>( *kdTree );
	if (NULL != cpuTree)
	{
		delete cpuTree;
	}
	*kdTree = NULL;

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	FINI_CPU_3D_LBT
  Desc:	Cleanup KDTree on CPU
---------------------------------------------------------*/

bool FINI_CPU_3D_LBT
(
	void ** kdTree		// IN/OUT - KD Tree pointer
)
{
	CPUTree_3D_LBT * cpuTree = static_cast<CPUTree_3D_LBT *>( *kdTree );
	if (NULL != cpuTree)
	{
		delete cpuTree;
	}
	*kdTree = NULL;

	// Success
	return true;
}

/*---------------------------------------------------------
  Name:	FINI_CPU_4D_LBT
  Desc:	Cleanup KDTree on CPU
---------------------------------------------------------*/

bool FINI_CPU_4D_LBT
(
	void ** kdTree		// IN/OUT - KD Tree pointer
)
{
	CPUTree_4D_LBT * cpuTree = static_cast<CPUTree_4D_LBT *>( *kdTree );
	if (NULL != cpuTree)
	{
		delete cpuTree;
	}
	*kdTree = NULL;

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	KD_TREE_TEST_SELECT
  Desc:	Builds KD Tree on CPU
---------------------------------------------------------*/
#if 0
bool KD_TREE_TEST_SELECT
(
	unsigned int   nPoints,			// IN - Number of Points
	unsigned int   kth,				// IN - kth element to select on
	unsigned int   axis				// IN - axis to select on
)
{
	bool bResult = false;
	float4 * h_Points = NULL;
	CPUTree_2D_MED * cpuKDTree = NULL;

	// allocate host memory for 3D points
	int mem_size_Points = nPoints * sizeof(float4);
	h_Points = (float4*) malloc( (size_t)mem_size_Points );
	if (NULL == h_Points)
	{
		goto lblCLEANUP;
	}

	float minS = 0.0;
	float maxS = 1.0;

	// Build list of points
	unsigned int idx;
	for (idx = 0; idx < nPoints; idx++)	// Large Points
	{
		h_Points[idx].x = RandomFloat( minS, maxS );
		h_Points[idx].y = RandomFloat( minS, maxS );
		h_Points[idx].z = RandomFloat( minS, maxS );

		// Store point index in 'w' channel
		h_Points[idx].w = (float)idx;
	}

	// Create KDTree
	cpuKDTree = new CPUTree_2D_MED();
	if (NULL == cpuKDTree) 
	{ 
		goto lblCLEANUP;
	}

	// Test Select for KD Tree
	bResult = cpuKDTree->TestSelect( nPoints, h_Points, kth, axis );

lblCLEANUP:
	// Cleanup KD Tree
	if (NULL != cpuKDTree)
	{
		delete cpuKDTree;
	}

	// Cleanup point list
	if (NULL != h_Points)
	{
		delete [] h_Points;
	}

	// Return result
	return bResult;
}
#endif

/*---------------------------------------------------------
  Name:	COPY_NODES_2D_MEDIAN
  Desc:	Copy GPU nodes from CPU KDTree nodes
---------------------------------------------------------*/
#if 0
bool COPY_NODES_2D_MEDIAN
(
	void *          kdTree,			// IN - KDTree pointer
	unsigned int	nSearch,		// IN - Count of items in search list
	unsigned int    nPadSearch,		// IN - Count of items in padded search list
	void *          nodes_GPU,		// OUT - GPU Node List
	unsigned int *  ids_GPU			// OUT - ID list for GPU nodes
)
{
	CPUTree_2D_MED          * cpuKDTree = static_cast<CPUTree_2D_MED *>( kdTree );
	GPUNode_2D_MED * gpuNodes = static_cast<GPUNode_2D_MED *>( nodes_GPU );

	if (NULL == cpuKDTree) { return false; }
	if (NULL == gpuNodes)  { return false; }
	if (NULL == ids_GPU)   { return false; }

	// Copy KD Tree Nodes into GPU list
	unsigned int i;
	for (i = 0; i < nSearch; i++)
	{
		CPUNode_2D_MED      * currCPU = cpuKDTree->NODE_PTR( i );
		GPUNode_2D_MED * currGPU = &(gpuNodes[i]);

		// Set GPU Node values
		currGPU->pos[0] = currCPU->X();
		currGPU->pos[1] = currCPU->Y();
		currGPU->Left   = currCPU->Left();
		currGPU->Right  = currCPU->Right();

		// Set ID for current Node
		ids_GPU[i] = currCPU->ID();
	}

	// Fill in extra search nodes with extra copies of the 1st element
	if (nPadSearch >= nSearch)
	{
		GPUNode_2D_MED * firstGPU = &(gpuNodes[1]);
		float firstX = firstGPU->pos[0];
		float firstY = firstGPU->pos[1];
		unsigned int padIdx;
		for (padIdx = nSearch; padIdx < nPadSearch; padIdx++)
		{
			GPUNode_2D_MED * currGPU = &(gpuNodes[padIdx]);

			// Initialize extra search nodes to "first" values
			currGPU->pos[0] = firstX;
			currGPU->pos[1] = firstY;
			currGPU->Left   = 0xFFFFFFFFu;
			currGPU->Right  = 0xFFFFFFFFu;
		}
	}


	// Success
	return true;
}
#endif


/*---------------------------------------------------------
  Name:	COPY_NODES_2D_LBT
  Desc:	copy GPU kd-nodes from CPU kd-nodes
---------------------------------------------------------*/

bool COPY_NODES_2D_LBT
(
	void *          kdTree,			// IN - KDTree pointer
	unsigned int	nSearch,		// IN - Count of items in search list
	unsigned int    nPadSearch,		// IN - Count of items in padded search list
	void *          nodes_GPU,		// OUT - GPU Node List
	unsigned int *  ids_GPU			// OUT - ID list for GPU nodes
)
{
	CPUTree_2D_LBT * cpuTree  = static_cast<CPUTree_2D_LBT *>( kdTree );
	GPUNode_2D_LBT * gpuNodes = static_cast<GPUNode_2D_LBT *>( nodes_GPU );

	if (NULL == cpuTree)   { return false; }
	if (NULL == gpuNodes)  { return false; }
	if (NULL == ids_GPU)   { return false; }

	// Initialize zero position to default "Don't care" values
	if (nPadSearch >= 1)
	{
		CPUNode_2D_LBT * currCPU = cpuTree->NODE_PTR( 1 );

		float rootX = currCPU->X();
		float rootY = currCPU->Y();

		// BUGBUG:  To avoid 2 elements colliding on results at position zero
			// We map this special zeroth element to a position that can't hurt 
			// any of the the other elements outcomes.
		gpuNodes[0].pos[0] = rootX;
		gpuNodes[0].pos[1] = rootY;
		ids_GPU[0] = nSearch+1;	 // Maps to special location
	}

	// Copy KD Tree Nodes into GPU Nodes
	unsigned int nodeIdx;
	unsigned int pointIdx;
	float currX, currY;
	for (nodeIdx = 1; nodeIdx <= nSearch; nodeIdx++)
	{
		CPUNode_2D_LBT * currCPU = cpuTree->NODE_PTR( nodeIdx );
		GPUNode_2D_LBT * currGPU = &(gpuNodes[nodeIdx]);

		currX = currCPU->X();
		currY = currCPU->Y();
		pointIdx = currCPU->SearchID();

		// Set GPU Node values
		currGPU->pos[0] = currX;
		currGPU->pos[1] = currY;

		// Set Search ID for current Node
		ids_GPU[nodeIdx] = pointIdx;
	}

	// Fill in extra search nodes with extra copies of the root for querying
	if (nPadSearch > nSearch)
	{
		CPUNode_2D_LBT * rootCPU = cpuTree->NODE_PTR( 1 );
		float rootX = rootCPU->X();
		float rootY = rootCPU->Y();
		unsigned int padIdx;
		for (padIdx = nSearch+1; padIdx < nPadSearch; padIdx++)
		{
			GPUNode_2D_LBT * currGPU = &(gpuNodes[padIdx]);

			// Dump result for debugging
#if 0
			printf( "Extra Node[%u] = <PID=%u, X=%3.6f, Y=%3.6f>\n", padIdx, padIdx, rootX, rootY );
#endif

			// Initialize extra search nodes to "root" values
			currGPU->pos[0] = rootX;
			currGPU->pos[1] = rootY;	
			ids_GPU[padIdx] = padIdx;				 // Map extra nodes back onto themselves
		}
	}

	// Success
	return true;
}

/*---------------------------------------------------------
  Name:	COPY_NODES_3D_LBT
  Desc:	copy GPU kd-nodes from CPU kd-nodes
---------------------------------------------------------*/

bool COPY_NODES_3D_LBT
(
	void *          kdTree,			// IN - KDTree pointer
	unsigned int	nSearch,		// IN - Count of items in search list
	unsigned int    nPadSearch,		// IN - Count of items in padded search list
	void *          nodes_GPU,		// OUT - GPU Node List
	unsigned int *  ids_GPU			// OUT - ID list for GPU nodes
)
{
	CPUTree_3D_LBT * cpuTree  = static_cast<CPUTree_3D_LBT *>( kdTree );
	GPUNode_3D_LBT * gpuNodes = static_cast<GPUNode_3D_LBT *>( nodes_GPU );

	if (NULL == cpuTree)   { return false; }
	if (NULL == gpuNodes)  { return false; }
	if (NULL == ids_GPU)   { return false; }

	// Initialize zero position to default "Don't care" values
	if (nPadSearch >= 1)
	{
		CPUNode_3D_LBT * rootCPU = cpuTree->NODE_PTR( 1 );
		float rootX = rootCPU->X();
		float rootY = rootCPU->Y();
		float rootZ = rootCPU->Z();

		// BUGBUG:  To avoid 2 elements colliding on results at position zero
			// We map this special zeroth element to a position that can't hurt 
			// any of the the other original elements outcomes.
		gpuNodes[0].pos[0] = rootX;
		gpuNodes[0].pos[1] = rootY;
		gpuNodes[0].pos[2] = rootZ;
		ids_GPU[0] = nSearch+1;		// Map to values past where we care about
	}

	// Copy KD Tree Nodes into GPU Nodes
	unsigned int nodeIdx;
	unsigned int pointIdx;
	float currX, currY, currZ;
	for (nodeIdx = 1; nodeIdx <= nSearch; nodeIdx++)
	{
		CPUNode_3D_LBT * currCPU = cpuTree->NODE_PTR( nodeIdx );
		GPUNode_3D_LBT * currGPU = &(gpuNodes[nodeIdx]);

		currX = currCPU->X();
		currY = currCPU->Y();
		currZ = currCPU->Z();

		pointIdx = currCPU->SearchID();

		// Set GPU Node values
		currGPU->pos[0] = currX;
		currGPU->pos[1] = currY;
		currGPU->pos[2] = currZ;

		// Set Search ID for current Node
		ids_GPU[nodeIdx] = pointIdx;
	}

	// Fill in extra search nodes with extra copies of the root for querying
	if (nPadSearch > nSearch)
	{
		CPUNode_3D_LBT * rootCPU = cpuTree->NODE_PTR( 1 );
		float rootX = rootCPU->X();
		float rootY = rootCPU->Y();
		float rootZ = rootCPU->Z();

		unsigned int padIdx;
		for (padIdx = nSearch+1; padIdx < nPadSearch; padIdx++)
		{
			GPUNode_3D_LBT * currGPU = &(gpuNodes[padIdx]);

			// Dump result for debugging
#if 0
			printf( "Extra Node[%u] = <PID=%u, X=%3.6f, Y=%3.6f, Z=%3.6f>\n", 
				     padIdx, padIdx, rootX, rootY, rootZ );
#endif

			// Initialize extra search nodes to "root" values
			currGPU->pos[0] = rootX;
			currGPU->pos[1] = rootY;	
			currGPU->pos[2] = rootZ;	
			ids_GPU[padIdx] = padIdx;	 // Map extra nodes back onto themselves
		}
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	COPY_NODES_4D_LBT
  Desc:	copy GPU kd-nodes from CPU kd-nodes
---------------------------------------------------------*/

bool COPY_NODES_4D_LBT
(
	void *          kdTree,			// IN - KDTree pointer
	unsigned int	nSearch,		// IN - Count of items in search list
	unsigned int    nPadSearch,		// IN - Count of items in padded search list
	void *          nodes_GPU,		// OUT - GPU Node List
	unsigned int *  ids_GPU			// OUT - ID list for GPU nodes
)
{
	CPUTree_4D_LBT * cpuTree  = static_cast<CPUTree_4D_LBT *>( kdTree );
	GPUNode_4D_LBT * gpuNodes = static_cast<GPUNode_4D_LBT *>( nodes_GPU );

	if (NULL == cpuTree)   { return false; }
	if (NULL == gpuNodes)  { return false; }
	if (NULL == ids_GPU)   { return false; }

	// Initialize zero position to default "Don't care" values
	if (nPadSearch >= 1)
	{
		CPUNode_4D_LBT * rootCPU = cpuTree->NODE_PTR( 1 );
		float rootX = rootCPU->X();
		float rootY = rootCPU->Y();
		float rootZ = rootCPU->Z();
		float rootW = rootCPU->W();

		// BUGBUG:  To avoid 2 elements colliding on results at position zero
			// We map this special zeroth element to a position that can't hurt 
			// any of the the other original elements outcomes.
		gpuNodes[0].pos[0] = rootX;
		gpuNodes[0].pos[1] = rootY;
		gpuNodes[0].pos[2] = rootZ;
		gpuNodes[0].pos[3] = rootW;

		ids_GPU[0] = nSearch+1;		// Map to values past where we care about
	}

	// Copy KD Tree Nodes into GPU Nodes
	unsigned int nodeIdx;
	unsigned int pointIdx;
	float currX, currY, currZ, currW;
	for (nodeIdx = 1; nodeIdx <= nSearch; nodeIdx++)
	{
		CPUNode_4D_LBT * currCPU = cpuTree->NODE_PTR( nodeIdx );
		GPUNode_4D_LBT * currGPU = &(gpuNodes[nodeIdx]);

		currX = currCPU->X();
		currY = currCPU->Y();
		currZ = currCPU->Z();
		currW = currCPU->W();

		pointIdx = currCPU->SearchID();

		// Set GPU Node values
		currGPU->pos[0] = currX;
		currGPU->pos[1] = currY;
		currGPU->pos[2] = currZ;
		currGPU->pos[3] = currW;

		// Set Search ID for current Node
		ids_GPU[nodeIdx] = pointIdx;
	}

	// Fill in extra search nodes with extra copies of the root for querying
	if (nPadSearch > nSearch)
	{
		CPUNode_4D_LBT * rootCPU = cpuTree->NODE_PTR( 1 );

		float rootX = rootCPU->X();
		float rootY = rootCPU->Y();
		float rootZ = rootCPU->Z();
		float rootW = rootCPU->W();

		unsigned int padIdx;
		for (padIdx = nSearch+1; padIdx < nPadSearch; padIdx++)
		{
			GPUNode_4D_LBT * currGPU = &(gpuNodes[padIdx]);

			// Dump result for debugging
#if 0
			printf( "Extra Node[%u] = <PID=%u, X=%3.6f, Y=%3.6f, Z=%3.6f, W=%3.6f>\n", 
				     padIdx, padIdx, rootX, rootY, rootZ, rootW );
#endif

			// Initialize extra search nodes to "root" values
			currGPU->pos[0] = rootX;
			currGPU->pos[1] = rootY;	
			currGPU->pos[2] = rootZ;	
			currGPU->pos[2] = rootW;

			ids_GPU[padIdx] = padIdx;	 // Map extra nodes back onto themselves
		}
	}

	// Success
	return true;
}
