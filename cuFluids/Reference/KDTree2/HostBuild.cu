/*-----------------------------------------------------------------------------
  File:  HostBuild.cpp
  Desc:  Host CPU API scaffolding for running kd-tree build kernels on GPU
         This supports kd-trees using a left-balanced binary tree array layout

  Log:   Created by Shawn D. Brown (5/12/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// System includes
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CUDA includes
#include <cutil_inline.h>

// Project includes
#include "CPUTree_API.h"
#include "GPUTree_API.h"
#include "QueryResult.h"

#include "CPUTree_LBT.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

extern AppGlobals g_app;

// Lookup tables for calculating left-balanced Median for small 'n'
static unsigned int g_leftTableHOST[32] = 
{ 
	0u,			// Wasted space (but necessary for 1-based indexing)
	1u,							// Level 1
	2u,2u,						// Level 2
	3u,4u,4u,4u,				// Level 3
	5u,6u,7u,8u,8u,8u,8u,8u,	// Level 4
	9u,10u,11u,12u,13u,14u,15u,16u,16u,16u,16u,16u,16u,16u,16u,16u // Level 5
};

static unsigned int g_halfTableHOST[32] = 
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
  Name:	 KD_IntLog2_GPU
  Desc:  Find the log base 2 for a 32-bit unsigned integer
  Note:  Does a binary search to find log2(val)
	     Takes O( log n ) time where n is input value
---------------------------------------------------------*/

unsigned int KD_IntLog2_HOST( unsigned int inVal )
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


/*---------------------------------------------------------
  Name:	 KD_LBM_HOST()
  Desc:  Find the left balanced median of 'n' elements
  Note:  Also returns the 'half' value for possible use 
		 in kd-tree build algorithm for faster performance
		 
		 half = root(1 element) + 
		        size of complete left sub-tree (minus last row)
				(2^h-2)-1 elements
			  = (2^h-2)
---------------------------------------------------------*/

void KD_LBM_HOST
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
		half   = g_halfTableHOST[n];
		median = g_leftTableHOST[n];
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
	unsigned int height  = KD_IntLog2_GPU( n+1 );	// Get height of tree
	unsigned int h = height+1;
#endif

	unsigned int lastRow;

	// Compute Left-balanced median
	half    = 1 << (h-2);						// 2^(h-2), Get size of left sub-tree (minus last row)
	lastRow = KD_Min( half, n-(2*half)+1 );	// Get # of elements to include from last row
	median  = half + lastRow;					// Return left-balanced median
	return;
}


/*---------------------------------------------------------
  Name:	 DumpP1Stats
---------------------------------------------------------*/

#ifdef _BUILD_STATS
bool DumpP1Stats( GPU_BUILD_STATS & stats )
{
	// Dump Totals
	fprintf( stdout, "\n\nGPU P1 Totals {\n" );
	fprintf( stdout, "\tRoot Reads       = %u\n", stats.cRootReads );
	fprintf( stdout, "\tPivot Reads      = %u\n", stats.cPivotReads );
	fprintf( stdout, "\tCount Reads      = %u\n", stats.cCountReads );
	fprintf( stdout, "\tCount Writes     = %u\n", stats.cCountWrites );
	fprintf( stdout, "\tPartition Reads  = %u\n", stats.cPartReads );
	fprintf( stdout, "\tPartition Writes = %u\n", stats.cPartWrites );
	fprintf( stdout, "\tStore Reads      = %u\n", stats.cStoreReads );
	fprintf( stdout, "\tStore Writes     = %u\n", stats.cStoreWrites );
	fprintf( stdout, "\tNode Loops       = %u\n", stats.cNodeLoops );
	fprintf( stdout, "\tPartition Loops  = %u\n", stats.cPartLoops );
	fprintf( stdout, "\tDev2Host Reads   = %u\n", stats.cD2HReads );
	fprintf( stdout, "\tDev2Host Writes  = %u\n", stats.cD2HWrites );
	fprintf( stdout, "\tHost2Dev Reads   = %u\n", stats.cH2DReads );
	fprintf( stdout, "\tHost2Dev Writes  = %u\n", stats.cH2DWrites );
	fprintf( stdout, "}\n\n" );

	return true;
}
#endif


/*---------------------------------------------------------
  Name:	 DumpP2Stats
---------------------------------------------------------*/

#ifdef _BUILD_STATS
bool DumpP2Stats( unsigned int nItems, GPU_BUILD_STATS * statsList )
{
	// Check Parameters
	if ((nItems == 0) || (NULL == statsList))
	{
		return false;
	}

	GPU_BUILD_STATS gpuTotal;
	gpuTotal.cRootReads   = 0;
	gpuTotal.cPivotReads  = 0;
	gpuTotal.cCountReads  = 0;
	gpuTotal.cCountWrites = 0;
	gpuTotal.cPartReads   = 0;
	gpuTotal.cPartWrites  = 0;
	gpuTotal.cStoreReads  = 0;
	gpuTotal.cStoreWrites = 0;
	gpuTotal.cNodeLoops   = 0;
	gpuTotal.cPartLoops   = 0;

	// Dump Per Thread (sub-tree results) & accumulate counts
	unsigned int currIdx;
	for (currIdx = 0; currIdx < nItems; currIdx++)
	{
		GPU_BUILD_STATS & currStats = statsList[currIdx];
		fprintf( stdout, "GPU P2 Thread[%u] {\n", currIdx );
		fprintf( stdout, "\tRoot Reads       = %u\n", gpuTotal.cRootReads );
		fprintf( stdout, "\tPivot Reads      = %u\n", currStats.cPivotReads );
		fprintf( stdout, "\tCount Reads      = %u\n", currStats.cCountReads );
		fprintf( stdout, "\tCount Writes     = %u\n", currStats.cCountWrites );
		fprintf( stdout, "\tPartition Reads  = %u\n", currStats.cPartReads );
		fprintf( stdout, "\tPartition Writes = %u\n", currStats.cPartWrites );
		fprintf( stdout, "\tStore Reads      = %u\n", currStats.cStoreReads );
		fprintf( stdout, "\tStore Writes     = %u\n", currStats.cStoreWrites );
		fprintf( stdout, "\tNode Loops       = %u\n", currStats.cNodeLoops );
		fprintf( stdout, "\tPartition Loops  = %u\n", currStats.cPartLoops );
		fprintf( stdout, "}\n" );

		gpuTotal.cRootReads   += currStats.cRootReads;
		gpuTotal.cPivotReads  += currStats.cPivotReads;
		gpuTotal.cCountReads  += currStats.cCountReads;
		gpuTotal.cCountWrites += currStats.cCountWrites;
		gpuTotal.cPartReads   += currStats.cPartReads;
		gpuTotal.cPartWrites  += currStats.cPartWrites;
		gpuTotal.cStoreReads  += currStats.cStoreReads;
		gpuTotal.cStoreWrites += currStats.cStoreWrites;
		gpuTotal.cNodeLoops   += currStats.cNodeLoops;
		gpuTotal.cPartLoops   += currStats.cPartLoops;
	}

	// Dump Totals
	fprintf( stdout, "\n\nGPU P2 Totals {\n" );
	fprintf( stdout, "\tRoot Reads       = %u\n", gpuTotal.cRootReads );
	fprintf( stdout, "\tPivot Reads      = %u\n", gpuTotal.cPivotReads );
	fprintf( stdout, "\tCount Reads      = %u\n", gpuTotal.cCountReads );
	fprintf( stdout, "\tCount Writes     = %u\n", gpuTotal.cCountWrites );
	fprintf( stdout, "\tPartition Reads  = %u\n", gpuTotal.cPartReads );
	fprintf( stdout, "\tPartition Writes = %u\n", gpuTotal.cPartWrites );
	fprintf( stdout, "\tStore Reads      = %u\n", gpuTotal.cStoreReads );
	fprintf( stdout, "\tStore Writes     = %u\n", gpuTotal.cStoreWrites );
	fprintf( stdout, "\tNode Loops       = %u\n", gpuTotal.cNodeLoops );
	fprintf( stdout, "\tPartition Loops  = %u\n", gpuTotal.cPartLoops );
	fprintf( stdout, "}\n\n" );

	return true;
}
#endif

/*-------------------------------------------------------------------------
  Name:	Host_Build2D
  Desc:	Build left-balanced kd-tree from point list
-------------------------------------------------------------------------*/

bool Host_Build2D( HostDeviceParams & params )
{
	bool bResult = true;
	cudaError_t cudaResult = cudaSuccess;

	/*-----------------------
	  Check Parameters
	-----------------------*/

	unsigned int nOrigPoints = params.nSearch;
	unsigned int nPadPoints  = params.nPadSearch;

	if (NULL == params.searchList) { return false; }
	if (0 == nOrigPoints)		   { return false; }
	if (0 == nPadPoints)		   { return false; }
	if (nPadPoints <= nOrigPoints) { return false; }


	/*----------------------------
	  Compute Grid & block
	  layouts for phase 1 & 2
	----------------------------*/

	//unsigned int nThreads   = 16;
	unsigned int nStarts    = BUILD_TOTAL_THREADS;
	unsigned int nCounts    = BUILD_TOTAL_THREADS;
	unsigned int maxBuildQ  = 256;
	unsigned int nPivots    = 1;

	/*--------------------------------------
	  Validate GPU memory usage
	--------------------------------------*/

	// Make sure memory usage for NN search is not to big to use up all device memory
		// 1 GB on Display Card


	unsigned int size_NodesLBT, size_NodesMED, size_Scratch, size_IDs, totalRequested;
	unsigned int size_BuildQ, size_Counts, size_Starts, size_Pivot;
	size_NodesLBT  = nPadPoints * sizeof(GPUNode_2D_LBT);
	size_NodesMED  = nPadPoints * sizeof(GPUNode_2D_MED);
	size_Scratch   = nPadPoints * sizeof(GPUNode_2D_MED);
	size_IDs       = nPadPoints * sizeof(unsigned int);
	size_BuildQ    = maxBuildQ  * sizeof( GPU_BUILD_ITEM );
	size_Counts    = nCounts * sizeof( GPU_COUNTS_STARTS );
	size_Starts    = nStarts * sizeof( GPU_COUNTS_STARTS );
	size_Pivot	   = nPivots * sizeof( unsigned int );

#if _BUILD_STATS
	unsigned int size_StatsQ = maxBuildQ * sizeof( GPU_BUILD_STATS );
#endif

	totalRequested = size_NodesLBT + size_NodesMED + size_Scratch + size_IDs +
					 size_BuildQ + size_Counts + size_Starts + size_Pivot;

	// Make sure memory required to perform this operation doesn't exceed display device memory
	unsigned int totalMemAvail = (unsigned int)(params.cudaProps->totalGlobalMem);
	if (totalRequested >= totalMemAvail)
	{
		printf( "GPU Build 2D - Inputs (%u) are too large for available device memory (%u), running test will crash...\n",
				totalRequested, totalMemAvail );
		printf( "\tsizeNodesLBT = %u\n", size_NodesLBT );
		printf( "\tsizeNodesMED = %u\n", size_NodesMED );
		printf( "\tsizeScratch  = %u\n", size_Scratch );
		printf( "\tsizeIDs      = %u\n", size_IDs );
		printf( "\tsizeBuildQ   = %u\n", size_BuildQ );
		printf( "\tsizeCounts   = %u\n", size_Counts );
		printf( "\tsizeStarts   = %u\n", size_Starts );
		printf( "\tsizePivot    = %u\n", size_Pivot );
		return false;
	}


	/*----------------------------
	  Initialize Root Build Item
	----------------------------*/

	/*-------------------------------------------
	  Allocate space on CPU host & GPU device
	-------------------------------------------*/
	
	GPUNode_2D_LBT * h_NodesLBT = NULL;
	GPUNode_2D_LBT * d_NodesLBT = NULL;

	GPUNode_2D_MED * h_NodesMED = NULL;
	GPUNode_2D_MED * d_NodesMED = NULL;

	//GPUNode_2D_MED * h_Scratch  = NULL;
	GPUNode_2D_MED * d_Scratch  = NULL;

	GPU_BUILD_ITEM * h_BuildQ   = NULL;
	GPU_BUILD_ITEM * h_ScratchQ = NULL;
	GPU_BUILD_ITEM * d_BuildQ   = NULL;

	GPU_COUNTS_STARTS * h_Counts = NULL;
	GPU_COUNTS_STARTS * d_Counts = NULL;

	GPU_COUNTS_STARTS * h_Starts = NULL;
	GPU_COUNTS_STARTS * d_Starts = NULL;

#ifdef _BUILD_STATS
	GPU_BUILD_STATS * h_StatsQ = NULL;
	GPU_BUILD_STATS * d_StatsQ = NULL;
#endif

	//unsigned int * h_Pivot    = NULL;
	unsigned int * d_Pivot = NULL;

	unsigned int * h_PointIDs = NULL;
	unsigned int * d_PointIDs   = NULL;

		//---------------------------------------------
		// Allocate memory for point index lookup table
		//---------------------------------------------

	if (size_IDs > 0u)
	{
		// Allocate Device memory for left-balanced kd-tree nodes
		cudaResult = cudaMalloc( (void **) &(d_PointIDs), size_IDs );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for left-balanced  kd-tree nodes
		if (g_app.dumpVerbose >= 1)
		{
			h_PointIDs = (unsigned int *)AllocHostMemory( size_IDs, params.bPinned );
			if (NULL == h_PointIDs)
			{
				fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
				exit( EXIT_FAILURE );
			}
		}
	}


		//---------------------------------------------
		// Allocate memory for LBT Nodes
		//---------------------------------------------

	if (size_NodesLBT > 0u)
	{
		// Allocate Device memory for left-balanced kd-tree nodes
		cudaResult = cudaMalloc( (void **) &(d_NodesLBT), size_NodesLBT );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for left-balanced  kd-tree nodes
		if (g_app.dumpVerbose >= 1)
		{
			h_NodesLBT = (GPUNode_2D_LBT *)AllocHostMemory( size_NodesLBT, params.bPinned );
			if (NULL == h_NodesLBT)
			{
				fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
				exit( EXIT_FAILURE );
			}
		}
	}

		//---------------------------------------------
		// Allocate memory for Median Nodes
		//---------------------------------------------	

	if (size_NodesMED > 0u)
	{
		// Allocate Device memory for median kd-tree nodes
		cudaResult = cudaMalloc( (void **) &(d_NodesMED), size_NodesMED );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Device memory for median scratch buffer
		cudaResult = cudaMalloc( (void **) &(d_Scratch), size_NodesMED );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for median  kd-tree nodes
		h_NodesMED = (GPUNode_2D_MED *)AllocHostMemory( size_NodesMED, params.bPinned );
		if (NULL == h_NodesMED)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}

		//h_Scratch = (GPUNode_2D_MED *)AllocHostMemory( size_NodesMED, params.bPinned );
		//if (NULL == h_Scratch)
		//{
			//fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			//exit( EXIT_FAILURE );
		//}
	}

		//---------------------------------------------
		// Allocate memory for Build Queue (Phase 1)
		//---------------------------------------------	

	if (size_BuildQ > 0u)
	{
		// Allocate Device memory for GPU Build Queue
		cudaResult = cudaMalloc( (void **) &(d_BuildQ), size_BuildQ );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for Build Queue
		h_BuildQ = (GPU_BUILD_ITEM *)malloc( size_BuildQ );
		if (NULL == h_BuildQ)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for Build Queue
		h_ScratchQ = (GPU_BUILD_ITEM *)malloc( size_BuildQ );
		if (NULL == h_ScratchQ)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}

#if _BUILD_STATS
		//------------------------------------------------
		// Allocate memory for Build Stats Queue (Phase 2)
		//------------------------------------------------

	if (size_StatsQ > 0u)
	{
		// Allocate Device memory for GPU Build Stats
		cudaResult = cudaMalloc( (void **) &(d_StatsQ), size_StatsQ );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for Build Stats
		h_StatsQ = (GPU_BUILD_STATS *)malloc( size_StatsQ );
		if (NULL == h_StatsQ)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#endif

		//--------------------------------------------------
		// Allocate memory for Counts 
		//--------------------------------------------------

	if (size_Counts > 0u)
	{
		// Allocate Device memory for counts
		cudaResult = cudaMalloc( (void **) &(d_Counts), size_Counts );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for counts
		h_Counts = (GPU_COUNTS_STARTS *)AllocHostMemory( size_Counts, params.bPinned );
		if (NULL == h_Counts)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}

		//--------------------------------------------------
		// Allocate memory for Starts
		//--------------------------------------------------

	if (size_Starts > 0u)
	{
		// Allocate Device memory for starts
		cudaResult = cudaMalloc( (void **) &(d_Starts), size_Starts );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for starts
		h_Starts = (GPU_COUNTS_STARTS *)AllocHostMemory( size_Counts, params.bPinned );
		if (NULL == h_Starts)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}

		//---------------------------------------------
		// Allocate memory for pivot location
		//---------------------------------------------

	if (size_Pivot > 0u)
	{
		// Allocate Device memory for pivot index
		cudaResult = cudaMalloc( (void **) &(d_Pivot), size_Pivot );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMalloc() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Allocate Host memory for pivot
		//h_Pivot = (unsigned int *)AllocHostMemory( size_Pivot, params.bPinned );
		//if (NULL == h_Pivot)
		//{
		//	fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
		//	exit( EXIT_FAILURE );
		//}
	}


	/*---------------------------------
	  Initialize Inputs
	---------------------------------*/

		//--------------------------------------------
		// Copy initial Median Nodes from pointList
		//--------------------------------------------

	// BUGBUG:  Would this initialization be faster on GPU ???

	float firstX, firstY;
	firstX = params.searchList[0].x;
	firstY = params.searchList[0].y;

	// Initialize zeroth node (which is effectively wasted space)
	h_NodesMED[0].pos[0]      = firstX;
	h_NodesMED[0].pos[1]      = firstY;
	h_NodesMED[0].m_searchIdx = nOrigPoints+1;	// Move collision problem to where it can't hurt us
	h_NodesMED[0].m_nodeIdx   = 0u;

	// Copy pointList into <unordered> median nodes list
	unsigned int i, i1;
	for (i = 1; i <= nOrigPoints; i++)
	{
		i1 = i-1;
		h_NodesMED[i].pos[0]      = params.searchList[i1].x;
		h_NodesMED[i].pos[1]      = params.searchList[i1].y;
		h_NodesMED[i].m_searchIdx = i1;	
		h_NodesMED[i].m_nodeIdx   = (unsigned int)(-1);
	}

	// Initialize extra padded nodes to something
	for (i = nOrigPoints+1; i < nPadPoints; i++)
	{
		h_NodesMED[i].pos[0]      = firstX;
		h_NodesMED[i].pos[1]      = firstY;
		h_NodesMED[i].m_searchIdx = i;
		h_NodesMED[i].m_nodeIdx   = (unsigned int)(-1);
	}

		//-----------------------------------
		// Init BuildQ to single element
		//-----------------------------------

	// Empty build queue
	unsigned int maxPhase2 = P2_BUILD_TOTAL_THREADS;
	unsigned int maxQ     = 256;
	unsigned int startQ   = 0;
	unsigned int endQ     = 0;
	unsigned int cntQ     = 0;
	memset( h_BuildQ, 0x00, size_BuildQ );
	memset( h_ScratchQ, 0x00, size_BuildQ );

	unsigned int startRange = 1;			// Range [1,nPoints]
	unsigned int endRange   = nOrigPoints;
	unsigned int N          = (endRange - startRange) + 1;
	unsigned int origStart, origEnd, currStart, currEnd;
	unsigned int currTarget, currFlags, origN;
	unsigned int currMedian, currHalf, currLBM, lastRow;		
	unsigned int currAxis, nextAxis, currLeft, currRight;
	bool bDone;

	// Kernel Grid & Block Sizes
	dim3 pivotGrid( 1, 1, 1 );
	dim3 pivotBlock( 1, 1, 1 );
	dim3 countGrid( BUILD_BLOCKS_PER_ROW, 1, 1 );
	dim3 countBlock( BUILD_THREADS_PER_ROW, BUILD_ROWS_PER_BLOCK, 1 );
	dim3 csScanGrid( 1, 1, 1 );
	dim3 csScanBlock( BUILD_CS_SCAN_THREADS, 1, 1 );
	dim3 partGrid( BUILD_BLOCKS_PER_ROW,1,1);
	dim3 partBlock( BUILD_THREADS_PER_ROW, BUILD_ROWS_PER_BLOCK, 1 );
	dim3 storeGrid( 1, 1, 1 );
	dim3 storeBlock( 1, 1, 1 );
	dim3 p2BuildGrid( P2_BUILD_BLOCKS_PER_ROW, P2_BUILD_ROWS_PER_GRID, 1 );
	dim3 p2BuildBlock( P2_BUILD_THREADS_PER_ROW, P2_BUILD_ROWS_PER_BLOCK, 1 );
	dim3 p2CopyGrid( BUILD_BLOCKS_PER_ROW, 1, 1 );
	dim3 p2CopyBlock( BUILD_THREADS_PER_ROW, BUILD_ROWS_PER_BLOCK, 1 );

#ifdef _BUILD_STATS
	// Initialize Stats
	GPU_BUILD_STATS p1Stats;
	p1Stats.cRootReads   = 0;
	p1Stats.cPivotReads  = 0;
	p1Stats.cCountReads  = 0;
	p1Stats.cCountWrites = 0;
	p1Stats.cPartReads   = 0;
	p1Stats.cPartWrites  = 0;
	p1Stats.cStoreReads	 = 0;
	p1Stats.cStoreWrites = 0;
	p1Stats.cNodeLoops   = 0;
	p1Stats.cPartLoops   = 0;
	p1Stats.cD2HReads    = 0;
	p1Stats.cD2HWrites   = 0;
	p1Stats.cH2DReads    = 0;
	p1Stats.cH2DWrites   = 0;
#endif

	// Get Left-balanced median position for root
	KD_LBM_HOST( N, currLBM, currHalf );
	currMedian = startRange + currLBM - 1;

	// Push root info onto build queue
	h_BuildQ[endQ].start     = startRange;		// Root Range = [1,n]
	h_BuildQ[endQ].end       = endRange;
	h_BuildQ[endQ].targetID  = 1;				// Root is always at index 1
	h_BuildQ[endQ].flags     = (currHalf & NODE_INDEX_MASK) | 
		                       ((X_AXIS << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
	cntQ++;
	endQ++;
	if (endQ >= maxQ) { endQ = 0u; }

#ifdef _BUILD_STATS
	p1Stats.cRootReads++;
#endif

	/*----------------------------
	  Transfer inputs onto GPU
	  for phase 1
    ----------------------------*/
	
	// Copy median 'kd-nodes' from host memory to device memory
	if ((NULL != d_NodesMED) && (NULL != h_NodesMED))
	{
		cudaResult = cudaMemcpy( d_NodesMED, h_NodesMED, 
			                     size_NodesMED, cudaMemcpyHostToDevice );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMemcpy() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}
	}

#ifdef _BUILD_STATS
	p1Stats.cH2DReads  += nPadPoints;
	p1Stats.cH2DWrites += nPadPoints;
#endif

	/*----------------------------
	  Phase 1 Processing
		// Hybrid CPU/GPU solution
    ----------------------------*/

	// Process children ranges until we finish or fill up build items for phase 2.
	while ((0 < cntQ) && (cntQ < maxPhase2))
	{
		// Grab Build Info from front of queue
		origStart  = (h_BuildQ[startQ].start & NODE_INDEX_MASK);
		origEnd    = (h_BuildQ[startQ].end & NODE_INDEX_MASK);
		currTarget = h_BuildQ[startQ].targetID;
		currFlags  = h_BuildQ[startQ].flags;

#ifdef _BUILD_STATS
	p1Stats.cNodeLoops++;
#endif

		cntQ--;
		startQ++;
		if (startQ >= maxQ) { startQ = 0; }

		// Compute axes
		currAxis   = ((currFlags & SPLIT_AXIS_MASK) >> SPLIT_AXIS_SHIFT);
		nextAxis   = ((currAxis == 1u) ? 0u : 1u);

		// No need to do median sort if only one element is in range (IE a leaf node)
		origN = origEnd-origStart + 1;
		if (origN > 1)
		{
			//-------------------------
			// Compute Median Index
			//-------------------------

			if (origN <= 31)
			{
				// Lookup answer from table
				currHalf = g_halfTableHOST[origN];
				currLBM  = g_leftTableHOST[origN];
			}
			else
			{			
				// Compute Left-balanced Median
				currHalf   = (currFlags & NODE_INDEX_MASK); 
				lastRow    = KD_Min( currHalf, origN-(2*currHalf)+1 );	// Get # of elements to include from last row
				currLBM    = currHalf + lastRow;					// Get left-balanced median
			}
			currMedian = origStart + (currLBM - 1);			// Get actual median 


			//---------------------------------------------
			// Apply Median selection algorithm 
			// on specified range [start,end] 
			// and requested left-balanced median 
			//---------------------------------------------

			currStart = origStart;
			currEnd   = origEnd;

			bDone = false;
			while (! bDone)
			{
#ifdef _BUILD_STATS
	p1Stats.cPartLoops++;
#endif

				//---------------------
				// Pick a Pivot value
				//---------------------

				GPU_2D_NODE_MED_PICK_PIVOT<<< pivotGrid, pivotBlock >>>
				(
					d_Pivot,		// Pivot memory location
					d_NodesMED,		// List of nodes
					currStart,		// Range [start,end]
					currEnd,		
					currAxis		// Curr axis
				);

#ifdef _BUILD_STATS
	p1Stats.cPivotReads += 3;
	//p1Stats.cPivotWrites += 0;
#endif 

				// Check if we had an ERROR
				cudaThreadSynchronize();
				cudaResult = cudaGetLastError();
				if (cudaSuccess != cudaResult) 
				{
					fprintf( stderr, "Host_Build(): GPU_2D_PICK_PIVOT kernel failed with error (%d = %s) in file '%s' at line %i.\n",
							 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
					exit( EXIT_FAILURE );
				}


				//--------------------------------------------
				// Count # of elements in each partition
				// {L}, {M}, and {R} formed by pivot
				//--------------------------------------------

				GPU_2D_NODE_MED_COUNTS<<< countGrid, countBlock >>>
				(
					d_Counts,	// OUT: counts
					d_NodesMED,	// IN:  median node list (source)
					d_Scratch,  // OUT: scratch buffer (dest)
					d_Pivot,	// IN: pivot location
					endRange,	// IN: number of nodes
					currStart,	// IN: range [start, end] to count
					currEnd,	// IN: end of range to count
					currAxis	// IN: axis of dimension to work with
				);

#ifdef _BUILD_STATS
	p1Stats.cPivotReads  += BUILD_TOTAL_THREADS;	// Read pivot at begin of algorithm (per thread)
	p1Stats.cCountReads  += (currEnd - currStart + 1);	// Read nodes from orig buffer
	p1Stats.cCountWrites += (currEnd - currStart + 1);	// Write notes to scratch buffer
	p1Stats.cCountWrites += BUILD_TOTAL_THREADS;	// Store counts at end of algorithm (per thread)
#endif 

				// Check if we had an ERROR
				cudaThreadSynchronize();
				cudaResult = cudaGetLastError();
				if (cudaSuccess != cudaResult) 
				{
					fprintf( stderr, "Host_Build(): GPU_2D_MED_COUNTS kernel failed with error (%d = %s) in file '%s' at line %i.\n",
							 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
					exit( EXIT_FAILURE );
				}


				//--------------------------------------------
				// Compute thread starts from thread counts
				//--------------------------------------------

				size_t sizeSharedMem = BUILD_CS_SCAN_MAX_ITEMS * sizeof(GPU_COUNTS_STARTS);
				GPU_2D_COUNTS_TO_STARTS<<< csScanGrid, csScanBlock, sizeSharedMem >>>
				( 
					d_Starts,	// OUT - starts (store prefix sums here)
					d_Counts,	// IN  - Counts (to total)
					nCounts,	// IN  - number of items in count list
					currStart,	// IN  - range[start,end]
					currEnd		//       ditto
				);

				/*
				// Copy counts onto CPU host
				cudaResult = cudaMemcpy( h_Counts, d_Counts, 
										 size_Counts, cudaMemcpyDeviceToHost );
				if (cudaSuccess != cudaResult) 
				{
					fprintf( stderr, "Host_Build(): cudaMemcpy() failed with error (%d = %s) in file '%s' at line %i.\n",
							 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
					exit( EXIT_FAILURE );
				}

#ifdef _BUILD_STATS
	p1Stats.cD2HReads   += BUILD_TOTAL_THREADS;  // Read pivot at begin of algorithm
	p1Stats.cD2HWrites  += BUILD_TOTAL_THREADS;	// Read nodes from orig buffer
#endif 

				// Compute partition totals
				unsigned int beforeTotal = 0u;
				unsigned int afterTotal  = 0u;
				unsigned int equalTotal  = 0u;

				for (currIdx = 0; currIdx < BUILD_TOTAL_THREADS; currIdx++)
				{
					beforeTotal += h_Counts[currIdx].before;
					afterTotal  += h_Counts[currIdx].after;
					equalTotal  += h_Counts[currIdx].equal;
				}

				// Double check totals are correct
				unsigned int totalCount = beforeTotal + afterTotal + equalTotal;
				unsigned int nRange = currEnd - currStart + 1;
				if (totalCount != nRange)
				{
					// Error - we have a bug
					fprintf( stdout, "Count Totals(%d) != Range Size(%d)\n", totalCount, nRange );
					//exit( 0 );
				}

				// Initialize bases for first thread
				unsigned beforeBase = currStart;
				unsigned equalBase  = beforeBase + beforeTotal;
				unsigned afterBase  = equalBase + equalTotal;

				unsigned beforeStart = beforeBase;
				unsigned equalStart  = equalBase;
				unsigned afterStart  = afterBase;

				// Compute starts from counts and bases
				for (currIdx = 0; currIdx < BUILD_TOTAL_THREADS; currIdx++)
				{
					// Set starts for current thread
					h_Starts[currIdx].before = beforeStart;
					h_Starts[currIdx].after  = afterStart;
					h_Starts[currIdx].equal  = equalStart;

					// Update running starts for next thread
					beforeStart += h_Counts[currIdx].before;
					afterStart  += h_Counts[currIdx].after;
					equalStart  += h_Counts[currIdx].equal;
				}
				
				// Copy starts onto GPU device
				cudaResult = cudaMemcpy( d_Starts, h_Starts, size_Starts, 
					                     cudaMemcpyHostToDevice );
				if (cudaSuccess != cudaResult) 
				{
					fprintf( stderr, "Host_Build(): cudaMemcpy() failed with error (%d = %s) in file '%s' at line %i.\n",
							 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
					exit( EXIT_FAILURE );
				}

#ifdef _BUILD_STATS
	p1Stats.cH2DReads   += BUILD_TOTAL_THREADS;  // Read pivot at begin of algorithm
	p1Stats.cH2DWrites  += BUILD_TOTAL_THREADS;	// Read nodes from orig buffer
#endif 
				*/

				//--------------------------------------------
				// Partition elements
				// sets {L}, {M}, and {R} using pivot
				//--------------------------------------------

				GPU_2D_NODE_MED_PARTITION<<< partGrid, partBlock >>>
				(
					d_Starts,		// OUT: starts
					d_Scratch,		// IN: scratch list (source)
					d_NodesMED,		// OUT: median node list (dest)
					nOrigPoints,	// IN: number of nodes
					currStart,		// IN: range [start, end] to count
					currEnd,		// IN: end of range to count
					currAxis,		// IN: axis of dimension to work with
					d_Pivot			// IN: pivot index
				);

#ifdef _BUILD_STATS
	p1Stats.cPivotReads += BUILD_TOTAL_THREADS;  // Read pivot at begin of algorithm (per thread)
	p1Stats.cPartReads  += BUILD_TOTAL_THREADS;	 // Read in starts at beginning of algorithm (per threads)
	p1Stats.cPartReads  += (currEnd - currStart + 1);	// Read nodes from orig buffer
	p1Stats.cPartWrites += (currEnd - currStart + 1);	// Write notes to scratch buffer
#endif 

				// Check if we had an ERROR
				cudaThreadSynchronize();
				cudaResult = cudaGetLastError();
				if (cudaSuccess != cudaResult) 
				{
					fprintf( stderr, "Host_Build(): GPU_2D_MED_COUNTS kernel failed with error (%d = %s) in file '%s' at line %i.\n",
							 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
					exit( EXIT_FAILURE );
				}


				//--------------------------------------------
				// Check if we are done
				//--------------------------------------------

				// Read beginning start off of GPU
				cudaResult = cudaMemcpy( h_Starts, d_Starts, sizeof(GPU_COUNTS_STARTS), 
					                     cudaMemcpyDeviceToHost );
				//unsigned int beforeBase = h_Starts[0].before;
				unsigned int afterBase  = h_Starts[0].after;
				unsigned int equalBase  = h_Starts[0].equal;

				if (currMedian < equalBase)	
				{
					// Not done, iterate on {L} partition = [currStart, equalBase - 1]
					currEnd   = equalBase - 1;
				}
				else if (currMedian >= afterBase)	// Median in after partition {R}
				{
					// Not done, iterate on {R} partition = range [afterBase, currEnd]
					currStart = afterBase;
				}
				else // Median is in median partition {M}
				{
					// Done, the left-balanced median is where we want it
					bDone = true;
				}				
			} // end while (currPivot != currMedian)
		}
		else
		{
			currMedian = origStart;
		}

		// Store current median node 
		// in left balanced list at target
		GPU_2D_NODE_STORE<<< storeGrid, storeBlock  >>>
		(
			d_NodesMED,		// IN:  Median nodes
			d_NodesLBT,		// OUT:  LBT nodes
			d_PointIDs,		// OUT:  point ids
			currMedian,		// IN:   left balanced median index
			currTarget		// IN:   target index for storing node
		);

#ifdef _BUILD_STATS
	p1Stats.cStoreReads  += 1;  // Read pivot at begin of algorithm (per thread)
	p1Stats.cStoreWrites += 2;	 // Read in starts at beginning of algorithm (per threads)
#endif 

		cudaThreadSynchronize();
		cudaResult = cudaGetLastError();
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): GPU_2D_NODE_STORE kernel failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		currLeft  = currTarget << 1;
		currRight = currLeft + 1;

		if (currLeft <= nOrigPoints)
		{
			// enqueue left child
			h_BuildQ[endQ].start    = (origStart & NODE_INDEX_MASK);
			h_BuildQ[endQ].end      = ((currMedian-1) & NODE_INDEX_MASK);
			h_BuildQ[endQ].targetID = (currLeft & NODE_INDEX_MASK);
			h_BuildQ[endQ].flags    = ((currHalf >> 1) & NODE_INDEX_MASK) 
								        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			cntQ++;
			endQ++;
			if (endQ >= maxQ) { endQ = 0u; }
		}

		if (currRight <= nOrigPoints)
		{
			// enqueue right child
			h_BuildQ[endQ].start    = ((currMedian + 1) & NODE_INDEX_MASK);
			h_BuildQ[endQ].end      = (origEnd & NODE_INDEX_MASK);
			h_BuildQ[endQ].targetID = (currRight & NODE_INDEX_MASK);
			h_BuildQ[endQ].flags    = ((currHalf >> 1) & NODE_INDEX_MASK) 
								        | ((nextAxis << SPLIT_AXIS_SHIFT) & SPLIT_AXIS_MASK);
			cntQ++;
			endQ++;
			if (endQ >= maxQ) { endQ = 0u; }
		}
	}

#ifdef _BUILD_STATS
	DumpP1Stats( p1Stats );
#endif

	/*---------------------------------
	  Phase 2 Processing
		// Hybrid CPU/GPU solution
    ---------------------------------*/

	bool bPhase2 = false;
	if (cntQ > 0)
	{
		bPhase2 = true;
	}

	if (bPhase2)
	{
		//-------------------------------------------------
		// Copy Individual Thread Build Queues onto GPU
		//-------------------------------------------------

		unsigned int n1, n2;
		unsigned int size_Part1, size_Part2;
		GPU_BUILD_ITEM * src_Part1;
		GPU_BUILD_ITEM * src_Part2;
		GPU_BUILD_ITEM * dst_Part1;
		GPU_BUILD_ITEM * dst_Part2;

		// Is build queue range currently represented 
		// by a single or double partition ?!?
		if (endQ >= startQ) 
		{ 
			// Queue range has a single partition [start,end]
			src_Part1 = h_BuildQ + startQ;
			dst_Part1 = h_ScratchQ;
			n1 = cntQ;

			// Shift valid range back over to start at position zero
			size_Part1 = n1 * sizeof( GPU_BUILD_ITEM );
			memcpy( dst_Part1, src_Part1, size_Part1 );

			// Pad rest of queue with invalid build items
			if (n1 < maxPhase2)
			{
				for (i = n1; i < maxPhase2; i++)
				{
					// Make sure 'end < start' to indicate invalid build item
					h_ScratchQ[i].start    = (1 & NODE_INDEX_MASK);
					h_ScratchQ[i].end      = (0 & NODE_INDEX_MASK);
					h_ScratchQ[i].targetID = 0;
					h_ScratchQ[i].flags    = 0;
				}
			}

			// Copy build range onto GPU
			cudaResult = cudaMemcpy( d_BuildQ, h_ScratchQ, size_BuildQ, cudaMemcpyHostToDevice );
			if (cudaSuccess != cudaResult) 
			{
				fprintf( stderr, "Host_Build(): cudaMemcpy() failed with error (%d = %s) in file '%s' at line %i.\n",
						 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
				exit( EXIT_FAILURE );
			}
		}
		else
		{
			// Queue Range has two partitions [start,maxQ-1] and [0, endQ]
			n1 = maxQ  - startQ;
			n2 = endQ + 1;
			size_Part1 = n1 * sizeof( GPU_BUILD_ITEM );
			size_Part2 = n2 * sizeof( GPU_BUILD_ITEM );
			src_Part1 = h_BuildQ + startQ;
			src_Part2 = h_BuildQ;
			dst_Part1 = h_ScratchQ;
			dst_Part2 = h_ScratchQ + n1;

			if ((n1+n2) != cntQ) 
			{
				fprintf( stdout, "nRange(%u) != cntQ(%u), at line (%d) in file ('%s')\n", 
						 n1+n2, cntQ, __LINE__, __FILE__ );
			}

			// Shift 1st range [startQ, maxQ] back over to start at position zero
			size_Part1 = n1 * sizeof( GPU_BUILD_ITEM );
			memcpy( dst_Part1, src_Part1, size_Part1 );

			// Shift 2nd range [0,endQ] into position
			size_Part2 = n2 * sizeof( GPU_BUILD_ITEM );
			memcpy( dst_Part2, src_Part2, size_Part2 );

			if ((n1+n2) < maxPhase2)
			{
				// Pad rest of queue with invalid build items
				for (i = (n1+n2); i < maxPhase2; i++)
				{
					// Make sure end < start to indicate invalid build item
					h_ScratchQ[i].start    = (1 & NODE_INDEX_MASK);
					h_ScratchQ[i].end      = (0 & NODE_INDEX_MASK);
					h_ScratchQ[i].targetID = 0;
					h_ScratchQ[i].flags    = 0;
				}
			}

			// Copy build range onto GPU
			cudaResult = cudaMemcpy( d_BuildQ, h_ScratchQ, size_BuildQ, cudaMemcpyHostToDevice );
			if (cudaSuccess != cudaResult) 
			{
				fprintf( stderr, "Host_Build(): cudaMemcpy() failed with error (%d = %s) in file '%s' at line %i.\n",
						 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
				exit( EXIT_FAILURE );
			}
		}
	}

	//-------------------------------------------------------
	// Phase 2 - build 'm' thread partitions in parallel
	//-------------------------------------------------------

	if (bPhase2)
	{
		// Build Median kd-tree from queue of build items
			// 1 thread per sub-tree (build item)
		//P2_2D_BUILD_MED<<< p2BuildGrid, p2BuildBlock >>>
		//(
		//	d_NodesMED, d_Scratch, d_BuildQ, nOrigPoints
		//);
#if _BUILD_STATS
		P2_2D_BUILD_STATS<<< p2BuildGrid, p2BuildBlock >>>
		(
			d_NodesLBT,	// OUT: lbt node list
			d_PointIDs,	// OUT: point indices are stored in this array
			d_StatsQ,	// OUT: build stats
			d_NodesMED,	// IN: median node list
			d_Scratch,	// IN: scratch space for temporary copying
			d_BuildQ,	// IN: build queue (per thread)
			nOrigPoints	// IN: maximum # of points
		);

		// Copy Phase2 stats back to host
		cudaResult = cudaMemcpy( h_StatsQ, d_StatsQ, size_StatsQ, cudaMemcpyDeviceToHost );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): cudaMemcpy() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			exit( EXIT_FAILURE );
		}

		// Dump Phase2 stats
		unsigned int maxStatsQ = P2_BUILD_TOTAL_THREADS;
		DumpP2Stats( maxStatsQ, h_StatsQ );
#else
		P2_2D_BUILD_LBT<<< p2BuildGrid, p2BuildBlock >>>
		(
			d_NodesLBT,	// OUT: lbt node list
			d_PointIDs,	// OUT: point indices are stored in this array
			d_NodesMED,	// IN: median node list
			d_Scratch,	// IN: scratch space for temporary copying
			d_BuildQ,	// IN: build queue (per thread)
			nOrigPoints	// IN: maximum # of points
		);
#endif


		// Check if we had an ERROR
		cudaThreadSynchronize();
		cudaResult = cudaGetLastError();
		if( cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "Host_Build(): GPU_2D_BUILD_PHASE2 kernel failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			//exit( EXIT_FAILURE );
		}

		// Build Left-balanced kd-tree from median kd-tree
		//P2_2D_COPY<<< p2CopyGrid, p2CopyBlock >>>
		//(
		//	d_NodesLBT, d_PointIDs, d_NodesMED, 1, nOrigPoints
		//);

		// Check if we had an ERROR
		//cudaThreadSynchronize();
		//cudaResult = cudaGetLastError();
		//if( cudaSuccess != cudaResult) 
		//{
		//	fprintf( stderr, "Host_Build(): GPU_2D_BUILD_PHASE2 kernel failed with error (%d = %s) in file '%s' at line %i.\n",
		//			 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
			//exit( EXIT_FAILURE );
		//}
	}


	/*--------------------
	  Cleanup
	---------------------*/

	if (g_app.dumpVerbose >= 1)
	{
		if ((NULL != h_NodesLBT) && (NULL != d_NodesLBT))
		{
			cudaResult = cudaMemcpy( h_NodesLBT, d_NodesLBT, size_NodesLBT, cudaMemcpyDeviceToHost );
			if( cudaSuccess != cudaResult) 
			{
				fprintf( stderr, "Host_Build(): cudaMemcpy from GPU to CPU failed with error (%d = %s) in file '%s' at line %i.\n",
						 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
				exit( EXIT_FAILURE );
			}
		}

		if ((NULL != h_PointIDs) && (NULL != d_PointIDs))
		{
			cudaResult = cudaMemcpy( h_PointIDs, d_PointIDs, size_IDs, cudaMemcpyDeviceToHost );
		    if( cudaSuccess != cudaResult) 
			{
				fprintf( stderr, "Host_Build(): cudaMemcpy from GPU to CPU failed with error (%d = %s) in file '%s' at line %i.\n",
						 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
				exit( EXIT_FAILURE );
			}
		}
	}

	// Cleanup temporary Memory Resources
	FreeHostMemory( h_Starts, params.bPinned );
	cutilSafeCall( cudaFree( d_Starts ) );

	FreeHostMemory( h_Counts, params.bPinned );
	cutilSafeCall( cudaFree( d_Counts ) );

#ifdef _BUILD_STATS
	if (NULL != h_StatsQ)   { free( h_StatsQ ); h_StatsQ = NULL; }
	cutilSafeCall( cudaFree( d_StatsQ ) );
#endif

	if (NULL != h_BuildQ)   { free( h_BuildQ ); h_BuildQ = NULL; }
	if (NULL != h_ScratchQ) { free( h_ScratchQ ); h_ScratchQ = NULL; }
	cutilSafeCall( cudaFree( d_BuildQ ) );

	//FreeHostMemory( h_Scratch, params.bPinned );
	cutilSafeCall( cudaFree( d_Scratch ) );

	FreeHostMemory( h_NodesMED, params.bPinned );
	cutilSafeCall( cudaFree( d_NodesMED ) );

	//FreeHostMemory( h_Pivot, params.bPinned );
	cutilSafeCall( cudaFree( d_Pivot ) );

	// BUGBUG - don't cleanup semi-permanent memory structures now, clean them up later

	//FreeHostMemory( h_NodesLBT, params.bPinned );
	//cutilSafeCall( cudaFree( d_NodesLBT ) );
	
	//FreeHostMemory( h_PointsID, params.bPinned );
	//cutilSafeCall( cudaFree( d_PointIDs ) );

	// Cleanup GPU Device Memory

	/*------------------
	  Return Results
	------------------*/

	params.mem_size_Nodes = size_NodesLBT;
	params.mem_size_IDs   = size_IDs;

	params.h_Nodes        = h_NodesLBT;
	params.d_Nodes        = d_NodesLBT; 

	params.h_IDs          = h_PointIDs;
	params.d_IDs          = d_PointIDs;

	return bResult;
}


/*-------------------------------------------------------------------------
  Name: Fini_Host_Build
  Desc:	Cleanup GPU kd-tree structures from build
-------------------------------------------------------------------------*/

void FiniHostBuild( HostDeviceParams & params )
{
	// Free LBT kd-nodes
	FreeHostMemory( params.h_Nodes, params.bPinned );
	params.h_Nodes = NULL;

	if (params.d_Nodes != NULL)
	{
		cutilSafeCall( cudaFree( params.d_Nodes ) );
		params.d_Nodes = NULL;
	}

	// Free mapping IDs (node indices to point indices)
	FreeHostMemory( params.h_IDs, params.bPinned );
	params.h_IDs = NULL;

	if (params.d_IDs != NULL)
	{
		cutilSafeCall( cudaFree( params.d_IDs ) );
		params.d_IDs = NULL;
	}
}

