/*-----------------------------------------------------------------------------
  File:  GPU_API_LBT.cpp
  Desc:  Host CPU API scaffolding for running kd-tree NN searches
         for left-balanced binary tree array layouts

  Log:   Created by Shawn D. Brown (3/22/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
//#include <string_app.h>
#include <math.h>

// includes, CUDA
#include <cutil_inline.h>

// includes, project
#include "CPUTREE_API.h"
#include "GPUTREE_API.h"
#include "QueryResult.h"

#include "CPUTree_LBT.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

extern AppGlobals g_app;


/*-------------------------------------
  Function Declarations
-------------------------------------*/

/*---------------------------------------------------------
  Name:	DefaultParams
---------------------------------------------------------*/

void DefaultParams
(
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	dim3 tmpBlock( 0, 0, 0 );
	dim3 tmpGrid( 0, 0, 0 );

	// Query Type
	params.nnType		= NN_UNKNOWN;
	params.nDims		= 0u;
	params.kVal         = 0u;
	//params.kPadVal      = 0u;

	// Search Info
	params.nSearch		= 0u;
	params.nPadSearch	= 0u;
	params.searchList	= NULL;

	// Query Info
	params.nQuery		= 0u;
	params.nPadQuery	= 0u;
	params.queryList	= NULL;

	// Results Info
	params.nOrigResult	= 0u;
	params.nPadResult	= 0u;
	params.resultList	= NULL;

	// CUDA Properties
	params.rawProps		= NULL;
	params.cudaProps	= NULL;

	// Thread Block Shape
	params.bgShape.nElems			= 0u;
	params.bgShape.nPadded			= 0u;
	params.bgShape.threadsPerRow	= 0u;
	params.bgShape.rowsPerBlock		= 0u;
	params.bgShape.threadsPerBlock  = 0u;
	params.bgShape.blocksPerRow     = 0u;
	params.bgShape.rowsPerGrid      = 0u;
	params.bgShape.blocksPerGrid	= 0u;
	params.bgShape.W				= 0u;
	params.bgShape.H				= 0u;

	params.nnBlock = tmpBlock;
	params.nnGrid  = tmpGrid;

	params.kdtree		= NULL;
	params.reservedPtr1 = NULL;

	params.mem_size_Query = 0u;
	params.h_Query = NULL;
	params.d_Query = NULL;

	params.mem_size_Nodes = 0u;
	params.h_Nodes	= NULL;
	params.d_Nodes	= NULL;

	params.mem_size_IDs = 0u;			
	params.h_IDs		= NULL;
	params.d_IDs		= NULL;

	// Results
	params.mem_size_Results_GPU = 0u;			
	params.mem_size_Results_CPU = 0u;			
	params.d_Results_GPU		= NULL;
	params.h_Results_CPU		= NULL;

	// Misc
	params.buildGPU = true;
	//params.buildGPU = false;
	
	params.bPinned = false;
	//params.bPinned = true;

	params.bRowByRow = g_app.rowByRow;
}


/*---------------------------------------------------------
  Name:	InitTimer
---------------------------------------------------------*/

void InitTimer()
{
	cudaError_t cudaResult;

	// Initialize Timer
	if (g_app.profile)
	{
		if (g_app.hTimer == 0)
		{
			cutCreateTimer( &(g_app.hTimer) );
			cutStartTimer( g_app.hTimer );
			cutStopTimer( g_app.hTimer );
		}	

		// Compute Base Timer Cost
		unsigned int idx;
		double totalTime = 0.0;
		float elapsedTime;
		for (idx = 0; idx < 100; idx++)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );

			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			elapsedTime = cutGetTimerValue( g_app.hTimer );

			totalTime += (double)elapsedTime;
		}
		g_app.baseTimerCost = totalTime / 100.0;

		// Compute CUDA Timer Cost
		cudaEvent_t weirdBug    = 0;
		cudaEvent_t searchStart = 0;
		cudaEvent_t searchStop  = 0;

		// Work around weird bug (first call to cudaEventCreate in emulation mode always fails)
		cudaResult = cudaEventCreate( &weirdBug );

		cudaResult = cudaEventCreate( &searchStart );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "InitTimer(): cudaEventCreate() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
		}

		cudaResult = cudaEventCreate( &searchStop );
		if (cudaSuccess != cudaResult) 
		{
			fprintf( stderr, "InitTimer(): cudaEventCreate() failed with error (%d = %s) in file '%s' at line %i.\n",
					 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
		}

		totalTime = 0.0;
		for (idx = 0; idx < 100; idx++)
		{
			// Start Timer
			cudaEventRecord( searchStart, 0 );

			// Stop Timer and save performance measurement
			cudaEventRecord( searchStop, 0 );
			cudaEventSynchronize( searchStop );
			cudaEventElapsedTime( &elapsedTime, searchStart, searchStop );

			totalTime += (double)elapsedTime;
		}
		g_app.cudaTimerCost = totalTime / 100.0;

		cudaEventDestroy( searchStop );
		cudaEventDestroy( searchStart );

		// Work around weird bug by setting cudaGetLastError back to cudaSuccess
		cudaEventDestroy( weirdBug );
		cudaResult = cudaGetLastError();

		if (g_app.dumpVerbose > 0)
		{
			printf( "Base Timer Cost = %f\n", g_app.baseTimerCost );
			printf( "Cuda Timer Cost = %f\n", g_app.cudaTimerCost );
		}
	}
}

void FiniTimer()
{
	// Cleanup timer
	if (g_app.hTimer != 0)
	{
		cutDeleteTimer( g_app.hTimer );
		g_app.hTimer = 0;
	}

	if (g_app.start != 0)
	{
		cudaEventDestroy( g_app.start );
		g_app.start = 0;
	}

	if (g_app.stop != 0)
	{
		cudaEventDestroy( g_app.stop );
		g_app.stop = 0;
	}
}


/*---------------------------------------------------------
  Name:	InitCuda
---------------------------------------------------------*/

void InitCuda
(
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	// CUDA Properties
	params.rawProps		= &(g_app.rawProps);
	params.cudaProps	= &(g_app.cudaProps);
}


/*---------------------------------------------------------
  Name:	InitNN
---------------------------------------------------------*/

bool InitNN
(
	HostDeviceParams & params,	// IN/OUT - input, output parameters
	unsigned int nnType,		// IN - NN search type
	unsigned int nDims,			// IN - d-dimension of NN search
	unsigned int kVal			// IN - 'k' value for kNN and All-kNN searches
)
{
	if ((nnType < NN_QNN) || (nnType > NN_ALL_KNN))
	{
		// Error - unknown NN search type
		fprintf( stderr, "Unknown NN search type[%u]: only QNN, All-NN, kNN, and All-kNN supported - in file '%s' at line %i.\n",
				 nnType, __FILE__, __LINE__ );
		return false;
	}

	if ((nDims < 2) || (nDims > 4))
	{
		// Error - unsupported dimensions
		fprintf( stderr, "Invalid dims[%u]: only 2d,3d, and 4d supported - in file '%s' at line %i.\n",
				 nDims, __FILE__, __LINE__ );
		return false;
	}

	params.nnType = nnType;
	params.nDims  = nDims;

	if ((nnType == NN_KNN) || (nnType == NN_ALL_KNN))
	{
		// Clamp k-Value to reasonable range [1, KD_KNN_SIZE = 32]
		if (kVal < 1)				{ kVal = 1; }
		if (kVal >= KD_KNN_SIZE)	{ kVal = KD_KNN_SIZE - 1; }

		params.kVal = kVal;
	}

	return true;
}


/*---------------------------------------------------------
  Name:	InitSearch
---------------------------------------------------------*/

bool InitSearch
(
	HostDeviceParams & params,	// IN/OUT - input, output parameters
	unsigned int nSearch,		// IN - count of search points
	const float4 * searchList	// IN - points in search list
)
{
	// Check Parameters
	if ((nSearch <= 0) || (searchList == NULL))
	{
		fprintf( stderr, "Invalid Parameters: in file '%s' at line %i.\n",
				 __FILE__, __LINE__ );
		return false;
	}

	// Search Info
	params.nSearch		= nSearch;
	params.nPadSearch	= 0u;
	params.searchList	= searchList;

	return true;
}


/*---------------------------------------------------------
  Name:	InitQuery
---------------------------------------------------------*/

bool InitQuery
(
	HostDeviceParams & params,	// IN/OUT - input, output parameters
	unsigned int nQuery,		// IN - count of search points
	const float4 * queryList	// IN - points in search list
)
{
	// Check Parameters
	if ((params.nnType == NN_QNN) || 
		(params.nnType == NN_KNN))
	{
		// Make sure we have some queries.
		if ((nQuery <= 0) || (queryList == NULL))
		{
			fprintf( stderr, "Invalid Parameters: in file '%s' at line %i.\n",
				     __FILE__, __LINE__ );
			return false;
		}
	}

	// Query Info
	params.nQuery		= nQuery;
	params.nPadQuery	= 0u;
	params.queryList	= queryList;

	return true;
}


/*---------------------------------------------------------
  Name:	InitResult
---------------------------------------------------------*/

bool InitResult
(
	HostDeviceParams & params,	// IN/OUT - input, output parameters
	unsigned int nResult,		// IN  - count of results
	CPU_NN_Result * resultList	// OUT - result list
)
{
	// Check Parameters
	if ((nResult <= 0) || (resultList == NULL))
	{
		fprintf( stderr, "Invalid Parameters: in file '%s' at line %i.\n",
				 __FILE__, __LINE__ );
		return false;
	}

	bool bResult = true;

	// Make sure our result array is big enough to hold all the results
	unsigned int totalResults;
	switch (params.nnType)
	{
	case NN_QNN:
		if (nResult < params.nQuery)
		{
			fprintf( stderr, 
				     "QNN: result count [%u] less than nQuery[%u] - in file '%s' at line %i.\n",
					 nResult, params.nQuery, __FILE__, __LINE__ );
			return false;
		}
		break;

	case NN_ALL_NN:
		if (nResult < params.nSearch)
		{
			fprintf( stderr, 
				     "All-NN: result count [%u] less than nSearch[%u] - in file '%s' at line %i.\n",
					 nResult, params.nSearch, __FILE__, __LINE__ );
			return false;
		}
		break;

	case NN_KNN:
		totalResults = params.nQuery * params.kVal;
		if (nResult < totalResults)
		{
			fprintf( stderr, 
				     "kNN: result count(%u) is less than total(%u) = kVal(%u) * nQuery(%u) - in file '%s' at line %i.\n",
					 nResult, totalResults, params.kVal, params.nQuery, 
					 __FILE__, __LINE__ );
			return false;
		}
		break;

	case NN_ALL_KNN:
		totalResults = params.nSearch * params.kVal;
		if (nResult < totalResults)
		{
			fprintf( stderr, 
					 "All-kNN: result count(%u) is less than total(%u) = kVal(%u) * nSearch(%u) - in file '%s' at line %i.\n",
					 nResult, totalResults, params.kVal, params.nSearch, 
					 __FILE__, __LINE__ );
			return false;
		}
		break;

	default:
		fprintf( stderr, "Unknown NN search type(%u): in file '%s' at line %i.\n",
				 params.nnType, __FILE__, __LINE__ );
		bResult = false;
		break;
	}


	params.nOrigResult	= nResult;
	params.resultList	= resultList;
	params.nPadResult   = 0;

	return bResult;
}


/*---------------------------------------------------------
  Name:	InitBlockGrid
---------------------------------------------------------*/

bool InitBlockGrid
(
	HostDeviceParams & params,	// IN/OUT - input, output parameters
	unsigned int threadsPerRow,	// IN - threads per row
	unsigned int rowsPerBlock	// IN - rows per block
)
{
	bool bResult = true;

	// Check Parameters
	if (threadsPerRow <= 0u)
	{
		threadsPerRow = 1u;
	}
	if (threadsPerRow >= 512u)
	{
		threadsPerRow = 512u;
	}

	if (rowsPerBlock <= 0u)
	{
		rowsPerBlock = 1u;
	}
	if (rowsPerBlock >= 512u)
	{
		rowsPerBlock = 512u;
	}

	unsigned int threadsPerBlock = threadsPerRow * rowsPerBlock;
	if (threadsPerBlock > 512u)
	{
		fprintf( stderr, "<TPR = %u, RPB = %u, TPB = %u> Can't have more than 512 threads per block, problem in '%s' at line %i.\n",
				 threadsPerRow, rowsPerBlock, threadsPerBlock, __FILE__, __LINE__ );
		return false;
	}

	//-------------------------------------------
	// Request # of elements in thread block/grid
	//-------------------------------------------

	switch (params.nnType)
	{
	case NN_QNN:	
		params.bgShape.nElems = params.nQuery;
		break;

	case NN_ALL_NN:
		params.bgShape.nElems = params.nSearch + 1;
		break;

	case NN_KNN:	
		params.bgShape.nElems = params.nQuery;
		break;

	case NN_ALL_KNN:
		params.bgShape.nElems = params.nSearch + 1;
		break;

	default:
		// Error - unknown type
		break;
	}

	// Thread Block Shape
	params.bgShape.threadsPerRow  = threadsPerRow;
	params.bgShape.rowsPerBlock   = rowsPerBlock;

	// Compute Reasonable thread block/grid shape from request
	bResult = ComputeBlockShapeFromQueryVector( params.bgShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

	//----------------------------
	// Set Padded Sizes
	//	  from padded block shape
	//----------------------------

	// BUGBUG:  
	//      Interesting bug caused by remapping all extra padded threads to the 
	//		same query as the very first query.  These multiple threads collide 
	//		on this query position usually storing an incorrect result there.  
	//		This is most noticeable for small queries, n = 10 or 100.
	//
	// WORKAROUND:  
	//		- Increase memory allocation by one extra element
	//		- For the remapping vector have the extra threads point to themselves for remapping
	//			 to avoid conflicts with the 1st valid query element
	//      - For All-NN and All-kNN searches, 
	//			also have the unused 0th entry point in the remapping vector point to this extra
	//			 allocated element to avoid conflicts with the 1st valid query element.
	//		

	switch (params.nnType)
	{
	case NN_QNN:	
		params.nPadSearch  = params.nSearch + 1;
		params.nPadQuery   = params.bgShape.nPadded;
		params.nPadResult  = params.nPadQuery;
		break;

	case NN_ALL_NN:
		params.nPadSearch  = params.bgShape.nPadded;
		params.nQuery      = 0u;
		params.nPadQuery   = 0u;
		params.nPadResult  = params.nPadSearch + 1;		// +1 to work around bug
		break;

	case NN_KNN:	
		params.nPadSearch  = params.nSearch + 1;
		params.nPadQuery   = params.bgShape.nPadded;
		params.nPadResult  = params.nPadQuery * params.kVal;
		break;

	case NN_ALL_KNN:
		params.nPadSearch  = params.bgShape.nPadded;
		params.nQuery      = 0u;
		params.nPadQuery   = 0u;
		params.nPadResult  = (params.nPadSearch + 1) * params.kVal;	// +1 to work around bug
		break;

	default:
		// Error - unknown type
		break;
	}

	// Setup Kernel Thread block & grid parameters
	dim3 tmpBlock( params.bgShape.threadsPerRow, params.bgShape.rowsPerBlock, 1 );
	dim3 tmpGrid( params.bgShape.blocksPerRow, params.bgShape.rowsPerGrid, 1 );

	params.nnBlock = tmpBlock;
	params.nnGrid  = tmpGrid;

	return true;
}


/*---------------------------------------------------------
  Name:	DumpInitParams
---------------------------------------------------------*/

void DumpInitParams
(
	const HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	// Print out Initialization Parameters
	if (g_app.dumpVerbose > 0)
	{
		DumpBlockGridShape( params.bgShape );

		if (g_app.dumpVerbose > 1)
		{
			// Dump NN Search Type
			switch (params.nnType)
			{
			case NN_QNN:		printf( "QNN " );		break;
			case NN_ALL_NN:		printf( "All-NN " );	break;
			case NN_KNN:		printf( "kNN " );		break;
			case NN_ALL_KNN:	printf( "All-kNN " );	break;
			default:			printf( "Unknown " );	break;
			}
			// Dump # of dimensions
			switch (params.nDims)
			{
			case 2:		printf( "2D " );		break;
			case 3:		printf( "3D " );		break;
			case 4:		printf( "4D " );		break;
			default:	printf( "Bad Dim " );	break;
			}
			printf( " Left-balanced kd-tree\n\n" );
		}

		printf( "# Requested Search Points  = %u\n", params.nSearch );
		printf( "# Padded Search Points     = %u\n", params.nPadSearch );
		printf( "# Requested Query Points   = %u\n", params.nQuery );
		printf( "# Padded Query Points      = %u\n", params.nPadQuery );
		printf( "# Results                  = %u\n", params.nOrigResult );
		printf( "# Padded Results           = %u\n", params.nPadResult );

		if ((params.nnType == NN_KNN) || (params.nnType == NN_ALL_KNN))
		{
			unsigned int kVal = params.kVal;
			printf( "k-value                    = %u\n", kVal );
		}
		if ((params.nnType == NN_QNN) || (params.nnType == NN_ALL_NN))
		{
			printf( "Width                      = %u\n", params.bgShape.W );
			printf( "Height                     = %u\n", params.bgShape.H );
		}
	}
}


/*---------------------------------------------------------
  Name:	ValidateAvailMemory
---------------------------------------------------------*/

bool ValidateAvailMemory
(
	const HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	// Make sure memory usage for NN search is not to big to use up all device memory
		// 1 GB on Display Card
	unsigned int sizeResults, sizeQueries, sizeNodes, sizeIDs, totalMem;

	sizeResults = params.nPadResult * sizeof(GPU_NN_Result);
	sizeQueries = params.nPadQuery * sizeof(float2);

	sizeQueries = 0u;
	switch (params.nDims)
	{
	case 2:	sizeQueries = params.nPadQuery * sizeof(float2); break;
	case 3:	sizeQueries = params.nPadQuery * sizeof(float4); break;
	case 4:	sizeQueries = params.nPadQuery * sizeof(float4); break;
	default:
		// Error
		break;
	}

	sizeNodes = 0u;
	switch (params.nDims)
	{
	case 2:	sizeNodes = params.nPadSearch * sizeof(GPUNode_2D_LBT); break;
	case 3:	sizeNodes = params.nPadSearch * sizeof(GPUNode_3D_LBT); break;
	case 4:	sizeNodes = params.nPadSearch * sizeof(GPUNode_4D_LBT); break;
	default:
		// Error
		break;
	}

	sizeIDs		= params.nPadSearch * sizeof(unsigned int);

	totalMem    = sizeNodes + sizeIDs + sizeQueries + sizeResults;

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= params.cudaProps->totalGlobalMem)
	{
		printf( "KD Tree Inputs (%d) are too large for available device memory (%d), running test will crash...\n",
				totalMem, params.cudaProps->totalGlobalMem );
		printf( "\tsizeNodes = %d\n", sizeNodes );
		printf( "\tsizeIDs   = %d\n", sizeIDs );
		printf( "\tsizeQueries = %d\n", sizeQueries );
		printf( "\tsizeResults = %d\n", sizeResults );
		return false;
	}

	if (g_app.dumpVerbose >= 1)
	{
		printf( "# Inputs  Memory    = %d\n", sizeNodes + sizeIDs + sizeQueries );
		printf( "# Outputs Memory    = %d\n", sizeResults );
		printf( "# Total Memory      = %d\n", totalMem );
	}

	// Make Sure total search points doesn't overflow stack space
	unsigned int stackHeight;

	// Get Stack Height according to NN type
	switch (params.nnType)
	{
	case NN_QNN:		stackHeight   = QNN_STACK_SIZE;		break;
	case NN_ALL_NN:		stackHeight   = ALL_NN_STACK_SIZE;  break;
	case NN_KNN:		stackHeight   = KNN_STACK_SIZE;		break;
	case NN_ALL_KNN:	stackHeight   = ALL_KNN_STACK_SIZE; break;
	default:
		// Error - unknown type
		stackHeight = 0;
		break;
	}

	unsigned int maxStackElems = 1u;
	unsigned int idx;
	for (idx = 0; idx < stackHeight; idx++)
	{
		maxStackElems *= 2u;
	}
	if (params.nPadSearch > maxStackElems)
	{
		printf( "Padded Search Points (%u) are too many for DFS stack[H=%u,S=%u], NN GPU Search will crash...\n",
				params.nPadSearch, stackHeight, maxStackElems );
		return false;
	}

	return true;
}


/*---------------------------------------------------------
  Name:	AllocMem
---------------------------------------------------------*/

bool AllocMem
(
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	cudaError_t cudaResult;

	//---------------------------------------------
	// Allocate memory for KD Tree Nodes
	//---------------------------------------------

	switch (params.nDims)
	{
	case 2:
		params.mem_size_Nodes = params.nPadSearch * sizeof(GPUNode_2D_LBT);
		break;
	case 3:
		params.mem_size_Nodes = params.nPadSearch * sizeof(GPUNode_3D_LBT);
		break;
	case 4:
		params.mem_size_Nodes = params.nPadSearch * sizeof(GPUNode_4D_LBT);
		break;
	default:
		// Error
		break;
	}

	params.h_Nodes = NULL;	
	params.d_Nodes = NULL;
	if ((false == params.buildGPU) && 
		(params.mem_size_Nodes > 0))
	{
		// Allocate Host Memory for GPU kd-tree nodes
		params.h_Nodes = AllocHostMemory( params.mem_size_Nodes, params.bPinned );
		if (NULL == params.h_Nodes)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
		
		// Allocate device memory for GPU KD Tree Nodes
		if (params.mem_size_Nodes > 0)
		{
			cutilSafeCall( cudaMalloc( (void **) &(params.d_Nodes), params.mem_size_Nodes ) );
		}
	}


	//---------------------------------------------
	// Allocate memory for mapping ID's
	//---------------------------------------------

	params.mem_size_IDs = params.nPadSearch * sizeof(unsigned int);
	params.h_IDs = NULL;
	params.d_IDs = NULL;

	if ((false == params.buildGPU) && 
		(params.mem_size_IDs > 0))
	{
		// Allocate host memory for GPU Node mapping ID's
		params.h_IDs = (unsigned int*) AllocHostMemory( params.mem_size_IDs, params.bPinned );
		if (NULL == params.h_IDs)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}

		// Allocate device memory for GPU Node mapping ID's
		if (params.mem_size_IDs > 0)
		{
			cutilSafeCall( cudaMalloc( (void **) &(params.d_IDs), params.mem_size_IDs ) );
		}
	}



	//---------------------------------------------
	// Allocate memory for query points
	//---------------------------------------------

	// Allocate host memory for GPU query points 
	switch (params.nDims)
	{
	case 2: 
		params.mem_size_Query = params.nPadQuery * sizeof(float2);
		break;
	case 3:
	case 4:
		params.mem_size_Query = params.nPadQuery * sizeof(float4);
		break;
	default:
		// Error
		break;
	}

	params.h_Query = NULL;
	if (params.mem_size_Query > 0)
	{
		params.h_Query = AllocHostMemory( params.mem_size_Query, params.bPinned );
		if (NULL == params.h_Query)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}

	// Allocate device memory for GPU query points 
	params.d_Query = NULL;
	if (params.mem_size_Query > 0)
	{
		cudaResult = cudaMalloc( (void **) &(params.d_Query), params.mem_size_Query );
		if (cudaSuccess != cudaResult)
		{
			fprintf( stderr, "cudaMalloc() Error: (%u)='%s' in file '%s' in line %i.\n", 
				     cudaResult, cudaGetErrorString( cudaResult ), __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}


	//---------------------------------------------
	// Allocate memory for Results
	//---------------------------------------------

	params.mem_size_Results_GPU = params.nPadResult  * sizeof(GPU_NN_Result);

	// Allocate host memory for GPU results
	/*
	if (params.mem_size_Results_GPU > 0)
	{
		params.h_Results_GPU = (GPU_NN_Result*) AllocHostMemory( params.mem_size_Results_GPU, params.bPinned );
		if (NULL == params.h_Results_GPU)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
	*/

	// Allocate device memory for GPU results
	params.d_Results_GPU = NULL;
	if (params.mem_size_Results_GPU > 0)
	{
		cudaResult = cudaMalloc( (void **) &(params.d_Results_GPU), params.mem_size_Results_GPU );
		if (cudaSuccess != cudaResult)
		{
			fprintf( stderr, "cudaMalloc() Error: (%u)='%s' in file '%s' in line %i.\n", 
				     cudaResult, cudaGetErrorString( cudaResult ), __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}

	// Allocate host memory for CPU Query Results
		// bugbug -- only allocate if we are checking GPU vs. CPU
	params.mem_size_Results_CPU = params.nOrigResult * sizeof(CPU_NN_Result);
	params.h_Results_CPU = NULL;
	if (params.mem_size_Results_CPU > 0)
	{
		params.h_Results_CPU = (CPU_NN_Result*) malloc( params.mem_size_Results_CPU );
	}

	return true;
}


/*---------------------------------------------------------
  Name:	Cleanup
---------------------------------------------------------*/

bool Cleanup
(
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{

	if (g_app.dumpVerbose > 1)
	{
		printf( "Cleaning up...\n" );
	}

	// Cleanup CUDA Timer
	FiniTimer();

	// Cleanup Query Memory
	FreeHostMemory( params.h_Query,       params.bPinned );
	cutilSafeCall( cudaFree( params.d_Query ) );

	// Cleanup Results Memory
	//FreeHostMemory( params.h_Results_GPU, params.bPinned );
	cutilSafeCall( cudaFree( params.d_Results_GPU ) );
	if (NULL != params.h_Results_CPU)
	{
		free( params.h_Results_CPU );
	}

	// Cleanup GPU kd-tree
	FiniHostBuild( params );

	// Cleanup CPU kd-tree
	switch (params.nDims)
	{
	case 2:
		FINI_CPU_2D_LBT( &(params.kdtree) );
		break;
	case 3:
		FINI_CPU_3D_LBT( &(params.kdtree) );
		break;
	case 4:
		FINI_CPU_4D_LBT( &(params.kdtree) );
		break;
	default:
		// Error
		break;
	}

	if (g_app.dumpVerbose > 1)
	{
		printf( "Cleanup complete...\n\n" );
	}

	return true;
}


/*---------------------------------------------------------
  Name:	CopyQuery
---------------------------------------------------------*/

bool CopyQuery
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	// Copy Query List
	if ((NULL != params.h_Query) && (NULL != params.queryList))
	{
		unsigned int qryIdx;

		switch (params.nDims)
		{
		case 2:	// 2D queries
		{
			const float4 * origList = params.queryList;
			float2 * hostList = (float2 *)(params.h_Query);

			// Copy actual 2D queries 
			for (qryIdx = 0; qryIdx < params.nQuery; qryIdx++)
			{
				hostList[qryIdx].x = origList[qryIdx].x;
				hostList[qryIdx].y = origList[qryIdx].y;
			}

			// Create some extra queries for thread block alignment
			float currX, currY;
			currX = origList[0].x;
			currY = origList[0].y;
			for (qryIdx = params.nQuery; qryIdx < params.nPadQuery; qryIdx++)
			{
				// Just repeat the first query a few times
				hostList[qryIdx].x = currX;
				hostList[qryIdx].y = currY;
			}
		}
			break;

		case 3:	// 3D queries
		{
			float4 * origList = (float4 *)(params.queryList);
			float4 * hostList = (float4 *)(params.h_Query);

			// Copy actual 3D queries 
			for (qryIdx = 0; qryIdx < params.nQuery; qryIdx++)
			{
				hostList[qryIdx].x = origList[qryIdx].x;
				hostList[qryIdx].y = origList[qryIdx].y;
				hostList[qryIdx].z = origList[qryIdx].z;
			}

			// Create some extra queries for thread block alignment
			float currX, currY, currZ;
			currX = origList[0].x;
			currY = origList[0].y;
			currZ = origList[0].z;
			for (qryIdx = params.nQuery; qryIdx < params.nPadQuery; qryIdx++)
			{
				// Just repeat the first query a few times
				hostList[qryIdx].x = currX;
				hostList[qryIdx].y = currY;
				hostList[qryIdx].z = currZ;
			}
		}
			break;

		case 4:	// 4D queries
		{
			float4 * origList = (float4 *)(params.queryList);
			float4 * hostList = (float4 *)(params.h_Query);

			// Copy actual 4D queries 
			for (qryIdx = 0; qryIdx < params.nQuery; qryIdx++)
			{
				hostList[qryIdx].x = origList[qryIdx].x;
				hostList[qryIdx].y = origList[qryIdx].y;
				hostList[qryIdx].z = origList[qryIdx].z;
				hostList[qryIdx].w = origList[qryIdx].w;
			}

			// Create some extra queries for thread block alignment
			float currX, currY, currZ, currW;
			currX = origList[0].x;
			currY = origList[0].y;
			currZ = origList[0].z;
			currW = origList[0].w;
			for (qryIdx = params.nQuery; qryIdx < params.nPadQuery; qryIdx++)
			{
				// Just repeat the first query a few times
				hostList[qryIdx].x = currX;
				hostList[qryIdx].y = currY;
				hostList[qryIdx].z = currZ;
				hostList[qryIdx].w = currW;
			}
		}
			break;

		default:
			// Error
			break;
		}
	}

	return true;
}


/*---------------------------------------------------------
  Name:	DumpSearch
---------------------------------------------------------*/

void DumpSearch
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	unsigned int searchIdx;
	double currX, currY, currZ, currW;

	// Dump search List (for debugging)
	if ((g_app.dumpVerbose >= 3) &&
		(params.nSearch > 0) &&
		(params.searchList != NULL))
	{
		switch (params.nDims)
		{
		case 2:	// 2D
		{
			const float4 * searchList = params.searchList;
			for (searchIdx = 0; searchIdx < params.nSearch; searchIdx++)
			{
				currX = (double)(searchList[searchIdx].x);
				currY = (double)(searchList[searchIdx].y);

				printf( "Search[%u] = <%3.6f, %3.6f>\n", 
						searchIdx, currX, currY );
			}
		}
			break;

		case 3:	// 3D
		{
			float4 * searchList = (float4 *)params.searchList;
			for (searchIdx = 0; searchIdx < params.nSearch; searchIdx++)
			{
				currX = (double)(searchList[searchIdx].x);
				currY = (double)(searchList[searchIdx].y);
				currZ = (double)(searchList[searchIdx].z);

				printf( "Search[%u] = <%3.6f, %3.6f, %3.6f>\n", 
					    searchIdx, currX, currY, currZ );
			}
		}
			break;

		case 4:	// 4D
		{
			float4 * searchList = (float4 *)params.searchList;
			for (searchIdx = 0; searchIdx < params.nSearch; searchIdx++)
			{
				currX = (double)(searchList[searchIdx].x);
				currY = (double)(searchList[searchIdx].y);
				currZ = (double)(searchList[searchIdx].z);
				currW = (double)(searchList[searchIdx].w);

				printf( "Search[%u] = <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
						searchIdx, currX, currY, currZ, currW );
			}
		}
			break;

		default:
			// Error
			break;
		} // end switch
	}
}


/*---------------------------------------------------------
  Name:	DumpQuery
---------------------------------------------------------*/

void DumpQuery
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	unsigned int queryIdx;
	double currX, currY, currZ, currW;

	// Dump query List (for debugging)
	if ((g_app.dumpVerbose >= 3) &&
		(params.nQuery > 0) &&
		(params.queryList != NULL))
	{
		switch (params.nDims)
		{
		case 2:	// 2D
		{
			const float4 * queryList = params.queryList;
			for (queryIdx = 0; queryIdx < params.nQuery; queryIdx++)
			{
				currX = (double)(queryList[queryIdx].x);
				currY = (double)(queryList[queryIdx].y);

				printf( "Query[%u] = <%3.6f, %3.6f>\n", 
						queryIdx, currX, currY );
			}
		}
			break;

		case 3:	// 3D
		{
			const float4 * queryList = params.queryList;
			for (queryIdx = 0; queryIdx < params.nQuery; queryIdx++)
			{
				currX = (double)(queryList[queryIdx].x);
				currY = (double)(queryList[queryIdx].y);
				currZ = (double)(queryList[queryIdx].z);

				printf( "Query[%u] = <%3.6f, %3.6f, %3.6f>\n", 
						queryIdx, currX, currY, currZ );
			}
		}
			break;

		case 4:	// 4D
		{
			const float4 * queryList = params.queryList;
			for (queryIdx = 0; queryIdx < params.nQuery; queryIdx++)
			{
				currX = (double)(queryList[queryIdx].x);
				currY = (double)(queryList[queryIdx].y);
				currZ = (double)(queryList[queryIdx].z);
				currW = (double)(queryList[queryIdx].w);

				printf( "Query[%u] = <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
						queryIdx, currX, currY, currZ, currW );
			}
		}
			break;

		default:
			// Error
			break;
		} // end switch
	}
}


/*---------------------------------------------------------
  Name:	DumpGPUNodes
---------------------------------------------------------*/

void DumpGPUNodes
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	unsigned int nodeIdx, searchIdx;
	unsigned int parent, left, right;
	double currX, currY, currZ, currW;

	// Dump query List (for debugging)
	if ((g_app.dumpVerbose >= 3) &&
		(params.nPadSearch > 0) &&
		(params.h_Nodes != NULL) &&
		(params.h_IDs != NULL))
	{
		switch (params.nDims)
		{
		case 2:	// 2D Nodes
		{
			GPUNode_2D_LBT * nodeList = (GPUNode_2D_LBT *)params.h_Nodes;
			unsigned int * idList = params.h_IDs;
			for (nodeIdx = 0; nodeIdx < params.nPadSearch; nodeIdx++)
			{
				currX     = (double)(nodeList[nodeIdx].pos[0]);
				currY     = (double)(nodeList[nodeIdx].pos[1]);
				searchIdx = idList[nodeIdx];
				
				parent = ((nodeIdx == 0u) ? 0xFFFFFFFFu : (nodeIdx >> 1));
				left   = ((nodeIdx > params.nSearch) ? 0xFFFFFFFFu : (nodeIdx << 1));
				right  = ((nodeIdx > params.nSearch) ? 0xFFFFFFFFu : ((nodeIdx << 1)+1));

				if (nodeIdx == 0)
				{
					// zeroth node
					printf( "NodeID[%u] (Zero) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY );
				}
				else if (nodeIdx == 1)
				{
					// Root Node
					printf( "NodeID[%u] (Root) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY );
				}
				else if (nodeIdx > params.nSearch)
				{
					// Extra padded node for searching 
					printf( "NodeID[%u] (PAD) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY );
				}
				else if (searchIdx > params.nSearch)
				{
					// Possible error
					printf( "NodeID[%u] (ERROR) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY );
				}
				else if ((left > params.nSearch) && (right > params.nSearch))
				{
					// Leaf Node
					printf( "NodeID[%u] (Leaf) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY );
				}
				else // Internal Node
				{
					printf( "NodeID[%u] (Internal) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY );
				}
			}
		}
			break;

		case 3:	// 3D Nodes
		{
			GPUNode_3D_LBT * nodeList = (GPUNode_3D_LBT *)params.h_Nodes;
			unsigned int * idList = params.h_IDs;
			for (nodeIdx = 0; nodeIdx < params.nPadSearch; nodeIdx++)
			{
				currX    = (double)(nodeList[nodeIdx].pos[0]);
				currY    = (double)(nodeList[nodeIdx].pos[1]);
				currZ    = (double)(nodeList[nodeIdx].pos[2]);
				searchIdx = idList[nodeIdx];
				
				parent = ((nodeIdx == 0u) ? 0xFFFFFFFFu : (nodeIdx >> 1));
				left   = ((nodeIdx > params.nSearch) ? 0xFFFFFFFFu : (nodeIdx << 1));
				right  = ((nodeIdx > params.nSearch) ? 0xFFFFFFFFu : ((nodeIdx << 1)+1));

				if (nodeIdx == 0)
				{
					// zeroth node
					printf( "NodeID[%u] (Zero) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ );
				}
				else if (nodeIdx == 1)
				{
					// Root Node
					printf( "NodeID[%u] (Root) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ );
				}
				else if (nodeIdx > params.nSearch)
				{
					// Extra padded node for searching 
					printf( "NodeID[%u] (PAD) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ );
				}
				else if (searchIdx > params.nSearch)
				{
					// Possible error
					printf( "NodeID[%u] (ERROR) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ );
				}
				else if ((left > params.nSearch) && (right > params.nSearch))
				{
					// Leaf Node
					printf( "NodeID[%u] (Leaf) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ );
				}
				else // Internal Node
				{
					printf( "NodeID[%u] (Internal) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ );
				}
			}
		}
			break;

		case 4:	// 4D Nodes
		{
			GPUNode_4D_LBT * nodeList = (GPUNode_4D_LBT *)params.h_Nodes;
			unsigned int * idList = params.h_IDs;
			for (nodeIdx = 0; nodeIdx < params.nPadSearch; nodeIdx++)
			{
				currX     = (double)(nodeList[nodeIdx].pos[0]);
				currY     = (double)(nodeList[nodeIdx].pos[1]);
				currZ     = (double)(nodeList[nodeIdx].pos[2]);
				currW     = (double)(nodeList[nodeIdx].pos[3]);
				searchIdx = idList[nodeIdx];
				
				parent = ((nodeIdx == 0u) ? 0xFFFFFFFFu : (nodeIdx >> 1));
				left   = ((nodeIdx > params.nSearch) ? 0xFFFFFFFFu : (nodeIdx << 1));
				right  = ((nodeIdx > params.nSearch) ? 0xFFFFFFFFu : ((nodeIdx << 1)+1));

				if (nodeIdx == 0)
				{
					// zeroth node
					printf( "NodeID[%u] (Zero) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ, currW );
				}
				else if (nodeIdx == 1)
				{
					// Root Node
					printf( "NodeID[%u] (Root) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ, currW );
				}
				else if (nodeIdx > params.nSearch)
				{
					// Extra padded node for searching 
					printf( "NodeID[%u] (PAD) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ, currW );
				}
				else if (searchIdx > params.nSearch)
				{
					// Possible error
					printf( "NodeID[%u] (ERROR) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ, currW );
				}
				else if ((left > params.nSearch) && (right > params.nSearch))
				{
					// Leaf Node
					printf( "NodeID[%u] (Leaf) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ, currW );
				}
				else // Internal Node
				{
					printf( "NodeID[%u] (Internal) = <S=%u, P=%u, L=%u, R=%u> <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
							nodeIdx, searchIdx, 
							parent, left, right,
							currX, currY, currZ, currW );
				}
			}
		}
			break;

		default:
			// Error
			break;
		} // end switch
	}
}

void DumpMapping
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	unsigned int nodeIdx, searchIdx;

	// Dump Mapping Array (for debugging)
	if ((g_app.dumpVerbose >= 3) &&
		(params.nPadSearch > 0) &&
		(params.h_IDs != NULL))
	{
		unsigned int * idList = params.h_IDs;
		for (nodeIdx = 0; nodeIdx < params.nPadSearch; nodeIdx++)
		{
			searchIdx = idList[nodeIdx];
			printf( "Map[%u] = %u\n", nodeIdx, searchIdx ); 
		}
	}
}


void DumpCPUNodes
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	if ((g_app.dumpVerbose >= 3) &&
		(params.kdtree != NULL))
	{
		switch (params.nDims)
		{
		case 2:
		{
			CPUTree_2D_LBT * currTree = (CPUTree_2D_LBT *)params.kdtree;
			if (NULL != currTree)
			{
				currTree->DumpNodes();
			}
		}
			break;

		case 3:
		{
			CPUTree_3D_LBT * currTree = (CPUTree_3D_LBT *)params.kdtree;
			if (NULL != currTree)
			{
				currTree->DumpNodes();
			}
		}
			break;

		case 4:
		{
			CPUTree_4D_LBT * currTree = (CPUTree_4D_LBT *)params.kdtree;
			if (NULL != currTree)
			{
				currTree->DumpNodes();
			}
		}
			break;

		default:
			// Error
			break;
		}
	}
}


/*---------------------------------------------------------
  Name:	Dump GPU Results
---------------------------------------------------------*/

void DumpResultsGPU
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	if (g_app.dumpVerbose >= 3)
	{
		unsigned int nResultsGPU, rIdx, gpuID;
		double gpuDist;

		switch (params.nnType)
		{
		case NN_QNN:
			nResultsGPU = params.nQuery;
			break;

		case NN_ALL_NN:
			nResultsGPU = params.nSearch;
			break;

		case NN_KNN:
			nResultsGPU = params.nQuery * params.kVal;
			break;

		case NN_ALL_KNN:
			nResultsGPU = params.nSearch * params.kVal;
			break;

		}

		// Dump GPU results
		const GPU_NN_Result * gpuResults = (const GPU_NN_Result *)(params.resultList);
		if (gpuResults != NULL)
		{
			for (rIdx = 0; rIdx < nResultsGPU; rIdx++)
			{
				gpuID   = gpuResults[rIdx].Id;
				gpuDist = gpuResults[rIdx].Dist;

				printf( "GPU Result[%u] = <GPU ID=%u, Dist=%3.6f>\n", 
						rIdx, gpuID, gpuDist );
			}
		}
	}
}


/*---------------------------------------------------------
  Name:	Dump CPU Results
---------------------------------------------------------*/

void DumpResultsCPU
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	if (g_app.dumpVerbose >= 3)
	{
		unsigned int rIdx, cpuID, nResultsCPU;
		double cpuDist;

		switch (params.nnType)
		{
		case NN_QNN:
			nResultsCPU = params.nQuery;
			break;

		case NN_ALL_NN:
			nResultsCPU = params.nQuery;
			break;

		case NN_KNN:
			nResultsCPU = params.nQuery * params.kVal;
			break;

		case NN_ALL_KNN:
			nResultsCPU = params.nSearch * params.kVal;
			break;

		}

		const CPU_NN_Result * cpuResults = (const CPU_NN_Result *)(params.h_Results_CPU);

		if (cpuResults != NULL)
		{
			// Dump CPU results
			for (rIdx = 0; rIdx < nResultsCPU; rIdx++)
			{
				cpuID   = cpuResults[rIdx].Id;
				cpuDist = cpuResults[rIdx].Dist;

				printf( "CPU Result[%u] = <CPU ID=%u, Dist=%3.6f>\n", 
						rIdx, cpuID, cpuDist );
			}
		}
	}
}


/*---------------------------------------------------------
  Name:	BUILD_KD_TREE
  Desc: Builds kd-Tree on GPU
---------------------------------------------------------*/

bool BuildKDTree_GPU
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	bool bResult = true;

	float KD_GPU_build        = 0.0f;

	/*-------------------------------------------
	  Build left-balanced KDTree (on CPU)
    -------------------------------------------*/

	if (g_app.profile)
	{
		// Start Timer
		cutResetTimer( g_app.hTimer );
		cutStartTimer( g_app.hTimer );
	}

	switch (params.nDims)
	{
	case 2:
		// Build 2D left-balanced KDTree (on CPU)
		bResult = Host_Build2D( params );
		break;

	case 3:
		// Build 3D left-balanced KDTree (on CPU)
		//bResult = Host_Build3D( params );
		break;

	case 4:
		// Build 4D left-balanced KDTree (on CPU)
		//bResult = Host_Build4D( params );
		break;

	default:
		// Error
		bResult = false;
		break;
	}

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g_app.hTimer );
		KD_GPU_build += cutGetTimerValue( g_app.hTimer );
	}

	// Dump Build Time
	if (g_app.dumpVerbose > 0)
	{
		double avgBuild = (double)KD_GPU_build - g_app.baseTimerCost;
		if (avgBuild <= 0.0) { avgBuild = (double)KD_GPU_build; }
		printf( "Build kd-tree on GPU,            time: %f msecs.\n", avgBuild );
	}

	return bResult;
}

bool BuildKDTree_CPU
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	bool bResult = true;

	float KD_CPU_build        = 0.0f;

	/*-------------------------------------------
	  Build left-balanced KDTree (on CPU)
    -------------------------------------------*/

	if (g_app.profile)
	{
		// Start Timer
		cutResetTimer( g_app.hTimer );
		cutStartTimer( g_app.hTimer );
	}

	switch (params.nDims)
	{
	case 2:
		// Build 2D left-balanced KDTree (on CPU)
		bResult = BUILD_CPU_2D_LBT( &params.kdtree, params.nSearch, params.searchList );
		break;

	case 3:
		// Build 3D left-balanced KDTree (on CPU)
		bResult = BUILD_CPU_3D_LBT( &params.kdtree, params.nSearch, params.searchList );
		break;

	case 4:
		// Build 4D left-balanced KDTree (on CPU)
		bResult = BUILD_CPU_4D_LBT( &params.kdtree, params.nSearch, params.searchList );
		break;

	default:
		// Error
		bResult = false;
		break;
	}

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g_app.hTimer );
		KD_CPU_build += cutGetTimerValue( g_app.hTimer );
	}

	// Dump Build Time
	if (g_app.dumpVerbose > 0)
	{
		double avgBuild = (double)KD_CPU_build - g_app.baseTimerCost;
		printf( "Build kd-tree on CPU,            time: %f msecs.\n", avgBuild );
	}

	return bResult;
}


/*---------------------------------------------------------
  Name:	CopyNodes
  Desc: Copy kd-nodes from CPU kd-tree onto GPU
---------------------------------------------------------*/

bool CopyNodes
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	bool bResult = true;

	float KD_copy_nodes	        = 0.0f;
	float GPU_nodes_onto_device = 0.0f;
	float GPU_ids_onto_device   = 0.0f;

	if (g_app.profile)
	{
		// Start Timer
		cutResetTimer( g_app.hTimer );
		cutStartTimer( g_app.hTimer );
	}


	/*-------------------------------------------
	  Copy kd-nodes from CPU to GPU
    -------------------------------------------*/

	switch (params.nDims)
	{
	case 2:
		// Copy 2D kd-tree nodes from CPU to GPU
		bResult = COPY_NODES_2D_LBT( params.kdtree, params.nSearch, params.nPadSearch, 
									 params.h_Nodes, params.h_IDs );
		break;
	case 3:
		// Copy 3D kd-tree nodes from CPU to GPU
		bResult = COPY_NODES_3D_LBT( params.kdtree, params.nSearch, params.nPadSearch, 
									 params.h_Nodes, params.h_IDs );
		break;
	case 4:
		// Copy 4D kd-tree nodes from CPU to GPU
		bResult = COPY_NODES_4D_LBT( params.kdtree, params.nSearch, params.nPadSearch, 
									 params.h_Nodes, params.h_IDs );
		break;
	default:
		// Error
		bResult = false;
		break;
	}

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g_app.hTimer );
		KD_copy_nodes += cutGetTimerValue( g_app.hTimer );
	}


	float elapsedTime;
	cudaEvent_t nodesOntoStart, nodesOntoStop;
	cudaEvent_t idsOntoStart, idsOntoStop;

	if (g_app.profile)
	{
		// Create Timer Events
		cudaEventCreate( &nodesOntoStart );
		cudaEventCreate( &nodesOntoStop );

		// Start Timer
		cudaEventRecord( nodesOntoStart, 0 );
	}

	// Copy 'kd-nodes' vector from host memory to device memory
	if ((false == params.buildGPU) &&
		(NULL != params.d_Nodes) && 
		(NULL != params.h_Nodes))
	{
		cutilSafeCall( cudaMemcpy( params.d_Nodes, params.h_Nodes, 
								   params.mem_size_Nodes, cudaMemcpyHostToDevice ) );
	}

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( nodesOntoStop, 0 );
		cudaEventSynchronize( nodesOntoStop );
		cudaEventElapsedTime( &elapsedTime, nodesOntoStart, nodesOntoStop );

		GPU_nodes_onto_device += elapsedTime;
	}

	if (g_app.profile)
	{
		// Create Timer Events
		cudaEventCreate( &idsOntoStart );
		cudaEventCreate( &idsOntoStop );

		// Start Timer
		cudaEventRecord( idsOntoStart, 0 );
	}

	// Copy 'search ids' mapping vector from host memory to device memory
	if ((false == params.buildGPU) && 
		(NULL != params.d_IDs) && 
		(NULL != params.h_IDs))
	{
		cutilSafeCall( cudaMemcpy( params.d_IDs, params.h_IDs, 
			                       params.mem_size_IDs, cudaMemcpyHostToDevice ) );
	}

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( idsOntoStop, 0 );
		cudaEventSynchronize( idsOntoStop );
		cudaEventElapsedTime( &elapsedTime, idsOntoStart, idsOntoStop );

		GPU_ids_onto_device += elapsedTime;
	}

	// Dump Times
	if (g_app.dumpVerbose > 0)
	{
		double avgCopy  = (double)KD_copy_nodes  - g_app.baseTimerCost;
		double avgNodes = (double)GPU_nodes_onto_device - g_app.cudaTimerCost;
		double avgIDs   = (double)GPU_ids_onto_device - g_app.cudaTimerCost;

		if (avgCopy  <= 0) { avgCopy  = (double)KD_copy_nodes; }
		if (avgNodes <= 0) { avgNodes = (double)GPU_nodes_onto_device; }
		if (avgIDs   <= 0) { avgIDs   = (double)GPU_ids_onto_device; }

		printf( "Map kd-nodes from CPU to GPU,      time: %f msecs.\n", avgCopy );
		printf( "Copy Host nodes onto GPU device,   time: %f msecs.\n", avgNodes );
		printf( "Copy Host ids onto GPU device,     time: %f msecs.\n", avgIDs );
	}

	// Success
	return bResult;
}


/*---------------------------------------------------------
  Name:	CopyInputsOntoGPU
  Desc: copy search & query vectors onto GPU device
---------------------------------------------------------*/

void CopyInputsOntoGPU
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	unsigned int currIter;
	unsigned int nProfileLoops = g_app.profileActualLoops;
	if (nProfileLoops <= 0)
	{
		nProfileLoops = 1;
	}
	if (nProfileLoops >= 102)
	{
		nProfileLoops = 102;
	}

	float KD_GPU_query_onto = 0.0f;

	float elapsedTime;
	cudaEvent_t queryOntoStart, queryOntoStop;

	if (g_app.profile)
	{
		// Create Timer Events
		cudaEventCreate( &queryOntoStart );
		cudaEventCreate( &queryOntoStop );
	}

// Profile Measurement Loop
for (currIter = 0; currIter < nProfileLoops; currIter++)
{
	/*---------------------------------------------------------------
	  Move Input Vectors from CPU main memory to GPU device memory
	---------------------------------------------------------------*/

	if (g_app.profile)
	{
		// Start Timer
		cudaEventRecord( queryOntoStart, 0 );
	}

	// Copy 'query' vector from host memory to device memory
	if ((NULL != params.d_Query) && (NULL != params.h_Query))
	{
		cutilSafeCall( cudaMemcpy( params.d_Query, params.h_Query, 
									params.mem_size_Query, cudaMemcpyHostToDevice ) );
	}

	if (g_app.profile)
	{
		// Stop Query Timer and save performance measurement
		cudaEventRecord( queryOntoStop, 0 );
		cudaEventSynchronize( queryOntoStop );
		cudaEventElapsedTime( &elapsedTime, queryOntoStart, queryOntoStop );

		if (g_app.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
			{
				KD_GPU_query_onto += elapsedTime;
			}
		}
		else
		{
			KD_GPU_query_onto += elapsedTime;
		}
	}

} // End Profile Loop

	// Dump Performance Timings
	if (g_app.dumpVerbose > 0)
	{
		if (g_app.profileActualLoops > 1)
		{
			float loops = (float)g_app.profileActualLoops;
			float o_l = 1.0f / loops;

			double avgQuery  = (double)(KD_GPU_query_onto * o_l)  - g_app.cudaTimerCost;
			if (avgQuery <= 0.0) { avgQuery = (double)(KD_GPU_query_onto * o_l); }
			printf( "Copy 'Query' vector onto GPU,    time: %f msecs.\n", avgQuery );
		}
		else
		{
			double ontoQuery  = (double)KD_GPU_query_onto - g_app.cudaTimerCost;
			if (ontoQuery <= 0.0) { ontoQuery = (double)KD_GPU_query_onto; }
			printf( "Copy 'Query' vector onto GPU,    time: %f msecs.\n", ontoQuery );
		}
	}

	// Success
}


/*---------------------------------------------------------
  Name:	CopyResultsFromGPU
  Desc: copy NN results vector from GPU device
---------------------------------------------------------*/

bool CopyResultsFromGPU
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	if ((0 == params.nOrigResult) || 
		(NULL == params.d_Results_GPU) || 
		(NULL == params.resultList))
	{
		// Error - invalid parameter
		return false;
	}

	cudaError_t cudaResult = cudaSuccess;

	unsigned int currIter;
	unsigned int nProfileLoops = g_app.profileActualLoops;
	if (nProfileLoops <= 0)
	{
		nProfileLoops = 1;
	}
	if (nProfileLoops >= 102)
	{
		nProfileLoops = 102;
	}

	float KD_GPU_results_from = 0.0f;

	float elapsedTime;
	cudaEvent_t resultsFromStart, resultsFromStop;

	if (g_app.profile)
	{
		// Create Timer Events
		cudaEventCreate( &resultsFromStart );
		cudaEventCreate( &resultsFromStop );
	}

	CPU_NN_Result * rowHost     = NULL;
	GPU_NN_Result * rowDevice   = NULL;
	unsigned int bytesToCopy     = 0u;
	unsigned int rowBytesToCopy  = 0u;
	unsigned int currRow;

// Profile Measurement Loop
for (currIter = 0; currIter < nProfileLoops; currIter++)
{
	/*---------------------------------------------------------------
	  Move Output Vectors from GPU device memory to CPU main memory
	---------------------------------------------------------------*/

	if (g_app.profile)
	{
		// Start Timer
		cudaEventRecord( resultsFromStart, 0 );
	}

	switch (params.nnType)
	{
		// Copy 'search ids' mapping vector from host memory to device memory
		// Copy result vector from GPU device to host CPU
	case NN_QNN:
		bytesToCopy = params.nQuery * sizeof(GPU_NN_Result);
		cutilSafeCall( cudaMemcpy( params.resultList, params.d_Results_GPU, 
			                       bytesToCopy, cudaMemcpyDeviceToHost ) );
		break;

	case NN_ALL_NN:
		bytesToCopy = params.nSearch * sizeof(GPU_NN_Result);
		cutilSafeCall( cudaMemcpy( params.resultList, params.d_Results_GPU, 
			                       bytesToCopy, cudaMemcpyDeviceToHost ) );
		break;

	case NN_KNN:
		// Copy Results from GPU to CPU row by row
		rowHost   = (CPU_NN_Result *)(params.resultList);
		rowDevice = (GPU_NN_Result *)(params.d_Results_GPU);
		rowBytesToCopy = params.nQuery * sizeof(CPU_NN_Result);
		for (currRow = 0; currRow < params.kVal; currRow++)
		{
			// Read current row of k-values from Host Results onto Actual Results
			//memcpy( (void *)rowResult, (void *)rowGPU, rowBytesToCopy );
			cudaResult = cudaMemcpy( rowHost, rowDevice,
				                     rowBytesToCopy, cudaMemcpyDeviceToHost );
			if( cudaSuccess != cudaResult) 
			{
				// Error - failed to copy results from GPU to CPU
				fprintf( stderr, "KNN ERROR: cudaMemcpy from GPU to CPU failed with error(%d = %s) in file '%s' at line %i.\n",
						 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
				exit( 1 );
			}

			// Move to next row of k-values
			rowHost   += params.nQuery;
			rowDevice += params.nPadQuery;
		}
		break;

	case NN_ALL_KNN:
		// Have to read results one row at a time to skip wasted first entry
		rowHost   = (CPU_NN_Result *)(params.resultList);
		rowDevice = (GPU_NN_Result *)(params.d_Results_GPU);
		//rowDevice = &(rowDevice[1]);
		rowBytesToCopy = params.nSearch * sizeof(GPU_NN_Result);
		for (currRow = 0; currRow < params.kVal; currRow++)
		{
			// Read current row of k-values from results on GPU device
			cudaResult = cudaMemcpy( rowHost, rowDevice, 
			                          rowBytesToCopy, cudaMemcpyDeviceToHost );
		    if( cudaSuccess != cudaResult) 
			{
				// Error - failed to copy results from GPU to CPU
				fprintf( stderr, "ALL_KNN ERROR: cudaMemcpy from GPU to CPU failed with error(%d = %s) in file '%s' at line %i.\n",
						 cudaResult, cudaGetErrorString(cudaResult), __FILE__, __LINE__  );
				fprintf( stderr, "currRow = %d, maxRow = %d\n", currRow, params.kVal );
				fprintf( stderr, "rowHost = 0x%016X, rowDevice = 0x%016X, bytesToCopy = %u\n",
					     rowHost, rowDevice, rowBytesToCopy );
				fprintf( stderr, "nQuery = %u, nPadQuery = %u, kVal = %u, sizeof(GPU_NN_Result) = %u\n",
					     params.nQuery, params.nPadQuery, params.kVal, sizeof(GPU_NN_Result) );

				// Is this the best solution ?!?
				exit( 1 );
			}

			// Move to next row of k-values
			rowHost   += params.nSearch;
			rowDevice += params.nPadSearch;
		}
		break;

	default:
		// Error
		break;
	} // end switch (nnType)

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( resultsFromStop, 0 );
		cudaEventSynchronize( resultsFromStop );
		cudaEventElapsedTime( &elapsedTime, resultsFromStart, resultsFromStop );

		if (g_app.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
			{
				KD_GPU_results_from += elapsedTime;
			}
		}
		else
		{
			KD_GPU_results_from += elapsedTime;
		}
	}

} // End Profile Loop

	// Dump Performance Timings
	if (g_app.dumpVerbose > 0)
	{
		if (g_app.profileActualLoops > 1)
		{
			float loops = (float)g_app.profileActualLoops;
			float o_l = 1.0f / loops;

			double avgResults = (double)(KD_GPU_results_from * o_l) - g_app.cudaTimerCost;

			printf( "Copy 'Results' vector from GPU,  time: %f msecs.\n", avgResults );
		}
		else
		{
			double fromResults = (double)KD_GPU_results_from - g_app.cudaTimerCost;
			printf( "Copy 'Results' vector from GPU,  time: %f msecs.\n", fromResults );
		}
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CallGPUKernel
  Desc: Wrapper around GPU NN search kernel
  Notes:  
  
  1.) BUGBUG - TDR service
	  
	  Problem:  The Microsoft Windows Timeout Display and 
	  Recovery (TDR) service will kill and restart the 
	  display driver (including our GPU NN Kernels) after 
	  5 seconds of non-response.

	  This effectively means that we have at most 5 seconds
	  to complete our GPU call before Windows jumps in 
	  and causes a non preventable error.

	  WORK-A-ROUND:  If you experience this bug, try using 
	  the alternate CallGPUKernelRowByRow() function 
	  <see below> instead.  Which computes answers 
	  row by row instead of all by once to try and prevent 
	  this TDR 5 second bug from happening.
               
  2.) Assumes all neccessary inputs already copied onto GPU

Inputs:
  kd-nodes - search nodes representing left-balanced kd-tree
  ids      - maps node indices to search point indices
  queries  - query points 
  kVal     - 'k' value for kNN and All-kNN searches
  W		   - padded width of thread grid / thread block layout             

Outputs:
  NN_results - search point ID and distance for each query
               point.
---------------------------------------------------------*/

bool CallGPUKernel
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	float elapsedTime	= 0.0f;
	float KD_GPU_kernel = 0.0f;

	unsigned int currIter;
	unsigned int nProfileLoops = g_app.profileActualLoops;
	if (nProfileLoops <= 0)
	{
		nProfileLoops = 1;
	}
	if (nProfileLoops >= 102)
	{
		nProfileLoops = 102;
	}

	cudaEvent_t searchStart, searchStop;

	if (g_app.profile)
	{
		// Create Timer Events
		cudaEventCreate( &searchStart );
		cudaEventCreate( &searchStop );
	}


// Profile Measurement Loop
for (currIter = 0; currIter < nProfileLoops; currIter++)
{
	/*---------------------------------
	  Call GPU NN Search Kernel
	----------------------------------*/

	cudaError_t cuda_err;

	if (g_app.profile)
	{
		// Start Timer
		cudaEventRecord( searchStart, 0 );
	}
	
	// Check if GPU kernel execution generated an error
	switch (params.nnType)
	{
	case NN_QNN:	// Query Nearest Neighbor Search
		switch (params.nDims)
		{
		case 2:	// 2D
			GPU_QNN_2D_LBT<<< params.nnGrid, params.nnBlock >>>
				(
					(GPU_NN_Result *)(params.d_Results_GPU), 
					(float2 *)(params.d_Query), 
					(GPUNode_2D_LBT *)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.bgShape.W  
				);
			break;

		case 3: // 3D
			GPU_QNN_3D_LBT<<< params.nnGrid, params.nnBlock >>>
				(
					(GPU_NN_Result *)(params.d_Results_GPU), 
					(float4 *)(params.d_Query), 
					(GPUNode_3D_LBT *)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.bgShape.W  
				);
			break;

		case 4: // 4D
			GPU_QNN_4D_LBT<<< params.nnGrid, params.nnBlock >>>
				(
					(GPU_NN_Result *)(params.d_Results_GPU), 
					(float4 *)(params.d_Query), 
					(GPUNode_4D_LBT *)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.bgShape.W  
				);
			break;

		default:
			// Error
			break;
		} // end switch (dims)
		break;

	case NN_ALL_NN:	// All Nearest Neighbor Search
		switch (params.nDims)
		{
		case 2:	// 2D
			GPU_ALL_NN_2D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result*)(params.d_Results_GPU), 
					(GPUNode_2D_LBT*)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.bgShape.W 
				);
			break;

		case 3:	// 3D
			GPU_ALL_NN_3D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result*)(params.d_Results_GPU), 
					(GPUNode_3D_LBT*)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.bgShape.W 
				);
			break;

		case 4:	// 4D
			GPU_ALL_NN_4D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result*)(params.d_Results_GPU), 
					(GPUNode_4D_LBT*)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.bgShape.W 
				);
			break;

		default:
			// Error
			break;
		} // end switch (dims)
		break;

	case NN_KNN:	// 'k' Nearest Neighbor Search
		switch (params.nDims)
		{
		case 2:	// 2D
			GPU_KNN_2D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result *)(params.d_Results_GPU), 
					(float2 *)(params.d_Query), 
					(GPUNode_2D_LBT *)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.kVal
				);
			break;

		case 3:  // 3D
			GPU_KNN_3D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result *)(params.d_Results_GPU), 
					(float4 *)(params.d_Query), 
					(GPUNode_3D_LBT *)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.kVal 
				);
			break;

		case 4:  // 4D
			GPU_KNN_4D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result *)(params.d_Results_GPU), 
					(float4 *)(params.d_Query), 
					(GPUNode_4D_LBT *)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.kVal 
				);
			break;

		default:
			// Error
			break;
		}
		break;

	case NN_ALL_KNN:	// All 'k' Nearest Neighbor Search
		switch (params.nDims)
		{
		case 2:	// 2D
			GPU_ALL_KNN_2D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result*)(params.d_Results_GPU), 
					(GPUNode_2D_LBT*)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.kVal  
				);
			break;

		case 3:	// 3D
			GPU_ALL_KNN_3D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result*)(params.d_Results_GPU), 
					(GPUNode_3D_LBT*)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.kVal  
				);
			break;

		case 4:	// 4D
			GPU_ALL_KNN_4D_LBT<<< params.nnGrid, params.nnBlock >>>
				( 
					(GPU_NN_Result*)(params.d_Results_GPU), 
					(GPUNode_4D_LBT*)(params.d_Nodes),
					params.d_IDs, 
					params.nSearch, 
					params.kVal  
				);
			break;
		
		default:
			// Error
			break;
		}

	default:
		// Error
		break;
	} // end switch(nnType)

	// Prevent other kernels (including ourselves) from running asynchronously
	cudaThreadSynchronize();

	// Get result from GPU kernel
	cuda_err = cudaGetLastError();

	if (g_app.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( searchStop, 0 );
		cudaEventSynchronize( searchStop );
		cudaEventElapsedTime( &elapsedTime, searchStart, searchStop );

		if (g_app.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
			{
				KD_GPU_kernel += elapsedTime;
			}
		}
		else
		{
			KD_GPU_kernel += elapsedTime;
		}

		// Check for Errors
		if( cudaSuccess != cuda_err)
		{
			fprintf( stderr, "NN GPU Kernel failed, Cuda error(%d) = '%s', in file '%s' at line %i.\n",
					 cuda_err, cudaGetErrorString( cuda_err ), __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
	else
	{
		// Check for Errors
		if( cudaSuccess != cuda_err)
		{
			fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
				     "NN left-balanced GPU kernel failed", __FILE__, __LINE__, 
					 cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}
	}
} // End Profile Loop

	// Dump Performance Timings
	if (g_app.dumpVerbose > 0)
	{
		if (g_app.profileActualLoops > 1)
		{
			float loops = (float)g_app.profileActualLoops;
			float o_l = 1.0f / loops;

			double avgGPU  = (double)(KD_GPU_kernel * o_l) - g_app.cudaTimerCost;

			printf( "Number of total iterations = %f.\n", loops );
			printf( "GPU NN Search Kernel,        avg time: %f msecs.\n", avgGPU );
		}
		else
		{
			double gpuKernel  = (double)KD_GPU_kernel - g_app.cudaTimerCost;
			printf( "GPU NN Search Kernel,            time: %f msecs.\n", gpuKernel );
		}
	}

	return true;
}



/*---------------------------------------------------------
  Name:	CallGPUKernelRowByRow
  Desc: Wrapper around GPU NN search kernel
  Notes:  
  
  Same as above, however this version of the function 
  calls the GPU kernels one row of thread blocks 
  at a time to try and avoid hitting the dreaded 
  Microsoft 5 second timeout bug !!! 
	           
---------------------------------------------------------*/

bool CallGPUKernelRowByRow
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	float elapsedTime	= 0.0f;
	double KD_GPU_kernel = 0.0;

	cudaEvent_t searchStart, searchStop;

	if (g_app.profile)
	{
		// Create Timer Events
		cudaEventCreate( &searchStart );
		cudaEventCreate( &searchStop );
	}

	// We assume our worst case searches are taking longer than 5 seconds
	// We will try to call the kernel functions one row of thread blocks 
	// at a time to work around this problem.

	// Call one row of thread blocks at a time...
	unsigned int rowIdx, startIdx;
	unsigned int rowsPerBlock = params.bgShape.rowsPerBlock;
	unsigned int nGridRows    = params.bgShape.rowsPerGrid;
	unsigned int rowWidth     = params.bgShape.W;
	GPU_NN_Result * origResults = (GPU_NN_Result *)(params.d_Results_GPU);
	GPU_NN_Result * currResults = NULL;
	float2 * origQuery2D = (float2 *)(params.d_Query);
	float4 * origQuery4D = (float4 *)(params.d_Query);
	float2 * currQuery2D = NULL;
	float4 * currQuery4D = NULL;
	dim3 rowGrid( params.bgShape.blocksPerRow, 1, 1 );
	dim3 rowBlock( params.bgShape.threadsPerRow, params.bgShape.rowsPerBlock, 1 );

	// Call Kernel one thread block row at a time
	for (rowIdx = 0; rowIdx < nGridRows ; rowIdx++)
	{
		// Move to current row
		startIdx     = rowIdx * (rowWidth*rowsPerBlock);
		currResults  = &(origResults[startIdx]);
		if (origQuery2D != NULL)
		{
			currQuery2D  = &(origQuery2D[startIdx]);
			currQuery4D  = &(origQuery4D[startIdx]);
		}

		/*---------------------------------
		  Call GPU NN Search Kernel
		----------------------------------*/

		if (g_app.profile)
		{
			// Start Timer
			cudaEventRecord( searchStart, 0 );
		}
		
		// Check if GPU kernel execution generated an error
		switch (params.nnType)
		{
		case NN_QNN:	// Query Nearest Neighbor Search
			switch (params.nDims)
			{
			case 2:	// 2D
				GPU_QNN_2D_LBT<<< rowGrid, rowBlock >>>
					(
						currResults, 
						currQuery2D, 
						(GPUNode_2D_LBT *)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.bgShape.W  
					);
				break;

			case 3: // 3D
				GPU_QNN_3D_LBT<<< rowGrid, rowBlock >>>
					(
						currResults, 
						currQuery4D, 
						(GPUNode_3D_LBT *)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.bgShape.W  
					);
				break;

			case 4: // 4D
				GPU_QNN_4D_LBT<<< rowGrid, rowBlock >>>
					(
						currResults, 
						currQuery4D, 
						(GPUNode_4D_LBT *)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.bgShape.W  
					);
				break;

			default:
				// Error
				break;
			} // end switch (dims)
			break;

		case NN_ALL_NN:	// All Nearest Neighbor Search
			switch (params.nDims)
			{
			case 2:	// 2D
				GPU_ALL_NN_2D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						(GPUNode_2D_LBT*)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.bgShape.W 
					);
				break;

			case 3:	// 3D
				GPU_ALL_NN_3D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						(GPUNode_3D_LBT*)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.bgShape.W 
					);
				break;

			case 4:	// 4D
				GPU_ALL_NN_4D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						(GPUNode_4D_LBT*)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.bgShape.W 
					);
				break;

			default:
				// Error
				break;
			} // end switch (dims)
			break;

		case NN_KNN:	// 'k' Nearest Neighbor Search
			switch (params.nDims)
			{
			case 2:	// 2D
				GPU_KNN_2D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						currQuery2D, 
						(GPUNode_2D_LBT *)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.kVal 
					);
				break;

			case 3:  // 3D
				GPU_KNN_3D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						currQuery4D, 
						(GPUNode_3D_LBT *)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.kVal 
					);
				break;

			case 4:  // 4D
				GPU_KNN_4D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						currQuery4D, 
						(GPUNode_4D_LBT *)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.kVal 
					);
				break;

			default:
				// Error
				break;
			}

		case NN_ALL_KNN:	// All 'k' Nearest Neighbor Search
			switch (params.nDims)
			{
			case 2:	// 2D
				GPU_ALL_KNN_2D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						(GPUNode_2D_LBT*)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.kVal  
					);
				break;

			case 3:	// 3D
				GPU_ALL_KNN_3D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						(GPUNode_3D_LBT*)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.kVal  
					);
				break;

			case 4:	// 4D
				GPU_ALL_KNN_4D_LBT<<< rowGrid, rowBlock >>>
					( 
						currResults, 
						(GPUNode_4D_LBT*)(params.d_Nodes),
						params.d_IDs, 
						params.nSearch, 
						params.kVal  
					);
				break;
			
			default:
				// Error
				break;
			}

		default:
			// Error
			break;
		} // end switch(nnType)

		// BUGBUG:  This function call is necessary to avoid queueing 
		//          up each row by row GPU Kernel calls,
		//          Without it we will definitely hit the 
		//          5 second Microsoft TDR bug !!!
		cudaThreadSynchronize();

		cudaError_t cuda_err = cudaGetLastError();
		if( cudaSuccess != cuda_err)
		{
			fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
					 "NN left-balanced GPU kernel failed", __FILE__, __LINE__, 
					 cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cudaEventRecord( searchStop, 0 );
			cudaEventSynchronize( searchStop );
			cudaEventElapsedTime( &elapsedTime, searchStart, searchStop );

			KD_GPU_kernel += (double)elapsedTime;
		}

		// Dump 'Still Alive' Indicator
		if (g_app.dumpVerbose > 0)
		{
			printf( "Row[%d] done!\n", rowIdx );
		}

	} // End row by row thread block loop

	// Dump Performance Timings
	if (g_app.dumpVerbose > 0)
	{
		float loops = (float)nGridRows;
		float o_l = 1.0f / loops;

		double avgGPU   = (KD_GPU_kernel * o_l) - g_app.cudaTimerCost;
		double totalGPU = KD_GPU_kernel;

		printf( "Number of total rows = %f.\n", loops );
		printf( "GPU NN Search Kernel,  avg time per row: %f msecs.\n", avgGPU );
		printf( "GPU NN Search Kernel,        total time: %f msecs.\n", totalGPU );
	}

	return true;
} // end CallGPUKernelRowByRow


/*---------------------------------------------------------
  Name:	CallCPUKernel
---------------------------------------------------------*/

bool CallCPUKernel
( 
	HostDeviceParams & params	// IN/OUT - input, output parameters
)
{
	if (g_app.doubleCheckDists)
	{
		bool bResult = true;

		float KD_CPU_kernel = 0.0f;

		unsigned int currIter;
		unsigned int nProfileLoops = g_app.profileActualLoops;
		if (nProfileLoops <= 0)
		{
			nProfileLoops = 1;
		}
		if (nProfileLoops >= 102)
		{
			nProfileLoops = 102;
		}

	// Profile Measurement Loop
	for (currIter = 0; currIter < nProfileLoops; currIter++)
	{
		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Determine Nearest Neighbors using KDTree
		switch (params.nnType)
		{
		case NN_QNN:	// CPU QNN search
			switch (params.nDims)
			{
			case 2: // 2D
				bResult = CPU_QNN_2D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								(const float4 *)(params.queryList),
								params.h_Results_CPU
							);
				break;

			case 3: // 3D
				bResult = CPU_QNN_3D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								(const float4 *)(params.queryList),
								params.h_Results_CPU
							);
				break;

			case 4: // 4D
				bResult = CPU_QNN_4D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								(const float4 *)(params.queryList),
								params.h_Results_CPU
							);
				break;

			default:
				// Error
				break;
			} // end switch (nDims)
			break;

		case NN_ALL_NN:	// CPU All-NN search
			switch (params.nDims)
			{
			case 2: // 2D
				bResult = CPU_ALL_NN_2D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								(const float4 *)(params.queryList),
								params.h_Results_CPU 
							);
				break;

			case 3: // 3D
				bResult = CPU_ALL_NN_3D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								(const float4 *)(params.queryList),
								params.h_Results_CPU 
							);
				break;

			case 4: // 4D
				bResult = CPU_ALL_NN_4D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								(const float4 *)(params.queryList),
								params.h_Results_CPU 
							);
				break;

			default:
				// Error
				break;
			} // end switch (nDims)
			break;

		case NN_KNN:	// CPU kNN search
			switch (params.nDims)
			{
			case 2: // 2D
				bResult = CPU_KNN_2D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								params.nPadQuery,
								(const float4 *)(params.queryList),
								params.kVal,
								params.h_Results_CPU
							 );
				break;

			case 3: // 3D
				bResult = CPU_KNN_3D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								params.nPadQuery,
								(const float4 *)(params.queryList),
								params.kVal,
								params.h_Results_CPU
							 );
				break;

			case 4: // 4D
				bResult = CPU_KNN_4D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								params.nPadQuery,
								(const float4 *)(params.queryList),
								params.kVal,
								params.h_Results_CPU
							 );
				break;

			default:
				// Error
				break;
			} // end switch (nDims)
			break;

		case NN_ALL_KNN:
			switch (params.nDims)
			{
			case 2: // 2D
				bResult = CPU_ALL_KNN_2D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								params.nPadSearch,
								(const float4 *)(params.queryList),
								params.kVal,
								params.h_Results_CPU
							);
				break;

			case 3: // 3D
				bResult = CPU_ALL_KNN_3D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								params.nPadSearch,
								(const float4 *)(params.queryList),
								params.kVal,
								params.h_Results_CPU
							);
				break;

			case 4: // 4D
				bResult = CPU_ALL_KNN_4D_LBT
							(
								params.kdtree,
								params.nSearch,
								(const float4 *)(params.searchList),
								params.nQuery,
								params.nPadSearch,
								(const float4 *)(params.queryList),
								params.kVal,
								params.h_Results_CPU
							);
				break;

			default:
				// Error
				break;
			} // end switch (nDims)
			break;

		default:
			// Error
			break;
		} // end switch (nnType)

		if (false == bResult)
		{
			// Error
			return false;
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					KD_CPU_kernel += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				KD_CPU_kernel += cutGetTimerValue( g_app.hTimer );
			}
		}

	} // End Profile Loop

		// Dump Performance Timings
		if (g_app.dumpVerbose > 0)
		{
			if (g_app.profileActualLoops > 1)
			{
				float loops = (float)g_app.profileActualLoops;
				float o_l = 1.0f / loops;

				double avgCPU  = (double)(KD_CPU_kernel * o_l) - g_app.baseTimerCost;

				printf( "CPU NN Search Kernel,        avg time: %f msecs.\n", avgCPU );
			}
			else
			{
				double cpuKernel  = (double)KD_CPU_kernel - g_app.baseTimerCost;
				printf( "CPU NN Search Kernel,            time: %f msecs.\n", cpuKernel );
			}
		}
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	CallCPUKernel
---------------------------------------------------------*/

bool CompareCPUToGPU( HostDeviceParams & params )
{
	if ((g_app.doubleCheckDists) && 
		(NULL != params.resultList) && 
		(NULL != params.h_Results_CPU))
	{
		bool checkDistResults = true;

		unsigned int nSearch = params.nSearch;
		unsigned int nQuery  = params.nQuery;

		const GPU_NN_Result * gpuResults = (const GPU_NN_Result *)(params.resultList);
		const CPU_NN_Result * cpuResults = (const CPU_NN_Result *)(params.h_Results_CPU);

		float eps = 1.0e-4f;
		unsigned int rIdx, kIdx, gpuIdx, cpuIdx, gpuID, cpuID;
		unsigned int kVal, nDims;
		float gpuDist, cpuDist;
		double qX, qY, qZ, qW;
		double gpu_D, cpu_D;
		//double gpuVisited = 0;
		//double cpuVisited = 0;
		//unsigned int gpuMin = 0xFFFFFFFFu;
		//unsigned int gpuMax = 0u;
		//unsigned int cpuMin = 0xFFFFFFFFu;
		//unsigned int cpuMax = 0u;

		const float4 * searchList = params.searchList;
		const float4 * queryList  = params.queryList;

		kVal  = params.kVal;
		nDims = params.nDims;

		switch (params.nnType)
		{
		case NN_QNN:
		{
			// Check each query result (GPU vs. CPU)
			for (rIdx = 0; rIdx < nQuery; rIdx++)
			{				
				gpuID   = gpuResults[rIdx].Id;
				gpuDist = gpuResults[rIdx].Dist;
				//unsigned int gpuVisits = gpuResults[rIdx].cVisited;
				//gpuVisited += (double)gpuVisits;

				cpuID   = cpuResults[rIdx].Id;
				cpuDist = cpuResults[rIdx].Dist;
				//unsigned int cpuVisits = cpuResults[rIdx].cVisited;
				//cpuVisited += (double)cpuVisits;

				// Get Min, Max Nodes visited for any query
				//if (gpuVisits < gpuMin) { gpuMin = gpuVisits; }
				//if (gpuVisits > gpuMax) { gpuMax = gpuVisits; }
				//if (cpuVisits < cpuMin) { cpuMin = cpuVisits; }
				//if (cpuVisits > cpuMax) { cpuMax = cpuVisits; }

				// Did we get the same answer ?!?
				if (gpuID != cpuID)
				{
					// Get Query Point
					switch (nDims)
					{
					case 2:
						qX = static_cast<double>( queryList[rIdx].x );
						qY = static_cast<double>( queryList[rIdx].y );
						break;
					case 3:
						qX = static_cast<double>( queryList[rIdx].x );
						qY = static_cast<double>( queryList[rIdx].y );
						qZ = static_cast<double>( queryList[rIdx].z );
						break;
					case 4:
						qX = static_cast<double>( queryList[rIdx].x );
						qY = static_cast<double>( queryList[rIdx].y );
						qZ = static_cast<double>( queryList[rIdx].z );
						qW = static_cast<double>( queryList[rIdx].w );
						break;
					default:
						// Non-supported # of dimensions
						break;
					}

					// Get CPU & GPU distances
					gpu_D = static_cast<double>( gpuDist );
					cpu_D = static_cast<double>( cpuDist );

					// Need to use a fuzzy compare on distance
					if ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps)))
					{
						switch (nDims)
						{
						case 2:
							printf( "[%u]=<%.6g, %.6g>: Different ID's and incompatible distances !!!\n",
								    rIdx, qX, qY );
							break;

						case 3:
							printf( "[%u]=<%.6g, %.6g, %.6g>: Different ID's and incompatible distances !!!\n",
								    rIdx, qX, qY, qZ );
							break;

						case 4:
							printf( "[%u]=<%.6g, %.6g, %.6g, %.6g>: Different ID's and incompatible distances !!!\n",
								    rIdx, qX, qY, qZ, qW );
							break;

						default:
							// Non-supported # of dimensions
							printf( "[%u]=Different ID's, incompatible distances, and Unknown # of dimensions !!!\n",
								     rIdx );
							break;
						}
						printf( "\t\t\tGPU KD[%u %.9g],  CPU KD[%u %.9g]\n",
								 gpuID, gpu_D, cpuID, cpu_D );
						checkDistResults = false;
					}
					else
					{
						switch (nDims)
						{
						case 2:
							printf( "[%u]=<%.6g, %.6g>: Different ID's but distances are compatible\n",
								    rIdx, qX, qY );
							break;

						case 3:
							printf( "[%u]=<%.6g, %.6g, %.6g>: Different ID's but distances are compatible\n",
								    rIdx, qX, qY, qZ );
							break;

						case 4:
							printf( "[%u]=<%.6g, %.6g, %.6g, %.6g>: Different ID's but distances are compatible\n",
								    rIdx, qX, qY, qZ, qW );
							break;

						default:
							// Non-supported # of dimensions
							printf( "[%u]=Different ID's but compatible distances, and Unknown # of dimensions !!!\n",
								     rIdx );
							break;
						}
						printf( "\t\t\tGPU KD[%u %.9g],  CPU KD[%u %.9g]\n",
								 gpuID, gpu_D, cpuID, cpu_D );
					}
				}
			}

			// double gpuAvg = gpuVisited / (double)nResults;
			// double cpuAvg = cpuVisited / (double)nResults;

			// printf( "GPU Node Visits [Min = %u, Max = %u, Avg = %f]\n", gpuMin, gpuMax, gpuAvg );
			// printf( "CPU Node Visits [Min = %u, Max = %u, Avg = %f]\n", cpuMin, cpuMax, cpuAvg );
		}
			break;

		case NN_ALL_NN:
		{
			// Check each query result (GPU vs. CPU)
			for (rIdx = 0; rIdx < nSearch; rIdx++)
			{
				gpuID   = gpuResults[rIdx].Id;
				gpuDist = gpuResults[rIdx].Dist;
				//unsigned int gpuVisits = gpuResults[rIdx].cVisited;
				//gpuVisited += (double)gpuVisits;

				cpuID   = cpuResults[rIdx].Id;
				cpuDist = cpuResults[rIdx].Dist;
				//unsigned int cpuVisits = cpuResults[rIdx].cVisited;
				//cpuVisited += (double)cpuVisits;

				// Get Min, Max Nodes visited for any query
				//if (gpuVisits < gpuMin) { gpuMin = gpuVisits; }
				//if (gpuVisits > gpuMax) { gpuMax = gpuVisits; }
				//if (cpuVisits < cpuMin) { cpuMin = cpuVisits; }
				//if (cpuVisits > cpuMax) { cpuMax = cpuVisits; }

				// Did we get the same answer ?!?
				if (gpuID != cpuID)
				{
					switch (nDims)
					{
					case 2:
						qX = static_cast<double>( searchList[rIdx].x );
						qY = static_cast<double>( searchList[rIdx].y );
						break;
					case 3:
						qX = static_cast<double>( searchList[rIdx].x );
						qY = static_cast<double>( searchList[rIdx].y );
						qZ = static_cast<double>( searchList[rIdx].z );
						break;
					case 4:
						qX = static_cast<double>( searchList[rIdx].x );
						qY = static_cast<double>( searchList[rIdx].y );
						qZ = static_cast<double>( searchList[rIdx].z );
						qW = static_cast<double>( searchList[rIdx].w );
						break;
					default:
						// Unsupported # of dimensions
						break;
					}
					gpu_D = static_cast<double>( gpuDist );
					cpu_D = static_cast<double>( cpuDist );

					// Need to use a fuzzy compare on distance
					if ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps)))
					{
						switch (nDims)
						{
						case 2:
							printf( "[%u]=<%.6g, %.6g>: Different ID's and incompatible distances !!!\n",
								    rIdx, qX, qY );
							break;
						case 3:
							printf( "[%u]=<%.6g, %.6g, %.6g>: Different ID's and incompatible distances !!!\n",
								    rIdx, qX, qY, qZ );
							break;
						case 4:
							printf( "[%u]=<%.6g, %.6g, %.6g, %.6g>: Different ID's and incompatible distances !!!\n",
								    rIdx, qX, qY, qZ, qW );
							break;
						default:
							// Non-supported # of dimensions
							printf( "[%u]=Different ID's, Incompatible distances, and Unknown # of dimensions !!!\n",
								    rIdx );
							break;
						}
						printf( "\t\t\tGPU KD[%u %.9g],  CPU KD[%u %.9g]\n",
								 gpuID, gpu_D, cpuID, cpu_D );
						checkDistResults = false;
					}
					else
					{
						switch (nDims)
						{
						case 2:
							printf( "[%u]=<%.6g, %.6g>: Different ID's but distances are compatible\n",
								    rIdx, qX, qY );
							break;
						case 3:
							printf( "[%u]=<%.6g, %.6g, %.6g>: Different ID's but distances are compatible\n",
								    rIdx, qX, qY, qZ );
							break;
						case 4:
							printf( "[%u]=<%.6g, %.6g, %.6g, %.6g>: Different ID's but distances are compatible\n",
								    rIdx, qX, qY, qZ, qW );
							break;
						default:
							// Non-supported # of dimensions
							printf( "[%u]=Different ID's but compatible distances, and Unknown # of dimensions !!!\n", 
								    rIdx );
							break;
						}

						printf( "\t\t\tGPU KD[%u %.9g],  CPU KD[%u %.9g]\n",
								 gpuID, gpu_D, cpuID, cpu_D );
					}
				}
			}

			// double gpuAvg = gpuVisited / (double)nResults;
			// double cpuAvg = cpuVisited / (double)nResults;

			// printf( "GPU Node Visits [Min = %u, Max = %u, Avg = %f]\n", gpuMin, gpuMax, gpuAvg );
			// printf( "CPU Node Visits [Min = %u, Max = %u, Avg = %f]\n", cpuMin, cpuMax, cpuAvg );
		}
			break;

		case NN_KNN:
		{
			if (g_app.dumpVerbose >= 3)
			{
				// Check each query result (GPU vs. CPU)
				for (rIdx = 0; rIdx < nQuery; rIdx++)
				{
					// Check each Nearest Neighbor
					for (kIdx = 0; kIdx < kVal; kIdx++)
					{
						gpuIdx = (kIdx * nQuery) + rIdx;
						cpuIdx = (kIdx * nQuery) + rIdx;

						gpuID   = gpuResults[gpuIdx].Id;
						gpuDist = gpuResults[gpuIdx].Dist;

						cpuID   = cpuResults[cpuIdx].Id;
						cpuDist = cpuResults[cpuIdx].Dist;

						gpu_D = static_cast<double>( gpuDist );
						cpu_D = static_cast<double>( cpuDist );

						switch (nDims)
						{
						case 2:
							qX = static_cast<double>( queryList[rIdx].x );
							qY = static_cast<double>( queryList[rIdx].y );
							printf( "[%d][%d]=<%g, %g> GPU[%d %g], CPU[%d %g]\n",
									 rIdx, kIdx, 
									 qX, qY, 
									 gpuID, gpu_D, 
									 cpuID, cpu_D );
							break;

						case 3:
							qX = static_cast<double>( queryList[rIdx].x );
							qY = static_cast<double>( queryList[rIdx].y );
							qZ = static_cast<double>( queryList[rIdx].z );
							printf( "[%d][%d]=<%g, %g, %g> GPU[%d %g], CPU[%d %g]\n",
									 rIdx, kIdx, 
									 qX, qY, qZ,
									 gpuID, gpu_D, 
									 cpuID, cpu_D );
							break;

						case 4:
							qX = static_cast<double>( queryList[rIdx].x );
							qY = static_cast<double>( queryList[rIdx].y );
							qZ = static_cast<double>( queryList[rIdx].z );
							qW = static_cast<double>( queryList[rIdx].w );
							printf( "[%d][%d]=<%g, %g, %g, %g> GPU[%d %g], CPU[%d %g]\n",
									 rIdx, kIdx, 
									 qX, qY, qZ, qW,
									 gpuID, gpu_D, 
									 cpuID, cpu_D );
							break;

						default:
							// Error - unsupported dimensionality
							break;
						}
					}
				}
			}
		}
			break;

		case NN_ALL_KNN:
		{
			if (g_app.dumpVerbose >= 3)
			{
				// Check each query result (GPU vs. CPU)
				for (rIdx = 0; rIdx < nSearch; rIdx++)
				{
					// Check each Nearest Neighbor
					for (kIdx = 0; kIdx < kVal; kIdx++)
					{
						gpuIdx = (kIdx * nSearch) + rIdx;
						cpuIdx = (kIdx * nSearch) + rIdx;

						gpuID   = gpuResults[gpuIdx].Id;
						gpuDist = gpuResults[gpuIdx].Dist;

						cpuID   = cpuResults[cpuIdx].Id;
						cpuDist = cpuResults[cpuIdx].Dist;

						gpu_D = static_cast<double>( gpuDist );
						cpu_D = static_cast<double>( cpuDist );

						switch (nDims)
						{
						case 2:
							qX = static_cast<double>( searchList[rIdx].x );
							qY = static_cast<double>( searchList[rIdx].y );

							printf( "[%d][%d]=<%g, %g>, GPU[%d %g], CPU[%d %g]\n",
									 rIdx, kIdx, 
									 qX, qY, 
									 gpuID, gpu_D, 
									 cpuID, cpu_D );
							break;

						case 3:
							qX = static_cast<double>( searchList[rIdx].x );
							qY = static_cast<double>( searchList[rIdx].y );
							qZ = static_cast<double>( searchList[rIdx].z );

							printf( "[%d][%d]=<%g, %g, %g>, GPU[%d %g], CPU[%d %g]\n",
									 rIdx, kIdx, 
									 qX, qY, qZ, 
									 gpuID, gpu_D, 
									 cpuID, cpu_D );
							break;

						case 4:
							qX = static_cast<double>( searchList[rIdx].x );
							qY = static_cast<double>( searchList[rIdx].y );
							qZ = static_cast<double>( searchList[rIdx].z );
							qW = static_cast<double>( searchList[rIdx].w );

							printf( "[%d][%d]=<%g, %g, %g, %g>, GPU[%d %g], CPU[%d %g]\n",
									 rIdx, kIdx, 
									 qX, qY, qZ, qW,
									 gpuID, gpu_D, 
									 cpuID, cpu_D );
							break;

						default:
							// Error - unsupported dimensionality
							break;
						}
					}
				}
			}
		}
			break;

		default:
			// Error
			break;
		}

		//printf ("Max GPU Nodes = %d, Avg GPU Nodes = %g\n", maxNodes, avgNodes );
		if (true == checkDistResults)
		{
			printf( "Distance check: CPU and GPU results agree within tolerance.\n" );
		}
		else
		{
			printf( "Distance check: CPU and GPU results don't agree within tolerance !!!\n" );
		}

	}

	return true;
}



/*---------------------------------------------------------
  Name:	NN_LBT_HOST
  Desc: Performs NN search on GPU as specified
  Notes:
	1. Thread Block size (threads per row, rows per block)
	   needs to be hardcoded at compile time as this info
	   is used to generate static query vectors, search 
	   stacks and knn arrays for GPU kernels

    2. 'k' value is only needed for kNN and 
	   All-kNN searches

    3. number of results needed
		QNN      nResults == nQuery
		All-NN   nResults == nSearch
		kNN      nResults == kVal * nQuery
		All-kNN  nResults == kVal * nSearch

    4. The output results from the kNN (and All-kNN) 
	   search are stored in the following manner
		0th entry [          0, ...,           m-1]
		1st entry [      m + 0, ...,     1*m + m-1]
        2nd entry [    2*m + 0, ...,     2*m + m-1]
		...
        ith entry [    i*m + 0, ...,     i*m + m-1]
		...
		kth entry [(k-1)*m + 0, ..., (k-1)*m + m-1]

	  where m = # of querys (or search points for All-kNN)
	        k = # of points found (IE 'k' value)

	  Note: This implies that if you are doing a kNN or
	        All-kNN search you need to have room to store
			m*k results
---------------------------------------------------------*/

bool NN_LBT_HOST
( 
	//unsigned int threadsPerRow,	// IN - threads per row in Thread Block
	//unsigned int rowsPerBlock,	// IN - rows per block in Thread Block
	unsigned int nnType,		// IN - NN Search type (QNN, All-NN, kNN, All-kNN)
	unsigned int nDims,			// IN - number of dimensions for search
	unsigned int kVal,			// IN - 'k' value for kNN and All-kNN searches
	unsigned int nSearch,		// IN - Count of Search Points
	const float4 * searchList,	// IN - List of Search Points
	unsigned int nQuery,		// IN - Count of Query Points
	const float4 * queryList,	// IN - List of Query Points
	unsigned int nResult,		// IN - number of results to store
	CPU_NN_Result * resultList	// OUT - List or results (one for each query point)
)
{
	unsigned int nTPR, nRPB;

	// Check Parameters
	switch (nnType)
	{
	case NN_QNN:
		if ((nQuery == 0u) || (queryList == NULL)) 
		{
			// Error - need a valid query list
			fprintf( stderr, "NN_LBT_HOST: (QNN) Need a valid query list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
			return false;
		}
		if ((nSearch == 0u) || (searchList == NULL))
		{
			// Error - need a valid search list
			fprintf( stderr, "NN_LBT_HOST: (QNN) Need a valid search list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
			return false;
		}
		if (nResult < nQuery)
		{
			// Error - need at least as many results as queries
			fprintf( stderr, "NN_LBT_HOST: (QNN) Need at least as many results[%u] as queries[%u] - in file '%s' at line %i.\n",
					 nResult, nQuery, __FILE__, __LINE__ );
		}
		if (resultList == NULL)
		{
			// Error - need a valid result list
			fprintf( stderr, "NN_LBT_HOST: (QNN) Need a valid result list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
		}
		break;

	case NN_ALL_NN:
		if ((nSearch == 0u) || (searchList == NULL))
		{
			// Error - need a valid search list
			fprintf( stderr, "NN_LBT_HOST: (All-NN) Need a valid search list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
			return false;
		}
		if (nResult < nSearch)
		{
			// Error - need at least as many results as searches (queries)
			fprintf( stderr, "NN_LBT_HOST: (All-NN) Need at least as many results[%u] as searches[%u] - in file '%s' at line %i.\n",
					 nResult, nSearch, __FILE__, __LINE__ );
		}
		if (resultList == NULL)
		{
			// Error - need a valid result list
			fprintf( stderr, "NN_LBT_HOST: (All-NN) Need a valid result list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
		}
		break;

	case NN_KNN:
	{
		if ((nQuery == 0u) || (queryList == NULL)) 
		{
			// Error - need a valid query list
			fprintf( stderr, "NN_LBT_HOST: (kNN) Need a valid query list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
			return false;
		}
		if ((nSearch == 0u) || (searchList == NULL))
		{
			// Error - need a valid search list
			fprintf( stderr, "NN_LBT_HOST: (kNN) Need a valid search list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
			return false;
		}
		unsigned totalResult = nQuery * kVal;
		if (nResult < totalResult)
		{
			// Error - need at least as many results as (queries * kVal)
			fprintf( stderr, "NN_LBT_HOST: (kNN) Need at least as many #results(%u) as <Total(%u) = #queries(%u) * #kVal(%u)> - in file '%s' at line %i.\n",
					 nResult, totalResult, nQuery, kVal, __FILE__, __LINE__ );
		}
		if (resultList == NULL)
		{
			// Error - need a valid result list
			fprintf( stderr, "NN_LBT_HOST: (kNN) Need a valid result list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
		}
	}
		break;

	case NN_ALL_KNN:
	{
		if ((nSearch == 0u) || (searchList == NULL))
		{
			// Error - need a valid search list
			fprintf( stderr, "NN_LBT_HOST: (All-kNN) Need a valid search list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
			return false;
		}
		unsigned totalResult = nSearch * kVal;
		if (nResult < totalResult)
		{
			// Error - need at least as many results as (queries * kVal)
			fprintf( stderr, "NN_LBT_HOST: (All-kNN) Need at least as many results[%u] as <Total[%u] = searches[%u] * kVal[%u]> - in file '%s' at line %i.\n",
					 nResult, totalResult, nSearch, kVal, __FILE__, __LINE__ );
		}
		if (resultList == NULL)
		{
			// Error - need a valid result list
			fprintf( stderr, "NN_LBT_HOST: (All-kNN) Need a valid result list - in file '%s' at line %i.\n",
					 __FILE__, __LINE__ );
		}
	}
		break;

	default:
		fprintf( stderr, "NN_LBT_HOST: Unknown search type [%u] - in file '%s' at line %i.\n",
				 nnType, __FILE__, __LINE__ );
		return false;
	}

	bool bResult = true;	// Assume success
	HostDeviceParams params;

	// Initialize Parameters to default values
	DefaultParams( params );

	// Initialize CUDA parameters
	InitCuda( params );
	InitTimer();


	// Initialize Query Type, Dimensions and kVal
	InitNN( params, nnType, nDims, kVal );

	// Init Search Points (for input)
	bResult = InitSearch( params, nSearch, searchList );
	if (!bResult)
	{
		goto lblCleanup;
	}

		// For Debugging
	DumpSearch( params );

	// Init Query Points (for input)
	bResult = InitQuery( params, nQuery, queryList );
	if (!bResult)
	{
		goto lblCleanup;
	}

		// For Debugging
	DumpQuery( params );

	// Init Results (for output)
	bResult = InitResult( params, nResult, resultList );
	if (!bResult)
	{
		goto lblCleanup;
	}

	// Setup Thread Block & Grid
	//bResult = InitBlockGrid( params, threadsPerRow, rowsPerBlock );
	switch (params.nnType)
	{
	case NN_QNN:
		nTPR = QNN_THREADS_PER_ROW;
		nRPB = QNN_ROWS_PER_BLOCK;
		break;
	case NN_ALL_NN:
		nTPR = QNN_THREADS_PER_ROW;
		nRPB = QNN_ROWS_PER_BLOCK;
		break;
	case NN_KNN:
		nTPR = KNN_THREADS_PER_ROW;
		nRPB = KNN_ROWS_PER_BLOCK;
		break;
	case NN_ALL_KNN:
		nTPR = ALL_KNN_THREADS_PER_ROW;
		nRPB = ALL_KNN_ROWS_PER_BLOCK;
		break;
	default:
		// Error - Unknown NN type
		nTPR = QNN_THREADS_PER_ROW;
		nRPB = QNN_ROWS_PER_BLOCK;
		break;
	}

	bResult = InitBlockGrid( params, nTPR, nRPB );
	if (!bResult)
	{
		goto lblCleanup;
	}
		
		// For Debugging
	DumpInitParams( params );

	// Make sure there is enough memory to perform operation
	bResult = ValidateAvailMemory( params );
	if (!bResult)
	{
		goto lblCleanup;
	}

	// Allocate CPU Host and GPU Device memory 
	bResult = AllocMem( params );
	if (!bResult)
	{
		goto lblCleanup;
	}

	// Copy Query List
	bResult = CopyQuery( params );
	if (!bResult)
	{
		goto lblCleanup;
	}

	// Build kd-tree
	if (params.buildGPU)
	{
		// Build on GPU
		bResult = BuildKDTree_GPU( params );
		if (!bResult)
		{
			goto lblCleanup;
		}

		if ((g_app.dumpVerbose >= 1) && 
			(g_app.doubleCheckDists >= 1))
		{
			bResult = BuildKDTree_CPU( params );
			if (!bResult)
			{
				goto lblCleanup;
			}
		}
	}
	else
	{
		// Build on CPU
		bResult = BuildKDTree_CPU( params );
		if (!bResult)
		{
			goto lblCleanup;
		}

		// Copy Nodes from CPU onto GPU
		bResult = CopyNodes( params );
		if (!bResult)
		{	
			goto lblCleanup;
		}
	}

		// For Debugging
	DumpGPUNodes( params );
	DumpMapping( params );
	DumpCPUNodes( params );

	// Copy Search & Query Vectors 
		// from host CPU memory onto GPU device memory
	CopyInputsOntoGPU( params );

	// Call GPU Kernel
	bResult = CallGPUKernel( params );
	if (!bResult)
	{
		goto lblCleanup;
	}

	// Copy Results (from device)
	bResult = CopyResultsFromGPU( params );
	if (!bResult)
	{
		goto lblCleanup;
	}

	DumpResultsGPU( params );


	// Call CPU Kernel
	bResult = CallCPUKernel( params );
	if (!bResult)
	{
		goto lblCleanup;
	}

	DumpResultsCPU( params );


	// Check GPU result against CPU result
	bResult = CompareCPUToGPU( params );
	if (!bResult)
	{
		goto lblCleanup;
	}


	// Success

lblCleanup:
	// Cleanup host and device memory
	Cleanup( params );

	return bResult;
}


/*---------------------------------------------------------
  Name:	Test_NN_API()
---------------------------------------------------------*/

bool Test_NN_API()
{
	bool bResult = true;

	unsigned int nSearch = 10000000u;	// 10 million
	unsigned int nQuery  = 10000000u;	// 10 million
	unsigned int nResult = 10000000u;	// 10 million
	unsigned int kMax    = 32;
	unsigned int totalResult = nResult * kMax;

	unsigned int mem_size_Search;
	unsigned int mem_size_Query;
	unsigned int mem_size_Result;

	float4 * searchList        = NULL;
	float4 * queryList		   = NULL;
	CPU_NN_Result * resultList = NULL;

	float minS = 0.0f;
	float maxS = 1.0f;

	float minQ = 0.0f;
	float maxQ = 1.0f;

	unsigned int idx, nn, nS, nQ, nR, nD, nK;
	//unsigned int TPR, RPB;

	const char * nnStrings[5] = 
	{ 
		"Unknown", 
		"QNN", 
		"All-NN",
		"kNN",
		"All-kNN"
	};

	const char * nnStr = NULL;

	// Allocate memory for Search vector
	mem_size_Search = nSearch * sizeof(float4);
	searchList = (float4*) malloc( mem_size_Search );
	if (NULL == searchList)
	{
		bResult = false;
		goto lblCleanup;
	}

	// Allocate memory for Query vector
	mem_size_Query = nQuery * sizeof(float4);
	queryList = (float4*) malloc( mem_size_Query );
	if (NULL == queryList)
	{
		bResult = false;
		goto lblCleanup;
	}

	// Allocate memory for Result vector
	mem_size_Result = totalResult * sizeof(CPU_NN_Result);
	resultList = (CPU_NN_Result*) malloc( mem_size_Result );
	if (NULL == resultList)
	{
		bResult = false;
		goto lblCleanup;
	}

	// Set seed for random number generator
	RandomInit( 2010 );

	// Generate Random Search Points
	minS = 0.0f;
	maxS = 1.0f;
	for (idx = 0; idx < nSearch; idx++)
	{
		searchList[idx].x = RandomFloat( minS, maxS );
		searchList[idx].y = RandomFloat( minS, maxS );
		searchList[idx].z = RandomFloat( minS, maxS );
		searchList[idx].w = RandomFloat( minS, maxS );
	}

	// Generate Random Query Points
	minQ = 0.0f;
	maxQ = 1.0f;
	for (idx = 0; idx < nQuery; idx++)
	{
		queryList[idx].x = RandomFloat( minQ, maxQ );
		queryList[idx].y = RandomFloat( minQ, maxQ );
		queryList[idx].z = RandomFloat( minQ, maxQ );
		queryList[idx].w = RandomFloat( minQ, maxQ );
	}

	// Initialize Result set to invalid results
	for (idx = 0; idx < totalResult; idx++)
	{
		resultList[idx].Id   = 0xFFFFFFFFu;
		resultList[idx].Dist = 0.0f;
	}

	//---------------------------------------------------
	// QNN <2D, 3D, and 4D>, increasing size test
	//---------------------------------------------------

	nn = NN_QNN;
	nK = 1;
	nnStr = nnStrings[nn];

	//nD = 2;
	for (nD = 2; nD <= 2; nD++)
	{
#if 0
		// Test n = 1 (# search, # queries, # results)
			// ceil(log2(n)) = 2
		if (QNN_STACK_SIZE >= 2)
		{
			nS = 1;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10 (# search, # queries, # results)
			// ceil(log2(n)) = 4
		if (QNN_STACK_SIZE >= 4)
		{
			nS = 10;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 100 (# search, # queries, # results)
			// ceil(log2(n)) = 7
		if (QNN_STACK_SIZE >= 7)
		{
			nS = 100;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}
#endif

		// Test n = 1,000 (# search, # queries, # results)
			// ceil(log2(n)) = 11
		if (QNN_STACK_SIZE >= 11)
		{
			nS = 1000;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000 (# search, # queries, # results)
			// ceil(log2(n)) = 14
		if (QNN_STACK_SIZE >= 14)
		{
			nS = 10000;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 100,000 (# search, # queries, # results)
			// ceil(log2(n)) = 17
		if (QNN_STACK_SIZE >= 17)
		{
			nS = 100000;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 20
		if (QNN_STACK_SIZE >= 20)
		{
			nS = 1000000;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

#if 0
		// Test n = 10,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 24
		if (QNN_STACK_SIZE >= 24)
		{
			nS = 10000000;
			nQ = nS;
			nR = nS;
			printf( "\n%s %uD - #search = %u, #query (#results) = %u\n\n", nnStr, nD, nS, nQ );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}
#endif
	}

#if 0

	//---------------------------------------------------
	// All-NN <2D, 3D, and 4D>, increasing size test
	//---------------------------------------------------

	nn = NN_ALL_NN;
	nQ = 0;
	nnStr = nnStrings[nn];

	//nD = 2;
	for (nD = 2; nD <= 4; nD++)
	{
		// Test n = 1 (# search, # queries, # results)
			// ceil(log2(n)) = 2
		if (ALL_NN_STACK_SIZE >= 2)
		{
			nS = 1;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10 (# search, # queries, # results)
			// ceil(log2(n)) = 4
		if (ALL_NN_STACK_SIZE >= 4)
		{
			nS = 10;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 100 (# search, # queries, # results)
			// ceil(log2(n)) = 7
		if (ALL_NN_STACK_SIZE >= 7)
		{
			nS = 100;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000 (# search, # queries, # results)
			// ceil(log2(n)) = 11
		if (ALL_NN_STACK_SIZE >= 11)
		{
			nS = 1000;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000 (# search, # queries, # results)
			// ceil(log2(n)) = 14
		if (ALL_NN_STACK_SIZE >= 14)
		{
			nS = 10000;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 100,000 (# search, # queries, # results)
			// ceil(log2(n)) = 17
		if (ALL_NN_STACK_SIZE >= 17)
		{
			nS = 100000;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 20
		if (ALL_NN_STACK_SIZE >= 20)
		{
			nS = 1000000;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 24
		if (ALL_NN_STACK_SIZE >= 24)
		{
			nS = 10000000;
			nR = nS;
			printf( "\n%s %uD - (#search, #query, and #results) = %u\n\n", nnStr, nD, nS );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}
	}

	//---------------------------------------------------
	// Special Test
	//---------------------------------------------------

	// Special test (50,000 4D kNN points, k=15)
		// ceil(log2(n) = 16
	if (KNN_STACK_SIZE >= 16)
	{
		nn = NN_KNN;
		nD = 4;
		nK = 15;		
		nS = 50000;
		nQ = 50000;
		nR = nQ * nK;
		printf( "\n%s %uD - #search = %u, #query (#results) = %u, k = %u\n\n", nnStr, nD, nS, nQ, nK );
		bResult = NN_LBT_HOST( nn, nD, nK, 
							   nS, searchList, 
							   nQ, queryList, 
							   nR, resultList );
		if (! bResult)
		{
			goto lblCleanup;
		}
	}

	//---------------------------------------------------
	// kNN <2D, 3D, and 4D>, increasing size test
	//---------------------------------------------------

	nn = NN_KNN;
	nK = 32;
	nnStr = nnStrings[nn];

	//nD = 2;
	for (nD = 2; nD <= 4; nD++)
	{
		// Test n = 100 (# search, # queries, # results)
			// ceil(log2(n)) = 7
		if (KNN_STACK_SIZE >= 7)
		{
			nS = 100;
			nQ = 100;
			nR = nQ * nK;
			printf( "\n%s %uD - #search = %u, #query = %u, #results = %u, k = %u\n\n", 
				    nnStr, nD, nS, nQ, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   //nQ, queryList, 
								   nS, searchList,
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000 (# search, # queries, # results)
			// ceil(log2(n)) = 11
		if (KNN_STACK_SIZE >= 11)
		{
			nS = 1000;
			nQ = 1000;
			nR = nQ * nK;
			printf( "\n%s %uD - #search = %u, #query = %u, #results = %u, k = %u\n\n", 
				    nnStr, nD, nS, nQ, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000 (# search, # queries, # results)
			// ceil(log2(n)) = 14
		if (KNN_STACK_SIZE >= 14)
		{
			nS = 10000;
			nQ = 10000;
			nR = nQ * nK;
			printf( "\n%s %uD - #search = %u, #query = %u, #results = %u, k = %u\n\n", 
				    nnStr, nD, nS, nQ, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 100,000 (# search, # queries, # results)
			// ceil(log2(n)) = 17
		if (KNN_STACK_SIZE >= 17)
		{
			nS = 100000;
			nQ = 100000;
			nR = nQ * nK;
			printf( "\n%s %uD - #search = %u, #query = %u, #results = %u, k = %u\n\n", 
				    nnStr, nD, nS, nQ, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 20
		if (KNN_STACK_SIZE >= 20)
		{
			nS = 1000000;
			nQ = 1000000;
			nR = nQ * nK;
			printf( "\n%s %uD - #search = %u, #query = %u, #results = %u, k = %u\n\n", 
				    nnStr, nD, nS, nQ, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 24
		if (KNN_STACK_SIZE >= 24)
		{
			nK = 8;		// We don't have enough memory for both k=32 and Stack Depth = 26
			nS = 10000000;
			nQ = 10000000;
			nR = nQ * nK;
			printf( "\n%s %uD - #search = %u, #query = %u, #results = %u, k = %u\n\n", 
				    nnStr, nD, nS, nQ, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, queryList, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}
	}

	//---------------------------------------------------
	// All-kNN <2D, 3D, and 4D>, increasing size test
	//---------------------------------------------------

	nn = NN_ALL_KNN;
	nK = 32;
	nQ = 0;
	nnStr = nnStrings[nn];

	//nD = 2;
	for (nD = 2; nD <= 4; nD++)
	{
		// Test n = 100 (# search, # queries, # results)
			// ceil(log2(n)) = 7
		if (ALL_KNN_STACK_SIZE >= 7)
		{
			nS = 100;
			nR = nS * nK;
			printf( "\n%s %uD - #search (#query) = %u, #results = %u, k = %u\n\n", nnStr, nD, nS, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000 (# search, # queries, # results)
			// ceil(log2(n)) = 11
		if (ALL_KNN_STACK_SIZE >= 11)
		{
			nS = 1000;
			nR = nS * nK;
			printf( "\n%s %uD - #search (#query) = %u, #results = %u, k = %u\n\n", nnStr, nD, nS, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000 (# search, # queries, # results)
			// ceil(log2(n)) = 14
		if (ALL_KNN_STACK_SIZE >= 14)
		{
			nS = 10000;
			nR = nS * nK;
			printf( "\n%s %uD - #search (#query) = %u, #results = %u, k = %u\n\n", nnStr, nD, nS, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK,
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 100,000 (# search, # queries, # results)
			// ceil(log2(n)) = 17
		if (ALL_KNN_STACK_SIZE >= 17)
		{
			nS = 100000;
			nR = nS * nK;
			printf( "\n%s %uD - #search (#query) = %u, #results = %u, k = %u\n\n", nnStr, nD, nS, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 1,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 20
		if (ALL_KNN_STACK_SIZE >= 20)
		{
			nS = 1000000;
			nR = nS * nK;
			printf( "\n%s %uD - #search (#query) = %u, #results = %u, k = %u\n\n", nnStr, nD, nS, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}

		// Test n = 10,000,000 (# search, # queries, # results)
			// ceil(log2(n)) = 24
		if (ALL_KNN_STACK_SIZE >= 24)
		{
			nK = 8;		// We don't have enough memory for both k=32 and Stack Depth = 26
			nS = 10000000;
			nR = nS * nK;
			printf( "\n%s %uD - #search (#query) = %u, #results = %u, k = %u\n\n", nnStr, nD, nS, nR, nK );
			bResult = NN_LBT_HOST( nn, nD, nK, 
								   nS, searchList, 
								   nQ, NULL, 
								   nR, resultList );
			if (! bResult)
			{
				goto lblCleanup;
			}
		}
	}
#endif

	// Success
lblCleanup:
	//
	// Cleanup
	//

		// Cleanup resultList
	if (NULL != queryList)
	{
		CPU_NN_Result * tempList = resultList;
		resultList = NULL;
		free( tempList );
	}

		// Cleanup queryList
	if (NULL != queryList)
	{
		float4 * tempList = queryList;
		queryList = NULL;
		free( tempList );
	}

		// Cleanup searchList
	if (NULL != searchList)
	{
		float4 * tempList = searchList;
		searchList = NULL;
		free( tempList );
	}

	return bResult;
}

