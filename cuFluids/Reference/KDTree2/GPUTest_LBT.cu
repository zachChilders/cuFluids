/*-----------------------------------------------------------------------------
  File:  GPUTest_LBT.cpp
  Desc:  Host CPU scaffolding for running and testing kd-tree NN searches
         for left-balanced binary tree array layouts

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
#include "CPUTree_API.h"
#include "GPUTree_API.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

extern AppGlobals g_app;


/*-------------------------------------
	CUDA Kernels
-------------------------------------*/

//#include <GPU_QNN2.cu>    // QNN kernel (register solution, 3x slower than shared memory)

// Left Balanced Binary Tree Layout GPU Kernels
	// root = 1
	// Given current node at 'i'
		// currIdx   = i
		// parentIdx = i/2
		// leftIdx   = i*2
		// rightIdx  = i*2+1
#if (APP_TEST == TEST_KD_QNN)
//	#include <GPU_QNN_LBT.cu>    // QNN kernel (shared memory solution)
#elif (APP_TEST == TEST_KD_ALL_NN)
//	#include <GPU_ALL_NN_LBT.cu>  // All-NN kernel
#elif (APP_TEST == TEST_KD_KNN)
//	#include <GPU_KNN_LBT.cu>    // kNN kernel
#elif (APP_TEST == TEST_KD_ALL_KNN)
//	#include <GPU_ALL_KNN_LBT.cu> // All-kNN Kernel
#else
#endif



/*-------------------------------------
  Function Declarations
-------------------------------------*/

/*---------------------------------------------------------
  Name:	CPUTest_2D_LBT()
---------------------------------------------------------*/

bool CPUTest_2D_LBT( AppGlobals & g )
{
	bool bResult = true;

	/*---------------------------------
	  Step 0.  Initialize
	---------------------------------*/

	cudaError_t cuda_err = cudaSuccess;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cu_err = CUDA_SUCCESS;
#endif

	// set seed for rand()
	RandomInit( 2010 );

	g.hTimer = 0;
	cutCreateTimer( &(g.hTimer) );

	/*-------------------------------------------
	  Step 1.  Create Search & Query Vectors
	-------------------------------------------*/

	// Get # of search and query points
#if (APP_TEST == TEST_KD_QNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = TEST_NUM_QUERY_POINTS;

	bool bInitSearch = true;
	bool bInitQuery  = true;
#elif (APP_TEST == TEST_KD_ALL_NN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = g.nSearch+1;

	bool bInitSearch = true;
	bool bInitQuery  = false;
#elif (APP_TEST == TEST_KD_KNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = TEST_NUM_QUERY_POINTS;

	bool bInitSearch = true;
	bool bInitQuery  = true;
	unsigned int kVal = 64;

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#elif (APP_TEST == TEST_KD_ALL_KNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = g.nSearch+1;

	bool bInitSearch = true;
	bool bInitQuery  = false;

	unsigned int kVal = 64;
	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#else
#endif

	// Create random search and query vectors
	bool bNonUniformSearch = false;
	bool bNonUniformQuery  = false;
	int scaleType = 0;
	bResult = InitSearchQueryVectors( g, bInitSearch, bInitQuery, 
		                              bNonUniformSearch, bNonUniformQuery, scaleType );
	if (false == bResult)
	{
		// Error
		return false;
	}


	/*-------------------------------------------
	  Step 2.  Setup Initial parameters
	-------------------------------------------*/

#if (APP_TEST == TEST_KD_QNN)
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = g.nQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	// Query == Search
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = 0;				
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = g.nQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Query == Search
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = 0;		
#else
#endif

	// Compute Reasonable Thread Block and Grid Shapes
	BlockGridShape kdShape;
	
#if (APP_TEST == TEST_KD_QNN)
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigSearch+1;		// Add 1 for zeroth element
#elif (APP_TEST == TEST_KD_KNN)
	kdShape.threadsPerRow = KNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	kdShape.threadsPerRow = KNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigSearch+1;		// Add 1 for zeroth element
#else
#endif
	bResult = ComputeBlockShapeFromQueryVector( kdShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

#if (APP_TEST == TEST_KD_QNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch + 1;		// Add 1 for zeroth element

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch + 1;		// Add 1 for zeroth element
#elif (APP_TEST == TEST_KD_ALL_KNN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;
#else
#endif

	//
	// Print out Initialization Parameters
	//
	DumpBlockGridShape( kdShape );

#if (APP_TEST == TEST_KD_QNN)
	printf( "\n QNN 2D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_NN)
	printf( "\n All-NN 2D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_KNN)
	printf( "\n kNN 2D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "\n All-kNN 2D Left-balanced Test\n\n" );
#else
#endif

	printf( "# Requested Search Points  = %d\n", nOrigSearch );
	printf( "# Padded Search Points     = %d\n", nPadSearch );
	printf( "# Requested Query Points   = %d\n", nOrigQuery );
	printf( "# Padded Query Points      = %d\n", nPadQuery );
#if (APP_TEST == TEST_KD_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#else
#endif


	// Make sure Matrix + vector is not to big to use up all device memory
		// 1 GB on Display Card
#if (APP_TEST == TEST_KD_QNN)
	unsigned int sizeResults = nPadQuery * sizeof(GPU_NN_Result);
	unsigned int sizeQueries = nPadQuery * sizeof(float2);
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int sizeResults = nPadSearch * sizeof(GPU_NN_Result);
	unsigned int sizeQueries = 0;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int sizeResults = nPadQuery * sizeof(GPU_NN_Result) * kVal;
	unsigned int sizeQueries = nPadQuery * sizeof(float2);
#elif (APP_TEST == TEST_KD_ALL_KNN)
	unsigned int sizeResults = nPadSearch * sizeof(GPU_NN_Result) * kVal;
	unsigned int sizeQueries = 0;
#else
#endif
	unsigned int sizeNodes   = nPadSearch * sizeof(GPUNode_2D_LBT);
	unsigned int sizeIDs	 = nPadSearch * sizeof(unsigned int);
	unsigned int totalMem    = sizeNodes + sizeIDs + sizeQueries + sizeResults;

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g.cudaProps.totalGlobalMem)
	{
		printf( "KD Tree Inputs (%d) are too large for available device memory (%d), running test will crash...\n",
				totalMem, g.cudaProps.totalGlobalMem );
		printf( "\tsizeNodes = %d\n", sizeNodes );
		printf( "\tsizeIDs   = %d\n", sizeIDs );
		printf( "\tsizeQueries = %d\n", sizeQueries );
		printf( "\tsizeResults = %d\n", sizeResults );
		return bResult;
	}

	printf( "# Onto Memory       = %d\n", sizeNodes + sizeIDs + sizeQueries );
	printf( "# From Memory       = %d\n", sizeResults );
	printf( "# Total Memory      = %d\n", totalMem );

	// Setup GPU Kernel execution parameters
		// KDTree Distance
	dim3 qryThreads( kdShape.threadsPerRow, kdShape.rowsPerBlock, 1 );
	dim3 qryGrid( kdShape.blocksPerRow, kdShape.rowsPerGrid, 1 );

	float KD_CPU_build        = 0.0f;
	float KD_GPU_copy_nodes   = 0.0f;
	float KD_GPU_onto_device  = 0.0f;
	float KD_GPU_from_device  = 0.0f;
	float KD_GPU_dist		  = 0.0f;
	float KD_CPU_dist		  = 0.0f;
	bool  checkDistResults    = true;
	//unsigned int maxNodes     = 0;
	//double avgNodes           = 0.0;


	/*-------------------------------------------
	  Step 3.  Allocate GPU Vectors
	-------------------------------------------*/

	// Allocate host memory for GPU KD Tree Nodes
	unsigned int mem_size_KDNodes = nPadSearch * sizeof(GPUNode_2D_LBT);
	GPUNode_2D_LBT * h_KDNodes = NULL;	
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_KDNodes > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_KDNodes, mem_size_KDNodes );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_KDNodes > 0)
	{
		h_KDNodes = (GPUNode_2D_LBT*) malloc( mem_size_KDNodes );
		if (NULL == h_KDNodes)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU KD Tree Nodes
	GPUNode_2D_LBT* d_KDNodes = NULL;
	if (mem_size_KDNodes > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_KDNodes, mem_size_KDNodes ) );
	}


	// Allocate host memory for GPU Node ID's
	unsigned int mem_size_IDs = nPadSearch * sizeof(unsigned int);
	unsigned int* h_IDs = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_IDs > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_IDs, mem_size_IDs );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_IDs > 0)
	{
		h_IDs = (unsigned int*) malloc( mem_size_IDs );
		if (NULL == h_IDs)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU Node ID's
	unsigned int* d_IDs = NULL;
	if (mem_size_IDs > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_IDs, mem_size_IDs ) );
	}


	// Allocate host memory for GPU query points 
	unsigned int mem_size_Query = nPadQuery * sizeof(float2);
	float2* h_Queries = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_Query > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_Queries, mem_size_Query );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_Query > 0)
	{
		h_Queries = (float2*) malloc( mem_size_Query );
		if (NULL == h_Queries)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU query points 
	float2* d_Queries = NULL;
	if (mem_size_Query > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_Queries, mem_size_Query ) );
	}

	// Allocate host memory for GPU Query Results
	unsigned int nQueryElems = 0;
	unsigned int nQueryCopy  = 0;
	unsigned int mem_size_Results_GPU = 0;
	unsigned int mem_copy_Results_GPU = 0;

	// BUGBUG:  Interesting bug caused by processing zeroth element as if it was a valid entry
	//			in GPU thread block processing, Currently, we end up with 2 elements
	//			colling on the zeroth entry for results causing at least one bug in the result set
	// WORKAROUND:  increase memory allocation by one extra element
	//				 and have zeroth element point to this extra location to avoid the 2 element collision
#if (APP_TEST == TEST_KD_QNN)
	nQueryElems = nPadQuery + 1;		// +1 to work around bug
	nQueryCopy  = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	nQueryElems = nPadSearch + 1;		// +1 to work around bug
	nQueryCopy  = nOrigSearch;
#elif (APP_TEST == TEST_KD_KNN)
	nQueryElems = (nPadQuery + 1) * kVal;	// +1 to work around bug
	nQueryCopy  = nOrigQuery * kVal;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	nQueryElems = (nPadSearch + 1) * kVal;	// +1 to work around bug
	nQueryCopy  = nOrigSearch * kVal;
#else
#endif
	mem_size_Results_GPU = nQueryElems * sizeof(GPU_NN_Result);
	mem_copy_Results_GPU = nQueryCopy * sizeof(GPU_NN_Result);

	GPU_NN_Result* h_Results_GPU = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_Results_GPU > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_Results_GPU, mem_size_Results_GPU );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_Results_GPU > 0)
	{
		h_Results_GPU = (GPU_NN_Result*) malloc( mem_size_Results_GPU );
		if (NULL == h_Results_GPU)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU query results
	GPU_NN_Result* d_Results_GPU = NULL;
	if (mem_size_Results_GPU > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_Results_GPU, mem_size_Results_GPU ) );
	}


	// Allocate host memory for CPU Query Results
	unsigned int mem_size_Results_CPU = nQueryElems * sizeof(CPU_NN_Result);
	CPU_NN_Result* h_Results_CPU = NULL;
	if (mem_size_Results_CPU > 0)
	{
		h_Results_CPU = (CPU_NN_Result*) malloc( mem_size_Results_CPU );
	}


	// Copy Query List
	if (NULL != h_Queries)
	{
		unsigned int qryIdx;
	
		// Copy actual queries 
		const float4 * queryList = (const float4 *)(g.queryList);
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			h_Queries[qryIdx].x = queryList[qryIdx].x;
			h_Queries[qryIdx].y = queryList[qryIdx].y;
		}

		// Create some extra queries for thread block alignment
		for (qryIdx = nOrigQuery; qryIdx < nPadQuery; qryIdx++)
		{
			// Just repeat the first query a few times
			h_Queries[qryIdx].x = queryList[0].x;
			h_Queries[qryIdx].y = queryList[0].y;
		}
	}


	/*-----------------------
	  Step 4. Build kd-tree
	-----------------------*/

	// Dump search List (for debugging)
#if 0
	unsigned int pointIdx;
	double currX, currY;
	for (pointIdx = 0; pointIdx < nOrigSearch; pointIdx++)
	{
		currX = g.searchList[pointIdx].x;
		currY = g.searchList[pointIdx].y;

		printf( "PointID[%u] = <%3.6f, %3.6f>\n", pointIdx, currX, currY );
	}
#endif

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Build left-balanced KDTree (on CPU)
	void * kdTree = NULL;
	const float4 * searchList = (const float4 *)g.searchList;
	bResult = BUILD_CPU_2D_LBT( &kdTree, nOrigSearch, searchList );
	if (false == bResult) 
	{
		// Error - goto cleanup
		return false;
	}	

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_CPU_build += cutGetTimerValue( g.hTimer );
	}

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Copy kd-tree from CPU to GPU
		// Note:  We have the zeroth element point to nPadSearch + 1 to work around a bug
	bResult = COPY_NODES_2D_LBT( kdTree, nOrigSearch, nPadSearch, (void*)h_KDNodes, h_IDs );
	if (!bResult)
	{
		// Error - goto cleanup
		return false;
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_GPU_copy_nodes += cutGetTimerValue( g.hTimer );
	}

	// Dump LBT Nodes and IDs (for debugging)
#if 0
	unsigned int nodeIdx;
	for (nodeIdx = 0; nodeIdx < nPadSearch; nodeIdx++)
	{
		currX    = (double)(h_KDNodes[nodeIdx].pos[0]);
		currY    = (double)(h_KDNodes[nodeIdx].pos[1]);
		pointIdx = h_IDs[nodeIdx];
		if (nodeIdx == 0)
		{
			// zeroth node
			printf( "ZERO:   NodeID[%u] = Point ID[%u] <%3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY );
		}
		else if (nodeIdx > nOrigSearch)
		{
			// Extra padded node for searching 
			printf( "Extra:  NodeID[%u] = Point ID[%u] <%3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY );
		}
		else if (pointIdx > nOrigSearch)
		{
			// Possible error
			printf( "ERROR:  NodeID[%u] = Point ID[%u] <%3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY );
		}
		else
		{
			// Normal behavior
			printf( "Normal: NodeID[%u] = Point ID[%u] <%3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY );
		}
	}
#endif



// Profile Measurement Loop
unsigned int currIter;
for (currIter = 0; currIter < g.profileActualLoops; currIter++)
{

	/*-------------------------------------------------------
	  Step 5.  Move Input Vectors 
	           from main memory to device memory
	-------------------------------------------------------*/

	float elapsedTime;
	cudaEvent_t moveOntoStart, moveOntoStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &moveOntoStart );
		cudaEventCreate( &moveOntoStop );

		// Start Timer
		cudaEventRecord( moveOntoStart, 0 );
	}

	// Copy 'KDNodes' vector from host memory to device memory
	if ((NULL != d_KDNodes) && (NULL != h_KDNodes))
	{
		cutilSafeCall( cudaMemcpy( d_KDNodes, h_KDNodes, mem_size_KDNodes, cudaMemcpyHostToDevice ) );
	}

	// Copy 'IDs' vector from host memory to device memory
	if ((NULL != d_IDs) && (NULL != h_IDs))
	{
		cutilSafeCall( cudaMemcpy( d_IDs, h_IDs, mem_size_IDs, cudaMemcpyHostToDevice ) );
	}

	// Copy 'Query Points' vector from host memory to device memory
	if ((NULL != d_Queries) && (NULL != h_Queries))
	{
		cutilSafeCall( cudaMemcpy( d_Queries, h_Queries, mem_size_Query, cudaMemcpyHostToDevice ) );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( moveOntoStop, 0 );
		cudaEventSynchronize( moveOntoStop );
		cudaEventElapsedTime( &elapsedTime, moveOntoStart, moveOntoStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_onto_device += elapsedTime;
			}
		}
		else
		{
			KD_GPU_onto_device += elapsedTime;
		}
	}


	/*-------------------------------------------------------
	  Step 6.  Call KDTree GPU Kernel
	-------------------------------------------------------*/

	cudaEvent_t searchStart, searchStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &searchStart );
		cudaEventCreate( &searchStop );

		// Start Timer
		cudaEventRecord( searchStart, 0 );
	}

	
	// Check if GPU kernel execution generated an error
#if (APP_TEST == TEST_KD_QNN)
	// Call 2D GPU KDTree QNN Kernel (single result per query)
	GPU_QNN_2D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, nOrigSearch, wQuery  );

#elif (APP_TEST == TEST_KD_ALL_NN)
	// Call 2D GPU KDTree ALL Kernel (single result per query)
	GPU_ALL_NN_2D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											      d_IDs, nOrigSearch, wQuery  );

#elif (APP_TEST == TEST_KD_KNN)
	// Call GPU KDTree kNN Kernel ('k' results per query)
	GPU_KNN_2D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, nOrigSearch, kVal  );

#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Call GPU KDTree kNN Kernel ('k' results per query)
	GPU_ALL_KNN_2D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											       d_IDs, nOrigSearch, kVal  );
#else
#endif

	cuda_err = cudaGetLastError();
	if( cudaSuccess != cuda_err) 
	{
		fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
		         "NN 2D left-balanced GPU kernel failed", __FILE__, __LINE__, 
				 cudaGetErrorString( cuda_err ) );
		exit( EXIT_FAILURE );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( searchStop, 0 );
		cudaEventSynchronize( searchStop );
		cudaEventElapsedTime( &elapsedTime, searchStart, searchStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_dist += elapsedTime;
			}
		}
		else
		{
			KD_GPU_dist += elapsedTime;
		}
	}

	/*-------------------------------------------------
	  Step 7.  Copy Outputs
	           from device memory to main memory
	-------------------------------------------------*/

	cudaEvent_t moveFromStart, moveFromStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &moveFromStart );
		cudaEventCreate( &moveFromStop );

		// Start Timer
		cudaEventRecord( moveFromStart, 0 );
	}

	// Copy result vector from GPU device to host CPU
	cutilSafeCall( cudaMemcpy( (void *) h_Results_GPU, d_Results_GPU, mem_copy_Results_GPU, cudaMemcpyDeviceToHost ) );

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( moveFromStop, 0 );
		cudaEventSynchronize( moveFromStop );
		cudaEventElapsedTime( &elapsedTime, moveFromStart, moveFromStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_from_device += elapsedTime;
			}
		}
		else
		{
			KD_GPU_from_device += elapsedTime;
		}
	}

	/*-------------------------------------------
	  Step 8:	Call KDTree CPU Algorithm
	-------------------------------------------*/

	if (g.doubleCheckDists)
	{
		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}

		// Determine Nearest Neighbors using KDTree
#if (APP_TEST == TEST_KD_QNN)
		bResult = CPU_QNN_2D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_ALL_NN)
		bResult = CPU_ALL_NN_2D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_KNN)
		bResult = CPU_KNN_2D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadQuery,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#elif (APP_TEST == TEST_KD_ALL_KNN)
		bResult = CPU_ALL_KNN_2D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadSearch,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#else
		bResult = true;
#endif
		if (false == bResult)
		{
			// Error
			return false;
		}

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					KD_CPU_dist += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				KD_CPU_dist += cutGetTimerValue( g.hTimer );
			}
		}
	}

} // End Profile Loop

	/*-------------------------------------------------
	  Step 9:  Double check GPU result 
			   against CPU result
	-------------------------------------------------*/

	if ((g.doubleCheckDists) && 
		(NULL != h_Results_GPU) && 
		(NULL != h_Results_CPU))
	{
		//double totalNodes = 0.0;
		//maxNodes = 0;

#if (APP_TEST == TEST_KD_QNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		float eps = 1.0e-4f;
		unsigned int qryIdx;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			float gpuDist = h_Results_GPU[qryIdx].Dist;
			float cpuDist = h_Results_CPU[qryIdx].Dist;
			unsigned int gpuID = h_Results_GPU[qryIdx].Id;
			unsigned int cpuID = h_Results_CPU[qryIdx].Id;

			//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			// Need to use a fuzzy compare on distance
			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g> - GPU KD[%5d %.9g] != CPU KD[%5d %.9g] !!!\n",
						 qryIdx, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#elif (APP_TEST == TEST_KD_ALL_NN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		float eps = 1.0e-4f;
		unsigned int qryIdx;
		for (qryIdx = 0; qryIdx < nOrigSearch; qryIdx++)
		{
			unsigned int gpuID = h_Results_GPU[qryIdx].Id;
			float gpuDist      = h_Results_GPU[qryIdx].Dist;

			unsigned int cpuID = h_Results_CPU[qryIdx].id;
			float cpuDist      = h_Results_CPU[qryIdx].dist;

			//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			// Need to use a fuzzy compare on distance
			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( g.searchList[qryIdx].x );
				double qY = static_cast<double>( g.searchList[qryIdx].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g> - GPU KD[%5u %.9g] != CPU KD[%5u %.9g] !!!\n",
						 qryIdx, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#elif (APP_TEST == TEST_KD_KNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		/*
		unsigned int qryIdx;
		unsigned int kIdx;
		//float eps = 1.0e-4;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			// Check each Nearest Neighbor
			for (kIdx = 0; kIdx < kVal; kIdx++)
			{
				unsigned int baseIdx = (kIdx * nPadQuery) + qryIdx;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 qryIdx, kIdx, 
						 qX, qY, 
						 gpuID, gpu_D, 
						 cpuID, cpu_D );
			}
		}
		*/
#elif (APP_TEST == TEST_KD_ALL_KNN)
		checkDistResults = true;
		/*
		// Check each query result (GPU vs. CPU)
		unsigned int qryIdx;
		unsigned int kIdx;
		//float eps = 1.0e-4;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			// Check each Nearest Neighbor
			for (kIdx = 0; kIdx < kVal; kIdx++)
			{
				unsigned int baseIdx = (kIdx * nPadQuery) + qryIdx;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 qryIdx, kIdx, 
						 qX, qY, 
						 gpuID, gpu_D, 
						 cpuID, cpu_D );
			}
		}
		*/
#else
	// Do nothing for now ...
#endif
		// Get Average Nodes Visited Per Query
		//avgNodes = totalNodes/(double)nOrigQuery;
	}


	/*--------------------------------------------------------
	  Step 10: Print out Results
	--------------------------------------------------------*/
	
	/*
	// GPU Results
	printf( "GPU Results\n" );
	unsigned int qryIdx;
	for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
	{
		double gpuDist = static_cast<double>( h_Results_GPU[qryIdx].Dist );
		unsigned int gpuID = h_Results_GPU[qryIdx].Id;
		//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", qryIdx, gpuID, gpuDist, gpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", i, gpuID, gpuDist );
	}

	// CPU Results
	printf( "CPU Results\n" );
	for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
	{
		double cpuDist = static_cast<double>( h_Results_CPU[qryIdx].dist );
		unsigned int cpuID = h_Results_CPU[qryIdx].id;
		//unsigned int cpuCount = h_Results_CPU[qryIdx].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", qryIdx, cpuID, cpuDist, cpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", qryIdx, cpuID, cpuDist );
	}
	*/


	if (g.doubleCheckDists)
	{
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


	/*--------------------------------------------------------
	  Step 11: Print out Profile Statistics
	--------------------------------------------------------*/

	if (g.profile)
	{
		// Dump Profile Statistics
		if (g.profileActualLoops > 1)
		{
			float loops = (float)g.profileActualLoops;
			float o_l = 1.0f / loops;

			float avgOnto    = KD_GPU_onto_device * o_l;
			float avgFrom    = KD_GPU_from_device * o_l;
			float avgGPUdist = KD_GPU_dist * o_l;
			float avgCPUdist = KD_CPU_dist * o_l;
			float avgBuild   = KD_CPU_build;
			float avgCopy    = KD_GPU_copy_nodes;

			// Verbose
			printf( "Number of total iterations = %f.\n", loops );
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", avgBuild );
			printf( "KD - Copy kd-nodes from CPU to GPU,   time: %f msecs.\n", avgCopy );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", avgOnto );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", avgFrom );
			printf( "KD - GPU Kernel computation,          time: %f msecs.\n", avgGPUdist );
			printf( "KD - CPU Kernel computation,          time: %f msecs.\n", avgCPUdist );

			// Terse
			//printf( "KD - In, Out, G_D, C_D, C_B\n" );
			//printf( "     %f, %f, %f, %f, %f\n\n", avgOnto, avgFrom, avgGPUdist, avgCPUdist, avgBuild );
		}
		else
		{
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", KD_CPU_build );
			printf( "KD - Copy kd-nodes from CPU to GPU,   time: %f msecs.\n", KD_GPU_copy_nodes );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", KD_GPU_onto_device );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", KD_GPU_from_device );
			printf( "KD - GPU Kernel computation,        time: %f msecs.\n", KD_GPU_dist );
			printf( "KD - CPU Kernel computation,        time: %f msecs.\n", KD_CPU_dist );
		}
	}


	/*--------------------------------------------------------
	  Step 13: Cleanup Resources
	--------------------------------------------------------*/


	printf( "Shutting Down...\n" );

	// cleanup CUDA Timer
	cutDeleteTimer( g.hTimer );

	// clean up allocations
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (NULL != h_KDNodes)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_KDNodes ) );
	}
	if (NULL != h_IDs)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_IDs ) );
	}
	if (NULL != h_Queries)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_Queries ) );
	}
	if (NULL != h_Results_GPU)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_Results_GPU ) );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (NULL != h_KDNodes)
	{
		free( h_KDNodes );
	}
	if (NULL != h_IDs)
	{
		free( h_IDs );
	}
	if (NULL != h_Queries)
	{
		free( h_Queries );
	}
	if (NULL != h_Results_CPU)
	{
		free( h_Results_GPU );
	}
#else
#endif
	if (NULL != h_Results_CPU)
	{
		free( h_Results_CPU );
	}

	cutilSafeCall( cudaFree( d_KDNodes ) );
	cutilSafeCall( cudaFree( d_IDs ) );
	cutilSafeCall( cudaFree( d_Queries ) );
	cutilSafeCall( cudaFree( d_Results_GPU ) );

	FINI_CPU_2D_LBT( &kdTree );

	FiniSearchQueryVectors( g );

	printf( "Shutdown done...\n\n" );

	// Success
	return true;
}



/*---------------------------------------------------------
  Name:	CPUTest_3D_LBT()
---------------------------------------------------------*/

bool CPUTest_3D_LBT( AppGlobals & g )
{
	bool bResult = true;

	/*---------------------------------
	  Step 0.  Initialize
	---------------------------------*/

	cudaError_t cuda_err = cudaSuccess;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cu_err = CUDA_SUCCESS;
#endif

	// set seed for rand()
	RandomInit( 2010 );

	g.hTimer = 0;
	cutCreateTimer( &(g.hTimer) );

	/*-------------------------------------------
	  Step 1.  Create Search & Query Vectors
	-------------------------------------------*/

	// Get # of search and query points
#if (APP_TEST == TEST_KD_QNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = TEST_NUM_QUERY_POINTS;

	bool bInitSearch = true;
	bool bInitQuery  = true;
#elif (APP_TEST == TEST_KD_ALL_NN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = g.nSearch;		

	bool bInitSearch = true;
	bool bInitQuery  = false;
#elif (APP_TEST == TEST_KD_KNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = TEST_NUM_QUERY_POINTS;

	bool bInitSearch = true;
	bool bInitQuery  = true;
	unsigned int kVal = 64;			

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#elif (APP_TEST == TEST_KD_ALL_KNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = g.nSearch;

	bool bInitSearch = true;
	bool bInitQuery  = false;

	unsigned int kVal = 64;
	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#else
#endif

	// Create random search and query vectors
	bool bNonUniformSearch = false;
	bool bNonUniformQuery  = false;
	int scaleType = 0;
	bResult = InitSearchQueryVectors( g, bInitSearch, bInitQuery, 
		                              bNonUniformSearch, bNonUniformQuery, scaleType );
	if (false == bResult)
	{
		// Error
		return false;
	}


	/*-------------------------------------------
	  Step 2.  Setup Initial parameters
	-------------------------------------------*/

#if (APP_TEST == TEST_KD_QNN)
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = g.nQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	// Query == Search
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = 0;				
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = g.nQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Query == Search
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = 0;		
#else
#endif

	// Compute Reasonable Thread Block and Grid Shapes
	BlockGridShape kdShape;
	
#if (APP_TEST == TEST_KD_QNN)
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigSearch+2;		// Add 2 to account for overlap between [0,n-1] and [1,n]
#elif (APP_TEST == TEST_KD_KNN)
	kdShape.threadsPerRow = KNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	kdShape.threadsPerRow = KNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigSearch+2;		// Add 2 to account for overlap between [0,n-1] and [1,n]
#else
#endif
	bResult = ComputeBlockShapeFromQueryVector( kdShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

#if (APP_TEST == TEST_KD_QNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch + 2;		// Add 2 to account for overlap between [0,n-1] and [1,n]

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch + 2;		// Add 2 to account for overlap between [0,n-1] and [1,n]
#elif (APP_TEST == TEST_KD_ALL_KNN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;
#else
#endif

	// Print out Initialization Parameters
	DumpBlockGridShape( kdShape );

#if (APP_TEST == TEST_KD_QNN)
	printf( "\n QNN 3D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_NN)
	printf( "\n All-NN 3D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_KNN)
	printf( "\n kNN 3D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "\n All-kNN 3D Left-balanced Test\n\n" );
#else
#endif

	printf( "# Requested Search Points  = %d\n", nOrigSearch );
	printf( "# Padded Search Points     = %d\n", nPadSearch );
	printf( "# Requested Query Points   = %d\n", nOrigQuery );
	printf( "# Padded Query Points      = %d\n", nPadQuery );
#if (APP_TEST == TEST_KD_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#else
#endif


	// Make sure Matrix + vector is not to big to use up all device memory
		// 1 GB on Display Card
#if (APP_TEST == TEST_KD_QNN)
	unsigned int sizeResults = nPadQuery * sizeof(GPU_NN_Result);
	unsigned int sizeQueries = nPadQuery * sizeof(float4);
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int sizeResults = nPadSearch * sizeof(GPU_NN_Result);
	unsigned int sizeQueries = 0;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int sizeResults = nPadQuery * sizeof(GPU_NN_Result) * kVal;
	unsigned int sizeQueries = nPadQuery * sizeof(float4);
#elif (APP_TEST == TEST_KD_ALL_KNN)
	unsigned int sizeResults = nPadSearch * sizeof(GPU_NN_Result) * kVal;
	unsigned int sizeQueries = 0;
#else
#endif
	unsigned int sizeNodes   = nPadSearch * sizeof(GPUNode_3D_LBT);
	unsigned int sizeIDs	 = nPadSearch * sizeof(unsigned int);
	unsigned int totalMem    = sizeNodes + sizeIDs + sizeQueries + sizeResults;

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g.cudaProps.totalGlobalMem)
	{
		printf( "KD Tree Inputs (%d) are too large for available device memory (%d), running test will crash...\n",
				totalMem, g.cudaProps.totalGlobalMem );
		printf( "\tsizeNodes = %d\n", sizeNodes );
		printf( "\tsizeIDs   = %d\n", sizeIDs );
		printf( "\tsizeQueries = %d\n", sizeQueries );
		printf( "\tsizeResults = %d\n", sizeResults );
		return bResult;
	}

	printf( "# Onto Memory       = %d\n", sizeNodes + sizeIDs + sizeQueries );
	printf( "# From Memory       = %d\n", sizeResults );
	printf( "# Total Memory      = %d\n", totalMem );

	// Setup GPU Kernel execution parameters
		// KDTree Distance
	dim3 qryThreads( kdShape.threadsPerRow, kdShape.rowsPerBlock, 1 );
	dim3 qryGrid( kdShape.blocksPerRow, kdShape.rowsPerGrid, 1 );

	float KD_CPU_build        = 0.0f;
	float KD_GPU_copy_nodes   = 0.0f;
	float KD_GPU_onto_device  = 0.0f;
	float KD_GPU_from_device  = 0.0f;
	float KD_GPU_dist		  = 0.0f;
	float KD_CPU_dist		  = 0.0f;
	bool  checkDistResults    = true;
	//unsigned int maxNodes     = 0;
	//double avgNodes           = 0.0;


	/*-------------------------------------------
	  Step 3.  Allocate GPU Vectors
	-------------------------------------------*/

	// Allocate host memory for GPU KD Tree Nodes
	unsigned int mem_size_KDNodes = nPadSearch * sizeof(GPUNode_3D_LBT);
	GPUNode_3D_LBT * h_KDNodes = NULL;	
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_KDNodes > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_KDNodes, mem_size_KDNodes );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_KDNodes > 0)
	{
		h_KDNodes = (GPUNode_3D_LBT*) malloc( mem_size_KDNodes );
		if (NULL == h_KDNodes)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU KD Tree Nodes
	GPUNode_3D_LBT* d_KDNodes = NULL;
	if (mem_size_KDNodes > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_KDNodes, mem_size_KDNodes ) );
	}


	// Allocate host memory for GPU Node ID's
	unsigned int mem_size_IDs = nPadSearch * sizeof(unsigned int);
	unsigned int* h_IDs = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_IDs > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_IDs, mem_size_IDs );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_IDs > 0)
	{
		h_IDs = (unsigned int*) malloc( mem_size_IDs );
		if (NULL == h_IDs)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU Node ID's
	unsigned int* d_IDs = NULL;
	if (mem_size_IDs > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_IDs, mem_size_IDs ) );
	}


	// Allocate host memory for GPU query points 
	unsigned int mem_size_Query = nPadQuery * sizeof(float4);
	float4* h_Queries = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_Query > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_Queries, mem_size_Query );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_Query > 0)
	{
		h_Queries = (float4*) malloc( mem_size_Query );
		if (NULL == h_Queries)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU query points 
	float4* d_Queries = NULL;
	if (mem_size_Query > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_Queries, mem_size_Query ) );
	}

	// Allocate host memory for GPU Query Results
	unsigned int nQueryElems = 0;
	unsigned int nQueryCopy  = 0;
	unsigned int mem_size_Results_GPU = 0;
	unsigned int mem_copy_Results_GPU = 0;

	// BUGBUG:  Interesting bug caused by processing zeroth element as if it was a valid entry
	//			in GPU thread block processing, Currently, we end up with 2 elements
	//			colling on the zeroth entry for results causing at least one bug in the result set
	// WORKAROUND:  increase memory allocation by one extra element
	//				 and have zeroth element point to this extra location to avoid the 2 element collision
#if (APP_TEST == TEST_KD_QNN)
	nQueryElems = nPadQuery;			
	nQueryCopy  = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	nQueryElems = nPadSearch;				
	nQueryCopy  = nOrigSearch;
#elif (APP_TEST == TEST_KD_KNN)
	nQueryElems = nPadQuery * kVal;	
	nQueryCopy  = nOrigQuery * kVal;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	nQueryElems = nPadSearch * kVal;	
	nQueryCopy  = nOrigSearch * kVal;
#else
#endif
	mem_size_Results_GPU = nQueryElems * sizeof(GPU_NN_Result);
	mem_copy_Results_GPU = nQueryCopy * sizeof(GPU_NN_Result);

	//printf( "\n#Query Elems = %d (%d bytes), #Copy Elems = %d (%d bytes)\n\n", 
	//	    nQueryElems, mem_size_Results_GPU, nQueryCopy, mem_copy_Results_GPU );

	GPU_NN_Result* h_Results_GPU = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_Results_GPU > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_Results_GPU, mem_size_Results_GPU );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_Results_GPU > 0)
	{
		h_Results_GPU = (GPU_NN_Result*) malloc( mem_size_Results_GPU );
		if (NULL == h_Results_GPU)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU query results
	GPU_NN_Result* d_Results_GPU = NULL;
	if (mem_size_Results_GPU > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_Results_GPU, mem_size_Results_GPU ) );
	}


	// Allocate host memory for CPU Query Results
	unsigned int mem_size_Results_CPU = nQueryElems * sizeof(CPU_NN_Result);
	CPU_NN_Result* h_Results_CPU = NULL;
	if (mem_size_Results_CPU > 0)
	{
		h_Results_CPU = (CPU_NN_Result*) malloc( mem_size_Results_CPU );
	}


	// Copy Query List
	if (NULL != h_Queries)
	{
		unsigned int qryIdx;
	
		// Copy actual queries 
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			h_Queries[qryIdx].x = g.queryList[qryIdx].x;
			h_Queries[qryIdx].y = g.queryList[qryIdx].y;
			h_Queries[qryIdx].z = g.queryList[qryIdx].z;
			h_Queries[qryIdx].w = 0.0f;
		}

		// Create some extra queries for thread block alignment
		float firstX = g.queryList[0].x;
		float firstY = g.queryList[0].y;
		float firstZ = g.queryList[0].z;

		for (qryIdx = nOrigQuery; qryIdx < nPadQuery; qryIdx++)
		{
			// Just repeat the first query a few times
			h_Queries[qryIdx].x = firstX;
			h_Queries[qryIdx].y = firstY;
			h_Queries[qryIdx].z = firstZ;
			h_Queries[qryIdx].w = 0.0f;
		}
	}


	/*-----------------------
	  Step 4. Build kd-tree
	-----------------------*/

	// Dump search List (for debugging)
#if 0
	unsigned int pointIdx;
	double currX, currY, currZ;
	for (pointIdx = 0; pointIdx < nOrigSearch; pointIdx++)
	{
		currX = g.searchList[pointIdx].x;
		currY = g.searchList[pointIdx].y;
		currZ = g.searchList[pointIdx].z;

		printf( "PointID[%u] = <%3.6f, %3.6f, %3.6f>\n", 
			    pointIdx, currX, currY, currZ );
	}
#endif

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Build left-balanced KDTree (on CPU)
	void * kdTree = NULL;
	bResult = BUILD_CPU_3D_LBT( &kdTree, nOrigSearch, g.searchList );
	if (false == bResult) 
	{
		// Error - goto cleanup
		return false;
	}	

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_CPU_build += cutGetTimerValue( g.hTimer );
	}

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Copy kd-tree from CPU to GPU
		// Note:  We have the zeroth element point to nPadSearch + 1 to work around a bug
	bResult = COPY_NODES_3D_LBT( kdTree, nOrigSearch, nPadSearch, (void*)h_KDNodes, h_IDs );
	if (!bResult)
	{
		// Error - goto cleanup
		return false;
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_GPU_copy_nodes += cutGetTimerValue( g.hTimer );
	}

	// Dump LBT Nodes and IDs (for debugging)
#if 0
	unsigned int nodeIdx;
	for (nodeIdx = 0; nodeIdx < nPadSearch; nodeIdx++)
	{
		currX    = (double)(h_KDNodes[nodeIdx].pos[0]);
		currY    = (double)(h_KDNodes[nodeIdx].pos[1]);
		currZ    = (double)(h_KDNodes[nodeIdx].pos[2]);
		pointIdx = h_IDs[nodeIdx];
		if (nodeIdx == 0)
		{
			// zeroth node
			printf( "ZERO:   NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ );
		}
		else if (nodeIdx > nOrigSearch)
		{
			// Extra padded node for searching 
			printf( "Extra:  NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ );
		}
		else if (pointIdx > nOrigSearch)
		{
			// Possible error
			printf( "ERROR:  NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ );
		}
		else
		{
			// Normal behavior
			printf( "Normal: NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ );
		}
	}
#endif


// Profile Measurement Loop
unsigned int currIter;
for (currIter = 0; currIter < g.profileActualLoops; currIter++)
{

	/*-------------------------------------------------------
	  Step 5.  Move Input Vectors 
	           from main memory to device memory
	-------------------------------------------------------*/

	float elapsedTime;
	cudaEvent_t moveOntoStart, moveOntoStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &moveOntoStart );
		cudaEventCreate( &moveOntoStop );

		// Start Timer
		cudaEventRecord( moveOntoStart, 0 );
	}

	// Copy 'KDNodes' vector from host memory to device memory
	if ((NULL != d_KDNodes) && (NULL != h_KDNodes))
	{
		cutilSafeCall( cudaMemcpy( d_KDNodes, h_KDNodes, mem_size_KDNodes, cudaMemcpyHostToDevice ) );
	}

	// Copy 'IDs' vector from host memory to device memory
	if ((NULL != d_IDs) && (NULL != h_IDs))
	{
		cutilSafeCall( cudaMemcpy( d_IDs, h_IDs, mem_size_IDs, cudaMemcpyHostToDevice ) );
	}

	// Copy 'Query Points' vector from host memory to device memory
	if ((NULL != d_Queries) && (NULL != h_Queries))
	{
		cutilSafeCall( cudaMemcpy( d_Queries, h_Queries, mem_size_Query, cudaMemcpyHostToDevice ) );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( moveOntoStop, 0 );
		cudaEventSynchronize( moveOntoStop );
		cudaEventElapsedTime( &elapsedTime, moveOntoStart, moveOntoStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_onto_device += elapsedTime;
			}
		}
		else
		{
			KD_GPU_onto_device += elapsedTime;
		}
	}


	/*-------------------------------------------------------
	  Step 6.  Call KDTree GPU Kernel
	-------------------------------------------------------*/

	cudaEvent_t searchStart, searchStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &searchStart );
		cudaEventCreate( &searchStop );

		// Start Timer
		cudaEventRecord( searchStart, 0 );
	}

	
	// Check if GPU kernel execution generated an error
#if (APP_TEST == TEST_KD_QNN)
	// Call 2D GPU KDTree QNN Kernel (single result per query)
	GPU_QNN_3D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, nOrigSearch, wQuery  );
#elif (APP_TEST == TEST_KD_ALL_NN)
	// Call 2D GPU KDTree ALL Kernel (single result per query)
	GPU_ALL_NN_3D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											      d_IDs, nOrigSearch, wQuery  );

#elif (APP_TEST == TEST_KD_KNN)
	// Call GPU KDTree kNN Kernel ('k' results per query)
	GPU_KNN_3D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, nOrigSearch, kVal  );

#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Call GPU KDTree kNN Kernel ('k' results per query)
	GPU_ALL_KNN_3D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											       d_IDs, nOrigSearch, kVal  );
#else
#endif

	cuda_err = cudaGetLastError();
	if( cudaSuccess != cuda_err) 
	{
		fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
		"NN 3D left-balanced GPU Kernel failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
		exit( EXIT_FAILURE );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( searchStop, 0 );
		cudaEventSynchronize( searchStop );
		cudaEventElapsedTime( &elapsedTime, searchStart, searchStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_dist += elapsedTime;
			}
		}
		else
		{
			KD_GPU_dist += elapsedTime;
		}
	}

	/*-------------------------------------------------
	  Step 7.  Copy Outputs
	           from device memory to main memory
	-------------------------------------------------*/

	cudaEvent_t moveFromStart, moveFromStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &moveFromStart );
		cudaEventCreate( &moveFromStop );

		// Start Timer
		cudaEventRecord( moveFromStart, 0 );
	}

	// Copy result vector from GPU device to host CPU
	cuda_err = cudaMemcpy( (void *) h_Results_GPU, d_Results_GPU, mem_copy_Results_GPU, cudaMemcpyDeviceToHost );
	if( cudaSuccess != cuda_err) 
	{
		fprintf( stderr, "Cuda error: %s in file '%s' in line %i : CUDA_ERROR[%d]=%s.\n",
				 "cudaMemcpy() failed", __FILE__, __LINE__, (unsigned int)cuda_err,
				 cudaGetErrorString( cuda_err ) );
		exit( EXIT_FAILURE );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( moveFromStop, 0 );
		cudaEventSynchronize( moveFromStop );
		cudaEventElapsedTime( &elapsedTime, moveFromStart, moveFromStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_from_device += elapsedTime;
			}
		}
		else
		{
			KD_GPU_from_device += elapsedTime;
		}
	}

	/*-------------------------------------------
	  Step 8:	Call KDTree CPU Algorithm
	-------------------------------------------*/

	if (g.doubleCheckDists)
	{
		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}

		// Determine Nearest Neighbors using KDTree
#if (APP_TEST == TEST_KD_QNN)
		bResult = CPU_QNN_3D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_ALL_NN)
		bResult = CPU_ALL_NN_3D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_KNN)
		bResult = CPU_KNN_3D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadQuery,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#elif (APP_TEST == TEST_KD_ALL_KNN)
		bResult = CPU_ALL_KNN_3D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadSearch,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#else
		bResult = true;
#endif
		if (false == bResult)
		{
			// Error
			return false;
		}

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					KD_CPU_dist += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				KD_CPU_dist += cutGetTimerValue( g.hTimer );
			}
		}
	}

} // End Profile Loop

	/*-------------------------------------------------
	  Step 9:  Double check GPU result 
			   against CPU result
	-------------------------------------------------*/

	if ((g.doubleCheckDists) && 
		(NULL != h_Results_GPU) && 
		(NULL != h_Results_CPU))
	{
		//double totalNodes = 0.0;
		//maxNodes = 0;

#if (APP_TEST == TEST_KD_QNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		float eps = 1.0e-4f;
		unsigned int qryIdx;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			float gpuDist = h_Results_GPU[qryIdx].Dist;
			float cpuDist = h_Results_CPU[qryIdx].Dist;
			unsigned int gpuID = h_Results_GPU[qryIdx].Id;
			unsigned int cpuID = h_Results_CPU[qryIdx].Id;

			//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			// Need to use a fuzzy compare on distance
			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double qZ = static_cast<double>( h_Queries[qryIdx].z );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g, %.6g> - GPU KD[%5d %.9g] != CPU KD[%5d %.9g] !!!\n",
						 qryIdx, qX, qY, qZ, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#elif (APP_TEST == TEST_KD_ALL_NN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		float eps = 1.0e-4f;
		unsigned int qryIdx;
		for (qryIdx = 0; qryIdx < nOrigSearch; qryIdx++)
		{
			unsigned int gpuID = h_Results_GPU[qryIdx].Id;
			float gpuDist      = h_Results_GPU[qryIdx].Dist;

			unsigned int cpuID = h_Results_CPU[qryIdx].id;
			float cpuDist      = h_Results_CPU[qryIdx].dist;

			//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			// Need to use a fuzzy compare on distance
			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( g.searchList[qryIdx].x );
				double qY = static_cast<double>( g.searchList[qryIdx].y );
				double qZ = static_cast<double>( g.searchList[qryIdx].z );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g, %.6g> - GPU KD[%5u %.9g] != CPU KD[%5u %.9g] !!!\n",
						 qryIdx, qX, qY, qZ, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#elif (APP_TEST == TEST_KD_KNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		/*
		unsigned int qryIdx;
		unsigned int kIdx;
		//float eps = 1.0e-4;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			// Check each Nearest Neighbor
			for (kIdx = 0; kIdx < kVal; kIdx++)
			{
				unsigned int baseIdx = (kIdx * nPadQuery) + qryIdx;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double qZ = static_cast<double>( h_Queries[qryIdx].z );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 qryIdx, kIdx, 
						 qX, qY, qZ,
						 gpuID, gpu_D, 
						 cpuID, cpu_D );
			}
		}
		*/
#elif (APP_TEST == TEST_KD_ALL_KNN)
		checkDistResults = true;
		/*
		// Check each query result (GPU vs. CPU)
		unsigned int qryIdx;
		unsigned int kIdx;
		//float eps = 1.0e-4;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			// Check each Nearest Neighbor
			for (kIdx = 0; kIdx < kVal; kIdx++)
			{
				unsigned int baseIdx = (kIdx * nPadQuery) + qryIdx;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 qryIdx, kIdx, 
						 qX, qY, qZ,
						 gpuID, gpu_D, 
						 cpuID, cpu_D );
			}
		}
		*/
#else
	// Do nothing for now ...
#endif
		// Get Average Nodes Visited Per Query
		//avgNodes = totalNodes/(double)nOrigQuery;
	}


	/*--------------------------------------------------------
	  Step 10: Print out Results
	--------------------------------------------------------*/
	
	/*
	// GPU Results
	printf( "GPU Results\n" );
	unsigned int qryIdx;
	for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
	{
		double gpuDist = static_cast<double>( h_Results_GPU[qryIdx].Dist );
		unsigned int gpuID = h_Results_GPU[qryIdx].Id;
		//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", qryIdx, gpuID, gpuDist, gpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", i, gpuID, gpuDist );
	}

	// CPU Results
	printf( "CPU Results\n" );
	for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
	{
		double cpuDist = static_cast<double>( h_Results_CPU[qryIdx].dist );
		unsigned int cpuID = h_Results_CPU[qryIdx].id;
		//unsigned int cpuCount = h_Results_CPU[qryIdx].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", qryIdx, cpuID, cpuDist, cpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", qryIdx, cpuID, cpuDist );
	}
	*/


	if (g.doubleCheckDists)
	{
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


	/*--------------------------------------------------------
	  Step 11: Print out Profile Statistics
	--------------------------------------------------------*/

	if (g.profile)
	{
		// Dump Profile Statistics
		if (g.profileActualLoops > 1)
		{
			float loops = (float)g.profileActualLoops;
			float o_l = 1.0f / loops;

			float avgOnto    = KD_GPU_onto_device * o_l;
			float avgFrom    = KD_GPU_from_device * o_l;
			float avgGPUdist = KD_GPU_dist * o_l;
			float avgCPUdist = KD_CPU_dist * o_l;
			float avgBuild   = KD_CPU_build;
			float avgCopy    = KD_GPU_copy_nodes;

			// Verbose
			printf( "Number of total iterations = %f.\n", loops );
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", avgBuild );
			printf( "KD - Copy kd-nodes from CPU to GPU,   time: %f msecs.\n", avgCopy );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", avgOnto );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", avgFrom );
			printf( "KD - GPU Kernel computation,          time: %f msecs.\n", avgGPUdist );
			printf( "KD - CPU Kernel computation,          time: %f msecs.\n", avgCPUdist );

			// Terse
			//printf( "KD - In, Out, G_D, C_D, C_B\n" );
			//printf( "     %f, %f, %f, %f, %f\n\n", avgOnto, avgFrom, avgGPUdist, avgCPUdist, avgBuild );
		}
		else
		{
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", KD_CPU_build );
			printf( "KD - Copy kd-nodes from CPU to GPU,   time: %f msecs.\n", KD_GPU_copy_nodes );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", KD_GPU_onto_device );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", KD_GPU_from_device );
			printf( "KD - GPU Kernel computation,        time: %f msecs.\n", KD_GPU_dist );
			printf( "KD - CPU Kernel computation,        time: %f msecs.\n", KD_CPU_dist );
		}
	}


	/*--------------------------------------------------------
	  Step 13: Cleanup Resources
	--------------------------------------------------------*/


	printf( "Shutting Down...\n" );

	// cleanup CUDA Timer
	cutDeleteTimer( g.hTimer );

	// clean up allocations
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (NULL != h_KDNodes)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_KDNodes ) );
	}
	if (NULL != h_IDs)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_IDs ) );
	}
	if (NULL != h_Queries)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_Queries ) );
	}
	if (NULL != h_Results_GPU)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_Results_GPU ) );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (NULL != h_KDNodes)
	{
		free( h_KDNodes );
	}
	if (NULL != h_IDs)
	{
		free( h_IDs );
	}
	if (NULL != h_Queries)
	{
		free( h_Queries );
	}
	if (NULL != h_Results_CPU)
	{
		free( h_Results_GPU );
	}
#else
#endif
	if (NULL != h_Results_CPU)
	{
		free( h_Results_CPU );
	}

	cutilSafeCall( cudaFree( d_KDNodes ) );
	cutilSafeCall( cudaFree( d_IDs ) );
	cutilSafeCall( cudaFree( d_Queries ) );
	cutilSafeCall( cudaFree( d_Results_GPU ) );

	FINI_CPU_3D_LBT( &kdTree );

	FiniSearchQueryVectors( g );

	printf( "Shutdown done...\n\n" );

	// Success
	return true;
}



/*---------------------------------------------------------
  Name:	CPUTest_4D_LBT()
---------------------------------------------------------*/

bool CPUTest_4D_LBT( AppGlobals & g )
{
	bool bResult = true;

	/*---------------------------------
	  Step 0.  Initialize
	---------------------------------*/

	cudaError_t cuda_err = cudaSuccess;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cu_err = CUDA_SUCCESS;
#endif

	// set seed for rand()
	RandomInit( 2010 );

	g.hTimer = 0;
	cutCreateTimer( &(g.hTimer) );

	/*-------------------------------------------
	  Step 1.  Create Search & Query Vectors
	-------------------------------------------*/

	// Get # of search and query points
#if (APP_TEST == TEST_KD_QNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = TEST_NUM_QUERY_POINTS;

	bool bInitSearch = true;
	bool bInitQuery  = true;
#elif (APP_TEST == TEST_KD_ALL_NN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = g.nSearch+1;

	bool bInitSearch = true;
	bool bInitQuery  = false;
#elif (APP_TEST == TEST_KD_KNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = TEST_NUM_QUERY_POINTS;

	bool bInitSearch = true;
	bool bInitQuery  = true;
	unsigned int kVal = 64;

	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#elif (APP_TEST == TEST_KD_ALL_KNN)
	g.nSearch = TEST_NUM_SEARCH_POINTS;
	g.nQuery  = g.nSearch+1;

	bool bInitSearch = true;
	bool bInitQuery  = false;

	unsigned int kVal = 64;
	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#else
#endif

	// Create random search and query vectors
	bool bNonUniformSearch = false;
	bool bNonUniformQuery  = false;
	int scaleType = 0;
	bResult = InitSearchQueryVectors( g, bInitSearch, bInitQuery, 
		                              bNonUniformSearch, bNonUniformQuery, scaleType );
	if (false == bResult)
	{
		// Error
		return false;
	}


	/*-------------------------------------------
	  Step 2.  Setup Initial parameters
	-------------------------------------------*/

#if (APP_TEST == TEST_KD_QNN)
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = g.nQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	// Query == Search
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = 0;				
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = g.nQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Query == Search
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nOrigQuery  = 0;		
#else
#endif

	// Compute Reasonable Thread Block and Grid Shapes
	BlockGridShape kdShape;
	
#if (APP_TEST == TEST_KD_QNN)
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigSearch+1;		// Add 1 for zeroth element
#elif (APP_TEST == TEST_KD_KNN)
	kdShape.threadsPerRow = KNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	kdShape.threadsPerRow = KNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KNN_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigSearch+1;		// Add 1 for zeroth element
#else
#endif
	bResult = ComputeBlockShapeFromQueryVector( kdShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

#if (APP_TEST == TEST_KD_QNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch + 1;		// Add 1 for zeroth element

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch + 1;		// Add 1 for zeroth element
#elif (APP_TEST == TEST_KD_ALL_KNN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;
#else
#endif

	//
	// Print out Initialization Parameters
	//
	DumpBlockGridShape( kdShape );

#if (APP_TEST == TEST_KD_QNN)
	printf( "\n QNN 4D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_NN)
	printf( "\n All-NN 4D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_KNN)
	printf( "\n kNN 4D Left-balanced Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "\n All-kNN 4D Left-balanced Test\n\n" );
#else
#endif

	printf( "# Requested Search Points  = %d\n", nOrigSearch );
	printf( "# Padded Search Points     = %d\n", nPadSearch );
	printf( "# Requested Query Points   = %d\n", nOrigQuery );
	printf( "# Padded Query Points      = %d\n", nPadQuery );
#if (APP_TEST == TEST_KD_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#else
#endif


	// Make sure Matrix + vector is not to big to use up all device memory
		// 1 GB on Display Card
#if (APP_TEST == TEST_KD_QNN)
	unsigned int sizeResults = nPadQuery * sizeof(GPU_NN_Result);
	unsigned int sizeQueries = nPadQuery * sizeof(float4);
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int sizeResults = nPadSearch * sizeof(GPU_NN_Result);
	unsigned int sizeQueries = 0;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int sizeResults = nPadQuery * sizeof(GPU_NN_Result) * kVal;
	unsigned int sizeQueries = nPadQuery * sizeof(float4);
#elif (APP_TEST == TEST_KD_ALL_KNN)
	unsigned int sizeResults = nPadSearch * sizeof(GPU_NN_Result) * kVal;
	unsigned int sizeQueries = 0;
#else
#endif
	unsigned int sizeNodes   = nPadSearch * sizeof(GPUNode_4D_LBT);
	unsigned int sizeIDs	 = nPadSearch * sizeof(unsigned int);
	unsigned int totalMem    = sizeNodes + sizeIDs + sizeQueries + sizeResults;

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g.cudaProps.totalGlobalMem)
	{
		printf( "KD Tree Inputs (%d) are too large for available device memory (%d), running test will crash...\n",
				totalMem, g.cudaProps.totalGlobalMem );
		printf( "\tsizeNodes = %d\n", sizeNodes );
		printf( "\tsizeIDs   = %d\n", sizeIDs );
		printf( "\tsizeQueries = %d\n", sizeQueries );
		printf( "\tsizeResults = %d\n", sizeResults );
		return bResult;
	}

	printf( "# Onto Memory       = %d\n", sizeNodes + sizeIDs + sizeQueries );
	printf( "# From Memory       = %d\n", sizeResults );
	printf( "# Total Memory      = %d\n", totalMem );

	// Setup GPU Kernel execution parameters
		// KDTree Distance
	dim3 qryThreads( kdShape.threadsPerRow, kdShape.rowsPerBlock, 1 );
	dim3 qryGrid( kdShape.blocksPerRow, kdShape.rowsPerGrid, 1 );

	float KD_CPU_build        = 0.0f;
	float KD_GPU_copy_nodes   = 0.0f;
	float KD_GPU_onto_device  = 0.0f;
	float KD_GPU_from_device  = 0.0f;
	float KD_GPU_dist		  = 0.0f;
	float KD_CPU_dist		  = 0.0f;
	bool  checkDistResults    = true;
	//unsigned int maxNodes     = 0;
	//double avgNodes           = 0.0;


	/*-------------------------------------------
	  Step 3.  Allocate GPU Vectors
	-------------------------------------------*/

	// Allocate host memory for GPU KD Tree Nodes
	unsigned int mem_size_KDNodes = nPadSearch * sizeof(GPUNode_4D_LBT);
	GPUNode_4D_LBT * h_KDNodes = NULL;	
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_KDNodes > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_KDNodes, mem_size_KDNodes );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_KDNodes > 0)
	{
		h_KDNodes = (GPUNode_4D_LBT*) malloc( mem_size_KDNodes );
		if (NULL == h_KDNodes)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU KD Tree Nodes
	GPUNode_4D_LBT* d_KDNodes = NULL;
	if (mem_size_KDNodes > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_KDNodes, mem_size_KDNodes ) );
	}


	// Allocate host memory for GPU Node ID's
	unsigned int mem_size_IDs = nPadSearch * sizeof(unsigned int);
	unsigned int* h_IDs = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_IDs > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_IDs, mem_size_IDs );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_IDs > 0)
	{
		h_IDs = (unsigned int*) malloc( mem_size_IDs );
		if (NULL == h_IDs)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU Node ID's
	unsigned int* d_IDs = NULL;
	if (mem_size_IDs > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_IDs, mem_size_IDs ) );
	}


	// Allocate host memory for GPU query points 
	unsigned int mem_size_Query = nPadQuery * sizeof(float4);
	float4* h_Queries = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_Query > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_Queries, mem_size_Query );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_Query > 0)
	{
		h_Queries = (float4*) malloc( mem_size_Query );
		if (NULL == h_Queries)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU query points 
	float4* d_Queries = NULL;
	if (mem_size_Query > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_Queries, mem_size_Query ) );
	}

	// Allocate host memory for GPU Query Results
	unsigned int nQueryElems = 0;
	unsigned int nQueryCopy  = 0;
	unsigned int mem_size_Results_GPU = 0;
	unsigned int mem_copy_Results_GPU = 0;

	// BUGBUG:  Interesting bug caused by processing zeroth element as if it was a valid entry
	//			in GPU thread block processing, Currently, we end up with 2 elements
	//			colling on the zeroth entry for results causing at least one bug in the result set
	// WORKAROUND:  increase memory allocation by one extra element
	//				 and have zeroth element point to this extra location to avoid the 2 element collision
#if (APP_TEST == TEST_KD_QNN)
	nQueryElems = nPadQuery + 2;		// +2 to work around bug
	nQueryCopy  = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	nQueryElems = nPadSearch + 2;		// +2 to work around bug
	nQueryCopy  = nOrigSearch;
#elif (APP_TEST == TEST_KD_KNN)
	nQueryElems = (nPadQuery + 2) * kVal;	// +2 to work around bug
	nQueryCopy  = nOrigQuery * kVal;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	nQueryElems = (nPadSearch + 2) * kVal;	// +2 to work around bug
	nQueryCopy  = nOrigSearch * kVal;
#else
#endif
	mem_size_Results_GPU = nQueryElems * sizeof(GPU_NN_Result);
	mem_copy_Results_GPU = nQueryCopy * sizeof(GPU_NN_Result);

	GPU_NN_Result* h_Results_GPU = NULL;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (mem_size_Results_GPU > 0)
	{
		cu_err = cuMemAllocHost( (void**)&h_Results_GPU, mem_size_Results_GPU );
		if( CUDA_SUCCESS != cu_err) 
		{
			fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
					 cu_err, __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (mem_size_Results_GPU > 0)
	{
		h_Results_GPU = (GPU_NN_Result*) malloc( mem_size_Results_GPU );
		if (NULL == h_Results_GPU)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU query results
	GPU_NN_Result* d_Results_GPU = NULL;
	if (mem_size_Results_GPU > 0)
	{
		cutilSafeCall( cudaMalloc( (void **) &d_Results_GPU, mem_size_Results_GPU ) );
	}


	// Allocate host memory for CPU Query Results
	unsigned int mem_size_Results_CPU = nQueryElems * sizeof(CPU_NN_Result);
	CPU_NN_Result* h_Results_CPU = NULL;
	if (mem_size_Results_CPU > 0)
	{
		h_Results_CPU = (CPU_NN_Result*) malloc( mem_size_Results_CPU );
	}


	// Copy Query List
	if (NULL != h_Queries)
	{
		unsigned int qryIdx;
	
		// Copy actual queries 
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			h_Queries[qryIdx].x = g.queryList[qryIdx].x;
			h_Queries[qryIdx].y = g.queryList[qryIdx].y;
			h_Queries[qryIdx].z = g.queryList[qryIdx].z;
			h_Queries[qryIdx].w = g.queryList[qryIdx].w;
		}

		// Create some extra queries for thread block alignment
		for (qryIdx = nOrigQuery; qryIdx < nPadQuery; qryIdx++)
		{
			// Just repeat the first query a few times
			h_Queries[qryIdx].x = g.queryList[0].x;
			h_Queries[qryIdx].y = g.queryList[0].y;
			h_Queries[qryIdx].z = g.queryList[0].z;
			h_Queries[qryIdx].w = g.queryList[0].w;
		}
	}


	/*-----------------------
	  Step 4. Build kd-tree
	-----------------------*/

	// Dump search List (for debugging)
#if 0
	unsigned int pointIdx;
	double currX, currY, currZ, currW;
	for (pointIdx = 0; pointIdx < nOrigSearch; pointIdx++)
	{
		currX = g.searchList[pointIdx].x;
		currY = g.searchList[pointIdx].y;
		currZ = g.searchList[pointIdx].z;
		currW = g.searchList[pointIdx].w;

		printf( "PointID[%u] = <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
			    pointIdx, currX, currY, currZ, currW );
	}
#endif

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Build left-balanced KDTree (on CPU)
	void * kdTree = NULL;
	bResult = BUILD_CPU_4D_LBT( &kdTree, nOrigSearch, g.searchList );
	if (false == bResult) 
	{
		// Error - goto cleanup
		return false;
	}	

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_CPU_build += cutGetTimerValue( g.hTimer );
	}

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Copy kd-tree from CPU to GPU
		// Note:  We have the zeroth element point to nPadSearch + 1 to work around a bug
	bResult = COPY_NODES_4D_LBT( kdTree, nOrigSearch, nPadSearch, (void*)h_KDNodes, h_IDs );
	if (!bResult)
	{
		// Error - goto cleanup
		return false;
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_GPU_copy_nodes += cutGetTimerValue( g.hTimer );
	}

	// Dump LBT Nodes and IDs (for debugging)
#if 0
	unsigned int nodeIdx;
	for (nodeIdx = 0; nodeIdx < nPadSearch; nodeIdx++)
	{
		currX    = (double)(h_KDNodes[nodeIdx].pos[0]);
		currY    = (double)(h_KDNodes[nodeIdx].pos[1]);
		currZ    = (double)(h_KDNodes[nodeIdx].pos[2]);
		currW    = (double)(h_KDNodes[nodeIdx].pos[3]);
		pointIdx = h_IDs[nodeIdx];
		if (nodeIdx == 0)
		{
			// zeroth node
			printf( "ZERO:   NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ, currW );
		}
		else if (nodeIdx > nOrigSearch)
		{
			// Extra padded node for searching 
			printf( "Extra:  NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ, currW );
		}
		else if (pointIdx > nOrigSearch)
		{
			// Possible error
			printf( "ERROR:  NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ, currW );
		}
		else
		{
			// Normal behavior
			printf( "Normal: NodeID[%u] = Point ID[%u] <%3.6f, %3.6f, %3.6f, %3.6f>\n", 
				    nodeIdx, pointIdx, currX, currY, currZ, currW );
		}
	}
#endif



// Profile Measurement Loop
unsigned int currIter;
for (currIter = 0; currIter < g.profileActualLoops; currIter++)
{

	/*-------------------------------------------------------
	  Step 5.  Move Input Vectors 
	           from main memory to device memory
	-------------------------------------------------------*/

	float elapsedTime;
	cudaEvent_t moveOntoStart, moveOntoStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &moveOntoStart );
		cudaEventCreate( &moveOntoStop );

		// Start Timer
		cudaEventRecord( moveOntoStart, 0 );
	}

	// Copy 'KDNodes' vector from host memory to device memory
	if ((NULL != d_KDNodes) && (NULL != h_KDNodes))
	{
		cutilSafeCall( cudaMemcpy( d_KDNodes, h_KDNodes, mem_size_KDNodes, cudaMemcpyHostToDevice ) );
	}

	// Copy 'IDs' vector from host memory to device memory
	if ((NULL != d_IDs) && (NULL != h_IDs))
	{
		cutilSafeCall( cudaMemcpy( d_IDs, h_IDs, mem_size_IDs, cudaMemcpyHostToDevice ) );
	}

	// Copy 'Query Points' vector from host memory to device memory
	if ((NULL != d_Queries) && (NULL != h_Queries))
	{
		cutilSafeCall( cudaMemcpy( d_Queries, h_Queries, mem_size_Query, cudaMemcpyHostToDevice ) );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( moveOntoStop, 0 );
		cudaEventSynchronize( moveOntoStop );
		cudaEventElapsedTime( &elapsedTime, moveOntoStart, moveOntoStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_onto_device += elapsedTime;
			}
		}
		else
		{
			KD_GPU_onto_device += elapsedTime;
		}
	}


	/*-------------------------------------------------------
	  Step 6.  Call KDTree GPU Kernel
	-------------------------------------------------------*/

	cudaEvent_t searchStart, searchStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &searchStart );
		cudaEventCreate( &searchStop );

		// Start Timer
		cudaEventRecord( searchStart, 0 );
	}

	
	// Check if GPU kernel execution generated an error
#if (APP_TEST == TEST_KD_QNN)
	// Call 4D GPU KDTree QNN Kernel (single result per query)
	GPU_QNN_4D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, nOrigSearch, wQuery  );

	/*
	// If searches take longer than 5 seconds
	if (nPadSearch <= 200000)
	{
		// All at once

		// Call 4D GPU KDTree QNN Kernel (single result per query)
		GPU_QNN_4D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
											       d_IDs, nOrigSearch, wQuery  );
	}
	else
	{
		// Call one row at a time...
		unsigned int rowIdx;
		unsigned int startIdx;
		unsigned int nGridRows = kdShape.rowsPerGrid;
		unsigned int rowWidth  = kdShape.W;
		GPU_NN_Result * currResults = NULL;
		float4 * currQuerys = NULL;
		dim3 rowGrid( kdShape.blocksPerRow, 1, 1 );
		dim3 rowThreads( kdShape.threadsPerRow, kdShape.rowsPerBlock, 1 );
		for (rowIdx = 0; rowIdx < nGridRows ; rowIdx++)
		{
			startIdx = rowIdx * rowWidth;
			currResults = &(d_Results_GPU[startIdx]);
			currQuerys = &(d_Queries[startIdx]);

			GPU_QNN_4D_LBT<<< rowGrid, rowThreads >>>( currResults, currQuerys, 
													   d_KDNodes, d_IDs, 
													   nOrigSearch, wQuery  );
			cudaThreadSynchronize();

			printf( "Row[%d] done!\n", rowIdx );
		}
	}
	*/
#elif (APP_TEST == TEST_KD_ALL_NN)
	// Call 4D GPU KDTree ALL Kernel (single result per query)
	GPU_ALL_NN_4D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											      d_IDs, nOrigSearch, wQuery  );

#elif (APP_TEST == TEST_KD_KNN)
	// Call 4D GPU KDTree kNN Kernel ('k' results per query)
	GPU_KNN_4D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, nOrigSearch, kVal  );

#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Call 4D GPU KDTree kNN Kernel ('k' results per query)
	GPU_ALL_KNN_4D_LBT<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											       d_IDs, nOrigSearch, kVal  );
#else
#endif

	cuda_err = cudaGetLastError();
	if( cudaSuccess != cuda_err) 
	{
		fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
		         "NN 4D left-balanced GPU kernel failed", __FILE__, __LINE__, 
				 cudaGetErrorString( cuda_err ) );
		exit( EXIT_FAILURE );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( searchStop, 0 );
		cudaEventSynchronize( searchStop );
		cudaEventElapsedTime( &elapsedTime, searchStart, searchStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_dist += elapsedTime;
			}
		}
		else
		{
			KD_GPU_dist += elapsedTime;
		}
	}

	printf( "\nDone with GPU Kernel\n\n" );

	/*-------------------------------------------------
	  Step 7.  Copy Outputs
	           from device memory to main memory
	-------------------------------------------------*/

	cudaEvent_t moveFromStart, moveFromStop;

	if (g.profile)
	{
		// Create Timer Events
		cudaEventCreate( &moveFromStart );
		cudaEventCreate( &moveFromStop );

		// Start Timer
		cudaEventRecord( moveFromStart, 0 );
	}

	// Copy result vector from GPU device to host CPU
	cutilSafeCall( cudaMemcpy( (void *) h_Results_GPU, d_Results_GPU, mem_copy_Results_GPU, cudaMemcpyDeviceToHost ) );

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cudaEventRecord( moveFromStop, 0 );
		cudaEventSynchronize( moveFromStop );
		cudaEventElapsedTime( &elapsedTime, moveFromStart, moveFromStop );

		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_from_device += elapsedTime;
			}
		}
		else
		{
			KD_GPU_from_device += elapsedTime;
		}
	}

	/*-------------------------------------------
	  Step 8:	Call KDTree CPU Algorithm
	-------------------------------------------*/

	if (g.doubleCheckDists)
	{
		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}

		// Determine Nearest Neighbors using KDTree
#if (APP_TEST == TEST_KD_QNN)
		bResult = CPU_QNN_4D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_ALL_NN)
		bResult = CPU_ALL_NN_4D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_KNN)
		bResult = CPU_KNN_4D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadQuery,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#elif (APP_TEST == TEST_KD_ALL_KNN)
		bResult = CPU_ALL_KNN_4D_LBT
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadSearch,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#else
		bResult = true;
#endif
		if (false == bResult)
		{
			// Error
			return false;
		}

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					KD_CPU_dist += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				KD_CPU_dist += cutGetTimerValue( g.hTimer );
			}
		}
	}

} // End Profile Loop

	/*-------------------------------------------------
	  Step 9:  Double check GPU result 
			   against CPU result
	-------------------------------------------------*/

	if ((g.doubleCheckDists) && 
		(NULL != h_Results_GPU) && 
		(NULL != h_Results_CPU))
	{
		//double totalNodes = 0.0;
		//maxNodes = 0;

#if (APP_TEST == TEST_KD_QNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		float eps = 1.0e-4f;
		unsigned int qryIdx;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			float gpuDist = h_Results_GPU[qryIdx].Dist;
			float cpuDist = h_Results_CPU[qryIdx].Dist;
			unsigned int gpuID = h_Results_GPU[qryIdx].Id;
			unsigned int cpuID = h_Results_CPU[qryIdx].Id;

			//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			// Need to use a fuzzy compare on distance
			if ( ((gpuDist < (cpuDist-eps)) || 
				  (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double qZ = static_cast<double>( h_Queries[qryIdx].z );
				double qW = static_cast<double>( h_Queries[qryIdx].w );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g, %.6g, %.6g> - GPU KD[%d %.9g] != CPU KD[%d %.9g] !!!\n",
						 qryIdx, qX, qY, qZ, qW, 
						 gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#elif (APP_TEST == TEST_KD_ALL_NN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		float eps = 1.0e-4f;
		unsigned int qryIdx;
		for (qryIdx = 0; qryIdx < nOrigSearch; qryIdx++)
		{
			unsigned int gpuID = h_Results_GPU[qryIdx].Id;
			float gpuDist      = h_Results_GPU[qryIdx].Dist;

			unsigned int cpuID = h_Results_CPU[qryIdx].id;
			float cpuDist      = h_Results_CPU[qryIdx].dist;

			//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			// Need to use a fuzzy compare on distance
			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( g.searchList[qryIdx].x );
				double qY = static_cast<double>( g.searchList[qryIdx].y );
				double qZ = static_cast<double>( g.searchList[qryIdx].z );
				double qW = static_cast<double>( g.searchList[qryIdx].w );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g, %.6g, %.6g> - GPU KD[%u %.9g] != CPU KD[%u %.9g] !!!\n",
						 qryIdx, qX, qY, qZ, qW,
						 gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#elif (APP_TEST == TEST_KD_KNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		/*
		unsigned int qryIdx;
		unsigned int kIdx;
		//float eps = 1.0e-4;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			// Check each Nearest Neighbor
			for (kIdx = 0; kIdx < kVal; kIdx++)
			{
				unsigned int baseIdx = (kIdx * nPadQuery) + qryIdx;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[qryIdx].x );
				double qY = static_cast<double>( h_Queries[qryIdx].y );
				double qZ = static_cast<double>( h_Queries[qryIdx].z );
				double qW = static_cast<double>( h_Queries[qryIdx].w );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g, %.6g, %.6g>: GPU [%u %.9g], CPU [%u %.9g]\n",
						 qryIdx, kIdx, 
						 qX, qY, qZ, qW,
						 gpuID, gpu_D, 
						 cpuID, cpu_D );
			}
		}
		*/
#elif (APP_TEST == TEST_KD_ALL_KNN)
		checkDistResults = true;
		/*
		// Check each query result (GPU vs. CPU)
		unsigned int qryIdx;
		unsigned int kIdx;
		//float eps = 1.0e-4;
		for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
		{
			// Check each Nearest Neighbor
			for (kIdx = 0; kIdx < kVal; kIdx++)
			{
				unsigned int baseIdx = (kIdx * nPadQuery) + qryIdx;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( g.searchList[qryIdx].x );
				double qY = static_cast<double>( g.searchList[qryIdx].y );
				double qZ = static_cast<double>( g.searchList[qryIdx].z );
				double qW = static_cast<double>( g.searchList[qryIdx].w );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g, %.6g, %.6g>: GPU [%u %.9g], CPU [%u %.9g]\n",
						 qryIdx, kIdx, 
						 qX, qY, qZ, qW,
						 gpuID, gpu_D, 
						 cpuID, cpu_D );
			}
		}
		*/
#else
	// Do nothing for now ...
#endif
		// Get Average Nodes Visited Per Query
		//avgNodes = totalNodes/(double)nOrigQuery;
	}


	/*--------------------------------------------------------
	  Step 10: Print out Results
	--------------------------------------------------------*/
	
	/*
	// GPU Results
	printf( "GPU Results\n" );
	unsigned int qryIdx;
	for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
	{
		double gpuDist = static_cast<double>( h_Results_GPU[qryIdx].Dist );
		unsigned int gpuID = h_Results_GPU[qryIdx].Id;
		//unsigned int gpuCount = h_Results_GPU[qryIdx].cNodes;
		//printf( "QR[%3d]: <ID=%u, Dist=%.6g, cNodes=%u>\n", qryIdx, gpuID, gpuDist, gpuCount );
		printf( "QR[%3d]: <ID=%u, Dist=%.6g>\n", i, gpuID, gpuDist );
	}

	// CPU Results
	printf( "CPU Results\n" );
	for (qryIdx = 0; qryIdx < nOrigQuery; qryIdx++)
	{
		double cpuDist = static_cast<double>( h_Results_CPU[qryIdx].dist );
		unsigned int cpuID = h_Results_CPU[qryIdx].id;
		//unsigned int cpuCount = h_Results_CPU[qryIdx].cNodes;
		//printf( "QR[%3d]: <ID=%u, Dist=%.6g, cNodes=%u>\n", qryIdx, cpuID, cpuDist, cpuCount );
		printf( "QR[%3d]: <ID=%u, Dist=%.6g>\n", qryIdx, cpuID, cpuDist );
	}
	*/


	if (g.doubleCheckDists)
	{
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


	/*--------------------------------------------------------
	  Step 11: Print out Profile Statistics
	--------------------------------------------------------*/

	if (g.profile)
	{
		// Dump Profile Statistics
		if (g.profileActualLoops > 1)
		{
			float loops = (float)g.profileActualLoops;
			float o_l = 1.0f / loops;

			float avgOnto    = KD_GPU_onto_device * o_l;
			float avgFrom    = KD_GPU_from_device * o_l;
			float avgGPUdist = KD_GPU_dist * o_l;
			float avgCPUdist = KD_CPU_dist * o_l;
			float avgBuild   = KD_CPU_build;
			float avgCopy    = KD_GPU_copy_nodes;

			// Verbose
			printf( "Number of total iterations = %f.\n", loops );
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", avgBuild );
			printf( "KD - Copy kd-nodes from CPU to GPU,   time: %f msecs.\n", avgCopy );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", avgOnto );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", avgFrom );
			printf( "KD - GPU Kernel computation,          time: %f msecs.\n", avgGPUdist );
			printf( "KD - CPU Kernel computation,          time: %f msecs.\n", avgCPUdist );

			// Terse
			//printf( "KD - In, Out, G_D, C_D, C_B\n" );
			//printf( "     %f, %f, %f, %f, %f\n\n", avgOnto, avgFrom, avgGPUdist, avgCPUdist, avgBuild );
		}
		else
		{
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", KD_CPU_build );
			printf( "KD - Copy kd-nodes from CPU to GPU,   time: %f msecs.\n", KD_GPU_copy_nodes );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", KD_GPU_onto_device );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", KD_GPU_from_device );
			printf( "KD - GPU Kernel computation,        time: %f msecs.\n", KD_GPU_dist );
			printf( "KD - CPU Kernel computation,        time: %f msecs.\n", KD_CPU_dist );
		}
	}


	/*--------------------------------------------------------
	  Step 13: Cleanup Resources
	--------------------------------------------------------*/


	printf( "Shutting Down...\n" );

	// cleanup CUDA Timer
	cutDeleteTimer( g.hTimer );

	// clean up allocations
#if (CUDA_PLATFORM == CUDA_DEVICE)
	if (NULL != h_KDNodes)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_KDNodes ) );
	}
	if (NULL != h_IDs)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_IDs ) );
	}
	if (NULL != h_Queries)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_Queries ) );
	}
	if (NULL != h_Results_GPU)
	{
		CU_SAFE_CALL( cuMemFreeHost( h_Results_GPU ) );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	if (NULL != h_KDNodes)
	{
		free( h_KDNodes );
	}
	if (NULL != h_IDs)
	{
		free( h_IDs );
	}
	if (NULL != h_Queries)
	{
		free( h_Queries );
	}
	if (NULL != h_Results_CPU)
	{
		free( h_Results_GPU );
	}
#else
#endif
	if (NULL != h_Results_CPU)
	{
		free( h_Results_CPU );
	}

	cutilSafeCall( cudaFree( d_KDNodes ) );
	cutilSafeCall( cudaFree( d_IDs ) );
	cutilSafeCall( cudaFree( d_Queries ) );
	cutilSafeCall( cudaFree( d_Results_GPU ) );

	FINI_CPU_2D_LBT( &kdTree );

	FiniSearchQueryVectors( g );

	printf( "Shutdown done...\n\n" );

	// Success
	return true;
}
