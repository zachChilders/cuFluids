/*-----------------------------------------------------------------------------
  File:  GPUTest_MED.cpp
  Desc:  Host CPU scaffolding for running and testing kd-tree NN searches
         for median layout kd-trees

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
#include "GPUTREE_API.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

extern AppGlobals g_app;

/*-------------------------------------
	CUDA Kernels
-------------------------------------*/

// Median Array Layout GPU Kernels
	// root range = [0, n-1] = [low,high]
	// rootIdx = (low+high)/2 = (n-1)/2
	// Given current range [low,high]
		// currIdx   = M = (low+high)/2
		// parentIdx = ??? (can't determine)
		// leftIdx   = (low+M-1)/2
		// rightIdx  = (M+1+high)/2
//#include <GPU_QNN_MED.cu>    // QNN kernel (shared memory solution)
//#include <GPU_ALL_NN_MED.cu>  // All-NN kernel
//#include <GPU_KNN_MED.cu>    // kNN kernel
//#include <GPU_ALL_KNN_MED.cu> // All-kNN Kernel


/*-------------------------------------
  Function Declarations
-------------------------------------*/

/*---------------------------------------------------------
  Name:	CPUTest_2D_MED()
  Desc:	Run a simple test of "KD Tree nearest neighbor" 
        search functionality on CUDA GPU framework
---------------------------------------------------------*/

bool CPUTest_2D_MED( AppGlobals & g )
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
	
	kdShape.threadsPerRow = QNN_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = QNN_ROWS_PER_BLOCK;
#if (APP_TEST == TEST_KD_QNN)
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	kdShape.nElems        = nOrigSearch;
#elif (APP_TEST == TEST_KD_KNN)
	kdShape.nElems        = nOrigQuery;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	kdShape.nElems        = nOrigSearch;
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
	unsigned int nPadSearch  = nOrigSearch;

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_ALL_NN)
	unsigned int nPadQuery   = 0;
	unsigned int nPadSearch  = kdShape.nPadded;

	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#elif (APP_TEST == TEST_KD_KNN)
	unsigned int nPadQuery   = kdShape.nPadded;
	unsigned int nPadSearch  = nOrigSearch;
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
	printf( "\n QNN Median Layout Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_NN)
	printf( "\n All-NN Median Layout Test\n\n" );
#elif (APP_TEST == TEST_KD_KNN)
	printf( "\n kNN Median Layout Test\n\n" );
#elif (APP_TEST == TEST_KD_ALL_KNN)
	printf( "\n All-kNN Median Layout Test\n\n" );
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
	unsigned int sizeNodes   = nPadSearch * sizeof(GPUNode_2D_MED);
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

	unsigned int rootIdx = (nOrigSearch-1)>>1;		// (start+end)/2

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
	unsigned int mem_size_KDNodes = nPadSearch * sizeof(GPUNode_2D_MED);
	GPUNode_2D_MED * h_KDNodes = NULL;	
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
		h_KDNodes = (GPUNode_2D_MED*) malloc( mem_size_KDNodes );
		if (NULL == h_KDNodes)
		{
			fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
			exit( EXIT_FAILURE );
		}
	}
#else
#endif

	// Allocate device memory for GPU KD Tree Nodes
	GPUNode_2D_MED* d_KDNodes = NULL;
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
	unsigned int mem_size_Results_GPU = 0;

#if (APP_TEST == TEST_KD_QNN)
	nQueryElems = nPadQuery;
#elif (APP_TEST == TEST_KD_ALL_NN)
	nQueryElems = nPadSearch;
#elif (APP_TEST == TEST_KD_KNN)
	nQueryElems = (nPadQuery) * kVal;
#elif (APP_TEST == TEST_KD_ALL_KNN)
	nQueryElems = (nPadSearch) * kVal;
#else
#endif
	mem_size_Results_GPU = nQueryElems * sizeof(GPU_NN_Result);


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

	const float4 * queryList = (const float4 *)g.queryList;
	if (NULL != h_Queries)
	{
		unsigned int qryIdx;
	
		// Copy actual queries 
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


	/*-------------------------------------------
	  Step 4.  Build kd-tree
	-------------------------------------------*/

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Build KDTree (on CPU)
	void * kdTree = NULL;
	const float4 * searchList = (const float4 *)g.searchList;
	bResult = BUILD_CPU_2D_MEDIAN( &kdTree, nOrigSearch, searchList );
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
	searchList = (const float4 *)g.searchList;
	bResult = COPY_NODES_2D_MEDIAN( kdTree, nOrigSearch, nPadSearch, (void*)h_KDNodes, h_IDs );
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
	cutilSafeCall( cudaMemcpy( d_KDNodes, h_KDNodes, mem_size_KDNodes, cudaMemcpyHostToDevice ) );

	// Copy 'IDs' vector from host memory to device memory
	cutilSafeCall( cudaMemcpy( d_IDs, h_IDs, mem_size_IDs, cudaMemcpyHostToDevice ) );

	// Copy 'Query Points' vector from host memory to device memory
	cutilSafeCall( cudaMemcpy( d_Queries, h_Queries, mem_size_Query, cudaMemcpyHostToDevice ) );

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
	GPU_QNN_2D_MED<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
											   d_IDs, rootIdx, wQuery  );

#elif (APP_TEST == TEST_KD_ALL_NN)
	// Call 2D GPU KDTree ALL Kernel (single result per query)
	GPU_ALL_NN_2D_MED<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											      d_IDs, rootIdx, wQuery  );

#elif (APP_TEST == TEST_KD_KNN)
	// Call GPU KDTree kNN Kernel ('k' results per query)
	GPU_KNN_2D_MED<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
										       d_IDs, rootIdx, kVal  );

#elif (APP_TEST == TEST_KD_ALL_KNN)
	// Call GPU KDTree kNN Kernel ('k' results per query)
	GPU_ALL_KNN_2D_MED<<< qryGrid, qryThreads >>>( d_Results_GPU, d_KDNodes,
											       d_IDs, rootIdx, kVal  );
#endif

	cuda_err = cudaGetLastError();
	if( cudaSuccess != cuda_err) 
	{
		fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
				 "NN 2D Median layout GPU kernel failed", __FILE__, __LINE__, 
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

	// copy result vector Z from device to host
	cutilSafeCall( cudaMemcpy( (void *) h_Results_GPU, d_Results_GPU, mem_size_Results_GPU, cudaMemcpyDeviceToHost ) );

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
		bResult = CPU_QNN_2D_MEDIAN
					(
					kdTree,
					g.nSearch,
					(const float4 *)g.searchList,
					g.nQuery,
					(const float4 *)g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_ALL_NN)
		bResult = CPU_ALL_NN_2D_MEDIAN
					(
					kdTree,
					g.nSearch,
					(const float4 *)g.searchList,
					g.nQuery,
					(const float4 *)g.queryList,
					h_Results_CPU 
					);
#elif (APP_TEST == TEST_KD_KNN)
		bResult = CPU_KNN_2D_MEDIAN
					(
					kdTree,
					g.nSearch,
					(const float4 *)g.searchList,
					g.nQuery,
					nPadQuery,
					(const float4 *)g.queryList,
					kVal,
					h_Results_CPU
					);
#elif (APP_TEST == TEST_KD_ALL_KNN)
		bResult = CPU_ALL_KNN_2D_MEDIAN
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

	if (g.doubleCheckDists)
	{
		//double totalNodes = 0.0;
		//maxNodes = 0;

#if (APP_TEST == TEST_KD_QNN)
		checkDistResults = true;

		/*
		// Check each query result (GPU vs. CPU)
		for (i = 0; i < nOrigQuery; i++)
		{
			// Need to use a fuzzy compare on distance
			float gpuDist = h_Results_GPU[i].Dist;
			float cpuDist = h_Results_CPU[i].dist;
			float eps = 1.0e-4f;
			unsigned int gpuID = h_Results_GPU[i].Id;
			unsigned int cpuID = h_Results_CPU[i].id;

			//unsigned int gpuCount = h_Results_GPU[i].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( h_Queries[i].x );
				double qY = static_cast<double>( h_Queries[i].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g> - GPU KD[%5d %.9g] != CPU KD[%5d %.9g] !!!\n",
						 i, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
		*/
#elif (APP_TEST == TEST_ALL_NN)
		checkDistResults = true;

		/*
		// Check each query result (GPU vs. CPU)
		for (i = 0; i < nOrigQuery; i++)
		{
			// Need to use a fuzzy compare on distance
			float gpuDist = h_Results_GPU[i].Dist;
			float cpuDist = h_Results_CPU[i].dist;
			float eps = 1.0e-4f;
			unsigned int gpuID = h_Results_GPU[i].Id;
			unsigned int cpuID = h_Results_CPU[i].id;

			//unsigned int gpuCount = h_Results_GPU[i].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( h_Queries[i].x );
				double qY = static_cast<double>( h_Queries[i].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g> - GPU KD[%5d %.9g] != CPU KD[%5d %.9g] !!!\n",
						 i, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
		*/
#elif (APP_TEST == TEST_KD_KNN)
		checkDistResults = true;

		// Check each query result (GPU vs. CPU)
		/*
		unsigned int j;
		for (i = 0; i < nOrigQuery; i++)
		{
			// Check each Nearest Neighbor
			for (j = 0; j < kVal; j++)
			{
				unsigned int baseIdx = (j * nPadQuery) + i;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				//float eps = 1.0e-4;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[i].x );
				double qY = static_cast<double>( h_Queries[i].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 i,j, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
			}
		}
		*/
#elif (APP_TEST == TEST_ALL_KNN)
		checkDistResults = true;
		/*
		// Check each query result (GPU vs. CPU)
		unsigned int j;
		for (i = 0; i < nOrigQuery; i++)
		{
			// Check each Nearest Neighbor
			for (j = 0; j < kVal; j++)
			{
				unsigned int baseIdx = (j * nPadQuery) + i;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				//float eps = 1.0e-4;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[i].x );
				double qY = static_cast<double>( h_Queries[i].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 i,j, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
			}
		}
		*/
#else
	// BUGBUG - do nothing for now ...
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
	for (i = 0; i < nOrigQuery; i++)
	{
		double gpuDist = static_cast<double>( h_Results_GPU[i].Dist );
		unsigned int gpuID = h_Results_GPU[i].Id;
		//unsigned int gpuCount = h_Results_GPU[i].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", i, gpuID, gpuDist, gpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", i, gpuID, gpuDist );
	}

	// CPU Results
	printf( "CPU Results\n" );
	for (i = 0; i < nOrigQuery; i++)
	{
		double cpuDist = static_cast<double>( h_Results_CPU[i].dist );
		unsigned int cpuID = h_Results_CPU[i].id;
		//unsigned int cpuCount = h_Results_CPU[i].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", i, cpuID, cpuDist, cpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", i, cpuID, cpuDist );
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
			printf( "KD - GPU Kernel computation,          time: %f msecs.\n", KD_GPU_dist );
			printf( "KD - CPU Kernel computation,          time: %f msecs.\n", KD_CPU_dist );
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
	if (NULL != h_KDNodes)     { free( h_KDNodes ); }
	if (NULL != h_IDs)         { free( h_IDs ); }
	if (NULL != h_Queries)     { free( h_Queries ); }
	if (NULL != h_Results_GPU) { free( h_Results_GPU ); }
#endif
	if (NULL != h_Results_CPU) { free( h_Results_CPU ); }

	if (NULL != d_KDNodes)
	{
		cutilSafeCall( cudaFree( d_KDNodes ) );
	}
	if (NULL != d_IDs)
	{
		cutilSafeCall( cudaFree( d_IDs ) );
	}
	if (NULL != d_Queries)
	{
		cutilSafeCall( cudaFree( d_Queries ) );
	}
	if (NULL != d_Results_GPU)
	{
		cutilSafeCall( cudaFree( d_Results_GPU ) );
	}

	FINI_CPU_2D_MEDIAN( &kdTree );

	FiniSearchQueryVectors( g );

	printf( "Shutdown done...\n\n" );

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	InitSearchQueryVectors
  Desc:	Create & Initialize Search Vectors
---------------------------------------------------------*/

bool InitSearchQueryVectors
( 
	AppGlobals & g, 
	bool bInitSearch,
	bool bInitQuery,
	bool bNonUniformSearch, 
	bool bNonUniformQuery, 
	int scale 
)
{
	/*-------------------------------------------
	  Check Parameters
	-------------------------------------------*/

	if (bInitSearch)
	{
		if (g.nSearch == 0) 
		{ 
			return false; 
		}
	}
	if (bInitQuery)
	{
		if (g.nQuery == 0) 
		{ 
			return false; 
		}
	}


	/*-------------------------------------------
	  Allocate Memory
	-------------------------------------------*/

	// Allocate memory for Search vector
	unsigned int mem_size_Search = g.nSearch * sizeof(float4);
	if (bInitSearch)
	{
		g.searchList = (float4*) malloc( mem_size_Search );
	}
	else 
	{
		g.searchList = NULL;
	}

	// Allocate memory for Query vector
	unsigned int mem_size_Query = g.nQuery * sizeof(float4);
	if (bInitQuery)
	{
		g.queryList = (float4*) malloc( mem_size_Query );
	}
	else
	{
		g.queryList = NULL;
	}
	

	/*-------------------------------------------
	  Setup Bounds
	-------------------------------------------*/

	float minS, maxS, minQ, maxQ;
	switch (scale)
	{
	case 1:
		// Search box 2x2 != query box 3x3
		minS = -1.0f;
		maxS =  1.0f;
		minQ = -1.5f;
		maxQ =  1.5f;
		break;

	case 2:
		// Search box 200x200 != Query Box 300x300
		minS = -100.0f;
		maxS =  100.0f;
		minQ = -150.0f;	
		maxQ =  150.0f;
		break;

	case 3:
		// Search box 3x3 != Query Box 2x2
		minS = -1.5f;
		maxS =  1.5f;
		minQ = -1.0f;	
		maxQ =  1.0f;
		break;

	default:
		// Search = query small box (1x1)
		minS = 0.0f;
		maxS = 1.0f;
		minQ = 0.0f;
		maxQ = 1.0f;
		break;
	}

	float minS_small = minS * 0.01f;
	float maxS_small = maxS * 0.01f;

	float minQ_small = minQ * 0.1f;
	float maxQ_small = maxQ * 0.1f;

	//float centerS = (maxS - minS)/2.0f;
	float centerS_small = (maxS_small - minS_small)/2.0f;

	float s_xOffset = minS + (maxS-minS) * 0.83f;
	float s_yOffset = minS + (maxS-minS) * 0.27f;
	float s_zOffset = minS + (maxS-minS) * 0.19f;

	float centerQ_small = (maxQ_small - minQ_small)/2.0f;

	float q_xOffset = minQ + (maxQ-minQ) * 0.83f;
	float q_yOffset = minQ + (maxQ-minQ) * 0.27f;
	float q_zOffset = minQ + (maxQ-minQ) * 0.19f;


	/*------------------------------------------------
	  Initialize Search points (to query against)
	------------------------------------------------*/

	float4 * searchList = (float4 *)g.searchList;

	int i;
	if (bInitSearch)
	{
		if (true == bNonUniformSearch)
		{
			// Put half the points in unit cube
			int oneHalf = g.nSearch>>1;
			for (i = 0; i < oneHalf; i++)	// Large Points
			{
				// BUGBUG - for now just randomly generate points
				// In future - we should read them in from a file...
				searchList[i].x = RandomFloat( minS, maxS );
				searchList[i].y = RandomFloat( minS, maxS );
				searchList[i].z = RandomFloat( minS, maxS );
				searchList[i].w = RandomFloat( minS, maxS );

				// Store point index in this channel
				//searchList[i].w = (float)i;
			}

			// Put other half of points in 1/100th of the cube

			for (i = oneHalf; i < g.nSearch; i++)	// Small Points
			{
				// BUGBUG - for now just randomly generate points
				// In future - we should read them in from a file...
				searchList[i].x = RandomFloat( minS_small, maxS_small ) - centerS_small + s_xOffset;
				searchList[i].y = RandomFloat( minS_small, maxS_small ) - centerS_small + s_yOffset;
				searchList[i].z = RandomFloat( minS_small, maxS_small ) - centerS_small + s_zOffset;
				searchList[i].w = RandomFloat( minS_small, maxS_small ) - centerS_small + s_zOffset;

				// Store point index in this channel
				//searchList[i].w = (float)i;
			}

		}
		else
		{
			for (i = 0; i < g.nSearch; i++)	// Original Points
			{
				// BUGBUG - for now just randomly generate points
				// In future - we should read them in from a file...
				searchList[i].x = RandomFloat( minS, maxS );
				searchList[i].y = RandomFloat( minS, maxS );
				searchList[i].z = RandomFloat( minS, maxS );
				searchList[i].w = RandomFloat( minS, maxS );

				// Store point index in this channel
				//searchList[i].w = (float)i;
			}
		}
	}


	//float r[4];
	//r[0] = RandomFloat( 0.0, 1.0 );
	//r[1] = RandomFloat( 0.0, 1.0 );
	//r[2] = RandomFloat( 0.0, 1.0 );
	//r[3] = RandomFloat( 0.0, 1.0 );


	/*---------------------------------
	  Initialize Query Points
	---------------------------------*/

	float4 * queryList = (float4 *)g.queryList;
	
	if (bInitQuery)
	{
		if (true == bNonUniformQuery)
		{
			// Put half the points in large cube
			int oneHalf = g.nSearch>>1;
			for (i = 0; i < oneHalf; i++)	// Large Points
			{
				// BUGBUG - for now just randomly generate points
				// In future - we should read them in from a file...
				queryList[i].x = RandomFloat( minQ, maxQ );
				queryList[i].y = RandomFloat( minQ, maxQ );
				queryList[i].z = RandomFloat( minQ, maxQ );
				queryList[i].w = RandomFloat( minQ, maxQ );

				// Store point index in this channel
				//queryList[i].w = (float)i;
			}

			// Put other half of points in smaller cube
			for (i = oneHalf; i < g.nSearch; i++)	// Small Points
			{
				// BUGBUG - for now just randomly generate points
				// In future - we should read them in from a file...
				queryList[i].x = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_xOffset;
				queryList[i].y = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_yOffset;
				queryList[i].z = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_zOffset;
				queryList[i].w = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_zOffset;

				// Store point index in this channel
				//queryList[i].w = (float)i;
			}
		}
		else
		{
			for (i = 0; i < g.nQuery; i++)	// Original Query Points
			{
				// BUGBUG - for now just randomly generate points
				// In future - we should read them in from a file...
				queryList[i].x = RandomFloat( minQ, maxQ );
				queryList[i].y = RandomFloat( minQ, maxQ );
				queryList[i].z = RandomFloat( minQ, maxQ );
				queryList[i].w = RandomFloat( minQ, maxQ );

				// ALL THE SAME HACK
				//queryList[i].x = r[0];
				//queryList[i].y = r[1];
				//queryList[i].z = r[2];
				//queryList[i].w = r[3];

				// Store point index in this channel
				//queryList[i].w = (float)i;
			}
		}
	}
	
	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	FiniSearchQueryVectors
  Desc:	Cleanup Search Vectors
---------------------------------------------------------*/

void FiniSearchQueryVectors( AppGlobals & g )
{
	// Cleanup Query List
	if (NULL != g.queryList)
	{
		float4 * tempList = (float4 *)(g.queryList);
		g.queryList = NULL;
		free( tempList );
	}
	g.nQuery = 0;

	// Cleanup Search List
	if (NULL != g.searchList)
	{
		float4 * tempList = (float4 *)(g.searchList);
		g.searchList = NULL;
		free( tempList );
	}
	g.nSearch = 0;
}
