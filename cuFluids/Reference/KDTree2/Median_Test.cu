/*-----------------------------------------------------------------------------
  File: Median_Test.cpp
  Desc: Runs Median Test 
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
#include "KD_API.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

extern AppGlobals g_app;


/*-------------------------------------
  CUDA Kernels
-------------------------------------*/

//#include <Median_GPU.cu>


/*-------------------------------------
  Function Declarations
-------------------------------------*/

bool RunMedianTest();


/*---------------------------------------------------------
  Name:	RunMedianTest()
  Desc:	Run a simple test of "Median Partition" 
        functionality on CUDA GPU framework
---------------------------------------------------------*/

bool RunMedianTest()
{
	bool bResult = false;

#if 0
	/*---------------------------------
	  Step 0.  Initialize Cuda
	---------------------------------*/

	cudaError_t cuda_err = cudaSuccess;

	// set seed for rand()
	srand( 2009 );

	g_app.hTimer = 0;
	cutCreateTimer( &(g_app.hTimer) );


	/*-------------------------------------------
	  Step 1.  Setup Initial parameters
	-------------------------------------------*/

		// Hard Coded for now...
	g_app.bgShape.nElems = g_app.nSearch;
	g_app.bgShape.threadsPerRow = MEDIAN_THREADS_PER_ROW;
	g_app.bgShape.rowsPerBlock  = MEDIAN_ROWS_PER_BLOCK;

	bResult = ComputeBlockShapeFromVector( g_app.bgShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

	// Make sure Matrix + vector is not to big to use up all device memory  // 768 Meg on Display Card
	int sizePoints = g_app.bgShape.nPadded * sizeof(float4);
	int sizeDists  = g_app.bgShape.nPadded * sizeof(float2);
	int totalMem   = sizePoints + (2*sizeDists);

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g_app.cudaProps.totalGlobalMem)
	{
		// Error - not enough memory to perform operation
		printf( "Matrix + Vector are too large for available device memory, running test will crash..." );
		return false;
	}

	// Setup GPU Kernel execution parameters
		// Median Sort Kernel
	dim3 dimBlock( g_app.bgShape.threadsPerRow, g_app.bgShape.rowsPerBlock, 1 );
	dim3 dimGrid( g_app.bgShape.blocksPerRow, g_app.bgShape.rowsPerGrid, 1 );


	/*-------------------------------------------
	  Step 2.  Allocate Vectors
	-------------------------------------------*/

	int nOrig = g_app.bgShape.nElems;
	int nPad  = g_app.bgShape.nPadded;
	int w     = g_app.bgShape.W;
	int h     = g_app.bgShape.H;

	/*-----------------------
	  Host Memory
	-----------------------*/

    // allocate host memory for original points (before median sort)
    int mem_size_Points = nPad * sizeof(float4);
    float4* h_Points_Orig = (float4*) malloc( (size_t)mem_size_Points );

	// allocate host memory for point results (after median sort)
	float4 *h_Points_Result = (float4*) malloc( mem_size_Points );

	// allocate host memory for CPU point results (after median sort)
	float4 *h_Points_CPU = (float4*) malloc( mem_size_Points );

	// Allocate host memory for singleton median index result
	unsigned int mem_size_Result = 16 * sizeof(I32);
	I32 *h_result_GPU = (I32 *) malloc( mem_size_Result );
	h_result_GPU[0] =  -1;


	/*-----------------------
	  Device Memory
	-----------------------*/

	// allocate device memory for points
	float4* d_Points;
	cutilSafeCall( cudaMalloc( (void**) &d_Points, mem_size_Points ) );

	// allocate device memory for points
	I32* d_result_GPU;
	cutilSafeCall( cudaMalloc( (void**) &d_result_GPU, mem_size_Result ) );

	// allocate device memory for Reduction Vector
		// Used for reduction
		// IE Ping Pong between dists vector and reduce vector to get answer
		// to get final answer
	//bool bPingPong = true;
	//float4* d_Reduce;
	//cutilSafeCall( cudaMalloc( (void **) &d_Reduce, mem_size_Points ) ); 


	/*-------------------------------------------
	  Step 3.  Initialize Vectors
	-------------------------------------------*/

	// Initialize Input points (to query against)
	int i;
	for (i = 0; i < nOrig; i++)	// Original Points
	{
		// BUGBUG - for now just randomly generate points
		// In future - we should read them in from a file...
		h_Points_Orig[i].x = RandomFloat( 0.0, 1.0 );
		h_Points_Orig[i].y = RandomFloat( 0.0, 1.0 );
		h_Points_Orig[i].z = RandomFloat( 0.0, 1.0 );

		// Store point index in this channel
		h_Points_Orig[i].w = (float)i;
	}

	// Initialize padded points (to query against)
	for (i = nOrig; i < nPad; i++)	// Padded points
	{
		// We want padded points to always fail...
		//	1st Approach, 
		//		Use a point that is so far away it is guranteed to never get picked
		//		Cons:  Requires advance knowledge of input point range 
		//				and query point range to pick a point
		//			    so far outside range it doesn't matter
		//	2nd Approach, 
		//		Duplicate the 1st point many times
		//		Cons:  Can fail because of numerical round-off errors
		//			IE what if the 1st point is really the closest to the query point
		//			which point wins (1st point or one of it's duplicates)
		
		//
		// 1st Approach
		//
		h_Points[i].x = 400.0f;	// Note:  Any number much larger than 46,000 and we will overflow on squaring the float
		h_Points[i].y = 400.0f;
		h_Points[i].z = 400.0f;
		h_Points[i].w = (float)-1;		// Store invalid point index in this channel

		//
		// 2nd Approach
		//
		//h_Points[i].x = h_Points[0].x;
		//h_Points[i].y = h_Points[0].y;
		//h_Points[i].z = h_Points[0].z;
		//h_Points[i].w = h_Points[0].w;
	}


	//
	// Profile Performance Metric Initialization
	//
	float MED_PNT_onto_device  = 0.0f;
	float MED_PNT_from_device  = 0.0f;
	float MED_M_from_device    = 0.0f;
	float MED_GPU_Kernel	   = 0.0f;
	float MED_CPU_Kernel	   = 0.0f;
	bool  checkMedianResults  = true;

	// Result values
	int gpuMedianIdx;			// Index of Median Point as computed on GPU
	int cpuMedianIdx;			// Index of Median Point as computed on CPU


	// Profile Measurement Loop
	unsigned int currIter;
	for (currIter = 0; currIter < g_app.profileActualLoops; currIter++)
	{

		//-------------------------------------------------------
		//	Step 3.  Move Points (& indices)
		//			 from main memory to device memory
		//-------------------------------------------------------

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Copy 'Points' vector from host memory to device memory
		cutilSafeCall( cudaMemcpy( d_Points, h_Points_Orig, mem_size_Points, cudaMemcpyHostToDevice ) );

		// Copy 'Initial' result vector from host memory to device memory
		cutilSafeCall( cudaMemcpy( d_result_GPU, h_result_GPU, mem_size_Results, cudaMemcpyHostToDevice ) );

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					MED_PNT_onto_device += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				MED_PNT_onto_device += cutGetTimerValue( g_app.hTimer );
			}
		}


		//---------------------------------
		//	Step 4.  Call Kernel Function
		//---------------------------------

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}
		
		// Excute the Brute Force Distance Kernel
		MedianSort_GPU<<< dimGrid, dimBlock >>>( d_Points, w, h  );
		
		// Check if GPU kernel execution generated an error
		cuda_err = cudaGetLastError();
		if( cudaSuccess != cuda_err) 
		{
			fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
			"MedianSort_GPU() failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					MED_GPU_Kernel += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				MED_GPU_Kernel += cutGetTimerValue( g_app.hTimer );
			}
		}


		//-------------------------------------------------
		//	Step 5.  Copy result vector (partitioned points)
		//			 from device memory to main memory
		//-------------------------------------------------

		if (g_app.doubleCheckDists)
		{
			// BUGBUG - this is a temporary step to verify brute force distance calculation
			if (g_app.profile)
			{
				// Start Timer
				cutResetTimer( g_app.hTimer );
				cutStartTimer( g_app.hTimer );
			}

			// copy result vector from device to host
			cutilSafeCall( cudaMemcpy( (void *) h_Points_Results, d_Points, mem_size_Points, cudaMemcpyDeviceToHost ) );

			// copy singleton median index from device to host
			cutilSafeCall( cudaMemcpy( (void *) h_results_GPU, d_results_GPU, mem_size_Results, cudaMemcpyDeviceToHost ) );

			if (g_app.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g_app.hTimer );
				if (g_app.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
					{
						MED_PNT_from_device += cutGetTimerValue( g_app.hTimer );
					}
				}
				else
				{
					MED_PNT_from_device += cutGetTimerValue( g_app.hTimer );
				}
			}
		}


		/*-------------------------------------------------
		  Step 6.  Double check GPU result 
		           against CPU result
		-------------------------------------------------*/

		if (g_app.doubleCheckDists)
		{
			if (g_app.profile)
			{
				// Start Timer
				cutResetTimer( g_app.hTimer );
				cutStartTimer( g_app.hTimer );
			}

			// Compute reference solution (distances) on CPU
			h_CPU_Idx = MedianSort_CPU( h_Points_CPU, h_Points_Orig, w, h );

			if (g_app.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g_app.hTimer );
				if (g_app.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
					{
						MED_CPU_Kernel += cutGetTimerValue( g_app.hTimer );
					}
				}
				else
				{
					MED_CPU_Kernel += cutGetTimerValue( g_app.hTimer );
				}
			}

			// Double check GPU Result against CPU result (for distances)
			int NCheck = nPad;
			int i;
			for (i = 0; i < NCheck; i++)
			{
				const float eps = 1.0e-2f;

				//printf( "[%d] GPU=%f, CPU=%f \n", i, gVal, cVal );

				if ( ((cVal - eps) >= gVal) ||
					 ((cVal + eps) <= gVal) )
				{
					// Error - Out of tolerance check range
					printf( "[%d] GPU %f != CPU %f \n", i, gVal, cVal );
					checkDistResults = false;
				}
			}
		}	// double check distances


		/*-------------------------------------------------
		  Step 7.  GPU Kernel to reduce distances 
		           (& index) vector
		           to single best result
		-------------------------------------------------*/

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Copy 'Distances' vector to 'Reduction' vector 
			// This is currently necessary to avoid garbage
			// results in output caused by unitialized values
		cutilSafeCall( cudaMemcpy( d_Reduce, d_Dists, mem_size_Dists_GPU, cudaMemcpyDeviceToDevice ) );

		int reduceElems  = nPad;
		dim3 reduceThreads;
		dim3 reduceGrid;
		BlockGridShape reduceShape;

		// Compute Initial Grid Shape
		reduceShape.nElems = reduceElems;
		reduceShape.threadsPerRow = BFMR_THREADS_PER_ROW;
		reduceShape.rowsPerBlock  = BFMR_ROWS_PER_BLOCK;

		ComputeBlockShapeFromVector( reduceShape );

		// Make sure we have an even number of blocks to work on
		if ((reduceShape.blocksPerRow % 2) != 0)
		{
			// Error - not an even number of blocks
			fprintf( stderr, "Error - not an even number of blocks\n" );
			return false;
		}

		reduceThreads.x = reduceShape.threadsPerRow;
		reduceThreads.y = reduceShape.rowsPerBlock;
		reduceThreads.z = 1;

		reduceGrid.x = reduceShape.blocksPerRow / 2;	// Divide by 2 (algorithm works on 2 blocks at a time)
		reduceGrid.y = reduceShape.rowsPerGrid;
		reduceGrid.z = 1;

				
		bool bReduced = false;
		bPingPong = true;
		while (!bReduced)
		{
			// Ping Pong between "Distances" and "Reduce" vectors
			if (bPingPong)
			{
				bPingPong = false;

				// Call GPU Kernel to reduce result vector by THREADS_PER_BLOCK
				Reduce_Min_GPU<<< reduceGrid, reduceThreads >>>( d_Reduce, d_Dists );
			}
			else
			{
				bPingPong = true;

				// Call GPU Kernel to reduce result vector by THREADS_PER_BLOCK
				Reduce_Min_GPU<<< reduceGrid, reduceThreads >>>( d_Dists, d_Reduce );
			}

			// Check if GPU kernel execution generated an error
			cuda_err = cudaGetLastError();
			if( cudaSuccess != cuda_err) 
			{
				fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
				"PLQ_GPU_BF_DIST() failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
				exit( EXIT_FAILURE );
			}
			
			// Update Number of elements in reduction vector
			reduceElems = reduceShape.blocksPerGrid / 2;	// Divide by 2 - Algorithm works on 2 columns of blocks at a time
			if (reduceElems == 1)
			{
				bReduced = true;
			}
			else
			{
				// Update Shape of Grid
				reduceShape.nElems = reduceElems;
				reduceShape.threadsPerRow = BFMR_THREADS_PER_ROW;
				reduceShape.rowsPerBlock  = BFMR_ROWS_PER_BLOCK;

				ComputeBlockShapeFromVector( reduceShape );

				// Make sure we have an even number of blocks to work on
				if ((reduceShape.blocksPerRow % 2) != 0)
				{
					// Error - not even number of blocks
					fprintf( stderr, "Error - not an even number of blocks" );
					return false;
				}

				reduceThreads.x = reduceShape.threadsPerRow;
				reduceThreads.y = reduceShape.rowsPerBlock;
				reduceThreads.z = 1;
		
				reduceGrid.x = reduceShape.blocksPerRow / 2;	// Divide by 2 (algorithm works on 2 blocks at a time)
				reduceGrid.y = reduceShape.rowsPerGrid;
				reduceGrid.z = 1;
			}
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					BF_GPU_min += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				BF_GPU_min += cutGetTimerValue( g_app.hTimer );
			}
		}


		/*-------------------------------------------------
		  Step 8.  Read Result from GPU
		-------------------------------------------------*/

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Copy closest point result from device to host memory (singleton distance & index)
		if (!bPingPong)
		{
			cuda_err = cudaMemcpy( h_result_GPU, d_Reduce, mem_size_Result, cudaMemcpyDeviceToHost );
		}
		else
		{
			cuda_err = cudaMemcpy( h_result_GPU, d_Dists, mem_size_Result, cudaMemcpyDeviceToHost );
		}
		if (cudaSuccess != cuda_err)
		{
		    fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
					 __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					BF_M_from_device += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				BF_M_from_device += cutGetTimerValue( g_app.hTimer );
			}
		}

		// Save Results 
		gpuMinDist = h_result_GPU[0].x;
		gpuMinIdx = (unsigned int)(h_result_GPU[0].y);


		/*-------------------------------------------------
		  Step 9.  Double check GPU result 
		           against CPU result
		-------------------------------------------------*/

		if (g_app.doubleCheckMin)
		{
			// BUGBUG - this is a temporary step to verify brute force distance calculation
			if (g_app.profile)
			{
				// Start Timer
				cutResetTimer( g_app.hTimer );
				cutStartTimer( g_app.hTimer );
			}

			// Compute reference solution (distances) on CPU
			Reduce_Min_CPU( cpuMinIdx, cpuMinDist, h_Points, queryPoint, nOrig );

			if (g_app.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g_app.hTimer );
				if (g_app.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
					{
						BF_CPU_min += cutGetTimerValue( g_app.hTimer );
					}
				}
				else
				{
					BF_CPU_min += cutGetTimerValue( g_app.hTimer );
				}
			}

			//
			// Double check GPU Result against CPU result
			//

				// Index check
			if (gpuMinIdx != cpuMinIdx)
			{
				// Warning - Indices are not the same
				// Note:  This is not truly an error unless 
				//		  the closest distances also don't match
				printf( "WARN - MIN GPU IDX %d != MIN CPU IDX %d \n", gpuMinIdx, cpuMinIdx );
			}

				// Distance Check
			const float minEps = 1.0e-4f;
			gVal = gpuMinDist;		
			cVal = cpuMinDist;
			if ( ((cVal - minEps) >= gVal) ||
				 ((cVal + minEps) <= gVal) )
			{
				// Error - Out of tolerance check range
				printf( "ERR  - MIN GPU DIST %f != MIN CPU DIST %f \n", i, gVal, cVal );
				checkMinResults = false;
			}
		}

	} // Profile Loops


	/*--------------------------------------------------------
	  Step 11. Print out Results
	--------------------------------------------------------*/

	int vectLen = g_app.nSearch;
	printf( "\n" );
	printf( "Search Vector Length = %d\n", vectLen );
	printf( "Query Point:          <%f %f %f>\n", 
			 queryPoint.x, queryPoint.y, queryPoint.z );
	printf( "GPU Closest Distance: %f\n", gpuMinDist );
	printf( "GPU Closest Index:    %d\n", gpuMinIdx );
	printf( "GPU Closest Point:    <%f %f %f>\n",
		    h_Points[gpuMinIdx].x, h_Points[gpuMinIdx].y, h_Points[gpuMinIdx].z );
	if (g_app.doubleCheckMin)
	{
		printf( "CPU Closest Distance: %f\n", cpuMinDist );
		printf( "CPU Closest Index:    %d\n", cpuMinIdx );
		printf( "CPU Closest Point:    <%f %f %f>\n",
		        h_Points[cpuMinIdx].x, h_Points[cpuMinIdx].y, h_Points[cpuMinIdx].z );
	}
	printf( "\n" );
	

	/*--------------------------------------------------------
	  Step 12. Print out Profile Performance Metrics
	--------------------------------------------------------*/

	// Does GPU Distance Kernel match up with CPU ?!?
	if (g_app.doubleCheckDists)
	{
		if (true == checkDistResults)
		{
			printf( "Distance check: CPU and GPU results agree within tolerance.\n" );
		}
		else
		{
			printf( "Distance check: CPU and GPU results don't agree within tolerance !!!\n" );
		}
	}

	// Does GPU Min Distance Kernel match up with CPU ?!?
	if (g_app.doubleCheckMin)
	{
		if (true == checkMinResults)
		{
			printf( "Min Distance check: CPU and GPU results agree within tolerance.\n" );
		}
		else
		{
			printf( "Min Distance check: CPU and GPU results don't agree within tolerance !!!\n" );
		}
	}

	// Dump Profile Info
	if (g_app.profile)
	{
		float loops = (float)g_app.profileActualLoops;
		float o_l = 1.0f / loops;

		float avgP   = BF_P_onto_device * o_l;
		float avgD   = BF_D_from_device * o_l;
		float avgM   = BF_M_from_device * o_l;
		float avgGPUdist = BF_GPU_dist * o_l;
		float avgCPUdist = BF_CPU_dist * o_l;
		float avgGPUmin  = BF_GPU_min * o_l;
		float avgCPUmin  = BF_CPU_min * o_l;

		// Verbose
		printf( "Number of profile loops = %f.\n", loops );
		printf( "BF - Copy 'Point' vector onto GPU,    time: %f msecs.\n", avgP );
		printf( "BF - Copy 'Dists' vector from GPU,    time: %f msecs.\n", avgD );
		printf( "BF - Copy 'Results' from GPU,         time: %f msecs.\n", avgM );
		printf( "BF - GPU Distance computation,        time: %f msecs.\n", avgGPUdist );
		printf( "BF - CPU Distance computation,        time: %f msecs.\n", avgCPUdist );
		printf( "BF - GPU Min Distance computation,    time: %f msecs.\n", avgGPUmin );
		printf( "BF - CPU Min Distance computation,    time: %f msecs.\n\n", avgCPUmin );

		// Terse
		//printf( "BF - P, D, M, G_D, C_D, G_M, C_M\n" );
		//printf( "     %f, %f, %f, %f, %f, %f, %f\n\n", avgP, avgD, avgM, avgGPUdist, avgCPUdist, avgGPUmin, avgCPUmin );

	}
	else
	{
		printf( "BF - Copy 'Point' vector onto GPU,    time: %f msecs.\n", BF_P_onto_device );
		printf( "BF - Copy 'Dists' vector from GPU,    time: %f msecs.\n", BF_D_from_device );
		printf( "BF - Copy 'Results' from GPU,         time: %f msecs.\n", BF_M_from_device );
		printf( "BF - GPU Distance computation,        time: %f msecs.\n", BF_GPU_dist );
		printf( "BF - CPU Distance computation,        time: %f msecs.\n", BF_CPU_dist );
		printf( "BF - GPU Min Distance computation,    time: %f msecs.\n", BF_GPU_min );
		printf( "BF - CPU Min Distance computation,    time: %f msecs.\n\n", BF_CPU_min );
	}


	/*---------------------------------
      Step 13.  Cleanup vector memory
	---------------------------------*/

    printf( "Shutting Down...\n" );

    // clean up allocations
    free( h_Points );
    free( h_Dists_GPU );
    free( h_Dists_CPU );
    free( h_result_GPU );

    cutDeleteTimer( g_app.hTimer );

    cutilSafeCall( cudaFree( d_Points ) );
    cutilSafeCall( cudaFree( d_Dists ) );
    cutilSafeCall( cudaFree( d_Reduce ) );

	printf( "Shutdown done...\n\n" );

#endif

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	ComputeBlockShapeFromVector
  Desc:	
  Notes:

	0. Assumes following members are initialized properly
	   before this funciton is called
		 shape.nElems = Number of original elements in vector
		 shape.tpr    = threads per row
		 shape.rpb    = rows per block

	1. Block Limits
		 Thread block has at most 512 theads per block

	2. Grid Limits
		Grid has at most 65,535 blocks in any dimension
			So a 1D grid is at most 65,535 x 1
			and a 2D grid is at most 65,535 x 65,535
		We use next smallest even number to these limits
			IE 65,535 - 1
			IE (65,535*65,535 - 1)
		It's useful to have an even number of columns
		in grid structure when doing reductions
---------------------------------------------------------*/

bool ComputeBlockShapeFromVector
( 
	BlockGridShape & bgShape	// IN/OUT - bgShape
)
{
	unsigned int remainder, extra;

	//---------------------------------
	//	1. Compute Threads Per Block
	//---------------------------------

	if ((bgShape.threadsPerRow > 512) || (bgShape.rowsPerBlock > 512))
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB) > 512 TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
	}

	// Compute threads per block
	bgShape.threadsPerBlock = bgShape.threadsPerRow * bgShape.rowsPerBlock;

	// Make sure we don't exceed block limits
	if (bgShape.threadsPerBlock > 512)
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB)  = %d TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock, bgShape.threadsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
		return false;
	}


	//---------------------------------
	//	2. Compute GRID structure
	//---------------------------------

	// Compute number of blocks needed to contain all elements in vector
	bgShape.blocksPerGrid  = bgShape.nElems / bgShape.threadsPerBlock;
	remainder			   = bgShape.nElems % bgShape.threadsPerBlock;
	extra				   = ((0 == remainder) ? 0 : 1);
	bgShape.blocksPerGrid += extra;

	// Check if need a 1D Grid or 2D Grid structure 
	// to contain all blocks in grid
	if (bgShape.blocksPerGrid <= 65534) 					  
	{
		// We can use a simple 1D grid of blocks
		bgShape.blocksPerRow = bgShape.blocksPerGrid;
		bgShape.rowsPerGrid  = 1;
	}
	else if (bgShape.blocksPerGrid <= 4294836224)	// 4294836224 = (65535 * 65535 - 1)
	{
		// We need to use a 2D Grid structure instead...
		// Use square root as an approximation for shape of 2D grid
		float sq_r = sqrtf( (float)( bgShape.blocksPerGrid ) );
		unsigned int uiSqrt = (unsigned int)sq_r;
		bgShape.blocksPerRow = uiSqrt;
		bgShape.rowsPerGrid  = uiSqrt;

		// Increment # of columns until we have enough space
		// in grid layout for all elements in vector
		while ((bgShape.blocksPerRow * bgShape.rowsPerGrid) < bgShape.blocksPerGrid)
		{
			bgShape.blocksPerRow++;
		}
	}
	else
	{
		// Error - Vector is too large for 2D Grid
		printf( "Vector is way too large...\n" );
		return false;
	}

	// Make sure # of columns in 1D or 2D grid is even
		// Which is useful to avoid special cases in reduction kernel
	remainder             = bgShape.blocksPerRow % 2;
	extra	              = ((0 == remainder) ? 0 : 1);
	bgShape.blocksPerRow += extra;

	// Compute # of padded blocks in Grid
	bgShape.blocksPerGrid = bgShape.blocksPerRow * bgShape.rowsPerGrid;

	// Make sure we don't exceed grid limits
	if ((bgShape.blocksPerRow >= 65535) || (bgShape.rowsPerGrid >= 65535))
	{
		// Error - Grid can't have more than 65535 blocks in any dimension
		printf( "Grid (%d BPR x %d RPG)  = %d BPG\n", 
				bgShape.blocksPerRow, bgShape.rowsPerGrid, bgShape.blocksPerGrid );
		printf( "Error - can't have more than 65535 blocks in any dimension\n" );
		return false;
	}

	// Compute Width and Height of 2D vector structure
	bgShape.W = bgShape.threadsPerRow * bgShape.blocksPerRow;	// Width (in elements) of 2D vector
	bgShape.H = bgShape.rowsPerBlock  * bgShape.rowsPerGrid;	// Height (in elements) of 2D vector

	// Compute padded length of 2D vector
	unsigned int sizeWH = bgShape.W * bgShape.H;
	unsigned int sizeBG = bgShape.threadsPerBlock * bgShape.blocksPerGrid;

	if (sizeWH != sizeBG)
	{
		// Programmer error-
		printf( "Error - sizes don't match\n" );
		return false;
	}

	// Compute number of elements in padded block structure
	bgShape.nPadded = sizeWH;

	// Success
	return true;
}


/*---------------------------------------------------------
  Name: ComputeBlockShapeFromQueryVector
  Desc:
  Notes:
    0. Assumes following members are initialized properly
       before this function is called
        shape.nElems = Number of original elements in query vector
        shape.tpr    = threads per row
        shape.rpb    = rows per block

    1. Block Limits
         Thread block has at most 512 theads per block

    2. Grid Limits
         Grid has at most 65,535 blocks in any dimension
           So a 1D grid is at most 65,535 x 1
           and a 2D grid is at most 65,535 x 65,535
         We use next smallest even number to these limits
           IE 65,535 - 1
           IE (65,535*65,535 - 1)
---------------------------------------------------------*/

bool ComputeBlockShapeFromQueryVector
(
	BlockGridShape & bgShape	// IN/OUT - bgShape
)
{
	unsigned int remainder, extra;

	//---------------------------------
	//	1. Compute Threads Per Block
	//---------------------------------

	if ((bgShape.threadsPerRow > 512) || (bgShape.rowsPerBlock > 512))
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB) > 512 TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
	}

	// Compute threads per block
	bgShape.threadsPerBlock = bgShape.threadsPerRow * bgShape.rowsPerBlock;

	// Make sure we don't exceed block limits
	if (bgShape.threadsPerBlock > 512)
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB)  = %d TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock, bgShape.threadsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
		return false;
	}


	//---------------------------------
	//	2. Compute GRID structure
	//---------------------------------

	// Compute number of blocks needed to contain all elements in vector
	bgShape.blocksPerGrid  = bgShape.nElems / bgShape.threadsPerBlock;
	remainder			   = bgShape.nElems % bgShape.threadsPerBlock;
	extra				   = ((0 == remainder) ? 0 : 1);
	bgShape.blocksPerGrid += extra;

	// Check if need a 1D Grid or 2D Grid structure 
	// to contain all blocks in grid
	if (bgShape.blocksPerGrid <= 65534)
	{
		// We can use a simple 1D grid of blocks
		bgShape.blocksPerRow = bgShape.blocksPerGrid;
		bgShape.rowsPerGrid  = 1;
	}
	else if (bgShape.blocksPerGrid <= 4294836224)	// 4294836224 = (65535 * 65535 - 1)
	{
		// We need to use a 2D Grid structure instead...
		// Use square root as an approximation for shape of 2D grid
		float sq_r = sqrtf( (float)( bgShape.blocksPerGrid ) );
		unsigned int uiSqrt = (unsigned int)sq_r;
		bgShape.blocksPerRow = uiSqrt;
		bgShape.rowsPerGrid  = uiSqrt;

		// Increment # of columns until we have enough space
		// in grid layout for all elements in vector
		while ((bgShape.blocksPerRow * bgShape.rowsPerGrid) < bgShape.blocksPerGrid)
		{
			bgShape.blocksPerRow++;
		}
	}
	else
	{
		// Error - Vector is too large for 2D Grid
		printf( "Vector is way too large...\n" );
		return false;
	}

	// Make sure we don't exceed grid limits
	if ((bgShape.blocksPerRow >= 65535) || (bgShape.rowsPerGrid >= 65535))
	{
		// Error - Grid can't have more than 65535 blocks in any dimension
		printf( "Grid (%d BPR x %d RPG)  = %d BPG\n", 
				bgShape.blocksPerRow, bgShape.rowsPerGrid, bgShape.blocksPerGrid );
		printf( "Error - can't have more than 65535 blocks in any dimension\n" );
		return false;
	}

	// Compute # of padded blocks in Grid
	bgShape.blocksPerGrid = bgShape.blocksPerRow * bgShape.rowsPerGrid;

	// Compute Width and Height of 2D vector structure
	bgShape.W = bgShape.threadsPerRow * bgShape.blocksPerRow;	// Width (in elements) of 2D vector
	bgShape.H = bgShape.rowsPerBlock  * bgShape.rowsPerGrid;	// Height (in elements) of 2D vector

	// Compute padded length of 2D vector
	unsigned int sizeWH = bgShape.W * bgShape.H;
	unsigned int sizeBG = bgShape.threadsPerBlock * bgShape.blocksPerGrid;

	if (sizeWH != sizeBG)
	{
		// Programmer error-
		printf( "Error - sizes don't match\n" );
		return false;
	}

	// Compute number of elements in padded block structure
	bgShape.nPadded = sizeWH;

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	RandomFloat
  Desc:	Generates a random float value in range [low,high]
---------------------------------------------------------*/

float RandomFloat( float low, float high )
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}


/*---------------------------------------------------------
  Name:	InitCUDA
  Desc:	Initialize CUDA system for GPU processing
---------------------------------------------------------*/

// Runtime API version...
bool InitCUDA( AppGlobals & g )
{
	bool bResult = false;
	int nDevices = 0;
	int deviceToUse = 0;
	unsigned int cudaContextFlags = 0;

	cudaError_t cudaResult = cudaSuccess;

#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cuda_Result = CUDA_SUCCESS;

	// Initialize CUDA
	unsigned int cudaFlags = 0;
	cuda_Result = cuInit( cudaFlags );
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cudaGetDeviceCount() failed
		fprintf( stderr, "InitCuda() - cuInit() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Get count of CUDA Devices
	cuda_Result = cuDeviceGetCount(&nDevices);
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cudaGetDeviceCount() failed
		fprintf( stderr, "InitCuda() - cuDeviceGetCount() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	if (nDevices <= 0)
	{
		// No Valid Display Device found
		cuda_Result = CUDA_ERROR_INVALID_DEVICE;
		fprintf( stderr, "InitCuda() - no valid display device found, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}
	else if (nDevices >= 2)
	{
		deviceToUse = 1;
	}

	// Get Specified Device
	cuda_Result = cuDeviceGet( &(g.currDevice), deviceToUse );
	if (CUDA_SUCCESS != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cuDeviceGet() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Get RAW Device Properties
	cuda_Result = cuDeviceGetProperties( &(g.rawProps), g.currDevice );
	if (CUDA_SUCCESS != cuda_Result )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cuDeviceGetProperties() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Set up the CUDA context
	cuda_Result = cuCtxCreate( &g.currContext, cudaContextFlags, g.currDevice );
	if ( CUDA_SUCCESS != cuda_Result )
	{
		// Error - cudaDeviceGet() failed
	    fprintf( stderr, "InitCuda() - cuCtxCreate() failed, error = %x in file '%s' in line %i.\n", 
			     cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Get CUDA Display Device Properties
	cudaResult = cudaGetDeviceProperties( &(g.cudaProps) , deviceToUse );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceProperties() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}


#elif (CUDA_PLATFORM == CUDA_CUDA)

	// Pick Display Device to perform GPU calculations on...
	cudaResult = cudaGetDeviceCount( &nDevices );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaGetDeviceCount() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceCount() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}

		// Note:  Assumes Device 0 = primary display device
		//		  Assumes Device 1 = work horse for CUDA
	if (nDevices <= 0)
	{
		// No Valid Display Device found
		cudaResult = cudaErrorInvalidDevice;
		fprintf( stderr, "InitCuda() - no valid display device found, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}
	else if (nDevices >= 2)
	{
		deviceToUse = 1;
	}

	// Get Display Device Properties
	cudaResult = cudaGetDeviceProperties( &(g.cudaProps) , deviceToUse );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceProperties() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}

	// Setup Display Device
	cudaResult = cudaSetDevice( deviceToUse );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaSetDevice() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}
#endif // CUDA_CUDA

	// Success
	bResult = true;

lblError:
	return bResult;
}


//---------------------------------------------------------
//	Name:	FiniCUDA
//	Desc:	Cleanup CUDA system
//---------------------------------------------------------

bool FiniCUDA()
{
#if (CUDA_PLATFORM == CUDA_DEVICE)
	// Detach CUDA from current thread
	CUresult cuda_Result = CUDA_SUCCESS;
	cuda_Result = cuCtxDetach( g_app.currContext );
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cuCtxDetach() failed
		fprintf( stderr, "FiniCUDA() - cuCtxDetach() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		return false;
	}
#endif

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	InitGlobals
  Desc:	Initialize Application Globals to Default
---------------------------------------------------------*/

bool InitGlobals( AppGlobals & g )
{
	//
	// Set Defaults
	//

	// Search Vectors
	g.nSearch	 = 100;
	g.searchList = NULL;
	g.nQuery	 = 10;
	g.queryList  = NULL;

	// Cuda Properties
	size_t byteSize;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	g.currDevice        = 0;

	// Initialize cuda device props to zero
	byteSize = sizeof( g.rawProps );
	memset( &g.rawProps, 0, byteSize );

#endif

	// Initialize cuda props to zero
	byteSize = sizeof( g.cudaProps );
	memset( &g.cudaProps, 0, byteSize );

	// Init Block Grid Shape
	InitShapeDefaults( g.bgShape );

	// App Properties
	g.nopromptOnExit    = 0;
	g.doubleCheckDists	= 1;

	// Profiling Info
	g.hTimer				= 0;
	g.profile				= 1;
	g.profileSkipFirstLast  = 0;
	g.profileRequestedLoops	= 1;
	g.profileActualLoops	= 1;

	return true;
}


/*---------------------------------------------------------
  Name:	GetCommandLineParameters
  Desc:	
---------------------------------------------------------*/

bool GetCommandLineParams
( 
	int argc,			// Count of Command Line Parameters
	const char** argv,	// List of Command Line Parameters
	AppGlobals & g		// Structure to store results in
)
{
	int iVal;

	// Prompt before exiting application ?!?
	if (cutCheckCmdLineFlag( argc, argv, "noprompt") ) 
	{
		g.nopromptOnExit = true;
	}
	else
	{
		g.nopromptOnExit = false;
	}

	// Double Check Distances
	if (cutCheckCmdLineFlag( argc, argv, "cdist") ) 
	{
		g.doubleCheckDists = true;
	}
	else
	{
		g.doubleCheckDists = false;
	}

	// Double Check Distances
	if (cutCheckCmdLineFlag( argc, argv, "cmin") ) 
	{
		g.doubleCheckMin = true;
	}
	else
	{
		g.doubleCheckMin = false;
	}


	// Get # Threads Per Row (block shape)
	if (cutGetCmdLineArgumenti( argc, argv, "TPR", &iVal )) 
	{
		if (iVal < 1) { iVal = 1; }
		g.bgShape.threadsPerRow = iVal;
	}

	// Get # Rows Per Block
	if (cutGetCmdLineArgumenti( argc, argv, "RPB", &iVal )) 
	{
		if (iVal < 1) { iVal = 1; }
		g.bgShape.rowsPerBlock = iVal;
	}

	// Calculate Threads Per Block
	g.bgShape.threadsPerBlock = g.bgShape.threadsPerRow * g.bgShape.rowsPerBlock;
	if (g.bgShape.threadsPerBlock > 512)
	{	
		// Error - Can't have more than 512 threads per block
		printf( "Max Threads Per Block is 512!!!\n\n" );
		return false;
	}

	// Get search Vector Length
	if (cutGetCmdLineArgumenti( argc, argv, "N", &iVal )) 
	{
		if (iVal < 1) { iVal = 10000; }
		g.nSearch = (int)iVal;
	}

	// Get Query Vector Length
	if (cutGetCmdLineArgumenti( argc, argv, "NQ", &iVal )) 
	{
		if (iVal < 1) { iVal = 100; }
		g.nQuery = (int)iVal;
	}

	// Should we do profiling (performance measurements) on application
	if (cutCheckCmdLineFlag( argc, argv, "profile") ) 
	{
		g.profile = true;
	}
	else
	{
		g.profile = false;
	}

	if (g.profile)
	{
		// Get Skip First Last flag
		if (cutCheckCmdLineFlag( argc, argv, "skip") ) 
		{
			g.profileSkipFirstLast = true;
		}
		else
		{
			g.profileSkipFirstLast = false;
		}

		// Get Number of Iterations for Profiling performance
		if (cutGetCmdLineArgumenti( argc, argv, "profile", &iVal )) 
		{
			if (iVal < 1) { iVal = 100; }
			g.profileRequestedLoops = iVal;
			if (g.profileSkipFirstLast)
			{
				g.profileActualLoops = g.profileRequestedLoops + 2;
			}
			else
			{
				g.profileActualLoops = g.profileRequestedLoops;
			}
		}
	}

	// Success
	return true;
}

/*---------------------------------------------------------
  Name:	InitShapeDefaults
---------------------------------------------------------*/

void InitShapeDefaults( BlockGridShape & bgShape )
{
	// Default Thread, Grid, Vector Properties
	bgShape.nElems			= 100;

	bgShape.threadsPerRow	= 1;
	bgShape.rowsPerBlock	= 1;
	bgShape.threadsPerBlock	= bgShape.threadsPerRow * bgShape.rowsPerBlock;

	bgShape.blocksPerRow	= 100;
	bgShape.rowsPerGrid		= 1;
	bgShape.blocksPerGrid	= bgShape.blocksPerRow * bgShape.rowsPerGrid;

	bgShape.W               = bgShape.threadsPerRow * bgShape.blocksPerRow;
	bgShape.H				= bgShape.rowsPerBlock * bgShape.rowsPerGrid;

	bgShape.nPadded			= bgShape.W * bgShape.H;
}


/*---------------------------------------------------------
  Name:	DumpBlockGridShape
---------------------------------------------------------*/

void DumpBlockGridShape( BlockGridShape & bgShape )
{
	printf( "N = %d, NPadded = %d\n",
		    bgShape.nElems, bgShape.nPadded );
	printf( "Block (%d TPR x %d RPB) = %d TPB\n",
		    bgShape.threadsPerRow, bgShape.rowsPerBlock,
			bgShape.threadsPerBlock );
	printf( "Grid (%d BPR x %d RPG)  = %d BPG\n",
		    bgShape.blocksPerRow, bgShape.rowsPerGrid,
			bgShape.blocksPerGrid );
	printf( "W = %d, H = %d\n",
		    bgShape.W, bgShape.H );
}

