/*-----------------------------------------------------------------------------
  File:  CPU_BF.cpp
  Desc:  Brute Force Query Nearest Neighbors algorithm on CPU

  Log:   Created by Shawn D. Brown (4/15/07)
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
#include "CPUTree_API.h"
#include "GPUTree_API.h"

#include "CPUTree_MED.h"	// CPU kd-tree (median layout)
#include "CPUTree_LBT.h"	// CPU kd-tree (left-balanced layout)


/*---------------------------------------------------------
  Name: ComputeDist3D_CPU
  Desc: Computes distance for each point in
        a vector against query point
  Note:
        The optimal thread block size appears to be 
        32 TPR x 1 RPB = 32 TPB
        determined after extensive testing
---------------------------------------------------------*/

void ComputeDist3D_CPU
(
	      float*   dists,       // OUT: 'dists', solution vector, dist from each point to qp
	const float4*  points,      // IN:  'points' vector, n elements in length
	const float4 & queryPoint,  // IN:  'qp' query point to locate
	      int   w,              // IN:  'W' number of cols in 2D padded point Vector
	      int   h               // IN:	'H' number of rows in 2D padded point Vector
)
{
	int n = w * h;
	int i;
	float d, d2;

	for (i = 0; i < n; i++)	// Iterate over all elements
	{
		// Get Difference Vector between p[i] and queryPoint
		float4 diff;
		diff.x = points[i].x - queryPoint.x;
		diff.y = points[i].y - queryPoint.y;
		diff.z = points[i].z - queryPoint.z;
		diff.w = 0.0f;

		// Compute Distance between p[i] and queryPoint
		d2 = (diff.x * diff.x) +
		     (diff.y * diff.y) +
		     (diff.z * diff.z);
		d = sqrt( d2 );

		// Save Result to dists vector
		dists[i] = d;
	}
}


/*---------------------------------------------------------
  Name: Reduce_Min_CPU
  Desc: Finds Point with Min distance to query point
  Note:
        1. algorithm done on CPU
           as check on GPU algorithm
---------------------------------------------------------*/

void Reduce_Min_CPU
(
          int    & closestIndex,    // OUT: 'closestIndex', index of closest point to query point
          float  & closestDist,     // OUT: 'closestDist', distance between closest point and query point
    const float4 * points,          // IN:  'points' vector, n elements in length
    const float4 & queryPoint,      // IN:  'qp', point to locate
          int      n                // IN:  'n', number of points in solution vector
)
{
	int i, bestIdx;
	float d, d2, bestDist;
	float4 diff;

	// Check Parameters
	if ((NULL == points) || (n == 0))
	{
		// Error - invalid parameters
		return;
	}

	// Compute Distance to 1st Point
	diff.x = points[0].x - queryPoint.x;
	diff.y = points[0].y - queryPoint.y;
	diff.z = points[0].z - queryPoint.z;
	diff.w = 0.0f;

	d2 = (diff.x * diff.x) +
	     (diff.y * diff.y) +
	     (diff.z * diff.z);
	d = sqrt( d2 );

	// Start off assuming 1st point is closest to query point
	bestDist = d;
	bestIdx  = 0;

	// Brute force compare of all remaining points
	for (i = 1; i < n; i++)	
	{
		// Get Difference Vector between p[i] and queryPoint
		diff.x = points[i].x - queryPoint.x;
		diff.y = points[i].y - queryPoint.y;
		diff.z = points[i].z - queryPoint.z;
		diff.w = 0.0f;

		// Compute Distance between p[i] and queryPoint
		d2 = (diff.x * diff.x) +
		     (diff.y * diff.y) +
		     (diff.z * diff.z);
		d = sqrtf( d2 );

		// Is this point closer than current best point?
		if (d < bestDist)
		{
			// Found a closer point, update
			bestDist = d;
			bestIdx  = i;
		}
	}

	// Success, return closest point (& distance)
	closestIndex  = bestIdx;
	closestDist   = bestDist;
}


