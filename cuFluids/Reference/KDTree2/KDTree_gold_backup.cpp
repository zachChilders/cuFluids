/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void vectorAddGold( const float* A, const float* B, float * C, int len );

/*-----------------------------------------------------------------------------
  Name:  vectorAddGold
  Desc:  C = A + B as a vector
-----------------------------------------------------------------------------*/

void
vectorAddGold
(
	const float* A,
	const float* B,
	float * C,
	int len
)
{
    for( int i = 0; i < len; ++i) 
    {
        C[i] = A[i] + B[i];
    }
}

