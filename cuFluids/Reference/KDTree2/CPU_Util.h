#pragma once
#ifndef _CPU_UTIL_H_
#define _CPU_UTIL_H_
/*-----------------------------------------------------------------------------
  File:  CPU_Util.cpp
  Desc:  Various CPU helper functions

  Log:   Created by Shawn D. Brown (4/15/07)
		 Modified by Shawn D. Brown (3/22/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/


/*-------------------------------------
  inline functions
-------------------------------------*/

/*---------------------------------------------------------
  Name:	 KD_Min
  Desc:  returns the minimum of two values
---------------------------------------------------------*/

unsigned int KD_Min
( 
	unsigned int a,		// IN: 1st of 2 values to compare
	unsigned int b		// IN: 2nd of 2 values to compare
)
{
	// returns minimum of two values
	return ((a <= b) ? a : b);
}

/*---------------------------------------------------------
  Name:	 KD_Max
  Desc:  returns the maximum of two values
---------------------------------------------------------*/

unsigned int KD_Max
( 
	unsigned int a,		// IN: 1st of 2 values to compare
	unsigned int b		// IN: 2nd of 2 values to compare
)
{
	// returns the maximum of two values
	return ((a >= b) ? a : b);
}


/*-------------------------------------
  Functions declarations
-------------------------------------*/

