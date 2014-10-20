#pragma once
#ifndef _QUERY_RESULT_H
#define _QUERY_RESULT_H

/*-----------------------------------------------------------------------------
  Name:  QueryResult.h
  Desc:  Defines Simple Query Result for CPU

  Log:   Created by Shawn D. Brown (4/15/07)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Type Definitons
-------------------------------------*/

typedef struct 
{
	unsigned int Id;			// ID of Closest Point in search List
	float		 Dist;			// Distance to closest point in search list
	//unsigned int cVisited;	// Number of Nodes Visited during processing
	//unsigned int reserved;	// Dummy value for alignment padding
} CPU_NN_Result;

#endif // _QUERY_RESULT_H
