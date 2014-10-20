#pragma once
#ifndef _KDTREE_CPU_H
#define _KDTREE_CPU_H
//-----------------------------------------------------------------------------
//	Name:	KDTree_CPU.h
//	Desc:	Defines Simple KDTree Class
//	Log:	Created by Shawn D. Brown 4/15/07
//	
//	Copyright (C) by UNC-Chapel Hill, All Rights Reserved
//-----------------------------------------------------------------------------

//-------------------------------------
//	Includes
//-------------------------------------

#include <math.h>
#include <vector>
#include <deque>
#include <string>

#include "Point_CPU.h"
#include "Box_CPU.h"
#include "Closest_CPU.h"


//-------------------------------------
//	Enumerations
//-------------------------------------


//-------------------------------------
//
//	Functions
//
//-------------------------------------

//-------------------------------------------------------------------------
//	Name:	AxisToString
//	Desc:	Get Human Readable string for Axis
//-------------------------------------------------------------------------

static char * AxisToString( Axis currAxis )
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
	default:
		break;
	}
	return sz;
}



//-------------------------------------
//
//	Classes
//
//-------------------------------------

//---------------------------------------------------------
//	Class:	KDTreeNode
//	Desc:	Simple Node for use in KD Tree
//---------------------------------------------------------


//---------------------------------------------------------
//	Class:	KDTree
//	Desc:	Simple 'n' dimensional KDTree 
//	T = Value Type
//	MD = max dimension of KD Tree typically 2 or 3
//---------------------------------------------------------

template 
<typename T, 
 unsigned int MD >
class KDTree_T
{
public:
	//---------------------------------
	// Type Definitions
	//---------------------------------

	typedef unsigned int				SIZE_TYPE;
	typedef T							VALUE_TYPE;	
	typedef Point2D_T<T>				POINT2D_TYPE;
	typedef Point3D_T<T>				POINT_TYPE;
	typedef std::vector<POINT_TYPE>		POINT_LIST_TYPE;			// Vector (Dynamic Array) of 3D Points
	typedef Box2D_T<T>					BOX2D_TYPE;
	typedef Box3D_T<T>					BOX_TYPE;
	typedef unsigned int				NODE_TYPE;
	typedef std::vector<unsigned int>	NODE_LIST_TYPE;				// Vector (Dynamic Array) of Nodes
	typedef ClosestNodes_T<T>			CLOSEST_LIST_TYPE;			// Vector of Closest Nodes

		// Range [start, end]
	typedef struct _RANGE_TYPE {
		SIZE_TYPE	start;		// Start of Range (inclusive)
		SIZE_TYPE	end;		// End of Range (inclusive
	} RANGE_TYPE;

protected:

	//---------------------------------
	// Fields
	//---------------------------------
	ISurfaceModuleQueryInterface * m_querySurface;	// Surface that This KD Tree Wraps
	NODE_LIST_TYPE	  m_nodes;		// List of Nodes (indices to points of surface) in KD Tree
	BOX_TYPE		  m_bounds;		// Bounds of entire KD Tree
	SIZE_TYPE		  m_startAxis;	// Starting Axis


	//---------------------------------
	// Helper Methods
	//---------------------------------

	static SIZE_TYPE NextDimension( SIZE_TYPE currDim )
	{
		SIZE_TYPE nextDim = currDim + 1;
		if (nextDim >= static_cast<SIZE_TYPE>( MD ))
		{
			nextDim = static_cast<SIZE_TYPE>( 0 );
		}
		return nextDim;
	}

	static SIZE_TYPE PrevDimension( SIZE_TYPE currDim )
	{
		SIZE_TYPE prevDim;
		if (currDim == 0)
		{
			prevDim = static_cast<SIZE_TYPE>( MD - 1 );
		}
		else
		{
			prevDim = currDim - 1;
		}
		return prevDim;
	}


	void Init()
	{
		m_nodes.clear();
		BOX_TYPE empty;
		m_bounds = empty;
	}

	// Get Length of Point in relevant dimensions
	static float KDLength2( POINT_TYPE & point ) 
	{
		float len = 0.0f;
		for (SIZE_TYPE i = 0; i < static_cast<SIZE_TYPE>( MD ); i++)
		{
			len += point[i] * point[i];
		}
		return len;
	}

	// Get Length of Point in relevant dimensions
	static float KDLength( POINT_TYPE & point ) 
	{
		float len = 0.0f;
		for (SIZE_TYPE i = 0; i < static_cast<SIZE_TYPE>( MD ); i++)
		{
			len += point[i] * point[i];
		}
		return sqrt( len );
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::FiniNodes
	//	Desc:	Cleanup Node Tree resource
	//-------------------------------------------------------------------------

	void FiniNodes()
	{
		m_nodes.clear();
	}

	//-------------------------------------------------------------------------
	//	Name:	KDTree::Fini
	//	Desc:	Cleanup resources
	//-------------------------------------------------------------------------

	void Fini()
	{
		FiniNodes();
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::CopyNodes
	//	Desc:	Copy nodes into this node list
	//-------------------------------------------------------------------------

	bool CopyNodes( const NODE_LIST_TYPE & nodes )
	{
		// Cleanup old list
		FiniNodes();

		// Get Size of new node list
		SIZE_TYPE nNodes = static_cast< SIZE_TYPE >( nodes.size() );
		if (nNodes == 0) { return true; }

		// Allocate Space for new node list
		m_nodes.resize( nNodes );

		// Copy nodes over
		for (SIZE_TYPE i = 0; i < nNodes; i++)
		{
			m_nodes[i] = nodes[i];
		}

		// Success
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::Copy
	//	Desc:	Copy 
	//-------------------------------------------------------------------------

	bool Copy( const KDTree_T<T,MD> & toCopy )
	{
		if (this == &toCopy) { return true; }

		Fini();

		CopyNodes( toCopy.m_nodes );

		m_querySurface = toCopy.m_querySurface;
		m_bounds	   = toCopy.m_bounds;
		m_startAxis    = toCopy.m_startAxis;

		// Success
		return true;
	}


public:
	//---------------------------------
	// Properties
	//---------------------------------

	const ISurfaceModuleQueryInterface * QUERY_SURFACE() const { return m_querySurface; }
	void QUERY_SURFACE( const ISurfaceModuleQueryInterface * value ) { m_querySurface = const_cast<ISurfaceModuleQueryInterface *>( value ); }

	//const POINT_LIST_TYPE & POINTLIST() const { return m_points; }
	//SIZE_TYPE NUM_POINTS() const { return m_points.size(); }

	const NODE_LIST_TYPE & NODELIST() const { return m_nodes; }
	SIZE_TYPE NUM_NODES() const { return m_nodes.size(); }

	static SIZE_TYPE MEDIAN_INDEX( SIZE_TYPE start, SIZE_TYPE end )
		{
			return ((start+end)/2);
		}

	NODE_TYPE & MEDIAN_NODE( SIZE_TYPE start, SIZE_TYPE end ) const 
		{ 
			SIZE_TYPE medianIndex = MEDIAN_INDEX( start, end );
			return m_nodes[medianIndex];
		}

	bool IsValid() const 
		{
			return (((m_querySurface != NULL) && (m_nodes.size() > 0)) ? true : false);
		}


	//---------------------------------
	// Constructors
	//---------------------------------

	//-------------------------------------------------------------------------
	//	Name:	KDTree::KDTree
	//	Desc:	Default Constructor
	//-------------------------------------------------------------------------

	KDTree_T<T,MD>() :
		m_querySurface( NULL ),
		m_nodes(),
		m_bounds(),
		m_startAxis( X_AXIS )
		{
		}

	//-------------------------------------------------------------------------
	//	Name:	KDTree::KDTree
	//	Desc:	Copy Constructor
	//-------------------------------------------------------------------------

	KDTree_T<T,MD>( const KDTree_T<T,MD> & toCopy ) :
		m_querySurface( NULL ),
		m_nodes(),
		m_bounds(),
		m_startAxis( X_AXIS )
		{
			Copy( toCopy );
		}

	//-------------------------------------------------------------------------
	//	Name:	KDTree::~KDTree
	//	Desc:	Destructor
	//-------------------------------------------------------------------------
	
	~KDTree_T<T,MD>()
		{
			Fini();
		}

	//---------------------------------
	// Operators
	//---------------------------------

	//-------------------------------------------------------------------------
	//	Name:	KDTree::operator =
	//	Desc:	Copy Operator
	//-------------------------------------------------------------------------

	KDTree_T<T,MD> & operator = ( const KDTree_T<T,MD> & toCopy )
		{
			Copy( toCopy );
			return (*this);
		}

	//---------------------------------
	// Methods
	//---------------------------------

	//-------------------------------------------------------------------------
	//	Name:	KDTree::SwapNodes
	//	Desc:	swaps two points in point list
	//-------------------------------------------------------------------------

	static bool SwapNodes( NODE_LIST_TYPE & nodes, int i, int j )
		{
			int nNodes = static_cast<int>( nodes.size() );
			if (( i < 0) || (i >= nNodes) || 
				( j < 0) || (j >= nNodes) )
			{
				// Error - out of range
				return false;
			}

			unsigned int temp = nodes[i];
			nodes[i] = nodes[j];
			nodes[j] = temp;
			return true;
		}

	static bool SwapNodes( NODE_LIST_TYPE & nodes, SIZE_TYPE i, SIZE_TYPE j )
		{
			SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( nodes.size() );
			if ( (i >= nNodes) || (j >= nNodes) )
			{
				// Error - out of range
				return false;
			}

			unsigned int temp = nodes[i];
			nodes[i] = nodes[j];
			nodes[j] = temp;
			return true;
		}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::GetPoint2D
	//	Desc:	get point at index in point list
	//-------------------------------------------------------------------------

	bool GetPoint2D( SIZE_TYPE index, POINT2D_TYPE & resultPoint ) const
		{
			//Assert( m_querySurface != NULL);
			m_querySurface->Query2DPointAtIndex( index, resultPoint );
			return true;
		}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::GetPoint3D
	//	Desc:	get point at index in point list
	//-------------------------------------------------------------------------

	bool GetPoint3D( SIZE_TYPE index, POINT_TYPE & resultPoint ) const
		{
			//Assert( m_querySurface != NULL);
			m_querySurface->Query3DPointAtIndex( index, resultPoint );
			return true;
		}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::GetNode
	//	Desc:	get node at index in node list
	//-------------------------------------------------------------------------

	static bool GetNode( NODE_LIST_TYPE & nodes, SIZE_TYPE index, NODE_TYPE & node )
		{
			SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( nodes.size() );
			if (index >= nNodes)
			{
				// Error - out of range
				return false;
			}

			// Success
			node = nodes[index];
			return true;
		}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::GetNodeValue
	//	Desc:	Helper Method
	//-------------------------------------------------------------------------

	static VALUE_TYPE GetNodeValue
	( 
		const NODE_LIST_TYPE & nodes,
		SIZE_TYPE index, 
		SIZE_TYPE axis,
		ISurfaceModuleQueryInterface * querySurface
	)
	{
		const NODE_TYPE currNode = nodes[index];
		POINT_TYPE currPoint;
		querySurface->Query3DPointAtIndex( currNode, currPoint );
		VALUE_TYPE currValue = currPoint[axis];
		return currValue;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::MedianSortNodes
	//	Desc:	Sorts nodes between [start,end] into 2 buckets
	//			nodes with points below median, and nodes with points above median
	//	Notes:	Should take O(N) time to process all points in input range
	//	Creaed:	Based on "Selection Sort" algorithm as
	//			presented by Robert Sedgewick in the book
	//			"Algorithms in C++"
	//-------------------------------------------------------------------------

	static bool MedianSortNodes
	(
		NODE_LIST_TYPE & nodes,		// IN/OUT - list of nodes
		SIZE_TYPE start,			// IN - start of range
		SIZE_TYPE end,				// IN - end of range
		SIZE_TYPE median,			// IN - approximate median number
		SIZE_TYPE axis,				// IN - dimension(axis) to split along (x,y,z)
		ISurfaceModuleQueryInterface * querySurface	// IN - Surface Module to query point info from...
	)
	{
		// Check Parameters
		int nNodes = static_cast<int>( nodes.size() );
		if (nNodes == 0) { return false; }

		int left  = static_cast<int>( start );
		int right = static_cast<int>( end );
		int middle = static_cast<int>( median );

		if (left >= nNodes) { left = 0; }
		if (right >= nNodes) { right = nNodes-1; }
		if (left > right) { std::swap( left, right ); }
		if ((middle < left) || (middle > right)) { middle = (left+right)/2; }

		while ( right > left ) 
		{
			VALUE_TYPE currVal = GetNodeValue( nodes, right, axis, querySurface );

			int i = left - 1;
			int j = right;

			for (;;) 
			{
				while ( GetNodeValue( nodes, ++i, axis, querySurface ) < currVal )
					;

				while ( (GetNodeValue( nodes, --j, axis, querySurface ) > currVal) && (j > left) )
					;

				if ( i >= j )
					break;

				SwapNodes( nodes, i, j );
			}

			SwapNodes( nodes, i, right );

			if ( i >= middle )
				right = i-1;
			if ( i <= middle )
				left = i+1;
		}

		// Success
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::Build
	//	Desc:	Build KDTree from point list
	//-------------------------------------------------------------------------

	bool Build()
	{
		// Make sure we have a valid surface to work with
		if (m_querySurface == NULL) { return false; }

		// Release current resources for KD Tree
		FiniNodes();

		// Get Surface Capabilities (needed to get number of points and bounds)
		VS_Caps qsCaps;
		m_querySurface->QueryCapabilities( qsCaps );

		// Get Number of Points in surface
		SIZE_TYPE nPoints = qsCaps.NumGroundTruthPoints();
		if (nPoints == static_cast<SIZE_TYPE>( 0 )) { return false; }

		// Update Node indices
		m_nodes.resize( nPoints );
		for (SIZE_TYPE i = 0; i < nPoints; i++)
		{
			m_nodes[i] = i;
		}

		// Get Bounding Box containing all points in surface
		m_bounds    = qsCaps.Bounds();
		m_startAxis = X_AXIS;

		// Allocate Space for Initial Node List
		m_nodes.resize( nPoints );

		// Add Root Range to Build Queue
		RANGE_TYPE currRange;
		currRange.start = 0;
		currRange.end   = nPoints - 1;
		SIZE_TYPE currAxis = m_startAxis;

		SIZE_TYPE currStart, currEnd;
		SIZE_TYPE median;
		SIZE_TYPE nextAxis;

		std::deque<RANGE_TYPE> rangeQueue;
		std::deque<SIZE_TYPE>  axisQueue;
		rangeQueue.push_back( currRange );
		axisQueue.push_back( currAxis );

		// Process child ranges until we reach 1 node per range (leaf nodes)
		bool bDone = false;
		while (! bDone)
		{
			// Is Build queue empty ?
			if (rangeQueue.empty())
			{
				bDone = true;
			}
			else
			{
				// Get top node from build queue to do work on
				currRange = rangeQueue.front();
				currAxis  = axisQueue.front();

				currStart = currRange.start;
				currEnd   = currRange.end;
				median    = (currStart + currEnd) / 2;		// Set subroot for this range
				nextAxis  = NextDimension( currAxis );

				// No need to do median sort if only one element is in range (IE a leaf node)
				if (currEnd > currStart)
				{
					// Sort nodes into 2 buckets (on axis plane)
					bool bResult = MedianSortNodes( m_nodes, currStart, currEnd, median, currAxis, m_querySurface );
					if (false == bResult) { return false; }
				}

				if (currStart < median)
				{
					// Add left child range to build queue
					currRange.start = currStart;
					currRange.end   = median - 1;
				
					rangeQueue.push_back( currRange );
					axisQueue.push_back( nextAxis );
				}

				if (median < currEnd)
				{
					// Add right child range to build queue
					currRange.start = median + 1;
					currRange.end   = currEnd;
					rangeQueue.push_back( currRange );
					axisQueue.push_back( nextAxis );
				}

				// Pop front element from build queue
				rangeQueue.pop_front();
				axisQueue.pop_front();
			}
		}

		// Success
		return true;
	}

	bool Build( const ISurfaceModuleQueryInterface * querySurface )
	{
		// Check Parameters
		if (querySurface == NULL) { return false; }

		QUERY_SURFACE( querySurface );

		return Build();
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::Validate
	//	Desc:	Validates Tree is correctly built
	//-------------------------------------------------------------------------

	bool Validate()
	{
		bool bResult = true;

		// Check Parameters
		if ( m_querySurface == NULL ) 
			{ 
				Dumpf( TEXT( "KDTree::Validate - No Valid Query Surface Pointer associated with this KD Tree" ) );
				return false; 
			}

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes == static_cast<SIZE_TYPE>( 0 )) 
			{ 
				Dumpf( TEXT( "KDTree::Validate - NO Nodes in NodeList" )
					   TEXT( "\r\n\tDid you forget to build KD Tree?" )
					   TEXT( "\r\n\tDoes associated surface have 2 or more points in list" ) );
				return false; 
			}

		// Validate Each Node Individually
		POINT_TYPE currPoint;

		for (SIZE_TYPE i = 0; i < nNodes; i++)
		{
			NODE_TYPE currNode = m_nodes[i];			
			GetPoint3D( currNode, currPoint );

			// Make sure each point is inside original bounding box
			bool bInBox = InBox( m_bounds, currPoint );
			if (!bInBox)
			{
				// Error - Point outside Bounding Box
				std::string pointString;
				currPoint.ToString( pointString );

				std::string boxString;
				m_bounds.ToString( boxString );
				unsigned int index = static_cast<unsigned int>( i );

				Dumpf( TEXT( "KDTree::Validate - Node[%d] = %s is outside original bounding box = %S" ), 
						index, pointString.c_str(), boxString.c_str() );
			}
		}

		// Validate KD Tree Layout of Nodes
		SIZE_TYPE currAxis, nextAxis;

		SIZE_TYPE start = 0;
		SIZE_TYPE end   = nNodes - 1;
		SIZE_TYPE median = (start+end)/2;

		RANGE_TYPE currRange;
		currRange.start = start;
		currRange.end   = end;

		BOX2D_TYPE currBounds;
		BOX2D_TYPE leftBounds;
		BOX2D_TYPE rightBounds;

		POINT_TYPE leftPoint, rightPoint;

		// Setup Search Queue
		currBounds.SetFromMinMax( m_bounds.MINX(), m_bounds.MINY(),
			                      m_bounds.MAXX(), m_bounds.MAXY() );

		std::deque<RANGE_TYPE> rangeQueue;
		std::deque<SIZE_TYPE> axisQueue;
		std::deque<BOX2D_TYPE> boundsQueue;

		rangeQueue.push_back( currRange );
		axisQueue.push_back( m_startAxis );
		boundsQueue.push_back( currBounds );

		int nNodesProcessed = 0;

		while (! rangeQueue.empty())
		{
			nNodesProcessed++;

			// Get Node off top of queue
			currRange	= rangeQueue.front();
			currAxis	= axisQueue.front();
			currBounds	= boundsQueue.front();

			nextAxis = NextDimension( currAxis );

			start  = currRange.start;
			end    = currRange.end;
			median = (start+end)/2;

			NODE_TYPE currNode = m_nodes[median];
			GetPoint3D( currNode, currPoint );

			VALUE_TYPE splitValue = currPoint[currAxis];

			// Make sure all values are properly ordered
			if (start < median)
			{
				for (SIZE_TYPE left = start; left < median; left++)
				{
					NODE_TYPE leftNode = m_nodes[left];
					GetPoint3D( leftNode, leftPoint );
					VALUE_TYPE leftValue = leftPoint[currAxis];
					if (leftValue > splitValue)
					{
						unsigned int st = static_cast<unsigned int>( start );
						unsigned int ed = static_cast<unsigned int>( end );
						unsigned int md = static_cast<unsigned int>( median );
						unsigned int index = static_cast<unsigned int>( left );

						double lv = static_cast<double>( leftValue );
						double sv = static_cast<double>( splitValue );

						Dumpf( TEXT( "KDTree::Validate - Left Range Node[S=%d,E=%d,M=%d] INDEX %d = %s, leftValue (%f) > splitValue = %f" ), 
								st, ed, md, index, lv, sv );
					}
				}
			}
			if (median < end)
			{
				for (SIZE_TYPE right = median+1; right <= end; right++)
				{
					NODE_TYPE rightNode = m_nodes[right];
					GetPoint3D( rightNode, rightPoint );
					VALUE_TYPE rightValue = rightPoint[currAxis];
					if (rightValue < splitValue)
					{
						unsigned int st = static_cast<unsigned int>( start );
						unsigned int ed = static_cast<unsigned int>( end );
						unsigned int md = static_cast<unsigned int>( median );
						unsigned int index = static_cast<unsigned int>( right );

						double rv = static_cast<double>( rightValue );
						double sv = static_cast<double>( splitValue );

						Dumpf( TEXT( "KDTree::Validate - Right Range Node[S=%d,E=%d,M=%d] INDEX %d = %s, rightValue (%f) < splitValue = %f" ), 
								st, ed, md, index, rv, sv );
					}
				}
			}

			// Is Median point inside curr Bounds ?!?
			bool bInBox = InBox( currBounds, currPoint );
			if (!bInBox)
			{
				// Error - Point outside current bounding box
				std::string pointString;
				currPoint.ToString( pointString );

				std::string boxString;
				currBounds.ToString( boxString );

				unsigned int st = static_cast<unsigned int>( start );
				unsigned int ed = static_cast<unsigned int>( end );
				unsigned int md = static_cast<unsigned int>( median );

				Dumpf( TEXT( "KDTree::Validate - Median Node[S=%d,E=%d,M=%d] = %s is outside bounding box = %S" ), 
					   st, ed, md, pointString.c_str(), boxString.c_str() );

				bResult = false;
			}

			if (start < end)
			{
				// Check Current Nodes Region against query region
				for (SIZE_TYPE j = start; j < end; j++)
				{
					NODE_TYPE subNode = m_nodes[j];
					POINT_TYPE subPoint;
					GetPoint3D( subNode, subPoint );

					bool bInSubBox = InBox( currBounds, subPoint );
					if (!bInSubBox)
					{
						// Error - Point outside current bounding box
						std::string pointString;
						subPoint.ToString( pointString );

						std::string boxString;
						currBounds.ToString( boxString );

						unsigned int st = static_cast<unsigned int>( start );
						unsigned int ed = static_cast<unsigned int>( end );
						unsigned int md = static_cast<unsigned int>( median );
						unsigned int index = static_cast<unsigned int>( j );

						Dumpf( TEXT( "KDTree::Validate - Node[S=%d,E=%d,M=%d], INDEX %d = %s is outside bounding box = %S" ), 
							   st, ed, md, index, pointString.c_str(), boxString.c_str() );

						bResult = false;
					}
				}

				// Get Split Value
				VALUE_TYPE splitValue = currPoint[currAxis];

				SplitBox( currBounds, currAxis, splitValue, leftBounds, rightBounds );

				bool bTestLeft = TestContains( currBounds, leftBounds );
				bool bTestRight = TestContains( currBounds, rightBounds );
				if ((bTestLeft == false) || (bTestRight == false))
				{
					unsigned int index = static_cast<unsigned int>( median );
					std::string boxCurrString, boxLeftString, boxRightString;
					currBounds.ToString( boxCurrString );
					leftBounds.ToString( boxLeftString );
					rightBounds.ToString( boxRightString );

					unsigned int st = static_cast<unsigned int>( start );
					unsigned int ed = static_cast<unsigned int>( end );
					unsigned int md = static_cast<unsigned int>( median );

					Dumpf( TEXT( "KDTree::Validate - Median Node[S=%d,E=%d,M=%d] currBounds = %s does not contain left bounds = %S and/or right bounds = %S" ), 
						  st, ed, md, index, boxCurrString.c_str(), boxLeftString.c_str(), boxRightString.c_str() );
					bResult = false;
				}

				if (start < median)
				{
					// Add Left Range to work queue
					currRange.start = start;
					currRange.end   = median-1;

					rangeQueue.push_back( currRange );
					axisQueue.push_back( nextAxis );
					boundsQueue.push_back( leftBounds );
				}

				if (median < end)
				{
					// Add Right Range to work queue
					currRange.start = median+1;
					currRange.end   = end;

					rangeQueue.push_back( currRange );
					axisQueue.push_back( nextAxis );
					boundsQueue.push_back( rightBounds );
				}
			}

			// Finished processing this node, get rid of it
			rangeQueue.pop_front();
			axisQueue.pop_front();
			boundsQueue.pop_front();
		}

		// Success
		return bResult;

	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::AddNodesInRangeToOutputList
	//	Desc:	Helper Function
	//-------------------------------------------------------------------------

	bool AddNodesInRangeToOutputList
	( 
		SIZE_TYPE start,			// IN - start of range to add
		SIZE_TYPE end,				// IN - end or range to add
		NODE_LIST_TYPE & results	// OUT - result list
	)
	{
		for (SIZE_TYPE currPos = start; currPos <= end; currPos++)
		{
			NODE_TYPE currNode = m_nodes[currPos];
			results.push_back( currNode );
		}

		// Success
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::AddNodesInRegionToOutputList
	//	Desc:	Helper Function
	//-------------------------------------------------------------------------

	bool AddNodesInRegionToOutputList
	( 
		const BOX2D_TYPE & queryRegion,	// IN - Input Region
		SIZE_TYPE start,				// IN - start of range to add
		SIZE_TYPE end,					// IN - end or range to add
		NODE_LIST_TYPE & results		// OUT - result list 
	)
	{
		for (SIZE_TYPE currPos = start; currPos <= end; currPos++)
		{
			NODE_TYPE currNode = m_nodes[currPos];
			POINT_TYPE currPoint;
			GetPoint3D( currNode, currPoint );
			bool bInBox = InBox( queryRegion, currPoint );
			if (bInBox)
			{
				// Add to result set
				results.push_back( currNode );
			}
		}

		// Success
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::AddNodeInRegionToOutputList
	//	Desc:	Helper Function
	//-------------------------------------------------------------------------

	bool AddNodeInRegionToOutputList
	( 
		const BOX2D_TYPE &	queryRegion,	// IN - Input Region
		const NODE_TYPE		currPos,		// IN - Node to check
		NODE_LIST_TYPE &	results			// OUT - result list 
	)
	{
		NODE_TYPE currNode = m_nodes[currPos];
		POINT_TYPE currPoint;
		GetPoint3D( currNode, currPoint );
		bool bInBox = InBox( queryRegion, currPoint );
		if (bInBox)
		{
			// Add to result set
			results.push_back( currNode );
		}

		// Success
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::BruteForceFindClosestPointIndex
	//	Desc:	Use Brute Force Algorithm to find closest Node (index)
	//			Takes O(N) time
	//-------------------------------------------------------------------------

	bool BruteForceFindClosestPointIndex
		(
			const POINT2D_TYPE & queryLocation,	// IN  - Location to sample
			NODE_TYPE & closestPointIndex		// OUT - Index of Closest Point
		)
	{
		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		// Get 1st Point
		SIZE_TYPE  bestIndex = 0;
		NODE_TYPE  bestNode  = m_nodes[bestIndex];
		POINT_TYPE currPoint;
		GetPoint3D( bestNode, currPoint );

		// Calculate distance from query location
		VALUE_TYPE dX = currPoint.X() - queryLocation.X();
		VALUE_TYPE dY = currPoint.Y() - queryLocation.Y();
		VALUE_TYPE bestDist2 = dX*dX + dY*dY;
		VALUE_TYPE diffDist2;

		for (SIZE_TYPE i = 1; i < nNodes; i++)
		{
			// Get Current Point
			NODE_TYPE currNode = m_nodes[i];
			GetPoint3D( currNode, currPoint );

			// Calculate Distance from query location
			dX = currPoint.X() - queryLocation.X();
			dY = currPoint.Y() - queryLocation.Y();
			diffDist2 = dX*dX + dY*dY;

			// Update Best Point Index
			if (diffDist2 < bestDist2)
			{
				bestIndex = currNode;
				bestDist2 = diffDist2;
			}
		}

		Dumpf( TEXT( "\r\n BruteForce Find Closest Point - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

		// Success
		closestPointIndex = bestIndex;
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::BruteForceFindKPointsIndices
	//	Desc:	Use Brute Force Algorithm to find 'k' closest points indices
	//			Takes O(N) time
	//-------------------------------------------------------------------------

	bool BruteForceFindKClosestPointIndices
		(
			const POINT2D_TYPE & queryLocation,	// IN  - Location to sample
			SIZE_TYPE		     k,				// IN  - find 'k' nearest points
			NODE_LIST_TYPE &	 results		// OUT - list of 'k' nearest point indices
		)
	{
		bool bResult = true;

		// Check Parameters
		if (k <= 0) { return false; }

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		if (nNodes <= k)
		{
			// Return as many points as we do have
			results.resize( nNodes );
			for (SIZE_TYPE i = 0; i < nNodes; i++)
			{
				results[i] = m_nodes[i];
			}
		}
		else
		{
			// Get 'k' closest nodes by Brute Point search
			CLOSEST_LIST_TYPE myClosestNodes( static_cast<unsigned int>( k ) );
			
			for (SIZE_TYPE i = 0; i < nNodes; i++)
			{
				// Get Point Associated with Node
				NODE_TYPE currNode = m_nodes[i];
				POINT_TYPE currPoint;
				GetPoint3D( currNode, currPoint );

				// Calculate distance from query Location
				VALUE_TYPE dX = currPoint.X() - queryLocation.X();
				VALUE_TYPE dY = currPoint.Y() - queryLocation.Y();
				VALUE_TYPE diffDist2 = dX*dX + dY*dY;

				// Insert Node into list (if closer than current farthest point in list)
				myClosestNodes.Insert( currNode, diffDist2 );
			}

			myClosestNodes.GetNodes( results );
		}

		Dumpf( TEXT( "\r\n BruteForce Find K Point Indices - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

		return bResult;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::BruteForceQueryRegion
	//	Desc:	Find all point indices in specified query region
	//-------------------------------------------------------------------------

	bool BruteForceQueryRegion
	( 
		const BOX2D_TYPE & queryRegion,	// IN - Region to Query
		NODE_LIST_TYPE &   results		// OUT - list of 'k' nearest point indices
	)
	{
		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() ); 
		if (nNodes <= 0) { return false; }

		for (SIZE_TYPE i = 0; i < nNodes; i++)
		{
			AddNodeInRegionToOutputList( queryRegion, i, results );
		}

		Dumpf( TEXT( "\r\n BruteForce Query Region - Num Nodes Processed = %d\r\n\r\n" ), nNodes );

		// Success
		return true;
	}



	//-------------------------------------------------------------------------
	//	Name:	KDTree::FindClosestIndex
	//	Desc:	Find closest point index to query location in KD Tree
	//-------------------------------------------------------------------------

	bool FindClosestPointIndex
	(
		const POINT2D_TYPE & queryLocation,	// IN  - Query Location
		NODE_TYPE &			 closestIndex	// OUT - closest point index to sample location
	) const
	{
		// Make sure we have something to search
		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		SIZE_TYPE start, end, median;
		SIZE_TYPE currAxis, nextAxis;
		VALUE_TYPE dX, dY;
		VALUE_TYPE diffDist2, bestDist2;

		// Setup Search Queue
		RANGE_TYPE currRange;
		currRange.start = 0;
		currRange.end   = nNodes - 1;
		median = (currRange.start + currRange.end)/2;

		SIZE_TYPE bestIndex	   = median;
		NODE_TYPE bestNode	   = m_nodes[median];
		POINT_TYPE currPoint;
		GetPoint3D( bestNode, currPoint );
		dX = currPoint.X() - queryLocation.X();
		dY = currPoint.Y() - queryLocation.Y();
		bestDist2 = dX*dX + dY*dY;

		std::deque<RANGE_TYPE> rangeQueue;		
		std::deque<SIZE_TYPE>  axisQueue;		

		rangeQueue.push_back( currRange );
		axisQueue.push_back( m_startAxis );

		//int nNodesProcessed = 0;

		// Start searching for closest points
		while (! rangeQueue.empty())
		{
			//nNodesProcessed++;

			// Get Current Node from front of queue
			currRange  = rangeQueue.front();
			currAxis   = axisQueue.front();

			// Get Median Node
			start    = currRange.start;
			end      = currRange.end;
			// Assert( start <= end );
			median   = (start+end)/2;		// Root Index (Split Index) for this range
			nextAxis = NextDimension( currAxis );

			// Calc Dist from Median Node to queryLocation
			NODE_TYPE currNode = m_nodes[median];
			POINT_TYPE currPoint;
			GetPoint3D( currNode, currPoint );
			dX = currPoint.X() - queryLocation.X();
			dY = currPoint.Y() - queryLocation.Y();
			diffDist2 = dX*dX + dY*dY;

			// Update closest point
			if (diffDist2 < bestDist2)
			{
				bestIndex = median;
				bestDist2 = diffDist2;
			}

			// Get Best Fit Dist for checking child ranges
			VALUE_TYPE queryValue = queryLocation[currAxis];
			VALUE_TYPE splitValue = currPoint[currAxis];

			if (start < end)
			{
				if (queryValue <= splitValue)
				{
					// [...QL...BD]...SV		-> Include Left range only
					//		or
					// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges

					// Always Add Left Sub-range to search path
					if (start < median)
					{
						currRange.start = start;
						currRange.end   = median-1;
						rangeQueue.push_back( currRange );
						axisQueue.push_back( nextAxis );
					}

					// Check if we should add Right Sub-range to search path
					VALUE_TYPE diff = splitValue - queryValue;
					VALUE_TYPE diff2 = diff*diff;
					if (diff2 < bestDist2)
					{
						// Add to Search Queue
						if (median < end)
						{
							currRange.start = median+1;
							currRange.end   = end;
							rangeQueue.push_back( currRange );
							axisQueue.push_back( nextAxis );						
						}
					}
				}
				else
				{
					// SV...[BD...QL...]		-> Include Right sub range only
					//		  or
					// [BD...SV...QL...]		-> Include Both Left and Right Sub Ranges

					// Check if we should add left sub-range to search path
					VALUE_TYPE diff = queryValue - splitValue;
					VALUE_TYPE diff2 = diff*diff;
					if (diff2 < bestDist2)
					{
						// Add to search queue
						if (start < median)
						{
							currRange.start = start;
							currRange.end   = median-1;
							rangeQueue.push_back( currRange );
							axisQueue.push_back( nextAxis );
						}
					}
						
					// Always Add Right Sub-range
					if (median < end)
					{
						currRange.start = median+1;
						currRange.end   = end;
						rangeQueue.push_back( currRange );
						axisQueue.push_back( nextAxis );
					}
				}
			}

			// Finished processing this node, get rid of it
			rangeQueue.pop_front();
			axisQueue.pop_front();
		}

		// Get Best Point Index
		bestNode = m_nodes[bestIndex];				
		closestIndex = bestNode;

		//Dumpf( TEXT( "\r\n Find Closest Point Index - Num Nodes Processed = %d\r\n\r\n" ), nNodesProcessed );

		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::FindClosestPointIndices
	//	Desc:	Find 'k' nearest point indices in KD Tree to query location
	//-------------------------------------------------------------------------
	
	bool FindKClosestPointIndices
	(
		const POINT2D_TYPE & queryLocation,	// IN  - Query Location
		SIZE_TYPE		     k,				// IN  - find 'k' nearest point indices to query location
		NODE_LIST_TYPE &     results		// OUT - list of 'k' nearest point indices
	) const
	{
		// Make sure we have something to search
		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		// Setup Closest Node List
		CLOSEST_LIST_TYPE myClosestNodes( static_cast<unsigned int>( k ) );

		SIZE_TYPE start, end, median;
		SIZE_TYPE currAxis, nextAxis;
		VALUE_TYPE dX, dY;
		VALUE_TYPE diffDist2, bestDist2;

		POINT_TYPE currPoint;

		// Setup Search Queue
		RANGE_TYPE currRange;
		currRange.start = 0;
		currRange.end   = nNodes - 1;
		median = (currRange.start + currRange.end)/2;

		std::deque<RANGE_TYPE> rangeQueue;		
		std::deque<SIZE_TYPE>  axisQueue;		

		rangeQueue.push_back( currRange );
		axisQueue.push_back( m_startAxis );

		//int nNodesProcessed = 0;

		// Start searching for closest points
		while (! rangeQueue.empty())
		{
			//nNodesProcessed++;

			// Get Current Node from front of queue
			currRange  = rangeQueue.front();
			currAxis   = axisQueue.front();

			// Get Median Node
			start    = currRange.start;
			end      = currRange.end;
			// Assert( start <= end );
			median   = (start+end)/2;		// Root Index (Split Index) for this range
			nextAxis = NextDimension( currAxis );

			// Calc Dist from Median Node to queryLocation
			NODE_TYPE currNode = m_nodes[median];
			GetPoint3D( currNode, currPoint );
			dX = currPoint.X() - queryLocation.X();
			dY = currPoint.Y() - queryLocation.Y();
			diffDist2 = dX*dX + dY*dY;

			// Update closest point
			myClosestNodes.Insert( currNode, diffDist2 );

			if (start < end)
			{
				// Get Best Fit Dist for checking child ranges
				VALUE_TYPE queryValue = queryLocation[currAxis];
				VALUE_TYPE splitValue = currPoint[currAxis];
				bestDist2 = myClosestNodes.BEST_DIST2();

				if (queryValue <= splitValue)
				{
					// [...QL...BD]...SV		-> Include Left range only
					//		or
					// [...QL...SV...BD]		-> Include Both Left and Right Sub Ranges

					// Always Add Left Sub-range to search path
					if (start < median)
					{
						currRange.start = start;
						currRange.end   = median-1;
						rangeQueue.push_back( currRange );
						axisQueue.push_back( nextAxis );
					}

					// Check if we should add Right Sub-range to search path
					VALUE_TYPE diff = splitValue - queryValue;
					VALUE_TYPE diff2 = diff*diff;
					if (diff2 < bestDist2)
					{
						// Add to Search Queue
						if (median < end)
						{
							currRange.start = median+1;
							currRange.end   = end;
							rangeQueue.push_back( currRange );
							axisQueue.push_back( nextAxis );						
						}
					}
				}
				else
				{
					// SV...[BD...QL...]		-> Include Right sub range only
					//		  or
					// [BD...SV...QL...]		-> Include Both Left and Right Sub Ranges

					// Check if we should add left sub-range to search path
					VALUE_TYPE diff = queryValue - splitValue;
					VALUE_TYPE diff2 = diff*diff;
					if (diff2 < bestDist2)
					{
						// Add to search queue
						if (start < median)
						{
							currRange.start = start;
							currRange.end   = median-1;
							rangeQueue.push_back( currRange );
							axisQueue.push_back( nextAxis );
						}
					}
						
					// Always Add Right Sub-range
					if (median < end)
					{
						currRange.start = median+1;
						currRange.end   = end;
						rangeQueue.push_back( currRange );
						axisQueue.push_back( nextAxis );						
					}
				}
			}

			// Finished processing this node, get rid of it
			rangeQueue.pop_front();
			axisQueue.pop_front();
		}

		// Get Result Point Indices
		myClosestNodes.GetNodes( results );

		//Dumpf( TEXT( "\r\n Find Closest Points - Num Nodes Processed = %d\r\n\r\n" ), nNodesProcessed );

		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::FindPointIndicesInRegion
	//	Desc:	Find all points in specified query region
	//-------------------------------------------------------------------------

	bool FindPointIndicesInRegion
	( 
		const BOX2D_TYPE & queryRegion,	// IN - Region to Query
		NODE_LIST_TYPE &   results		// OUT - list of 'k' point indices inisde query region
	)
	{
		results.clear();

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		SIZE_TYPE currAxis = m_startAxis;
		SIZE_TYPE nextAxis;

		SIZE_TYPE start = 0;
		SIZE_TYPE end   = nNodes - 1;
		SIZE_TYPE median = (start+end)/2;

		RANGE_TYPE currRange;
		currRange.start = start;
		currRange.end   = end;

		BOX2D_TYPE currBounds;
		BOX2D_TYPE leftBounds;
		BOX2D_TYPE rightBounds;

		// Setup Search Queue
		std::deque<RANGE_TYPE> rangeQueue;
		std::deque<SIZE_TYPE> axisQueue;
		std::deque<BOX2D_TYPE> boundsQueue;

		rangeQueue.push_back( currRange );
		axisQueue.push_back( currAxis );

		currBounds.SetFromMinMax( m_bounds.MINX(), m_bounds.MINY(),
			                      m_bounds.MAXX(), m_bounds.MAXY() );
		boundsQueue.push_back( currBounds );

		//int nNodesProcessed = 0;

		NODE_TYPE currNode;
		POINT_TYPE currPoint;

		while (! rangeQueue.empty())
		{
			//nNodesProcessed++;

			// Get Node off top of queue
			currRange	= rangeQueue.front();
			currAxis	= axisQueue.front();
			currBounds	= boundsQueue.front();

			nextAxis = NextDimension( currAxis );

			start = currRange.start;
			end   = currRange.end;
			median = (start+end)/2;

			currNode = m_nodes[median];
			GetPoint3D( currNode, currPoint );

			if (start == end)
			{
				// Leaf Node, compare point to query Region
				AddNodeInRegionToOutputList( queryRegion, median, results );
			}
			else
			{
				// Check Current Nodes Region against query region
				bool bContains = TestContains( queryRegion, currBounds );
				if (bContains)
				{
					// Node's region is completely contained in Query region
						// Include all points belonging to node in result set
					AddNodesInRangeToOutputList( start, end, results );
				}
				else
				{
					// Check if Node Region and Query Region overlap
					bool bIntersects = TestIntersects( queryRegion, currBounds );
					if (bIntersects)
					{
						// Test median point contained in this node
						AddNodeInRegionToOutputList( queryRegion, median, results );

						// Get Split Value
						VALUE_TYPE splitValue = currPoint[currAxis];

						SplitBox( currBounds, currAxis, splitValue, leftBounds, rightBounds );

						if (start < median)
						{
							// Add Left Range to work queue
							currRange.start = start;
							currRange.end   = median-1;

							rangeQueue.push_back( currRange );
							axisQueue.push_back( nextAxis );
							boundsQueue.push_back( leftBounds );
						}

						if (median < end)
						{
							// Add Right Range to work queue
							currRange.start = median+1;
							currRange.end   = end;

							rangeQueue.push_back( currRange );
							axisQueue.push_back( nextAxis );
							boundsQueue.push_back( rightBounds );
						}
					}
				}
			}

			// Finished processing this node, get rid of it
			rangeQueue.pop_front();
			axisQueue.pop_front();
			boundsQueue.pop_front();
		}

		//Dumpf( TEXT( "\r\n Query Region - Num Nodes Processed = %d\r\n\r\n" ), nNodesProcessed );

		// Success
		return true;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::QueryClosestPoint
	//	Desc:	Find nearest point to sample in KD Tree
	//-------------------------------------------------------------------------

	bool QueryClosestPoint
	( 
		const POINT2D_TYPE & queryLocation,	// IN  - Location to sample
		POINT_TYPE &  closestPoint			// OUT - closest point to sample
	)
	{
		bool bResult = true;

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		NODE_TYPE closestPointIndex;
		bResult = FindClosestPointIndex( queryLocation, closestPointIndex );
		GetPoint3D( closestPointIndex, closestPoint );
	
		return bResult;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::QueryClosestValue
	//	Desc:	Find nearest point (values) to sample in KD Tree
	//-------------------------------------------------------------------------

	bool QueryClosestValue
	( 
		const POINT2D_TYPE  & queryLocation,	// IN  - Location to sample
		const ValueEnumList	& valueTypes,		// IN:	Desired Value Types
		      ValueResultsList  & returnValues	// OUT: Return values here (for each location)
	)
	{
		bool bResult = true;

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		// Get Index of Closest Point
		NODE_TYPE closestIndex;
		bResult = FindClosestPointIndex( queryLocation, closestIndex );

		if (bResult)
		{
			// Extract requested values from surface at this index
			m_querySurface->QueryIndex( closestIndex, valueTypes, returnValues );
		}
	
		return bResult;
	}



	//-------------------------------------------------------------------------
	//	Name:	KDTree::QueryKClosestPoints
	//	Desc:	Find 'k' nearest points to sample in KD Tree
	//-------------------------------------------------------------------------

	bool QueryKClosestPoints
	( 
		const POINT2D_TYPE & queryLocation,	// IN  - Location to sample
		SIZE_TYPE		   k,				// IN  - find 'k' nearest points
		POINT_LIST_TYPE &  resultPoints		// OUT - list of 'k' nearest points
	)
	{
		bool bResult = true;

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		if (nNodes <= k)
		{
			// Return as many points as we do have
			if (resultPoints.size() != nNodes)
			{
				resultPoints.resize( nNodes );
			}

			POINT_TYPE currPoint;
			for (SIZE_TYPE i = 0; i < nNodes; i++)
			{
				NODE_TYPE currNode = m_nodes[i];
				GetPoint3D( currNode, currPoint );
				resultPoints[i] = currPoint;
			}
		}
		else
		{
			NODE_TYPE currNode;
			POINT_TYPE currPoint;
			NODE_LIST_TYPE resultIndices;
			bResult = FindKClosestPointIndices( queryLocation, k, resultIndices );
			if (bResult)
			{
				unsigned int numResults = static_cast<unsigned int>( resultIndices.size() );

				if (resultPoints.size() != numResults)
				{
					resultPoints.resize( numResults );
				}

				for (SIZE_TYPE i = 0; i < numResults; i++)
				{
					currNode = resultIndices[i];
					GetPoint3D( currNode, currPoint );
					resultPoints[i] = currPoint;
				}
			}
		}
	
		return bResult;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::QueryKClosestValues
	//	Desc:	Find 'k' nearest point (values) to sample in KD Tree
	//-------------------------------------------------------------------------

	bool QueryKClosestValues
	( 
		const POINT2D_TYPE  & queryLocation,	// IN  - Location to sample
		SIZE_TYPE		   k,					// IN  - find 'k' nearest points
		const ValueEnumList	& valueTypes,		// IN:	Desired Value Types
		      ValueResultsList  & returnValues	// OUT: Return values here (for each location)
	)
	{
		bool bResult = true;

		NODE_LIST_TYPE resultIndices;

		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes <= 0) { return false; }

		if (nNodes <= k)
		{
			// Return as many points as we do have
			if (resultIndices.size() != nNodes)
			{
				resultIndices.resize( nNodes );
			}

			for (SIZE_TYPE i = 0; i < nNodes; i++)
			{
				resultIndices[i] = m_nodes[i];
			}
		}
		else
		{
			// Find 'K' closest point indices in KDTree
			bResult = FindKClosestPointsIndices( queryLocation, k, resultIndices );
		}

		// Get Requested Value Types for each point index in result list
		if (bResult)
		{
			m_querySurface->QueryIndices( resultIndices, valueTypes, returnValues );
		}
	
		return bResult;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::QueryRegion
	//	Desc:	Find all points in specified query region
	//-------------------------------------------------------------------------

	bool QueryRegion
	( 
		const BOX2D_TYPE & queryRegion,	// IN - Region to Query
		POINT_LIST_TYPE & resultPoints	// OUT - list of 'k' points inside region
	)
	{
		resultPoints.clear();

		bool bResult = true;

		NODE_LIST_TYPE resultIndices;

		bResult = FindPointIndicesInRegion( queryRegion, resultIndices );
		if (bResult)
		{
			NODE_TYPE currNode;
			POINT_TYPE currPoint;

			SIZE_TYPE numResults = static_cast<SIZE_TYPE>( resultIndices.size() );

			if (resultPoints.size() != numResults)
			{
				resultPoints.resize( numResults );
			}

			for (SIZE_TYPE i = 0; i < numResults; i++)
			{
				currNode = resultIndices[i];
				GetPoint3D( currNode, currPoint );
				resultPoints[i] = currPoint;
			}
		}

		// Success
		return bResult;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::QueryRegionValues
	//	Desc:	Get point (values) for all points in specified query region
	//-------------------------------------------------------------------------

	bool QueryRegionValues
	( 
		const BOX2D_TYPE &        queryRegion,	// IN - Region to Query
		const ValueEnumList	&     valueTypes,	// IN:	Desired Value Types
		      ValueResultsList  & returnValues	// OUT: Return values here (for each location)
	)
	{
		bool bResult = true;

		NODE_LIST_TYPE resultIndices;
		bResult = FindPointIndicesInRegion( queryRegion, resultIndices );
		if (bResult)
		{
			m_querySurface->QueryIndices( resultIndices, valueTypes, returnValues );
		}

		// Success
		return bResult;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::ToString
	//	Desc:	write human readable string about KD Tree details to string
	//-------------------------------------------------------------------------

	void ToString( std::string & value ) const
	{
		char szMsg[512];
		std::string val( "KDTree = {\r\n" );

		SIZE_TYPE nNodes  = static_cast< SIZE_TYPE >( m_nodes.size() );
		StringPrintfA( szMsg, 512, "\tNum Nodes = %d", nNodes );
		val += szMsg;
	}


	//-------------------------------------------------------------------------
	//	Name:	KDTree::Dump
	//	Desc:	Dump human readable string about KD Tree details to debugger
	//			and/or standard output
	//-------------------------------------------------------------------------

	void Dump() const
	{
		char szMsg[512];
		int maxDim = static_cast<int>( MD );
		Dumpf( "KDTree<MD=%d> = {", maxDim );

		// Dump Nodes
		SIZE_TYPE nNodes = static_cast<SIZE_TYPE>( m_nodes.size() );
		if (nNodes == 0)
		{
			Dumpf( "   KD NODE List = { <NULL> }" );
		}
		else
		{
			Dumpf( "   Node List(%d) = {", nNodes );

			SIZE_TYPE start  = 0;
			SIZE_TYPE end    = nNodes - 1;
			SIZE_TYPE median = (start+end)/2;

			SIZE_TYPE leftIndex  = static_cast<SIZE_TYPE>( -1 );
			SIZE_TYPE rightIndex = static_cast<SIZE_TYPE>( -1 );

			SIZE_TYPE currAxis, nextAxis;

			RANGE_TYPE currRange;

			currRange.start = start;
			currRange.end   = end;

			// Setup Dump Queue
			std::deque<RANGE_TYPE> rangeQueue;
			std::deque<SIZE_TYPE> axisQueue;
			rangeQueue.push_back( currRange );
			axisQueue.push_back( m_startAxis );

			// Process all Nodes in Queue
			bool bDone = false;
			while (! bDone)
			{
				if (rangeQueue.empty())
				{
					bDone = true;
				}
				else
				{
					currRange = rangeQueue.front();
					currAxis  = axisQueue.front();

					nextAxis  = NextDimension( currAxis );

					start	  = currRange.start;
					end		  = currRange.end;
					median    = (start+end)/2;

					// Dump Current Node
					leftIndex = static_cast<SIZE_TYPE>( -1 );
					rightIndex = static_cast<SIZE_TYPE>( -1 );

					if (start < median)
					{
						// Left Child Range
						currRange.start = start;
						currRange.end   = median-1;
						rangeQueue.push_back( currRange );

						leftIndex = (currRange.start + currRange.end)/2;
					}
					if (median < end)
					{
						// Right Child Range
						currRange.start = median+1;
						currRange.end   = end;
						rangeQueue.push_back( currRange );

						rightIndex = (currRange.start + currRange.end)/2;
					}

					int iCurr   = static_cast<int>( median );
					int iStart  = static_cast<int>( start );
					int iEnd    = static_cast<int>( end );
					int iLeft   = ((leftIndex == static_cast<SIZE_TYPE>(-1)) ? -1 : static_cast<int>( leftIndex ));
					int iRight  = ((rightIndex == static_cast<SIZE_TYPE>(-1)) ? -1 : static_cast<int>( rightIndex ));
					char * szAxis = AxisToString( currAxis );
					
					NODE_TYPE & currNode   = m_nodes[median];
					POINT_TYPE currPoint;
					GetPoint3D( currNode, currPoint );
					double splitValue = static_cast<double>( currPoint[currAxis] );

					Dumpf( "      KD NODE(CIdx=%d)=<[Start,End]=[%d,%d]; Split[Axis,Value]=[%s, %3.6f]; [LIdx,RIdx]=[%d, %d]>", 
							      iCurr, iStart, iEnd, szAxis, splitValue, iLeft, iRight );

					rangeQueue.pop_front();
					axisQueue.pop_front();
				}
			}

			Dumpf( "   }" );
		}
		
	}
};	// End KDTree_T<T,MD>

typedef KDTree_T<float, 2> KDTree2D;		// 2D KDTree over <x,y> values -- planar, ideal for terrain
typedef KDTree_T<float, 3> KDTree3D;		// 3D KDTree over <x,y,z> values -- volumetric


#endif // _VS_KDTREE_H

