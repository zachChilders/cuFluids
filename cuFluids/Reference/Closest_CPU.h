#pragma once
#ifndef _VS_CLOSEST_NODES_H
#define _VS_CLOSEST_NODES_H
//-----------------------------------------------------------------------------
//	Name:	ClosestNodes.h
//	Desc:	Defines Simple Closest Points Class
//	Log:	Created by Shawn D. Brown 6/30/06
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

//#include "Heap.h"
#include "VS_Point.h"
#include "VS_Box.h"



//-------------------------------------
//
//	Classes
//
//-------------------------------------

//-----------------------------------------------------------------------------
//	Class:	ClosestNodes
//	Desc:	Helper Class
//			Represents the array of 'K' nodes (point indices) closest to a query location 'Q'
//	Notes:	
//		0. Ignores 1st element in Array to accomodate Heap Behavior
//		1. Behaves like an array until we reach 'K' points
//		2. Behaves like a MaxHeap after we reach 'K' points
//-----------------------------------------------------------------------------

template <typename T>
class ClosestNodes_T
{
public:
	// TYPEDEFS
	typedef unsigned int			SIZE_TYPE;			// Default Size type
	typedef unsigned int			NODE_TYPE;			// Default Node Type (Index)
	typedef T						VALUE_TYPE;			// Underlying value type
	typedef Point3D_T<T>			POINT_TYPE;			// Underlying 3D Point type
	typedef std::vector<NODE_TYPE>	NODE_LIST_TYPE;		// Vector of Node types

	typedef struct tagCloseNode // Structure to store a single closest node
	{
	public:
		// Fields
		unsigned int m_nodeIndex;	// Index for original node
		VALUE_TYPE	 m_dist2;		// Squared distance between node point and original query point
	} CloseNode; 

	typedef std::vector<CloseNode> CLOSENODE_LIST;

protected:
	//------------------------------------
	//	Fields
	//------------------------------------

	CLOSENODE_LIST		m_closeNodes;	// Array of Close Nodes
	SIZE_TYPE			m_maxNodes;		// Maximum number of nodes allowed in this set
	SIZE_TYPE			m_currNodes;	// current Number of nodes in this set
	VALUE_TYPE			m_maxDist2;		// Maximum Squared Distance of points allowed into this list
	VALUE_TYPE			m_currDist2;	// Current Maximum distance of points in this set


	//------------------------------------
	//	Helper Methods
	//------------------------------------

		// Default Initialization
	void Init()
	{
		m_closeNodes.clear();
		m_maxNodes	= 0;
		m_currNodes	= 0;
		m_maxDist2	= static_cast<VALUE_TYPE>( c_fHUGE );
		m_currDist2	= static_cast<VALUE_TYPE>( c_fHUGE );
	}

		// Fini - cleanup resources properly
	void Fini()
	{
		Init();
	}

		// Copy
	bool Copy( const ClosestNodes_T<T> & toCopy )
		{
			if (this == &toCopy) { return true; }

			m_closeNodes.clear();

			m_maxNodes  = toCopy.m_maxNodes;
			m_currNodes = toCopy.m_currNodes;
			m_maxDist2  = toCopy.m_maxDist2;
			m_currDist2 = toCopy.m_currDist2;
			
			m_closeNodes.resize( m_maxNodes+1 );

			// Copy over Closest Points List
			if (m_currNodes > 0)
			{	
				m_closeNodes[0].m_nodeIndex = static_cast<NODE_TYPE>( -1 );
				m_closeNodes[0].m_dist2     = static_cast<VALUE_TYPE>( -1.0 );

				for (SIZE_TYPE i = 1; i <= m_currNodes; i++)
				{
					m_closeNodes[i].m_nodeIndex = toCopy.m_closeNodes[i].m_nodeIndex;
					m_closeNodes[i].m_dist2     = toCopy.m_closeNodes[i].m_dist2;
				}
			}

			return true;
		}

		// Swap
	void Swap( SIZE_TYPE i, SIZE_TYPE j )
		{
			NODE_TYPE nodeTemp   = m_closeNodes[i].m_nodeIndex;
			VALUE_TYPE dist2Temp = m_closeNodes[i].m_dist2;

			m_closeNodes[i].m_nodeIndex = m_closeNodes[j].m_nodeIndex;
			m_closeNodes[i].m_dist2     = m_closeNodes[j].m_dist2;

			m_closeNodes[j].m_nodeIndex = nodeTemp;
			m_closeNodes[j].m_dist2     = dist2Temp; 
		}

	//-------------------------------------------------------------------------
	//	Name:	Promote
	//	Desc:	Promotes value at current index up parent chain
	//	Notes:	Assumes heap formated array
	//-------------------------------------------------------------------------

	void Promote( SIZE_TYPE currIndex )
	{
		SIZE_TYPE parentIndex = currIndex >> 1;	// Parent = Curr / 2;

		VALUE_TYPE cD2 = m_closeNodes[currIndex].m_dist2;
		VALUE_TYPE pD2 = m_closeNodes[parentIndex].m_dist2;

		// Compare currIndex with it's parent
		while ((currIndex > 1) && (pD2 < cD2))
		{
			// Promote currIndex by swapping with it's parent
			Swap( parentIndex, currIndex );

			// Update indices of k and it's parent
			currIndex = parentIndex;
			parentIndex = currIndex >> 1;	// Parent = Curr /2;

			// Update distances
			cD2 = m_closeNodes[currIndex].m_dist2;
			pD2 = m_closeNodes[parentIndex].m_dist2;
		}
	}

	//---------------------------------------------------------
	//	Name:	Demote
	//	Desc:	Demotes value at curr index down child chain
	//	Notes:	Assumes heap formated array
	//---------------------------------------------------------

	void Demote( SIZE_TYPE currIndex ) 
	{
		VALUE_TYPE currD2, childD2;

		SIZE_TYPE childIndex = 2*currIndex;	// left child of current

		// Compare current index to it's children
		while (childIndex <= m_maxNodes)
		{
			// Update Distances
			currD2  = m_closeNodes[currIndex].m_dist2;
			childD2 = m_closeNodes[childIndex].m_dist2;

			// Find largest child 
			if (childIndex < m_maxNodes)
			{
				VALUE_TYPE rightD2 = m_closeNodes[childIndex+1].m_dist2;
				if (childD2 < rightD2)
				{
					// Use right child
					childIndex++;	
					childD2 = rightD2;
				}
			}

			// Compare largest child to current
			if (currD2 >= childD2) 
			{
				// Current is larger than both children, exit
				break;
			}

			// Demote currIndex by swapping with it's largest child
			Swap( currIndex, childIndex );
			
			// Update indices
			currIndex  = childIndex;	
			childIndex = 2*currIndex;		// left child of current
		}
	}

	//---------------------------------------------------------
	//	Name:	make_heap
	//	Desc:	Move all values in heap to satisfy heap ordering
	//			IE largest element at top of heap.
	//			and all child nodes >= parent nodes
	//	Notes:	Takes O(N) time
	//---------------------------------------------------------

	void make_heap() 
	{
		unsigned int N = m_currNodes;
		unsigned int k;
		for (k = N/2; k >= 1; k--)
		{
			Demote( k );
		}
	}


	//---------------------------------------------------------
	//	Name:	Replace
	//	Desc:	replace top element on heap with new element
	//	Notes:	Takes O( log N ) time
	//---------------------------------------------------------

	void Replace( const NODE_TYPE newNode, VALUE_TYPE newDist2 )
	{
		// Replace Root Element with new element
		m_closeNodes[1].m_nodeIndex = newNode;
		m_closeNodes[1].m_dist2     = newDist2;

		// Demote new element to correct location in heap
		Demote( 1 );
	}


public:
	//------------------------------------
	//	Properties
	//------------------------------------
		// Maximum number of Points in this list (K)
	SIZE_TYPE MAX_NODES() const { return m_maxNodes; }
	void MAX_NODES( SIZE_TYPE value ) { m_maxNodes = value; }

		// Current number of points in this list M <= K
	SIZE_TYPE CURR_NODES() const { return m_currNodes; }
	void CURR_NODES( SIZE_TYPE value ) { m_currNodes = value; }

		// Maximum Squared distance of points allowed into this list
	VALUE_TYPE CUTOFF_DIST2() const { return m_maxDist2; }
	void CUTOFF_DIST2( VALUE_TYPE value ) { m_maxDist2 = value; }

		// Current Maximum Distance of all points in this list
	VALUE_TYPE CURR_MAX_DIST2() const { return m_currDist2; }
	void CURR_MAX_DIST2( VALUE_TYPE value ) { m_currDist2 = value; }

	bool IsEmpty() const
		{
			return ((m_currNodes == static_cast<SIZE_TYPE>( 0 )) ? true : false);
		}

	bool IsFull() const
		{
			return ((m_currNodes == m_maxNodes) ? true : false );
		}

	VALUE_TYPE BEST_DIST2() const 
		{
			return (IsFull() ? CURR_MAX_DIST2() : static_cast<VALUE_TYPE>( c_fHUGE ) );
		}


	//------------------------------------
	//	Constructors
	//------------------------------------
		// Default Constructor
	ClosestNodes_T<T>() :
		m_maxNodes( static_cast<SIZE_TYPE>( 0 ) ),
		m_currNodes( static_cast<SIZE_TYPE>( 0 ) ),
		m_maxDist2( static_cast<VALUE_TYPE>( c_fHUGE ) ),
		m_currDist2( static_cast<VALUE_TYPE>( c_fHUGE ) )
		{
		}

		// Constructor
	ClosestNodes_T<T>( SIZE_TYPE maxNodes ) :
		m_maxNodes( maxNodes ),
		m_currNodes( 0 ),
		m_maxDist2( static_cast<VALUE_TYPE>( c_fHUGE ) ),
		m_currDist2( static_cast<VALUE_TYPE>( c_fHUGE ) )
		{
			m_closeNodes.resize( maxNodes+1 );
		}

		// Constructor
	ClosestNodes_T<T>( SIZE_TYPE maxNodes, VALUE_TYPE maxDist2 ) :
		m_maxNodes( maxNodes ),
		m_currNodes( 0 ),
		m_maxDist2( maxDist2 ),
		m_currDist2( static_cast<VALUE_TYPE>( maxDist2 ) )
		{
			m_closeNodes.resize( maxNodes+1 );
		}

		// Copy Constructor
	ClosestNodes_T<T>( const ClosestNodes_T<T> & toCopy ) :
		m_maxNodes( toCopy.m_maxNodes ),
		m_currNodes( toCopy.m_currNodes ),
		m_maxDist2( toCopy.m_maxDist2 ),
		m_currDist2( toCopy.m_currDist2 )
		{
			SIZE_TYPE nNodes = toCopy.m_closeNodes.size();
			m_closeNodes.resize( m_maxNodes+1 );
			if (m_currNodes > 0)
			{	
				m_closeNodes[0].m_nodeIndex = static_cast<NODE_TYPE>( -1 );
				m_closeNodes[0].m_dist2     = static_cast<VALUE_TYPE>( -1.0 );

				for (SIZE_TYPE i = 1; i <= m_currNodes; i++)
				{
					m_closeNodes[i].m_nodeIndex = toCopy.m_closeNodes[i].m_nodeIndex;
					m_closeNodes[i].m_dist2     = toCopy.m_closeNodes[i].m_dist2;
				}

				return;
			}
		}

		// Destructor
	~ClosestNodes_T<T>()
		{
			Fini();
		}


	//------------------------------------
	//	Operators
	//------------------------------------
		// Copy Operator
	ClosestNodes_T<T> & operator = ( const ClosestNodes_T<T> & toCopy ) 
		{
			Copy( toCopy );
			return (*this);
		}

	const NODE_TYPE NodeAt( SIZE_TYPE i ) const 
		{
			// Shifts by 1 to account for heap behavior
			// ASSERT( (i >= 0) && (i < m_currNodes) );
			return m_closeNodes[i+1].m_nodeIndex;
		}

	const VALUE_TYPE Dist2At( SIZE_TYPE i ) const 
		{
			// Shifts by 1 to account for heap behavior
			// ASSERT( (i >= 0) && (i <= m_currNodes) );
			return m_closeNodes[i+1].m_dist2;
		}

	const VALUE_TYPE DistAt( SIZE_TYPE i ) const 
		{
			// Shifts by 1 to account for heap behavior
			// ASSERT( (i >= 1) && (i <= m_currNodes) );
			return (static_cast<VALUE_TYPE>( sqrt( static_cast<double>( m_closeNodes[i+1].m_dist2 ))));
		}

	
	//------------------------------------
	//	Methods
	//------------------------------------

	void Setup( SIZE_TYPE maxNodes, VALUE_TYPE maxDist2 )
		{
			Fini();

			if (maxNodes == 0) { return; }

			m_maxNodes   = maxNodes;
			m_maxDist2   = maxDist2;

			m_closeNodes.resize( m_maxNodes + 1 );
		}

		// Insert New Point into List
	void Insert( NODE_TYPE newNode, VALUE_TYPE dist2 ) 
		{
			// Check Cutoff Distance
			if (dist2 >= m_maxDist2) 
			{ 
				return; 
			}
		
			// Check Maximum Number of Points
			if (m_currNodes < m_maxNodes)
			{
				//-------------------------------
				// Do Array Insertion
				//-------------------------------

				SIZE_TYPE index = m_currNodes+1;

				// Get New Current Max Distance
				// Maintains largest Max Distance in initial 'k' points
				if (m_currNodes == 0)
				{
					CURR_MAX_DIST2( dist2 );
				}
				else
				{
					if (dist2 > m_currDist2) 
					{ 
						CURR_MAX_DIST2( dist2 ); 
					}
				}

				m_closeNodes[index].m_nodeIndex = newNode;
				m_closeNodes[index].m_dist2		= dist2;
				m_currNodes++;

				if (m_currNodes == m_maxNodes)
				{
					// Turn Simple Array into a MaxHeap on dist2
					make_heap();
				}
			}
			else if (dist2 < m_currDist2)
			{
				//-------------------------------
				// Do Heap Replacement
				//-------------------------------

				Replace( newNode, dist2 );

				// Get new Current Max Distance from root of heap
				// Reduces current max distance as root points get replaced by closer points
				CURR_MAX_DIST2( m_closeNodes[1].m_dist2 );
			}
		}

	void clear() 
		{ 
			Fini(); 
		}

	void reset()
		{
			m_currNodes = static_cast<SIZE_TYPE>( 0 );	// Reset number of points in list
			m_currDist2 = m_maxDist2;					// Reset current max distance
		}

	bool GetNodes( NODE_LIST_TYPE & results ) const
	{
		// Return as many nodes as we have
		SIZE_TYPE nNodes = m_currNodes;
		results.resize( nNodes );
		for (SIZE_TYPE i = 0; i < nNodes; i++)
		{
			results[i] = m_closeNodes[i+1].m_nodeIndex;
		}

		// Success
		return true;
	}
};


#endif // _VS_CLOSEST_NODES_H

