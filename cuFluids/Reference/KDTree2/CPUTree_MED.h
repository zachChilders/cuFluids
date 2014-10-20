#pragma once
#ifndef _CPUTREE_CPU_H
#define _CPUTREE_CPU_H
/*-----------------------------------------------------------------------------
  Name:	CPUTREE_MED.h
  Desc:	Defines Simple KDTree for CPU
  Notes: 
    - Kd-tree attributes
		static		-- we need to know all points "a priori" before building the kd-tree
		balanced	-- Tree has maximum height of O( log<2> n )
	    Median Array Layout
	        -- kd-nodes in kd-tree is stored in median array layout
			-- given 'n' points in kd-tree
			-- Root sequence is defined as [0, n-1]
			-- Root position can be found at the median: Root = (n-1)/2
			-- Given any sub-tree sequence from [low,high]
			-- Sub-tree node position is found at the median: M = (low+high)/2
			-- Left sub-tree sequence is [low, M-1]
			-- Left child position is found at: L = (low+M-1)/2
			-- Right sub-tree sequence is [M+1, high]
			-- Left child position is found at: R = (M+1+high)/2
			-- Median Partitioning Algorithm leaves elements in this order by default
		d-Dimensionality  -- 2D, 3D or 4D 
		cyclical	-- we follow a cyclical pattern in switching between axises
		               at each level of the tree, 
							for 2D <x,y,x,y,x,y,...>
							for 3D <x,y,z,x,y,z,...>
							for 4D <x,y,z,w,x,y,z,w,...>
		Point Storage -- 1 point is stored at each internal or leaf node
		Minimal     -- I have eliminated as many fields as possible
		               from the kd-node data structures

  Log:	Created by Shawn D. Brown (4/15/07)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// standard Includes
#ifndef _STRING_
	#include <string>
#endif
#ifndef _INC_MATH
	#include <math.h>
#endif
#ifndef _VECTOR_
	#include <vector>
#endif

// Cuda Includes
#ifndef _CUT_
	#include <cutil.h>
#endif
#if !defined(__VECTOR_TYPES_H__)
	#include <vector_types.h>	
#endif

// App Includes
#ifndef _KD_BASEDEFS_H
	#include "BaseDefs.h"
#endif

#include "QueryResult.h"


/*-------------------------------------
  Forward Declarations
-------------------------------------*/

class CPUNode_2D_MED;
class CPUTree_MED;


/*-------------------------------------
  Type Definitons
-------------------------------------*/

#define KD_STACK_SIZE_CPU 32

	// Range [start, end]
typedef struct
{
	unsigned int start;			// Start of Range (inclusive)
	unsigned int end;			// End of Range (inclusive)
	unsigned int parent;		// Parent Index
	unsigned short leftRight;	// Left Right Child
	unsigned short axis;		// Axis
} CPU_BUILD_MED;

typedef struct
{
	unsigned int   start;		// Start of Range (inclusive)	// 25 bits
	unsigned int   end;			// End of Range (inclusive)
	unsigned short InOut;		// Query Value (In/Out)			// 2 bits
	unsigned short axis;		// Axis							// 3 bits
} CPU_SEARCH_MED;

typedef struct
{
	unsigned int	nodeFlags;		
		//unsigned int	nodeIdx;	// Idx of Node [0..28]	// 29 bits
		//unsigned int    axis;		// Axis [29..30]		// 2 bits
		//unsigned int	inOut;		// InOut [31];			// 1 bits 
	float    splitValue;		// Parent's Split Value
} CPU_SEARCH_ALT_MED;



/*-------------------------------------------------------------------------
  Name:	AxisToString
  Desc:	Get Human Readable string for Axis
-------------------------------------------------------------------------*/

static const char * AxisToString( unsigned int currAxis )
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
	case W_AXIS:
		sz =  "W";
		break;
	case S_AXIS:
		sz =  "S";
		break;
	case T_AXIS:
		sz =  "T";
		break;
	case U_AXIS:
		sz =  "U";
		break;
	case V_AXIS:
		sz =  "V";
		break;
	default:
		break;
	}
	return sz;
}


/*-------------------------------------------------------------------------
  Name:	NextAxis2D
  Desc:	Return Next Axis (to partition on)
-------------------------------------------------------------------------*/

static unsigned int NextAxis2D( unsigned int currAxis )
{
	if (currAxis == Y_AXIS)
	{
		return X_AXIS;
	}
	return Y_AXIS;
}

static unsigned int PrevAxis2D( unsigned int currAxis )
{
	if (currAxis == X_AXIS)
	{
		return Y_AXIS;
	}
	return X_AXIS;
}

static unsigned int NextAxis3D( unsigned int currAxis )
{
	if (currAxis == Z_AXIS)
	{
		return X_AXIS;
	}
	return (currAxis+1);
}

static unsigned int PrevAxis3D( unsigned int currAxis )
{
	if (currAxis == X_AXIS)
	{
		return Z_AXIS;
	}
	return (currAxis-1);
}

inline float AxisValue( const float4 & p, unsigned int axis )
{
	const float * f = (const float *)&p;
	return f[axis];
}


/*-------------------------------------
  Classes
-------------------------------------*/

/*---------------------------------------------------------
  Name:	CPUNode_2D_MED
  Desc:	Simple kd-tree node for median array layout
---------------------------------------------------------*/

class CPUNode_2D_MED
{
public:
	/*------------------------------------
	  Type definitions
	------------------------------------*/

	typedef struct _val3 { 
		float x; 
		float y; 
		float z;
	} VAL3;

	/*------------------------------------
	  Constants
	------------------------------------*/
	
	static const unsigned int c_Invalid = static_cast<unsigned int>( -1 );

protected:
	/*------------------------------------
	  Fields
	------------------------------------*/
	
	friend class CPUTree_2D_MED;

	union {
		float m_v[3];
		VAL3  m_p;
	};

	unsigned int m_ID;			// Unique ID of node
	unsigned int m_Parent;		// Parent Node Index
	unsigned int m_Left;		// Left Child Index
	unsigned int m_Right;		// Right Child Index
	unsigned int m_SplitAxis;	// Split Axis

	// Bound Box
	//float m_box[4];			// X-Bounds[min,max], YBounds[min,max]

	/*------------------------------------
	  Helper Methods
	------------------------------------*/

public:
	/*------------------------------------
	  Properties
	------------------------------------*/

	static unsigned int MaxDim2D() { return static_cast<unsigned int>( 2 ); }
	static unsigned int MaxDim3D() { return static_cast<unsigned int>( 3 ); }

		// ID
	inline unsigned int ID() const { return m_ID; }
	inline void ID( unsigned int val ) { m_ID = val; }

		// X
	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }
	
		// Y
	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

		// Z
	inline float Z() const { return m_p.z; }
	inline void Z( float val ) { m_p.z = val; }

		// Length (2D)
	inline float len2_2D() const { return ( X()*X() + Y()*Y() ); }
	inline float len_2D() const { return static_cast<float>( sqrt( len2_2D() ) ); }

		// Length (3D)
	inline float len2_3D() const { return ( X()*X() + Y()*Y() + Z()*Z() ); }
	inline float len_3D() const { return static_cast<float>( sqrt( len2_3D() ) ); }

		// Root 
	inline bool IsRoot() const { return ((c_Invalid == m_Parent) ? true : false); }

		// Leaf
	inline bool IsLeaf() const { return (((c_Invalid == m_Left) && (c_Invalid == m_Right)) ? true : false); }

		// Bounds
	//inline float MINX() const { return m_box[0]; }
	//inline void MINX( float val ) { m_box[0] = val; }

	//inline float MAXX() const { return m_box[1]; }
	//inline void MAXX( float val ) { m_box[1] = val; }

	//inline float MINY() const { return m_box[2]; }
	//inline void MINY( float val ) { m_box[2] = val; }

	//inline float MAXY() const { return m_box[3]; }
	//inline void MAXY( float val ) { m_box[3] = val; }

	//inline void BOUNDS( float vals[4] )
	//	{
	//		m_box[0] = vals[0];
	//		m_box[1] = vals[1];
	//		m_box[2] = vals[2];
	//		m_box[3] = vals[3];
	//	}

		// Parent 
	inline unsigned int Parent() const { return m_Parent; }
	inline void Parent( unsigned int val ) { m_Parent = val; }

		// Left Child
	inline unsigned int Left() const { return m_Left; }
	inline void Left( unsigned int val ) { m_Left = val; }

		// Right Child
	inline unsigned int Right() const { return m_Right; }
	inline void Right( unsigned int val ) { m_Right = val; }

		// Axis
	inline unsigned int Axis() const { return m_SplitAxis; }
	inline void Axis( unsigned int val ) { m_SplitAxis = val; }


	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUNode_2D_MED() :
		m_ID( c_Invalid ),
		m_Parent( c_Invalid ),
		m_Left( c_Invalid ),
		m_Right( c_Invalid ),
		m_SplitAxis( INVALID_AXIS )
		{
			m_p.x = 0.0f;
			m_p.y = 0.0f;
			m_p.z = 0.0f;

			//m_box[0] = 0.0f;
			//m_box[1] = 0.0f;
			//m_box[2] = 0.0f;
			//m_box[3] = 0.0f;
		}

		// 2D Partial Constructor
	CPUNode_2D_MED( unsigned int id, float x, float y ) :
		m_ID( id ),
		m_Parent( c_Invalid ),
		m_Left( c_Invalid ),
		m_Right( c_Invalid ),
		m_SplitAxis( INVALID_AXIS )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = 0.0f;

			//m_box[0] = 0.0f;
			//m_box[1] = 0.0f;
			//m_box[2] = 0.0f;
			//m_box[3] = 0.0f;
		}

		// 3D Partial Constructor
	CPUNode_2D_MED( unsigned int id, float x, float y, float z ) :
		m_ID( id ),
		m_Parent( c_Invalid ),
		m_Left( c_Invalid ),
		m_Right( c_Invalid ),
		m_SplitAxis( INVALID_AXIS )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = 0.0f;

			//m_box[0] = 0.0f;
			//m_box[1] = 0.0f;
			//m_box[2] = 0.0f;
			//m_box[3] = 0.0f;
		}

		// Full Constructor
	CPUNode_2D_MED( unsigned int id, float x, float y, float z,
					unsigned int parent, unsigned int left, unsigned int right,
					unsigned int axis ) :
		m_ID( id ),
		m_Parent( parent ),
		m_Left( left ),
		m_Right( right ),
		m_SplitAxis( axis )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = 0.0f;

			//m_box[0] = 0.0f;
			//m_box[1] = 0.0f;
			//m_box[2] = 0.0f;
			//m_box[3] = 0.0f;
		}

		// Copy Constructor
	CPUNode_2D_MED( const CPUNode_2D_MED & toCopy ) :
		m_ID( toCopy.m_ID ),
		m_Parent( toCopy.m_Parent ),
		m_Left( toCopy.m_Left ),
		m_Right( toCopy.m_Right ),
		m_SplitAxis( toCopy.m_SplitAxis )
		{
			m_p.x    = toCopy.m_p.x;
			m_p.y    = toCopy.m_p.y;
			m_p.z    = toCopy.m_p.z;

			//m_box[0] = toCopy.m_box[0];
			//m_box[1] = toCopy.m_box[1];
			//m_box[2] = toCopy.m_box[2];
			//m_box[3] = toCopy.m_box[3];
		}

		// Destructor
	~CPUNode_2D_MED() 
		{
			m_ID     = c_Invalid;
		}



	/*------------------------------------
	  Operators
	------------------------------------*/

		// Copy Operator
	CPUNode_2D_MED & operator = ( const CPUNode_2D_MED & toCopy )
	{
		if (this == &toCopy) { return (*this); }

		m_ID     = toCopy.m_ID;
		m_p.x    = toCopy.m_p.x;
		m_p.y    = toCopy.m_p.y;
		m_p.z    = toCopy.m_p.z;
		m_Parent = toCopy.m_Parent;
		m_Left   = toCopy.m_Left;
		m_Right  = toCopy.m_Right;
		m_SplitAxis = toCopy.m_SplitAxis;

		//m_box[0] = toCopy.m_box[0];
		//m_box[1] = toCopy.m_box[1];
		//m_box[2] = toCopy.m_box[2];
		//m_box[3] = toCopy.m_box[3];

		return (*this);
	}

		// Point Index Operators
	inline float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 3 );
			return m_v[nIndex];
		}	
	inline const float & operator[] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 3 );
			return m_v[nIndex];
		}


	/*------------------------------------
	  Methods
	------------------------------------*/

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		unsigned int cchBuff = sizeof(szBuff)/sizeof(char);

		unsigned int id = ID();

		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );
		double z = static_cast< double >( Z() );

		//double minX = static_cast< double >( MINX() );
		//double maxX = static_cast< double >( MAXX() );
		//double minY = static_cast< double >( MINY() );
		//double maxY = static_cast< double >( MAXY() );

		unsigned int parentId = Parent();
		unsigned int leftId   = Left();
		unsigned int rightId  = Right();

		const char * szAxis = AxisToString( Axis() );
		
		sprintf_s( szBuff, cchBuff, "<%d>=<%3.6f, %3.6f, %3.6f> [P=%d, L=%d, R=%d, A=%s]", 
			       id, x, y, z, parentId, leftId, rightId, szAxis  );

		//sprintf_s( szBuff, cchBuff, "<%d>=<%3.6f, %3.6f, %3.6f> [P=%d, L=%d, R=%d, A=%s][mX=%3.6f, MX=%3.6f, mY=%3.6f, MY=%3.6f]", 
		//	       id, x, y, z, parentId, leftId, rightId, szAxis, minX, maxX, minY, maxY  );
		
		value = szBuff;
	}

	inline void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		printf( "%s", pointValue.c_str() );
	}


}; // end CPUNode_2D_MED


/*---------------------------------------------------------
  Name:	CPUTree_2D_MED
  Desc:	Simple static balanced cyclical kd-tree 
        stored one point per node in median array layout
---------------------------------------------------------*/

class CPUTree_2D_MED
{
public:
	/*------------------------------------
	  Type definitions
	------------------------------------*/

	/*------------------------------------
	  Constants
	------------------------------------*/
	
protected:
	/*------------------------------------
	  Fields
	------------------------------------*/

	unsigned int      m_cNodes;		// Count of Nodes
	CPUNode_2D_MED *  m_nodes;		// List of Nodes in Tree
	unsigned int	  m_startAxis;	// Starting Axis
	unsigned int      m_rootIdx;	// Root Index

	/*------------------------------------
	  Helper Methods
	------------------------------------*/

	inline void Init()
	{
		m_cNodes    = 0;
		m_nodes     = NULL;
		m_startAxis = X_AXIS;
	}

	inline void FiniNodes()
	{
		if (NULL != m_nodes)
		{
			delete [] m_nodes;
			m_nodes = NULL;
		}
		m_cNodes = 0;
	}

	inline void Fini()
	{
		FiniNodes();
	}

		// Copy Nodes
	bool CopyNodes( unsigned int cNodes, const CPUNode_2D_MED * nodes )
	{
		// Check Parameters
		if (NULL == nodes) { return false; }

		// Cleanup old list
		FiniNodes();

		// Anything in list ?!?
		if (cNodes == 0) { return true; }

		// Allocate Space for new node list
		m_nodes = new CPUNode_2D_MED[cNodes];
		if (NULL == m_nodes) { return false; }

		// Copy nodes over
		unsigned int i;
		for (i = 0; i < cNodes; i++)
		{
			// Copy Node over (with corrected pointers)
			m_nodes[i].m_ID     = nodes[i].m_ID;
			m_nodes[i].m_p.x    = nodes[i].m_p.x;
			m_nodes[i].m_p.y    = nodes[i].m_p.y;
			m_nodes[i].m_p.z    = nodes[i].m_p.z;
			m_nodes[i].m_Parent = nodes[i].m_Parent;
			m_nodes[i].m_Left   = nodes[i].m_Left;
			m_nodes[i].m_Right  = nodes[i].m_Right;
			m_nodes[i].m_SplitAxis = nodes[i].m_SplitAxis;
		}

		// Success
		return true;
	}

		// Copy 
	bool Copy( const CPUTree_2D_MED & toCopy )
	{
		if (this == &toCopy) { return true; }

		CopyNodes( toCopy.m_cNodes, toCopy.m_nodes );
		m_startAxis  = toCopy.m_startAxis;

		// Success
		return true;
	}

public:
	/*------------------------------------
	  Properties
	------------------------------------*/

	inline unsigned int NUM_NODES() const { return m_cNodes; }
	inline const CPUNode_2D_MED * NODES() const { return m_nodes; }

	inline const CPUNode_2D_MED & NODE( unsigned int idx ) const { return m_nodes[idx]; }
	inline CPUNode_2D_MED & NODE( unsigned int idx ) { return m_nodes[idx]; }

	inline const CPUNode_2D_MED * NODE_PTR( unsigned int idx ) const { return &(m_nodes[idx]); }
	inline CPUNode_2D_MED * NODE_PTR( unsigned int idx ) { return &(m_nodes[idx]); }

	inline unsigned int NODE_ID( unsigned int idx ) const { return m_nodes[idx].ID(); }
	inline float NODE_X( unsigned int idx ) const { return m_nodes[idx].X(); }
	inline float NODE_Y( unsigned int idx ) const { return m_nodes[idx].Y(); }
	inline float NODE_Z( unsigned int idx ) const { return m_nodes[idx].Z(); }
	
	inline unsigned int NODE_PARENT( unsigned int idx ) const { return m_nodes[idx].Parent(); }
	inline void NODE_PARENT( unsigned int idx, unsigned int val ) { m_nodes[idx].Parent( val ); }

	inline unsigned int NODE_LEFT( unsigned int idx ) const { return m_nodes[idx].Left(); }
	inline void NODE_LEFT( unsigned int idx, unsigned int val ) { m_nodes[idx].Left( val ); }

	inline unsigned int NODE_RIGHT( unsigned int idx ) const { return m_nodes[idx].Right(); }
	inline void NODE_RIGHT( unsigned int idx, unsigned int val ) { m_nodes[idx].Right( val ); }

	inline const CPUNode_2D_MED * NODE_PARENT_PTR( unsigned int idx ) const 
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == CPUNode_2D_MED::c_Invalid) ? NULL : &(m_nodes[parentIdx]));
		}
	inline CPUNode_2D_MED * NODE_PARENT_PTR( unsigned int idx )
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == CPUNode_2D_MED::c_Invalid) ? NULL : &(m_nodes[parentIdx]));
		}
	
	inline const CPUNode_2D_MED * NODE_LEFT_PTR( unsigned int idx ) const 
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx == CPUNode_2D_MED::c_Invalid) ? NULL : &(m_nodes[leftIdx]));
		}
	inline CPUNode_2D_MED * NODE_LEFT_PTR( unsigned int idx )
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx == CPUNode_2D_MED::c_Invalid) ? NULL : &(m_nodes[leftIdx]));
		}
	
	inline const CPUNode_2D_MED * NODE_RIGHT_PTR( unsigned int idx ) const 
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx == CPUNode_2D_MED::c_Invalid) ? NULL : &(m_nodes[rightIdx]));
		}
	inline CPUNode_2D_MED * NODE_RIGHT_PTR( unsigned int idx )
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx == CPUNode_2D_MED::c_Invalid) ? NULL : &(m_nodes[rightIdx]));
		}
	
	inline unsigned int NODE_AXIS( unsigned int idx ) const { return m_nodes[idx].Axis(); }

	inline bool IsRoot( unsigned int idx ) const
	{
		return ((m_nodes[idx].Parent() == CPUNode_2D_MED::c_Invalid) ? true : false);
	}

	inline bool IsLeaf( unsigned int idx ) const
	{
		return (((m_nodes[idx].Left()  == CPUNode_2D_MED::c_Invalid) &&
			     (m_nodes[idx].Right() == CPUNode_2D_MED::c_Invalid)) 
			     ? true : false); 
	}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUTree_2D_MED() :
		m_cNodes( 0 ),
		m_nodes( NULL ),
		m_startAxis( X_AXIS )
		{
		}

	CPUTree_2D_MED( const CPUTree_2D_MED & toCopy ) :
		m_cNodes( 0 ),
		m_nodes( NULL ),
		m_startAxis( X_AXIS )
		{
		}

	/*------------------------------------
	  Operators
	------------------------------------*/

	/*------------------------------------
	  Methods
	------------------------------------*/

	float GetNodeAxisValue( unsigned int index, unsigned int axis ) const;
	void SwapNodes( unsigned int idx1, unsigned int idx2 );

	// Find Median of 1 value
	unsigned int Median1
	(
		unsigned int start,	// IN - index of 1st element in set
		unsigned int shift,	// IN - shift factor to get subsequent elements
		unsigned int axis	// IN - axis to compare on
	);

	// Find Median of 2 values
	unsigned int Median2
	(
		unsigned int * v,
		unsigned int start,	// IN - index of 1st element in set
		unsigned int shift,	// IN - shift factor to get subsequent elements
		unsigned int axis	// IN - axis to compare on
	);

	// Find Median of 3 values
	unsigned int Median3
	(
		unsigned int * v,
		unsigned int start,	// IN - index of 1st element in set
		unsigned int shift,	// IN - shift factor to get subsequent elements
		unsigned int axis	// IN - axis to compare on
	);

	// Find Median of 4 values
	unsigned int Median4
	(
		unsigned int * v,
		unsigned int start,	// IN - index of 1st element in set
		unsigned int shift,	// IN - shift factor to get subsequent elements
		unsigned int axis	// IN - axis to compare on
	);

	// Find Median of 5 values
	unsigned int Median5
	(
		unsigned int * v,
		unsigned int start,	// IN - index of 1st element in set
		unsigned int shift,	// IN - shift factor to get subsequent elements
		unsigned int axis	// IN - axis to compare on
	);

	unsigned int Median5Sort
	(
		unsigned int start,	// IN - starting element of group of 5
		unsigned int shift,	// IN - shift increment to get to next element
		unsigned int axis	// IN - axis of value
	);


	// Median of 3 -- Helper Method
	void MedianOf3
	(
		unsigned int leftIdx,	// IN - left index
		unsigned int rightIdx,	// IN - right index
		unsigned int axis		// IN - axis to compare
	);

	bool MedianSortNodes
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end of range
		unsigned int & median,	// IN/OUT - approximate median number
								//          actual median number
		unsigned int axis		// IN - dimension(axis) to split along (x,y,z)
	);

	// Median of Median Sort to find median
	unsigned int FindMedianIndex
	(
		unsigned int left,		// IN - Left range to search
		unsigned int right,		// IN - Right range to search
		unsigned int shift,		// IN - Amount to shift
		unsigned int axis		// IN - axis <x,y,z,...> to work on
	);

	unsigned int FindMedianOfMedians
	(
		unsigned int left,		// IN - left of range to search
		unsigned int right,		// IN - right of range to search
		unsigned int axis		// IN - axis <x,y,z,...> to work on
	);

	bool MedianOfMediansSortNodes
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end of range
		unsigned int & median,	// IN/OUT - approximate median number
								//          actual median number
		unsigned int axis		// IN - dimension(axis) to split along (x,y,z)
	);

	// Alternate Median of Median Sort to find median
	unsigned int ChoosePivot
	(
		unsigned int start,	// IN - Starting index in range
		unsigned int end,	// IN - Ending index in range
		unsigned int axis	// IN - Axis to pivot on
	);

	unsigned int Partition
	(
		unsigned int start,	// IN - start of array
		unsigned int end,	// IN - end of array
		unsigned int pivot,	// IN - index of pivot value
		unsigned int axis	// IN - axis to do 'partition' on
	);

	unsigned int CPUTree_2D_MED::Select
	(
		unsigned int start,		// IN - start of array
		unsigned int end,		// IN - end of array
		unsigned int nth,		// IN - nth element from start of array to select
		unsigned int axis		// IN - axis to do 'select' on
	);

	bool TestMedian();

	bool TestSelect
	(
		unsigned int cPoints,		// Number of points in list
		const float4 * pointList,	// Raw points
		unsigned int kth,			// Find kth element with select
		unsigned int axis			// Axis to select on
	);


	bool ComputeBoundingBox
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end or range
		float        bounds[4]	// OUT - Bounding Box
	);

	bool Build2D( unsigned int cPoints, const float4 * pointList );
	bool Build2D( const std::vector<float4> & pointList );

	bool Build3D( unsigned int cPoints, const float4 * pointList );
	bool Build3D( const std::vector<float4> & pointList );

	bool BruteForceFindClosestPoint2D
	(
		const float4 & queryLocation,		// IN  - Location to sample
		unsigned int & closestPointIndex,	// OUT - Index of Closest Point
		unsigned int & closestID,			// OUT - ID of closest point
		float & bestDistance				// OUT - closest distance
	);

	bool BruteForceFindClosestPoint3D
	(
		const float4 & queryLocation,		// IN  - Location to sample
		unsigned int & closestPointIndex,	// OUT - Index of Closest Point
		unsigned int & closestID,			// OUT - ID of closest point
		float & bestDistance				// OUT - closest distance
	);

	bool FindClosestPoint2D
	(
		const float4 & queryLocation,	// IN  - Query Location
		unsigned int & closestIndex,	// OUT - closest point index to sample location
		unsigned int & closestID,		// OUT - ID of closest point
			   float &  bestDistance	// OUT - best distance
	) const;

	bool FindClosestPoint2DAlt
	(
		const float4 & queryLocation,	// IN  - Query Location
		unsigned int & closestIndex,	// OUT - closest point index to sample location
		unsigned int & closestID,		// OUT - ID of closest point
			   float &  bestDistance	// OUT - best distance
	) const;

	bool Find_QNN_2D
	( 
			CPU_NN_Result * queryResults,	// OUT: Results
			unsigned int      nQueries,		// IN: Number of Query points
			const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_NN_2D
	(
		CPU_NN_Result * queryResults		// OUT: Results
	);

	bool Find_KNN_2D
	( 
			CPU_NN_Result * queryResults,	// OUT: Results
			unsigned int      kVal,			// In: 'k' nearest neighbors to search for
			unsigned int      nQueries,		// IN: Number of Query points
			unsigned int      nPadQueries,	// IN: Number of padded query points
			const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_KNN_2D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nPadSearch	// In: Number of padded search points (query points)
	);


	bool Find_QNN_3D
	(
		const float4 & queryLocation,	// IN  - Query Location
		unsigned int & closestIndex,	// OUT - closest point index to sample location
		unsigned int & closestID,		// OUT - ID of closest point
			   float &  bestDistance	// OUT - best distance
	) const;


}; // end class CPUTree_2D_MED



/*-------------------------------------
  Function Declarations
-------------------------------------*/

bool FindClosestPoint2D
(
	const std::vector<float4> & searchList,	// IN - Points to put in Search List
	const std::vector<float4> & queryList,	// IN - Points to query against search list
	std::vector<CPU_NN_Result> & queryResults // OUT - Results of queries
);

bool FindClosestPoint3D
( 
	const std::vector<float4> & searchList,	// IN - Points to put in Search List
	const std::vector<float4> & queryList,	// IN - Points to query against search list
	std::vector<CPU_NN_Result> & queryResults // OUT - Results of queries
);

#endif // _CPUTree_2D_MED_H

