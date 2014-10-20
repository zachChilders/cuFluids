#pragma once
#ifndef _CPUTREE_LBT_H
#define _CPUTREE_LBT_H
/*-----------------------------------------------------------------------------
  Name:	CPUTREE_LBT.h
  Desc:	Defines Simple KDTree for CPU
  Notes: 
    - Kd-tree attributes
		static		-- we need to know all points "a priori" before building the kd-tree
		balanced	-- Tree has maximum height of O( log<2> n )
	    Left-Balanced tree array layout
	        -- kd-nodes in kd-tree is stored in left-balanced tree layout
			-- given 'n' points in kd-tree
			-- The Root is always found at index 1
			-- Given any node at position 'i'
				-- The parent is found at 'i/2'
				-- The left child is found at '2*i'
				-- The right child is found at '2*i+1'
		d-Dimensionality  -- 2D, 3D or 4D 
		cyclical	-- we follow a cyclical pattern in switching between axes
		               at each level of the tree, 
							for 2D <x,y,x,y,x,y,...>
							for 3D <x,y,z,x,y,z,...>
							for 4D <x,y,z,w,x,y,z,w,...>
							for 6D <x,y,z,w,s,t,x,y,z,w,s,t,...>
							etc.
		Point Storage -- 1 search point is stored at each internal or leaf node
		Minimal -- I have eliminated as many fields as possible
		           from the kd-node data structures.
				   The only remaining field is the stored search point

  Log:	Created by Shawn D. Brown (3/06/10)
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
#include <intrin.h>		// Intrinsics


// Cuda Includes
#ifndef _CUT_
	#include <cutil.h>
#endif
#if !defined(__VECTOR_TYPES_H__)
	#include <vector_types.h>	
#endif

// App Includes
//#include "CPUTree_MED.h"
#include "KD_Flags.h"
#include "QueryResult.h"


/*-------------------------------------
  Forward Declarations
-------------------------------------*/

class CPUNode_2D_LBT_LBT;
class CPUTree;


/*-------------------------------------
  Type Definitons
-------------------------------------*/

#define CPU_BUILD_STACK_SIZE  32
#define CPU_SEARCH_STACK_SIZE 32

#ifdef _BUILD_STATS
typedef struct
{
	unsigned int cNodeLoops,  cPartLoops;
	unsigned int cStoreReads, cStoreWrites;
	unsigned int cPivotReads, cPivotSwaps, cPivotWrites;
	unsigned int cPartReads,  cPartSwaps, cPartWrites;
} CPU_BUILD_STATS;
#endif


	// Range [start, end] for Left-Balanced Trees
typedef struct
{
	unsigned int start;			// Start of Range (inclusive)
	unsigned int end;			// End of Range (inclusive)
		// Note:  num elements n = (end - start) + 1
	unsigned int targetID;		// Target node index
	unsigned int flags;			// Build flags
		// unsigned int half;	// size of half a tree [0..28]	// 29 bits
		// unsigned int axis;	// curr axis [29..30]			//  2 bits
		// unsigned int res;	// reserved [31]				//  1 bit
} KD_BUILD_LBT;

typedef struct
{
	unsigned int flags;	
		// Node Index	(Bits [0..27])  Limits us to at most 2^28 (268+ million) nodes in search list
		// Split Axis	(Bits [28..30])	{x,y, z,w, s,t, u,v} Up to 8 dimensions (nDim <= 8 = 2^3)
		// On/Off       (Bits [31])		(Onside,offside node tracking for trim optimization)
		// 
		// NOTE: See search node flags in KD_flags.h for bit masks and shifts
	float        splitValue;	// Split Value
} KD_SEARCH_LBT;


/*-------------------------------------
  external functions
-------------------------------------*/

extern const char * AxisToString( unsigned int currAxis );


/*-------------------------------------
  Classes
-------------------------------------*/

/*--------------------------------------------------------------
  Name:	CPUNode_2D_LBT
  Desc:	Simple 2D kd-Tree node for Left Balanced tree on CPU
--------------------------------------------------------------*/

class CPUNode_2D_LBT
{
public:
	/*------------------------------------
	  Type definitions
	------------------------------------*/

	typedef struct _val2 { 
		float x; 
		float y; 
	} VAL2;

	/*------------------------------------
	  Constants
	------------------------------------*/
	
	static const unsigned int c_Invalid = static_cast<unsigned int>( -1 );
	static const unsigned int c_Root    = static_cast<unsigned int>( 1 );

protected:
	/*------------------------------------
	  Fields
	------------------------------------*/
	
	friend class CPUTree_2D_LBT;

	// 2D Point
	union {
		float m_v[2];
		VAL2  m_p;
	};

	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
	
	/*------------------------------------
	  Helper Methods
	------------------------------------*/

public:
	/*------------------------------------
	  Properties
	------------------------------------*/

	static unsigned int MaxDim() { return static_cast<unsigned int>( 2 ); }

	inline const float * BASE_PNT() const { return m_v; }
	inline float * BASE_PNT() { return m_v; }

		// X
	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }
	
		// Y
	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

	inline float GetVal( unsigned int nIndex ) const
		{
			// ASSERT( nIndex < 3 );
			return m_v[nIndex];
		}
	inline void SetVal( unsigned int nIndex, float newVal )
		{
			// ASSERT( nIndex < 3 );
			m_v[nIndex] = newVal;
		}

		// Search Index
	inline unsigned int SearchID() const     { return m_searchIdx; }
	inline void SearchID( unsigned int val ) { m_searchIdx = val; }

		// Node Index
	inline unsigned int NodeID() const     { return m_nodeIdx; }
	inline void NodeID( unsigned int val ) { m_nodeIdx = val; }

		// Length (2D)
	inline float length2() const { return ( X()*X() + Y()*Y() ); }
	inline float length() const  { return static_cast<float>( sqrt( length() ) ); }

		// Root 
	inline bool IsRoot() const { return ((c_Root == m_nodeIdx) ? true : false); }

		// Parent, Left & Right Children
	unsigned int Parent() const { return (m_nodeIdx >> 1); }
	unsigned int Left() const   { return (m_nodeIdx << 1); }
	unsigned int Right() const  { return ((m_nodeIdx << 1)+1); }

		// Height of node in kd-tree (SLOW)
	unsigned int Height() const 
		{
			// Binary search for height as log2(nodeIndex)
				// Portable solution
			unsigned int logVal = 0u;
			unsigned int v = m_nodeIdx;

			if (v >= (1u << 16u)) { v >>= 16u; logVal |= 16u; }
			if (v >= (1u <<  8u)) { v >>=  8u; logVal |=  8u; }
			if (v >= (1u <<  4u)) { v >>=  4u; logVal |=  4u; }
			if (v >= (1u <<  2u)) { v >>=  2u; logVal |=  2u; }
			if (v >= (1u <<  1u)) { logVal |= 1u; }

			return (logVal+1);
		}

		// Axis (SLOW)
	unsigned int Axis() const 
		{
			// Assume we start from x-axis at root
			unsigned int h = Height();
			unsigned int xy = (h & 0x1u);		// equivalent to xy = h % 2
			return ((xy == 0u) ? 1u : 0u);		// reverse order
		}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUNode_2D_LBT() 
		// :
		// Deliberately do nothing for higher performance 
		//m_searchIdx( c_Invalid ),
		//m_nodeIdx( c_Invalid )
		{
			//m_p.x = 0.0f;
			//m_p.y = 0.0f;
		}

	CPUNode_2D_LBT( float x, float y ) :
		m_searchIdx( c_Invalid ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
		}

	CPUNode_2D_LBT( unsigned int searchID, float x, float y ) :
		m_searchIdx( searchID ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
		}

		// Full constructor
	CPUNode_2D_LBT( unsigned int searchID, unsigned int nodeID, float x, float y ) :
		m_searchIdx( searchID ),
		m_nodeIdx( nodeID )
		{
			m_p.x = x;
			m_p.y = y;
		}

		// Copy Constructor
	CPUNode_2D_LBT( const CPUNode_2D_LBT & toCopy ) :
		m_searchIdx( toCopy.m_searchIdx ),
		m_nodeIdx( toCopy.m_nodeIdx )
		{
			m_p.x    = toCopy.m_p.x;
			m_p.y    = toCopy.m_p.y;
		}

		// Destructor
	~CPUNode_2D_LBT() 
		{
			// Do nothing for faster performance
			//m_searchIdx = c_Invalid;
			//m_nodeIdx   = c_Invalid;
		}

	/*------------------------------------
	  Operators
	------------------------------------*/

		// Copy Operator
	CPUNode_2D_LBT & operator = ( const CPUNode_2D_LBT & toCopy )
	{
		if (this != &toCopy) 
		{
			m_searchIdx = toCopy.m_searchIdx;
			m_nodeIdx   = toCopy.m_nodeIdx;
			m_p.x		= toCopy.m_p.x;
			m_p.y		= toCopy.m_p.y;
		}

		return (*this);
	}

		// Point Index Operators
	inline float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 3 );
			return m_v[nIndex];
		}	
	inline float operator[] ( unsigned int nIndex ) const 
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

		unsigned int searchID = SearchID();
		unsigned int nodeID   = NodeID();

		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );

		unsigned int parentID = Parent();
		unsigned int leftID   = Left();
		unsigned int rightID  = Right();

		unsigned int h = Height();
		unsigned int A = Axis();

		const char * axisString = AxisToString( A );

		sprintf_s( szBuff, cchBuff, "<%d, %d>=<%3.6f, %3.6f> [P=%d, L=%d, R=%d, H=%d, A=%s]", 
			       nodeID, searchID, x, y, 
				   parentID, leftID, rightID, h, axisString  );
		
		value = szBuff;
	}

	inline void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		printf( "%s", pointValue.c_str() );
	}

}; // end CPUNode_2D_LBT



/*--------------------------------------------------------------
  Name:	CPUNode_3D_LBT
  Desc:	Simple 3D kd-Tree node for Left Balanced tree on CPU
--------------------------------------------------------------*/

class CPUNode_3D_LBT
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
	static const unsigned int c_Root    = static_cast<unsigned int>( 1 );

protected:
	/*------------------------------------
	  Fields
	------------------------------------*/
	
	friend class CPUTree_3D_LBT;

	// 3D Point
	union {
		float m_v[3];
		VAL3  m_p;
	};

	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
	
	/*------------------------------------
	  Helper Methods
	------------------------------------*/

public:
	/*------------------------------------
	  Properties
	------------------------------------*/

	static unsigned int MaxDim() { return static_cast<unsigned int>( 3 ); }

	inline const float * BASE_PNT() const { return m_v; }
	inline float * BASE_PNT() { return m_v; }

		// X
	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }
	
		// Y
	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

		// Z
	inline float Z() const { return m_p.z; }
	inline void Z( float val ) { m_p.z = val; }

	inline float GetVal( unsigned int nIndex ) const
		{
			// ASSERT( nIndex < 4 );
			return m_v[nIndex];
		}
	inline void SetVal( unsigned int nIndex, float newVal )
		{
			// ASSERT( nIndex < 4 );
			m_v[nIndex] = newVal;
		}

		// Search Index
	inline unsigned int SearchID() const     { return m_searchIdx; }
	inline void SearchID( unsigned int val ) { m_searchIdx = val; }

		// Node Index
	inline unsigned int NodeID() const     { return m_nodeIdx; }
	inline void NodeID( unsigned int val ) { m_nodeIdx = val; }

		// Length (2D)
	inline float length2() const { return ( X()*X() + Y()*Y() + Z()*Z() ); }
	inline float length() const  { return static_cast<float>( sqrt( length() ) ); }

		// Root 
	inline bool IsRoot() const { return ((c_Root == m_nodeIdx) ? true : false); }

		// Parent, Left & Right Children
	unsigned int Parent() const { return (m_nodeIdx >> 1); }
	unsigned int Left() const   { return (m_nodeIdx << 1); }
	unsigned int Right() const  { return ((m_nodeIdx << 1)+1); }

		// Height of node in kd-tree (SLOW)
	unsigned int Height() const 
		{
			// Binary search for height as log2(nodeIndex)
				// Portable solution
			unsigned int logVal = 0u;
			unsigned int v = m_nodeIdx;

			if (v >= (1u << 16u)) { v >>= 16u; logVal |= 16u; }
			if (v >= (1u <<  8u)) { v >>=  8u; logVal |=  8u; }
			if (v >= (1u <<  4u)) { v >>=  4u; logVal |=  4u; }
			if (v >= (1u <<  2u)) { v >>=  2u; logVal |=  2u; }
			if (v >= (1u <<  1u)) { logVal |= 1u; }

			return (logVal+1);
		}

		// Axis (SLOW)
	unsigned int Axis() const 
		{
			// Assume we start from x-axis at root
			unsigned int h = Height();
			unsigned int axis = (h+2) % 3;
			return axis;
		}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUNode_3D_LBT() 
		// :
		// Deliberately do nothing for higher performance 
		//m_searchIdx( c_Invalid ),
		//m_nodeIdx( c_Invalid )
		{
			//m_p.x = 0.0f;
			//m_p.y = 0.0f;
			//m_p.z = 0.0f;
		}

	CPUNode_3D_LBT( float x, float y, float z ) :
		m_searchIdx( c_Invalid ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
		}

	CPUNode_3D_LBT( unsigned int searchID, float x, float y, float z ) :
		m_searchIdx( searchID ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
		}

		// Full constructor
	CPUNode_3D_LBT( unsigned int searchID, unsigned int nodeID, float x, float y, float z ) :
		m_searchIdx( searchID ),
		m_nodeIdx( nodeID )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
		}

		// Copy Constructor
	CPUNode_3D_LBT( const CPUNode_3D_LBT & toCopy ) :
		m_searchIdx( toCopy.m_searchIdx ),
		m_nodeIdx( toCopy.m_nodeIdx )
		{
			m_p.x = toCopy.m_p.x;
			m_p.y = toCopy.m_p.y;
			m_p.z = toCopy.m_p.z;
		}

		// Destructor
	~CPUNode_3D_LBT() 
		{
			// Do nothing for faster performance
			//m_searchIdx = c_Invalid;
			//m_nodeIdx   = c_Invalid;
		}

	/*------------------------------------
	  Operators
	------------------------------------*/

		// Copy Operator
	CPUNode_3D_LBT & operator = ( const CPUNode_3D_LBT & toCopy )
	{
		if (this != &toCopy) 
		{
			m_searchIdx = toCopy.m_searchIdx;
			m_nodeIdx   = toCopy.m_nodeIdx;
			m_p.x		= toCopy.m_p.x;
			m_p.y		= toCopy.m_p.y;
			m_p.z       = toCopy.m_p.z;
		}

		return (*this);
	}

		// Point Index Operators
	inline float operator [] ( unsigned int nIndex )
		{
			// ASSERT( nIndex < 4 );
			return m_v[nIndex];
		}	
	inline float operator [] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 4 );
			return m_v[nIndex];
		}


	/*------------------------------------
	  Methods
	------------------------------------*/

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		unsigned int cchBuff = sizeof(szBuff)/sizeof(char);

		unsigned int searchID = SearchID();
		unsigned int nodeID   = NodeID();

		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );
		double z = static_cast< double >( Z() );

		unsigned int parentID = Parent();
		unsigned int leftID   = Left();
		unsigned int rightID  = Right();

		unsigned int h = Height();
		unsigned int A = Axis();

		const char * axisString = AxisToString( A );

		sprintf_s( szBuff, cchBuff, "<%d, %d>=<%3.6f, %3.6f, %3.6f> [P=%d, L=%d, R=%d, H=%d, A=%s]", 
			       nodeID, searchID, x, y, z,
				   parentID, leftID, rightID, h, axisString  );
		
		value = szBuff;
	}

	inline void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		printf( "%s", pointValue.c_str() );
	}

}; // end CPUNode_3D_LBT


/*--------------------------------------------------------------
  Name:	CPUNode_4D_LBT
  Desc:	Simple 4D kd-Tree node for Left Balanced tree on CPU
--------------------------------------------------------------*/

class CPUNode_4D_LBT
{
public:
	/*------------------------------------
	  Type definitions
	------------------------------------*/

	typedef struct _val4 { 
		float x; 
		float y; 
		float z;
		float w;
	} VAL4;

	/*------------------------------------
	  Constants
	------------------------------------*/
	
	static const unsigned int c_Invalid = static_cast<unsigned int>( -1 );
	static const unsigned int c_Root    = static_cast<unsigned int>( 1 );

protected:
	/*------------------------------------
	  Fields
	------------------------------------*/
	
	friend class CPUTree_4D_LBT;

	// 4D Point
	union {
		float m_v[4];
		VAL4  m_p;
	};

	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
	
	/*------------------------------------
	  Helper Methods
	------------------------------------*/

public:
	/*------------------------------------
	  Properties
	------------------------------------*/

	static unsigned int MaxDim() { return static_cast<unsigned int>( 4 ); }

	inline const float * BASE_PNT() const { return m_v; }
	inline float * BASE_PNT() { return m_v; }

		// X
	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }
	
		// Y
	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

		// Z
	inline float Z() const { return m_p.z; }
	inline void Z( float val ) { m_p.z = val; }

		// W
	inline float W() const { return m_p.w; }
	inline void W( float val ) { m_p.w = val; }

	inline float GetVal( unsigned int nIndex ) const
		{
			// ASSERT( nIndex < 5 );
			return m_v[nIndex];
		}
	inline void SetVal( unsigned int nIndex, float newVal )
		{
			// ASSERT( nIndex < 5 );
			m_v[nIndex] = newVal;
		}

		// Search Index
	inline unsigned int SearchID() const     { return m_searchIdx; }
	inline void SearchID( unsigned int val ) { m_searchIdx = val; }

		// Node Index
	inline unsigned int NodeID() const     { return m_nodeIdx; }
	inline void NodeID( unsigned int val ) { m_nodeIdx = val; }

		// Length (4D)
	inline float length2() const { return ( X()*X() + Y()*Y() + Z()*Z() + W()*W() ); }
	inline float length() const  { return static_cast<float>( sqrt( length() ) ); }

		// Root 
	inline bool IsRoot() const { return ((c_Root == m_nodeIdx) ? true : false); }

		// Parent, Left & Right Children
	unsigned int Parent() const { return (m_nodeIdx >> 1); }
	unsigned int Left() const   { return (m_nodeIdx << 1); }
	unsigned int Right() const  { return ((m_nodeIdx << 1)+1); }

		// Height of node in kd-tree (SLOW)
	unsigned int Height() const 
		{
			// Binary search for height as log2(nodeIndex)
				// Portable solution
			unsigned int logVal = 0u;
			unsigned int v = m_nodeIdx;

			if (v >= (1u << 16u)) { v >>= 16u; logVal |= 16u; }
			if (v >= (1u <<  8u)) { v >>=  8u; logVal |=  8u; }
			if (v >= (1u <<  4u)) { v >>=  4u; logVal |=  4u; }
			if (v >= (1u <<  2u)) { v >>=  2u; logVal |=  2u; }
			if (v >= (1u <<  1u)) { logVal |= 1u; }

			return (logVal+1);
		}

		// Axis (SLOW)
	unsigned int Axis() const 
		{
			// Assume we start from x-axis at root
			unsigned int h = Height();
			unsigned int axis = (h+3) >> 2;
			return axis;
		}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUNode_4D_LBT() 
		// :
		// Deliberately do nothing for higher performance 
		//m_searchIdx( c_Invalid ),
		//m_nodeIdx( c_Invalid )
		{
			//m_p.x = 0.0f;
			//m_p.y = 0.0f;
			//m_p.z = 0.0f;
			//m_p.w = 0.0f;
		}

	CPUNode_4D_LBT( float x, float y, float z, float w ) :
		m_searchIdx( c_Invalid ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
		}

	CPUNode_4D_LBT( unsigned int searchID, float x, float y, float z, float w ) :
		m_searchIdx( searchID ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
		}

		// Full constructor
	CPUNode_4D_LBT( unsigned int searchID, unsigned int nodeID, float x, float y, float z, float w ) :
		m_searchIdx( searchID ),
		m_nodeIdx( nodeID )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
		}

		// Copy Constructor
	CPUNode_4D_LBT( const CPUNode_4D_LBT & toCopy ) :
		m_searchIdx( toCopy.m_searchIdx ),
		m_nodeIdx( toCopy.m_nodeIdx )
		{
			m_p.x = toCopy.m_p.x;
			m_p.y = toCopy.m_p.y;
			m_p.z = toCopy.m_p.z;
			m_p.w = toCopy.m_p.w;
		}

		// Destructor
	~CPUNode_4D_LBT() 
		{
			// Do nothing for faster performance
			//m_searchIdx = c_Invalid;
			//m_nodeIdx   = c_Invalid;
		}

	/*------------------------------------
	  Operators
	------------------------------------*/

		// Copy Operator
	CPUNode_4D_LBT & operator = ( const CPUNode_4D_LBT & toCopy )
	{
		if (this != &toCopy) 
		{
			m_searchIdx = toCopy.m_searchIdx;
			m_nodeIdx   = toCopy.m_nodeIdx;
			m_p.x		= toCopy.m_p.x;
			m_p.y		= toCopy.m_p.y;
			m_p.z       = toCopy.m_p.z;
			m_p.w       = toCopy.m_p.w;
		}

		return (*this);
	}

		// Point Index Operators
	inline float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 5 );
			return m_v[nIndex];
		}	
	inline float operator[] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 5 );
			return m_v[nIndex];
		}


	/*------------------------------------
	  Methods
	------------------------------------*/

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		unsigned int cchBuff = sizeof(szBuff)/sizeof(char);

		unsigned int searchID = SearchID();
		unsigned int nodeID   = NodeID();

		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );
		double z = static_cast< double >( Z() );
		double w = static_cast< double >( W() );

		unsigned int parentID = Parent();
		unsigned int leftID   = Left();
		unsigned int rightID  = Right();

		unsigned int h = Height();
		unsigned int A = Axis();

		const char * axisString = AxisToString( A );

		sprintf_s( szBuff, cchBuff, "<%d, %d>=<%3.6f, %3.6f, %3.6f, %3.6f> [P=%d, L=%d, R=%d, H=%d, A=%s]", 
			       nodeID, searchID, x, y, z, w,
				   parentID, leftID, rightID, h, axisString  );
		
		value = szBuff;
	}

	inline void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		printf( "%s", pointValue.c_str() );
	}

}; // end CPUNode_4D_LBT


/*--------------------------------------------------------------
  Name:	CPUNode_6D_LBT
  Desc:	Simple 6D kd-Tree node for Left Balanced tree on CPU
--------------------------------------------------------------*/

class CPUNode_6D_LBT
{
public:
	/*------------------------------------
	  Type definitions
	------------------------------------*/

	typedef struct _val6 { 
		float x; 
		float y; 
		float z;
		float w;
		float s;
		float t;
	} VAL6;

	/*------------------------------------
	  Constants
	------------------------------------*/
	
	static const unsigned int c_Invalid = static_cast<unsigned int>( -1 );
	static const unsigned int c_Root    = static_cast<unsigned int>( 1 );

protected:
	/*------------------------------------
	  Fields
	------------------------------------*/
	
	friend class CPUTree_6D_LBT;

	// 6D Point
	union {
		float m_v[6];
		VAL6  m_p;
	};

	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
	
	/*------------------------------------
	  Helper Methods
	------------------------------------*/

public:
	/*------------------------------------
	  Properties
	------------------------------------*/

	static unsigned int MaxDim() { return static_cast<unsigned int>( 4 ); }

	inline const float * BASE_PNT() const { return m_v; }
	inline float * BASE_PNT() { return m_v; }

		// X
	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }
	
		// Y
	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

		// Z
	inline float Z() const { return m_p.z; }
	inline void Z( float val ) { m_p.z = val; }

		// W
	inline float W() const { return m_p.w; }
	inline void W( float val ) { m_p.w = val; }

		// S
	inline float S() const { return m_p.s; }
	inline void S( float val ) { m_p.s = val; }

		// T
	inline float T() const { return m_p.t; }
	inline void T( float val ) { m_p.t = val; }

	inline float GetVal( unsigned int nIndex ) const
		{
			// ASSERT( nIndex < 7 );
			return m_v[nIndex];
		}
	inline void SetVal( unsigned int nIndex, float newVal )
		{
			// ASSERT( nIndex < 7 );
			m_v[nIndex] = newVal;
		}

		// Search Index
	inline unsigned int SearchID() const     { return m_searchIdx; }
	inline void SearchID( unsigned int val ) { m_searchIdx = val; }

		// Node Index
	inline unsigned int NodeID() const     { return m_nodeIdx; }
	inline void NodeID( unsigned int val ) { m_nodeIdx = val; }

		// Length (4D)
	inline float length2() const { return ( X()*X() + Y()*Y() + Z()*Z() + W()*W() + S()*S() + T()*T() ); }
	inline float length() const  { return static_cast<float>( sqrt( length() ) ); }

		// Root 
	inline bool IsRoot() const { return ((c_Root == m_nodeIdx) ? true : false); }

		// Parent, Left & Right Children
	unsigned int Parent() const { return (m_nodeIdx >> 1); }
	unsigned int Left() const   { return (m_nodeIdx << 1); }
	unsigned int Right() const  { return ((m_nodeIdx << 1)+1); }

		// Height of node in kd-tree (SLOW)
	unsigned int Height() const 
		{
			// Binary search for height as log2(nodeIndex)
				// Portable solution
			unsigned int logVal = 0u;
			unsigned int v = m_nodeIdx;

			if (v >= (1u << 16u)) { v >>= 16u; logVal |= 16u; }
			if (v >= (1u <<  8u)) { v >>=  8u; logVal |=  8u; }
			if (v >= (1u <<  4u)) { v >>=  4u; logVal |=  4u; }
			if (v >= (1u <<  2u)) { v >>=  2u; logVal |=  2u; }
			if (v >= (1u <<  1u)) { logVal |= 1u; }

			return (logVal+1);
		}

		// Axis (SLOW)
	unsigned int Axis() const 
		{
			// Assume we start from x-axis at root
			unsigned int h = Height();
			unsigned int axis = (h+5) % 6;
			return axis;
		}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUNode_6D_LBT() 
		// :
		// Deliberately do nothing for higher performance 
		//m_searchIdx( c_Invalid ),
		//m_nodeIdx( c_Invalid )
		{
			//m_p.x = 0.0f;
			//m_p.y = 0.0f;
			//m_p.z = 0.0f;
			//m_p.w = 0.0f;
		}

	CPUNode_6D_LBT( float x, float y, float z, float w, float s, float t ) :
		m_searchIdx( c_Invalid ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
			m_p.s = s;
			m_p.t = t;
		}

	CPUNode_6D_LBT( unsigned int searchID, 
		            float x, float y, float z, float w, float s, float t ) :
		m_searchIdx( searchID ),
		m_nodeIdx( c_Invalid )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
			m_p.s = s;
			m_p.t = t;
		}

		// Full constructor
	CPUNode_6D_LBT( unsigned int searchID, unsigned int nodeID, 
		            float x, float y, float z, float w, float s, float t ) :
		m_searchIdx( searchID ),
		m_nodeIdx( nodeID )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
			m_p.s = s;
			m_p.t = t;
		}

		// Copy Constructor
	CPUNode_6D_LBT( const CPUNode_6D_LBT & toCopy ) :
		m_searchIdx( toCopy.m_searchIdx ),
		m_nodeIdx( toCopy.m_nodeIdx )
		{
			m_p.x = toCopy.m_p.x;
			m_p.y = toCopy.m_p.y;
			m_p.z = toCopy.m_p.z;
			m_p.w = toCopy.m_p.w;
			m_p.s = toCopy.m_p.s;
			m_p.t = toCopy.m_p.t;
		}

		// Destructor
	~CPUNode_6D_LBT() 
		{
			// Do nothing for faster performance
			//m_searchIdx = c_Invalid;
			//m_nodeIdx   = c_Invalid;
		}

	/*------------------------------------
	  Operators
	------------------------------------*/

		// Copy Operator
	CPUNode_6D_LBT & operator = ( const CPUNode_6D_LBT & toCopy )
	{
		if (this != &toCopy) 
		{
			m_searchIdx = toCopy.m_searchIdx;
			m_nodeIdx   = toCopy.m_nodeIdx;
			m_p.x		= toCopy.m_p.x;
			m_p.y		= toCopy.m_p.y;
			m_p.z       = toCopy.m_p.z;
			m_p.w       = toCopy.m_p.w;
			m_p.s       = toCopy.m_p.s;
			m_p.t       = toCopy.m_p.t;
		}

		return (*this);
	}

		// Point Index Operators
	inline float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 7 );
			return m_v[nIndex];
		}	
	inline float operator[] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 7 );
			return m_v[nIndex];
		}


	/*------------------------------------
	  Methods
	------------------------------------*/

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		unsigned int cchBuff = sizeof(szBuff)/sizeof(char);

		unsigned int searchID = SearchID();
		unsigned int nodeID   = NodeID();

		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );
		double z = static_cast< double >( Z() );
		double w = static_cast< double >( W() );
		double s = static_cast< double >( S() );
		double t = static_cast< double >( T() );

		unsigned int parentID = Parent();
		unsigned int leftID   = Left();
		unsigned int rightID  = Right();

		unsigned int h = Height();
		unsigned int A = Axis();

		const char * axisString = AxisToString( A );

		sprintf_s( szBuff, cchBuff, "<%d, %d>=<%3.6f, %3.6f, %3.6f, %3.6f, %3.6f, %3.6f>\n\t[P=%d, L=%d, R=%d, H=%d, A=%s]", 
			       nodeID, searchID, 
				   x, y, z, w, s, t,
				   parentID, leftID, rightID, h, axisString  );
		
		value = szBuff;
	}

	inline void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		printf( "%s", pointValue.c_str() );
	}

}; // end CPUNode_6D_LBT


/*---------------------------------------------------------
  Name:	CPUTree_2D_LBT
  Desc:	Simple static balanced cyclical kd-tree 
        stored one point per node 
		in left balanced array layout
---------------------------------------------------------*/

class CPUTree_2D_LBT
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

	unsigned int     m_cNodes;		// Count of Nodes
	CPUNode_2D_LBT * m_nodes;		// List of 2D LBT kd-nodes in kd-Tree
	unsigned int     m_startAxis;	// Starting Axis
	unsigned int     m_rootIdx;		// Root Index
#ifdef _BUILD_STATS
	CPU_BUILD_STATS  m_cpuStats;	// CPU stats
#endif

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

	bool CopyNodes( unsigned int cNodes, const CPUNode_2D_LBT * nodes )
	{
		// Check Parameters
		if (NULL == nodes) { return false; }

		// Cleanup old list
		FiniNodes();

		// Anything in list ?!?
		if (cNodes == 0) { return true; }

		// Allocate Space for new node list
		m_nodes = new CPUNode_2D_LBT[cNodes+1];
		if (NULL == m_nodes) { return false; }

		// node at index 0 is wasted on purpose
			// so arithmetic becomes 1-based instead of 0-based
		m_nodes[0].m_searchIdx = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_nodeIdx   = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_p.x	   = 0.0f;
		m_nodes[0].m_p.y	   = 0.0f;

		// Copy nodes over
		unsigned int i;
		for (i = 0; i < cNodes; i++)
		{
			// Copy each node over (with corrected pointers)
			m_nodes[i+1].m_searchIdx = nodes[i].m_searchIdx;
			m_nodes[i+1].m_nodeIdx   = nodes[i].m_nodeIdx;
			m_nodes[i+1].m_p.x       = nodes[i].m_p.x;
			m_nodes[i+1].m_p.y       = nodes[i].m_p.y;
		}

		// Success
		return true;
	}

		// Copy 
	bool Copy( const CPUTree_2D_LBT & toCopy )
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
	inline const CPUNode_2D_LBT * NODES() const { return m_nodes; }

	inline const CPUNode_2D_LBT & NODE( unsigned int idx ) const { return m_nodes[idx]; }
	inline CPUNode_2D_LBT & NODE( unsigned int idx ) { return m_nodes[idx]; }

	inline const CPUNode_2D_LBT * NODE_PTR( unsigned int idx ) const { return &(m_nodes[idx]); }
	inline CPUNode_2D_LBT * NODE_PTR( unsigned int idx ) { return &(m_nodes[idx]); }

	inline unsigned int SEARCH_ID( unsigned int idx ) const { return m_nodes[idx].SearchID(); }
	inline unsigned int NODE_ID( unsigned int idx ) const { return m_nodes[idx].NodeID(); }
	inline float NODE_X( unsigned int idx ) const { return m_nodes[idx].X(); }
	inline float NODE_Y( unsigned int idx ) const { return m_nodes[idx].Y(); }

	inline unsigned int NODE_PARENT( unsigned int idx ) const { return m_nodes[idx].Parent(); }
	inline unsigned int NODE_LEFT( unsigned int idx ) const   { return m_nodes[idx].Left(); }
	inline unsigned int NODE_RIGHT( unsigned int idx ) const  { return m_nodes[idx].Right(); }

	inline const CPUNode_2D_LBT * NODE_PARENT_PTR( unsigned int idx ) const 
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	inline CPUNode_2D_LBT * NODE_PARENT_PTR( unsigned int idx )
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	
	inline const CPUNode_2D_LBT * NODE_LEFT_PTR( unsigned int idx ) const 
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	inline CPUNode_2D_LBT * NODE_LEFT_PTR( unsigned int idx )
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	
	inline const CPUNode_2D_LBT * NODE_RIGHT_PTR( unsigned int idx ) const 
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	inline CPUNode_2D_LBT * NODE_RIGHT_PTR( unsigned int idx )
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	
	inline unsigned int NODE_HEIGHT( unsigned int idx ) const { return m_nodes[idx].Height(); }
	inline unsigned int NODE_AXIS( unsigned int idx ) const { return m_nodes[idx].Axis(); }

	inline bool IsRoot( unsigned int idx ) const
	{
			//return (m_nodes[idx].IsRoot());
		return ((idx == CPUNode_2D_LBT::c_Root) ? true : false);
	}

	inline bool IsLeaf( unsigned int idx ) const
	{
		// If both node.left and node.right are beyond the max size of this array
		// then this node is a leaf
			//unsigned int L = m_nodes[idx].Left();
			//unsigned int R = m_nodes[idx].Right();
		unsigned int L = idx << 1;
		unsigned int R = (idx << 1) + 1;
		return ((L > m_cNodes) && (R > m_cNodes) ? true : false);
	}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUTree_2D_LBT() :
		m_cNodes( 0 ),
		m_nodes( NULL ),
		m_startAxis( X_AXIS )
		{
		}

	CPUTree_2D_LBT( const CPUTree_2D_LBT & toCopy ) :
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

	// GetNodeAxisValue -- Helper Method
	float GetNodeAxisValue
	( 
		const CPUNode_2D_LBT * currNodes,	// IN - node list
		unsigned int index,				// IN - index of node containing 2D point
		unsigned int axis				// IN - axis of 2D point to retrieve
	) const;

	// Swap Nodes -- Helper Method
	void SwapNodes
	( 
		CPUNode_2D_LBT * currNodes,			// IN - node list
		unsigned int idx1,				// IN - index of 1st node to swap
		unsigned int idx2				// IN - index of 2nd node to swap
	);

	// Median of 3 -- Helper Method
	void MedianOf3
	(
		CPUNode_2D_LBT * currNodes,			// IN - node list
		unsigned int leftIdx,			// IN - left index
		unsigned int rightIdx,			// IN - right index
		unsigned int axis				// IN - axis to compare
	);

	bool MedianSortNodes
	(
		CPUNode_2D_LBT * nodes,		// IN - node list
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end of range
		unsigned int & median,	// IN/OUT - approximate median number
								//          actual median number
		unsigned int axis		// IN - dimension(axis) to split along (x,y,z)
	);

	bool ComputeBoundingBox
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end or range
		float        bounds[4]	// OUT - Bounding Box
	);

	bool Build2D( unsigned int cPoints, const float2 * pointList );
	bool Build2D( unsigned int cPoints, const float3 * pointList );
	bool Build2D( unsigned int cPoints, const float4 * pointList );

	bool Build2DStats( unsigned int cPoints, const float2 * pointList );
	bool Build2DStats( unsigned int cPoints, const float3 * pointList );
	bool Build2DStats( unsigned int cPoints, const float4 * pointList );

	bool BF_FindNN_2D
	(
		const float4 & queryLocation,	// IN  - Location to sample
		unsigned int & nearestID,		// OUT - ID of nearest point
		float & nearestDistance			// OUT - nearest distance
	);

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
			unsigned int      kVal,			// IN: 'k' nearest neighbors to search for
			unsigned int      nQueries,		// IN: Number of Query points
			unsigned int      nPadQueries,	// IN: Number of padded query points
			const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_KNN_2D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// IN: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: Number of padded search points (query points)
	);

	void DumpNodes() const;
#ifdef _BUILD_STATS
	void DumpBuildStats() const;
#endif
	bool Validate() const;

}; // end class CPUTree_2D_LBT


/*---------------------------------------------------------
  Name:	CPUTree_3D_LBT
  Desc:	Simple static balanced cyclical kd-tree 
        stored one point per node 
		in left balanced array layout
---------------------------------------------------------*/

class CPUTree_3D_LBT
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

	unsigned int  m_cNodes;		// Count of Nodes
	CPUNode_3D_LBT * m_nodes;	// List of 3D LBT kd-nodes in kd-Tree
	unsigned int  m_startAxis;	// Starting Axis
	unsigned int  m_rootIdx;	// Root Index

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

	bool CopyNodes( unsigned int cNodes, const CPUNode_3D_LBT * nodes )
	{
		// Check Parameters
		if (NULL == nodes) { return false; }

		// Cleanup old list
		FiniNodes();

		// Anything in list ?!?
		if (cNodes == 0) { return true; }

		// Allocate Space for new node list
		m_nodes = new CPUNode_3D_LBT[cNodes+1];
		if (NULL == m_nodes) { return false; }

		// node at index 0 is wasted on purpose
			// so arithmetic becomes 1-based instead of 0-based
		m_nodes[0].m_searchIdx = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_nodeIdx   = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_p.x	   = 0.0f;
		m_nodes[0].m_p.y	   = 0.0f;
		m_nodes[0].m_p.z	   = 0.0f;

		// Copy nodes over
		unsigned int i;
		for (i = 0; i < cNodes; i++)
		{
			// Copy each node over (with corrected pointers)
			m_nodes[i+1].m_searchIdx = nodes[i].m_searchIdx;
			m_nodes[i+1].m_nodeIdx   = nodes[i].m_nodeIdx;
			m_nodes[i+1].m_p.x       = nodes[i].m_p.x;
			m_nodes[i+1].m_p.y       = nodes[i].m_p.y;
			m_nodes[i+1].m_p.z       = nodes[i].m_p.z;
		}

		// Success
		return true;
	}

		// Copy 
	bool Copy( const CPUTree_3D_LBT & toCopy )
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
	inline const CPUNode_3D_LBT * NODES() const { return m_nodes; }

	inline const CPUNode_3D_LBT & NODE( unsigned int idx ) const { return m_nodes[idx]; }
	inline CPUNode_3D_LBT & NODE( unsigned int idx ) { return m_nodes[idx]; }

	inline const CPUNode_3D_LBT * NODE_PTR( unsigned int idx ) const { return &(m_nodes[idx]); }
	inline CPUNode_3D_LBT * NODE_PTR( unsigned int idx ) { return &(m_nodes[idx]); }

	inline unsigned int SEARCH_ID( unsigned int idx ) const { return m_nodes[idx].SearchID(); }
	inline unsigned int NODE_ID( unsigned int idx ) const { return m_nodes[idx].NodeID(); }
	inline float NODE_X( unsigned int idx ) const { return m_nodes[idx].X(); }
	inline float NODE_Y( unsigned int idx ) const { return m_nodes[idx].Y(); }
	inline float NODE_Z( unsigned int idx ) const { return m_nodes[idx].Z(); }

	inline unsigned int NODE_PARENT( unsigned int idx ) const { return m_nodes[idx].Parent(); }
	inline unsigned int NODE_LEFT( unsigned int idx ) const   { return m_nodes[idx].Left(); }
	inline unsigned int NODE_RIGHT( unsigned int idx ) const  { return m_nodes[idx].Right(); }

	inline const CPUNode_3D_LBT * NODE_PARENT_PTR( unsigned int idx ) const 
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	inline CPUNode_3D_LBT * NODE_PARENT_PTR( unsigned int idx )
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	
	inline const CPUNode_3D_LBT * NODE_LEFT_PTR( unsigned int idx ) const 
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	inline CPUNode_3D_LBT * NODE_LEFT_PTR( unsigned int idx )
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	
	inline const CPUNode_3D_LBT * NODE_RIGHT_PTR( unsigned int idx ) const 
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	inline CPUNode_3D_LBT * NODE_RIGHT_PTR( unsigned int idx )
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	
	inline unsigned int NODE_HEIGHT( unsigned int idx ) const { return m_nodes[idx].Height(); }
	inline unsigned int NODE_AXIS( unsigned int idx ) const { return m_nodes[idx].Axis(); }

	inline bool IsRoot( unsigned int idx ) const
	{
		return ((idx == CPUNode_3D_LBT::c_Root) ? true : false);
	}

	inline bool IsLeaf( unsigned int idx ) const
	{
		// If both node.left and node.right are beyond the max size of this array
		// then this node is a leaf
			//unsigned int L = m_nodes[idx].Left();
			//unsigned int R = m_nodes[idx].Right();
		unsigned int L = idx << 1;
		unsigned int R = (idx << 1) + 1;
		return ((L > m_cNodes) && (R > m_cNodes) ? true : false);
	}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUTree_3D_LBT() :
		m_cNodes( 0 ),
		m_nodes( NULL ),
		m_startAxis( X_AXIS )
		{
		}

	CPUTree_3D_LBT( const CPUTree_3D_LBT & toCopy ) :
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

	// GetNodeAxisValue -- Helper Method
	float GetNodeAxisValue
	( 
		const CPUNode_3D_LBT * currNodes,	// IN - node list
		unsigned int index,				// IN - index of node containing 2D point
		unsigned int axis				// IN - axis of 2D point to retrieve
	) const;

	// Swap Nodes -- Helper Method
	void SwapNodes
	( 
		CPUNode_3D_LBT * currNodes,			// IN - node list
		unsigned int idx1,				// IN - index of 1st node to swap
		unsigned int idx2				// IN - index of 2nd node to swap
	);

	// Median of 3 -- Helper Method
	void MedianOf3
	(
		CPUNode_3D_LBT * currNodes,			// IN - node list
		unsigned int leftIdx,			// IN - left index
		unsigned int rightIdx,			// IN - right index
		unsigned int axis				// IN - axis to compare
	);

	bool MedianSortNodes
	(
		CPUNode_3D_LBT * nodes,		// IN - node list
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end of range
		unsigned int & median,	// IN/OUT - approximate median number
								//          actual median number
		unsigned int axis		// IN - dimension(axis) to split along (x,y,z)
	);

	bool ComputeBoundingBox
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end or range
		float        bounds[6]	// OUT - Bounding Box
	);

	bool Build3D( unsigned int cPoints, const float3 * pointList );
	bool Build3D( unsigned int cPoints, const float4 * pointList );

	bool BF_FindNN_3D
	(
		const float4 & queryLocation,	// IN  - Location to sample
		unsigned int & nearestID,		// OUT - ID of nearest point
		float & nearestDistance			// OUT - nearest distance
	);

	bool Find_QNN_3D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      nQueries,		// IN: Number of Query points
		const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_NN_3D
	(
		CPU_NN_Result * queryResults		// OUT: Results
	);

	bool Find_KNN_3D
	( 
			CPU_NN_Result * queryResults,	// OUT: Results
			unsigned int      kVal,			// In: 'k' nearest neighbors to search for
			unsigned int      nQueries,		// IN: Number of Query points
			unsigned int      nPadQueries,	// IN: Number of padded query points
			const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_KNN_3D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: Number of padded search points (query points)
	);

	void DumpNodes() const;
	bool Validate() const;

}; // end class CPUTree_3D_LBT


/*---------------------------------------------------------
  Name:	CPUTree_4D_LBT
  Desc:	Simple static balanced cyclical kd-tree 
        stored one point per node 
		in left balanced array layout
---------------------------------------------------------*/

class CPUTree_4D_LBT
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

	unsigned int  m_cNodes;		// Count of Nodes
	CPUNode_4D_LBT * m_nodes;	// List of 4D LBT kd-nodes in kd-Tree
	unsigned int  m_startAxis;	// Starting Axis
	unsigned int  m_rootIdx;	// Root Index

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

	bool CopyNodes( unsigned int cNodes, const CPUNode_4D_LBT * nodes )
	{
		// Check Parameters
		if (NULL == nodes) { return false; }

		// Cleanup old list
		FiniNodes();

		// Anything in list ?!?
		if (cNodes == 0) { return true; }

		// Allocate Space for new node list
		m_nodes = new CPUNode_4D_LBT[cNodes+1];
		if (NULL == m_nodes) { return false; }

		// node at index 0 is wasted on purpose
			// so arithmetic becomes 1-based instead of 0-based
		m_nodes[0].m_searchIdx = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_nodeIdx   = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_p.x	   = 0.0f;
		m_nodes[0].m_p.y	   = 0.0f;
		m_nodes[0].m_p.z	   = 0.0f;
		m_nodes[0].m_p.w       = 0.0f;

		// Copy nodes over
		unsigned int i;
		for (i = 1; i <= cNodes; i++)
		{
			// Copy each node over (with corrected pointers)
			m_nodes[i+1].m_searchIdx = nodes[i].m_searchIdx;
			m_nodes[i+1].m_nodeIdx   = nodes[i].m_nodeIdx;
			m_nodes[i+1].m_p.x       = nodes[i].m_p.x;
			m_nodes[i+1].m_p.y       = nodes[i].m_p.y;
			m_nodes[i+1].m_p.z       = nodes[i].m_p.z;
			m_nodes[i+1].m_p.w       = nodes[i].m_p.w;
		}

		// Success
		return true;
	}

		// Copy 
	bool Copy( const CPUTree_4D_LBT & toCopy )
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
	inline const CPUNode_4D_LBT * NODES() const { return m_nodes; }

	inline const CPUNode_4D_LBT & NODE( unsigned int idx ) const { return m_nodes[idx]; }
	inline CPUNode_4D_LBT & NODE( unsigned int idx ) { return m_nodes[idx]; }

	inline const CPUNode_4D_LBT * NODE_PTR( unsigned int idx ) const { return &(m_nodes[idx]); }
	inline CPUNode_4D_LBT * NODE_PTR( unsigned int idx ) { return &(m_nodes[idx]); }

	inline unsigned int SEARCH_ID( unsigned int idx ) const { return m_nodes[idx].SearchID(); }
	inline unsigned int NODE_ID( unsigned int idx ) const { return m_nodes[idx].NodeID(); }
	inline float NODE_X( unsigned int idx ) const { return m_nodes[idx].X(); }
	inline float NODE_Y( unsigned int idx ) const { return m_nodes[idx].Y(); }
	inline float NODE_Z( unsigned int idx ) const { return m_nodes[idx].Z(); }
	inline float NODE_W( unsigned int idx ) const { return m_nodes[idx].W(); }

	inline unsigned int NODE_PARENT( unsigned int idx ) const { return m_nodes[idx].Parent(); }
	inline unsigned int NODE_LEFT( unsigned int idx ) const   { return m_nodes[idx].Left(); }
	inline unsigned int NODE_RIGHT( unsigned int idx ) const  { return m_nodes[idx].Right(); }

	inline const CPUNode_4D_LBT * NODE_PARENT_PTR( unsigned int idx ) const 
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	inline CPUNode_4D_LBT * NODE_PARENT_PTR( unsigned int idx )
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	
	inline const CPUNode_4D_LBT * NODE_LEFT_PTR( unsigned int idx ) const 
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	inline CPUNode_4D_LBT * NODE_LEFT_PTR( unsigned int idx )
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	
	inline const CPUNode_4D_LBT * NODE_RIGHT_PTR( unsigned int idx ) const 
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	inline CPUNode_4D_LBT * NODE_RIGHT_PTR( unsigned int idx )
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	
	inline unsigned int NODE_HEIGHT( unsigned int idx ) const { return m_nodes[idx].Height(); }
	inline unsigned int NODE_AXIS( unsigned int idx ) const { return m_nodes[idx].Axis(); }

	inline bool IsRoot( unsigned int idx ) const
	{
		return ((idx == CPUNode_4D_LBT::c_Root) ? true : false);
	}

	inline bool IsLeaf( unsigned int idx ) const
	{
		// If both node.left and node.right are beyond the max size of this array
		// then this node is a leaf
			//unsigned int L = m_nodes[idx].Left();
			//unsigned int R = m_nodes[idx].Right();
		unsigned int L = idx << 1;
		unsigned int R = (idx << 1) + 1;
		return ((L > m_cNodes) && (R > m_cNodes) ? true : false);
	}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUTree_4D_LBT() :
		m_cNodes( 0 ),
		m_nodes( NULL ),
		m_startAxis( X_AXIS )
		{
		}

	CPUTree_4D_LBT( const CPUTree_4D_LBT & toCopy ) :
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

	// GetNodeAxisValue -- Helper Method
	float GetNodeAxisValue
	( 
		const CPUNode_4D_LBT * currNodes,	// IN - node list
		unsigned int index,				// IN - index of node containing 2D point
		unsigned int axis				// IN - axis of 2D point to retrieve
	) const;

	// Swap Nodes -- Helper Method
	void SwapNodes
	( 
		CPUNode_4D_LBT * currNodes,			// IN - node list
		unsigned int idx1,				// IN - index of 1st node to swap
		unsigned int idx2				// IN - index of 2nd node to swap
	);

	// Median of 3 -- Helper Method
	void MedianOf3
	(
		CPUNode_4D_LBT * currNodes,			// IN - node list
		unsigned int leftIdx,			// IN - left index
		unsigned int rightIdx,			// IN - right index
		unsigned int axis				// IN - axis to compare
	);

	bool MedianSortNodes
	(
		CPUNode_4D_LBT * nodes,		// IN - node list
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end of range
		unsigned int & median,	// IN/OUT - approximate median number
								//          actual median number
		unsigned int axis		// IN - dimension(axis) to split along (x,y,z)
	);

	bool ComputeBoundingBox
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end or range
		float        bounds[8]	// OUT - Bounding Box
	);

	bool Build4D( unsigned int cPoints, const float4 * pointList );
	//bool Build4D( const std::vector<float4> & pointList );

	bool BF_FindNN_4D
	(
		const float4 & queryLocation,	// IN  - Location to sample
		unsigned int & nearestID,		// OUT - ID of nearest point
		float & nearestDistance			// OUT - nearest distance
	);

	bool Find_QNN_4D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      nQueries,		// IN: Number of Query points
		const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_NN_4D
	(
		CPU_NN_Result * queryResults		// OUT: Results
	);

	bool Find_KNN_4D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nQueries,		// IN: Number of Query points
		unsigned int      nPadQueries,	// IN: Number of padded query points
		const float4	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_KNN_4D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: Number of padded search points (query points)
	);

	void DumpNodes() const;
	bool Validate() const;

}; // end class CPUTree_4D_LBT


/*---------------------------------------------------------
  Name:	CPUTree_6D_LBT
  Desc:	Simple static balanced cyclical kd-tree 
        stored one point per node 
		in left balanced array layout
---------------------------------------------------------*/

class CPUTree_6D_LBT
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

	unsigned int  m_cNodes;		// Count of Nodes
	CPUNode_6D_LBT * m_nodes;	// List of 6D LBT kd-nodes in kd-Tree
	unsigned int  m_startAxis;	// Starting Axis
	unsigned int  m_rootIdx;	// Root Index

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

	bool CopyNodes( unsigned int cNodes, const CPUNode_6D_LBT * nodes )
	{
		// Check Parameters
		if (NULL == nodes) { return false; }

		// Cleanup old list
		FiniNodes();

		// Anything in list ?!?
		if (cNodes == 0) { return true; }

		// Allocate Space for new node list
		m_nodes = new CPUNode_6D_LBT[cNodes+1];
		if (NULL == m_nodes) { return false; }

		// node at index 0 is wasted on purpose
			// so arithmetic becomes 1-based instead of 0-based
		m_nodes[0].m_searchIdx = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_nodeIdx   = CPUNode_2D_LBT::c_Invalid;
		m_nodes[0].m_p.x	   = 0.0f;
		m_nodes[0].m_p.y	   = 0.0f;
		m_nodes[0].m_p.z	   = 0.0f;
		m_nodes[0].m_p.w       = 0.0f;
		m_nodes[0].m_p.s	   = 0.0f;
		m_nodes[0].m_p.t       = 0.0f;

		// Copy nodes over
		unsigned int i;
		for (i = 1; i <= cNodes; i++)
		{
			// Copy each node over (with corrected pointers)
			m_nodes[i+1].m_searchIdx = nodes[i].m_searchIdx;
			m_nodes[i+1].m_nodeIdx   = nodes[i].m_nodeIdx;
			m_nodes[i+1].m_p.x       = nodes[i].m_p.x;
			m_nodes[i+1].m_p.y       = nodes[i].m_p.y;
			m_nodes[i+1].m_p.z       = nodes[i].m_p.z;
			m_nodes[i+1].m_p.w       = nodes[i].m_p.w;
			m_nodes[i+1].m_p.s       = nodes[i].m_p.s;
			m_nodes[i+1].m_p.t       = nodes[i].m_p.t;
		}

		// Success
		return true;
	}

		// Copy 
	bool Copy( const CPUTree_6D_LBT & toCopy )
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
	inline const CPUNode_6D_LBT * NODES() const { return m_nodes; }

	inline const CPUNode_6D_LBT & NODE( unsigned int idx ) const { return m_nodes[idx]; }
	inline CPUNode_6D_LBT & NODE( unsigned int idx ) { return m_nodes[idx]; }

	inline const CPUNode_6D_LBT * NODE_PTR( unsigned int idx ) const { return &(m_nodes[idx]); }
	inline CPUNode_6D_LBT * NODE_PTR( unsigned int idx ) { return &(m_nodes[idx]); }

	inline unsigned int SEARCH_ID( unsigned int idx ) const { return m_nodes[idx].SearchID(); }
	inline unsigned int NODE_ID( unsigned int idx ) const { return m_nodes[idx].NodeID(); }
	inline float NODE_X( unsigned int idx ) const { return m_nodes[idx].X(); }
	inline float NODE_Y( unsigned int idx ) const { return m_nodes[idx].Y(); }
	inline float NODE_Z( unsigned int idx ) const { return m_nodes[idx].Z(); }
	inline float NODE_W( unsigned int idx ) const { return m_nodes[idx].W(); }
	inline float NODE_S( unsigned int idx ) const { return m_nodes[idx].S(); }
	inline float NODE_T( unsigned int idx ) const { return m_nodes[idx].T(); }

	inline unsigned int NODE_PARENT( unsigned int idx ) const { return m_nodes[idx].Parent(); }
	inline unsigned int NODE_LEFT( unsigned int idx ) const   { return m_nodes[idx].Left(); }
	inline unsigned int NODE_RIGHT( unsigned int idx ) const  { return m_nodes[idx].Right(); }

	inline const CPUNode_6D_LBT * NODE_PARENT_PTR( unsigned int idx ) const 
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	inline CPUNode_6D_LBT * NODE_PARENT_PTR( unsigned int idx )
		{ 
			unsigned int parentIdx = m_nodes[idx].Parent();
			return ((parentIdx == 0u) ? NULL : &(m_nodes[parentIdx]));
		}
	
	inline const CPUNode_6D_LBT * NODE_LEFT_PTR( unsigned int idx ) const 
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	inline CPUNode_6D_LBT * NODE_LEFT_PTR( unsigned int idx )
		{ 
			unsigned int leftIdx = m_nodes[idx].Left();
			return ((leftIdx > m_cNodes) ? NULL : &(m_nodes[leftIdx]));
		}
	
	inline const CPUNode_6D_LBT * NODE_RIGHT_PTR( unsigned int idx ) const 
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	inline CPUNode_6D_LBT * NODE_RIGHT_PTR( unsigned int idx )
		{ 
			unsigned int rightIdx = m_nodes[idx].Right();
			return ((rightIdx > m_cNodes) ? NULL : &(m_nodes[rightIdx]));
		}
	
	inline unsigned int NODE_HEIGHT( unsigned int idx ) const { return m_nodes[idx].Height(); }
	inline unsigned int NODE_AXIS( unsigned int idx ) const { return m_nodes[idx].Axis(); }

	inline bool IsRoot( unsigned int idx ) const
	{
		return ((idx == CPUNode_6D_LBT::c_Root) ? true : false);
	}

	inline bool IsLeaf( unsigned int idx ) const
	{
		// If both node.left and node.right are beyond the max size of this array
		// then this node is a leaf
			//unsigned int L = m_nodes[idx].Left();
			//unsigned int R = m_nodes[idx].Right();
		unsigned int L = idx << 1;
		unsigned int R = (idx << 1) + 1;
		return ((L > m_cNodes) && (R > m_cNodes) ? true : false);
	}

	/*------------------------------------
	  Constructors
	------------------------------------*/

		// Default Constructor
	CPUTree_6D_LBT() :
		m_cNodes( 0 ),
		m_nodes( NULL ),
		m_startAxis( X_AXIS )
		{
		}

	CPUTree_6D_LBT( const CPUTree_6D_LBT & toCopy ) :
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

	// GetNodeAxisValue -- Helper Method
	float GetNodeAxisValue
	( 
		const CPUNode_6D_LBT * currNodes,	// IN - node list
		unsigned int index,				// IN - index of node containing 2D point
		unsigned int axis				// IN - axis of 2D point to retrieve
	) const;

	// Swap Nodes -- Helper Method
	void SwapNodes
	( 
		CPUNode_6D_LBT * currNodes,			// IN - node list
		unsigned int idx1,				// IN - index of 1st node to swap
		unsigned int idx2				// IN - index of 2nd node to swap
	);

	// Median of 3 -- Helper Method
	void MedianOf3
	(
		CPUNode_6D_LBT * currNodes,			// IN - node list
		unsigned int leftIdx,			// IN - left index
		unsigned int rightIdx,			// IN - right index
		unsigned int axis				// IN - axis to compare
	);

	bool MedianSortNodes
	(
		CPUNode_6D_LBT * nodes,		// IN - node list
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end of range
		unsigned int & median,	// IN/OUT - approximate median number
								//          actual median number
		unsigned int axis		// IN - dimension(axis) to split along (x,y,z)
	);

	bool ComputeBoundingBox
	(
		unsigned int start,		// IN - start of range
		unsigned int end,		// IN - end or range
		float        bounds[8]	// OUT - Bounding Box
	);

	bool Build6D( unsigned int cPoints, const CPU_Point6D * pointList );
	//bool Build6D( const std::vector<float4> & pointList );

	bool BF_FindNN_6D
	(
		const CPU_Point6D & queryLocation,	// IN  - Location to sample
		unsigned int & nearestID,		// OUT - ID of nearest point
		float & nearestDistance			// OUT - nearest distance
	);

	bool Find_QNN_6D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      nQueries,		// IN: Number of Query points
		const CPU_Point6D * queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_NN_6D
	(
		CPU_NN_Result * queryResults		// OUT: Results
	);

	bool Find_KNN_6D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nQueries,		// IN: Number of Query points
		unsigned int      nPadQueries,	// IN: Number of padded query points
		const CPU_Point6D	* queryPoints	// IN: query points to compute distance for (1D or 2D field)
	);

	bool Find_ALL_KNN_6D
	( 
		CPU_NN_Result * queryResults,	// OUT: Results
		unsigned int      kVal,			// In: 'k' nearest neighbors to search for
		unsigned int      nSearch,		// IN: Number of search points (query points)
		unsigned int      nPadSearch	// In: Number of padded search points (query points)
	);

	void DumpNodes() const;
	bool Validate() const;

}; // end class CPUTree_6D_LBT

#endif // _CPUTREE_LBT_H



