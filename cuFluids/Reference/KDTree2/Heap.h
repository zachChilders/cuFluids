#pragma once
#ifndef _HEAP_H
#define _HEAP_H
/*-----------------------------------------------------------------------------
  Name:	Heap.h
  Desc:	Defines Heap template

  Copyright:
    Copyright (C) by Shawn Brown - 2006

	This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

  History:	
    6/30/2006 - Created (Shawn Brown)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

//  Library Includes
#ifndef _TMV_BASEDEFS_H
	#include "BaseDefs.h"
#endif

/*-------------------------------------
  Classes / Templates
-------------------------------------*/

/*-----------------------------------------------------------------------------
  Template:	Binary Heap
  Desc:  stores objects in a priority heap layout
  Notes:
	a binary heap is an set of elements with the keys arranged in a
	complete heap-ordered binary tree, represented as an array.

Heap Definitions:
	Complete binary tree --  all levels except for possibly the last
	    are completely full.  The last level can be partially full but
	    is filled in sequentially from left to right with empty nodes
		until the last rightmost node has been reached.

		        Full           ..... A .....
						      /             \
				Full	     B               C 
					       /    \         /    \
				Full	  D      E       F      G
						 / \    / \     / \    / \
		Partially Full  H   I  J   K   L   .  .   .

    maxLength[A]  -- maximum size of array
	heapLength[A] -- size of heap stored in array

	Note:  heapLength[A] <= maxLength[A]

	Heap-Ordered -- a tree is Heap-Ordered if it is either Heap-Max-Ordered
	               or Heap-Min-ordered (see below)
	
	Heap-Max-Ordered:  a tree (or array) is max-ordered if the key
	in each node is larger than or equal to the keys in all of that node's
	children (if any).  No node in a heap-max-ordered tree has a key larger 
	than the key at the root.

	Heap-Min-Ordered:  a tree (or array) is min-ordered if the key in each
	node is smaller than or equal to the keys in all of that node's 
	children (if any).  No node in a heap-min-ordered tree has a key smaller 
	than the key at the root.

	Root[A] = A[1]		   -- root of heap
	parent[i] = floor[i/2] -- parent of node at index i
	left[i] = 2*i          -- left child of node at index i
	right[i] = 2*i+1       -- right child of node at index i
    
    MAX-HEAPIFY( A, i )
	  left = LEFT(i)
	  right = RIGHT(i)
	  if left <= heap-size[A] and A[left] > A[i]
	    then largest = left
		else largest = i
	  if right <= heap-size[A] and A[right] > A[largest]
	    then largest = right
	  if largest != i
	    then exchange A[i] and A[largest]
		  MAX-HEAPIFY( A, largest )

    MIN-HEAPIFY( A, i )
	  left = LEFT(i)
	  right = RIGHT(i)
	  if left <= heap-size[A] and A[left] < A[i]
	     then smallest = left
		 else smallest = i
      if right <= heap-size[A] and A[right] < A[smallest]
	     then smallest = right
	  if smallest != i
	     then exchange A[i] and A[smallest]
		    MIN-HEAPIFY( A, smallest )

	BUILD-MAX-HEAP( A )
	  heap-size[A] = curr-length[A]
      for i = floor( curr-length[A]/2 ) downto 1
	    do MAX-HEAPIFY( A, i )

	BUILD-MIN-HEAP( A )
	  heap-size[A] = curr-length[A]
	  for i = floor( curr-length[A]/2 ) downto 1
	    do MIN-HEAPIFY( A, i )

	HEAP-MAXIMUM( A )
	  return A[1]

    HEAP-MINIMUM( A )
	  return A[1]

    HEAP-EXTRACT-MAX( A )
	  if heap-size[A] < 1
	    then error "heap underflow"
	  max = A[1]
	  heap-size[A] = heap-size[A] - 1
	  MAX-HEAPIFY( A, 1 )
	  return max
   
    HEAP-EXTRACT-MIN( A )
	  if heap_size[A] < 1
	    then error "heap underflow"
	  min = A[1]
	  heap-size[A] = heap-size[A] - 1
	  MIN-HEAPIFY( A, 1 )
	  return min

    HEAP-INCREASE-KEY( A, i, key )
	  if key < A[i]
	    then error "new key is smaller than current key"
	  A[i] = key
	  while i > 1 and A[Parent(i)] < A[i]
	    do exchange A[i] and A[Parent(i)]
		  i = Parent(i)

    HEAP-DECREASE-KEY( A, i, key )
	   if key > A[i]
	     then error "new key is larger than current key"
	   A[i] = key
	   while i > 1 and A[Parent(i)] > A[i]
	     do exchange A[i] and A[Parent(i)]
		   i = Parent(i)

   MAX-HEAP-INSERT( A, key )
     heap-size[A] = heap-size[A] + 1
	 A[heap-size[A]] = -infinity
	 HEAP-INCREASE-KEY( A, heap-size[A], key )

   MIN-HEAP-INSERT( A, key )
     heap-size[A] = heap-size[A] + 1
	 A[heap-size[A]] = +infinity
	 HEAP-DECREASE-KEY( A, heap-size[A], key )
-----------------------------------------------------------------------------*/

template <typename T>
class BinaryHeap
{
public:
	/*-----------------------
	  TypeDefs
	-----------------------*/

protected:
	/*-----------------------
	  Fields
	-----------------------*/

	I32 m_maxSize;		// Max Size of Heap Array
	I32 m_currSize;		// Curr Size of Heap Array
	T * m_zeroHeap;		// Actual Heap Array (zero based indexing)
	T * m_oneHeap;		// one based indexing (one based indexing0
	bool m_minMax;		// true = maxHeap, false = minHeap

	/*-----------------------
	  Helper Methods
	-----------------------*/

		// Initialize fields to known good values
	void Init()
		{
			m_maxSize  = 0;
			m_currSize = 0;
			m_zeroHeap = NULL;
			m_oneHeap  = NULL;
			m_minMax   = true;
		}

		// Cleanup resource usage
	void Fini()
		{
			if (NULL != m_zeroHeap)
			{
				T * tempPtr = m_zeroHeap;
				m_zeroHeap = NULL;
				delete [] tempPtr;
			}
			m_oneHeap  = NULL;
			m_maxSize  = 0;
			m_currSize = 0;
		}

	bool Copy( const BinaryHeap<T> & toCopy )
		{
			if (this == &toCopy) { return true; }

			Fini();

			I32 newSize = toCopy.m_currSize;
			if (newSize > 0)
			{
				// Allocate space for heap elements
				T * newHeap = new T[newSize];
				if (NULL != newHeap)
				{
					return false;
				}

				// Copy heap elements over
				I64 heapIdx;
				const T * copyHeap = toCopy.m_zeroHeap;
				for (heapIdx = 0; heapIdx < newSize; heapIdx++)
				{
					newHeap[heapIdx] = copyHeap[heapIdx];
				}

				m_maxSize  = nElems;
				m_currSize = nElems;
				m_zeroHeap = newHeap;
				m_oneHeap  = newHeap - 1;
			}

			m_minMax = toCopy.m_minMax;

			return true;
		}

		// Exchange 2 elements in heap 
		// Note:  Assumes one based indexing 
	inline void Exchange( I32 i, I32 j )
		{
			T tempVal    = m_oneHeap[i];
			m_oneHeap[i] = m_oneHeap[j];
			m_oneHeap[j] = tempVal;
		}

		// Have we run out of space
	inline bool NeedsToGrow( I32 newSize ) { return ((newSize > m_maxSize) ? true : false); }

		// Grow heap to specified size
	bool Grow( I32 newSize )
		{
			// Check Parameters
			if (newSize <= 0)
			{
				// Error - Invalid parameter
				return false;
			}
			if ((newSize < m_maxSize) && (NULL != m_zeroHeap))
			{
				// Already big enough, do nothing
				return true;
			}

			// Allocate memory for new heap
			const T * oldHeap  = m_zeroHeap;
			I32 oldSize = m_currSize;
			T * newHeap = new T[newSize];
			if (NULL == newHeap) 
			{ 
				// Error - out of memory
				return false; 
			}

			// Copy over old elements to new heap
			if ((oldHeap) && (oldSize > 0))
			{
				// Copy valid elements over
				for (heapIdx = 0; heapIdx < oldSize; heapIdx++)
				{
					newHeap[heapIdx] = oldHeap[heapIdx];
				}
			}

			// Cleanup old heap memory
			if (NULL != oldHeap)
			{
				delete [] oldHeap;
			}

			m_zeroHeap = newHeap;
			m_oneHeap  = newHeap - 1;
			m_maxSize  = newSize;
			//m_currSize = oldSize;

			return true;
		}

		// Shrink heap to current usage
	bool Shrink() 
		{
			I32 newSize = m_currSize;
			I32 oldSize = m_maxSize;
			if (newSize < oldSize)
			{
				// Allocate memory for new heap
				T * oldHeap = m_zeroHeap;
				T * newHeap = NULL;
				if (newSize > 0)
				{
					newHeap = new T[newSize];
					if (NULL == newHeap) 
					{ 
						// Error  - out of memory
						return false; 
					}
				}

				// Copy over old elements to new heap
				if ((oldList) && (newSize > 0))
				{
					for (heapIdx = 0; heapIdx <= oldSize; heapIdx++)
					{
						newHeap[heapIdx] = oldHeap[heapIdx];
					}
				}

				// Cleanup old heap memory
				if (NULL != oldHeap)
				{
					delete [] oldHeap;
				}

				m_zeroHeap = newHeap;
				m_oneHeap  = newHeap - 1;
				m_maxSize  = newSize;
				//m_currSize = oldSize;
			}

			return true;
		}

	void MakeMaxHeap() 
	{
		I32 N = m_currNodes;
		I32 k;
		for (k = N/2; k >= 1; k--)
		{
			DemoteMax( k );
		}
	}

	void MakeMinHeap() 
	{
		I32 N = m_currNodes;
		I32 k;
		for (k = N/2; k >= 1; k--)
		{
			DemoteMin( k );
		}
	}

		// Root (Root = A[1])
	inline I32 RootIndex() const { return 1; }

		// Parent (Parent = A[ floor[i/2] ]
	inline I32 ParentIndex( I32 idx ) const { return idx >> 1; }

		// Left Child (Left = A[2*i])
	inline I32 LeftIndex( I32 idx ) { return 2*idx; }

		// Right Child (Right = A[2*i + 1]
	inline I32 RightIndex( I32 idx ) { return ((2*idx)+1); }

		// Get/Set Element at index
			// Assumes one based indexing
	inline T Element( I32 idx ) const { return m_oneHeap[idx]; }
	inline void Element( I32 idx, T newVal ) { m_oneHeap[idx] = newVal; }

		// Get Pointer to Element at specified index
			// Assumes one based indexing
	inline const T * ElemPtr( I32 idx ) const { return (const T *)&(m_oneHeap[idx]); }
	inline T * ElemPtr( I32 idx ) const { return (T *)&(m_oneHeap[idx]); }

	  // Promotes value at current index up parent chain
			// Assumes one based indexing
			// Uses extended swap for faster performance
	void PromoteMax( I32 idx )
	{
		I32 currIdx   = idx;
		I32 parentIdx = currIdx >> 1;

		// Save original value creating first hole for extended swap
		T   origVal   = m_oneHeap[currIdx];
		T   currVal   = origVal;

		// Compare currIndex with it's parent
		while ((currIdx > 1) && (parentVal < currVal))
		{
			// Move parent value into current hole
				// continuing extended swap
			m_oneHeap[currIdx] = m_oneHeap[parentIdx];

			// Update indices of currIndex and it's parent
			currIdx   = parentIdx;
			parentIdx = currIdx >> 1;	// Parent = Curr /2;
		}

		// Put original value back into final hole (ending extended swap)
		m_oneHeap[currIdx] = origVal;
	}

		// Promotes value at current index up parent chain
	void PromoteMin( I32 idx )
	{
		I32 currIdx   = idx;
		I32 parentIdx = currIdx >> 1;

		// Save original value creating first hole for extended swap
		T   origVal   = m_oneHeap[currIdx];
		T   currVal   = origVal;

		// Compare currIndex with it's parent
		while ((currIdx > 1) && (parentVal > currVal))
		{
			// Fill hole with parent value
				// continuing extended swap
			m_oneHeap[currIdx] = m_oneHeap[parentIdx];

			// Update indices of currIndex and it's parent
			currIdx   = parentIdx;
			parentIdx = currIdx >> 1;	// Parent = Curr /2;
		}

		// Put original value back into final hole (ending extended swap)
		m_oneHeap[currIdx] = origVal;
	}

		// Demotes value at curr index down child chain
	void DemoteMax( I32 idx ) 
	{
		I32 currIdx  = idx;
		I32 childIdx = idx << 1;	// left child of current index
		I32 rightIdx;

		// Save original value creating first hole for extended swap
		T   origVal   = m_oneHeap[currIdx];
		T   currVal   = origVal;
		T   childVal, rightVal;

		// Compare current index to it's children
		while (childIdx <= m_currNodes)
		{
			// Find largest child 
			childVal = m_oneHeap[childIdx];		// Left child
			rightIdx = childIdx+1;
			if (rightIdx < m_currNodes)
			{
				rightVal = m_oneHeap[rightIdx];	// Right child
				if (childVal < rightVal)
				{
					// Right child is largest
					childIdx = rightIdx;
					childVal = rightVal;
				}
			}

			// Compare largest child priority to current priority
			if (currVal >= childVal) 
			{
				// Current is larger than both children, exit loop
				break;
			}
			
			// Fill hole with child value
				// continuing extended swap
			m_oneHeap[currIdx] = m_oneHeap[childIdx];

			// Update indices
			currIndex  = childIndex;	
			childIndex = currIdx << 1; 
		}

		// Put original value back into final hole (ending extended swap)
		m_oneHeap[currIdx] = origVal;
	}

		// Demotes value at curr index down child chain
	void DemoteMin( I32 currIndex ) 
	{
		I32 currIdx  = idx;
		I32 childIdx = idx << 1;	// left child of current index
		I32 rightIdx;

		// Save original value creating first hole for extended swap
		T   origVal   = m_oneHeap[currIdx];
		T   currVal   = origVal;
		T   childVal, rightVal;

		// Compare current index to it's children
		while (childIdx <= m_currNodes)
		{
			// Find smallest child 
			childVal = m_oneHeap[childIdx];		// Left child
			rightIdx = childIdx+1;
			if (rightIdx < m_currNodes)
			{
				rightVal = m_oneHeap[rightIdx];	// Right child
				if (childVal > rightVal)
				{
					// Right child is smallest
					childIdx = rightIdx;
					childVal = rightVal;
				}
			}

			// Compare smallest child priority to current priority
			if (currVal <= childVal) 
			{
				// Current is smaller than both children, exit loop
				break;
			}
			
			// Fill hole with child value
				// continuing extended swap
			m_oneHeap[currIdx] = m_oneHeap[childIdx];

			// Update indices
			currIndex  = childIndex;	
			childIndex = currIdx << 1; 
		}

		// Put original value back into final hole (ending extended swap)
		m_oneHeap[currIdx] = origVal;
	}

		// removes element at specified index
	void Remove( I64 idx )
		{
			// Move current element to end of array
			Exchange( idx, m_currSize );
			if (m_minMax)
			{
				DemoteMax( idx );
			}
			else
			{
				DemoteMin( idx );
			}

			// Shrink heapsize by 1
			m_currSize--;
		}

public:
	/*-----------------------
	  Properties
	-----------------------*/

	inline bool isMaxHeap() { return ((m_minMax == true) ? true : false); }
	inline bool isMinHeap() { return ((m_minMax == false) ? true : false); }

	inline bool SetMaxHeap() 
		{
			bool oldMM = m_minMax;
			m_minMax = true;
			if (oldMM != true)
			{
				return MakeMaxHeap();
			}
		}

	inline bool SetMinHeap() 
		{
			bool oldMM = m_minMax;
			m_minMax = true;
			if (oldMM != true)
			{
				return MakeMinHeap();
			}
		}


	// MAX-HEAP-SIZE
		// Max number of elements allowed in heap
	inline I64 MaxHeapSize() const { return m_maxSize; }
	inline bool MaxHeapSize( I64 value ) { return Grow( value ); }

	// CURR-HEAP-SIZE
		// Current number of elements in the heap
	inline I64 CurrHeapSize() const { return m_currSize; }
	inline bool CurrHeapSize( I64 value ) 
		{ 
			I32 oldMax  = m_maxSize;
			I32 oldSize = m_currSize;

			if (value > oldMax)
			{
				Grow( value );
			}

			if (value > oldSize)
			{
				I32 heapIdx;

				// Heapify all new elements to correct positions
				if (m_minMax)
				{
					for (heapIdx = oldSize; heapIdx <= value; heapIdx++)
					{
						PromoteMax( heapIdx );
					}
				}
				else
				{
					for (heapIdx = oldSize; heapIdx <= value; heapIdx++)
					{
						PromoteMin( heapIdx );
					}
				}
			}
		}

		// Get Element at Idx (A[i])
			// Assumes one based indexing
	inline const T * ElementAt( I64 idx ) const 
		{ 
			return (const T *)&(m_oneHeap[idx]); 
		}
	inline T * ElementAt( I64 idx ) 
		{ 
			return (T *)&(m_oneHeap[idx]); 
		}

		// is heap empty
	inline bool empty() const { return ((m_currSize == 0) ? true : false); }

		// is heap full
	inline bool full() const { return ((m_currSize == m_maxSize) ? true : false ); }


	/*-----------------------
	  Constructors
	-----------------------*/

	BinaryHeap()
		{
			Init();
		}

	BinaryHeap( I32 maxElems = 0, bool minMax = true )
		{
			Init();

			m_minMax = minMax;
			if (maxElems > 0)
			{
				m_zeroHeap = new T[maxElems];
				if (NULL == m_zeroHeap)
				{
					// Error - out of memory in constructor
				}
			}
		}

	BinaryHeap( I32 nElems, const T * elements, bool minMax = true )
		{
			Init();

			// Check Parameters
			if ((nElems <= 0) || (NULL == elements))
			{
				// Error - invalid parameters in constructor
				return;
			}

			// Allocate memory for constructors
			T * newHeap = new T[maxElems];
			if (NULL == newHeap)
			{
				// Error - out of memory in constructor
			}

			// Copy elements into heap
			for (idx = 0; idx < nElems; idx++)
			{
				newHeap[idx] = elements[idx];
			}

			// Update member fields
			m_zeroHeap = newHeap;
			m_oneHeap  = newHeap - 1;
			m_currSize = maxElems;
			m_maxSize  = maxElems;
			m_minMax   = minMax;

			// Heapify new elements
			if (m_minMax)
			{
				MakeMaxHeap();
			}
			else
			{
				MakeMinHeap();
			}
		}

	BinaryHeap( const BinaryHeap<T> & toCopy )
		{
			Init();
			Copy( toCopy );
		}

	~BinaryHeap()
		{
			Fini();
		}


	/*------------------------------------
	  Operators
	------------------------------------*/

		// Copy Operator
	BinaryHeap<T> & operator = ( const BinaryHeap<T> & toCopy ) 
		{
			if (this != &toCopy)
			{
				Copy( toCopy );
			}
			return (*this);
		}


	/*-----------------------
	  Methods
	-----------------------*/

	inline void Clear() { Fini(); }


		// replace top element on heap with new element
	void Replace( const T & newElem )
	{
		// Replace Root Element with new element
		m_heap[1] = newElem;

		// Demote new element to correct location in heap
		if (m_minMax)
		{
			DemoteMax( 1 );
		}
		else
		{
			DemoteMin( 1 );
		}
	}

		// Insert new element into heap
	bool Insert( const T & newElem ) 
		{
			// Grow Heap, if we need to
			if (NeedsToGrow())
			{
				I32 newSize = ((m_maxSize == 0) ? 32 : (2*m_maxSize)); 
				bResult = Grow( newSize );
				if (! bResult)
				{
					return false;
				}
			}

			// Add Element to end of heap
			m_currSize += 1;
			m_oneHeap[m_currSize] = newElem;

			// Promote element to proper position in heap
			if (m_minMax)
			{
				PromoteMax( m_currSize );
			}
			else
			{
				PromoteMin( m_currSize );
			}

			return true;
		}

		// returns element at root of heap
	T   RootElem() const { return m_heap[1]; }
	T & RootElem() { return m_heap[1]; }

	// removes (and returns) element at root of heap
	T RemoveRoot()
		{
			// Get Element at root
			T elem = m_heap[1];

			// Move root element to end of array
			Exchange( 1, m_currSize );
			if (m_minMax)
			{
				DemoteMax( 1 );
			}
			else
			{
				DemoteMin( 1 );
			}

			// Shrink heapsize by 1
			m_currSize--;

			// Return original element
			return elem;
		}

	T RemoveAt( I32 idx )
		{
			T elem = m_oneHeap[idx];
			Remove( idx );
			return elem;
		}

	bool ChangeElem( I32 idx, const T & newElem )
		{
			T oldElem = m_oneHeap[idx];
			m_oneHeap[idx] = newElem;

			if (m_minMax)
			{
				if (newElem > oldElem)
				{
					PromoteMax( idx );
				}
				else if (newElem < oldElem)
				{
					DemoteMax( idx );
				}
			}
			else
			{
				if (newElem > oldElem)
				{
					DemoteMin( idx );
				}
				else if (newElem < oldElem)
				{
					PromoteMin( idx );
				}
			}

			return true;
		}

	bool Heapify()
	{
		if (m_minMax)
		{
			MakeMaxHeap();
		}
		else
		{
			MakeMinHeap();
		}
	}

}; // end binary heap


#endif // _HEAP_H

