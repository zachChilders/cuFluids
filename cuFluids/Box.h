#pragma once
#include "kdUtil.h"
#include "Point.h"

template
<typename T>
class Box
{

	protected:
		T min, max;

	public:
		//Constructors
		Box();
		~Box();

		//Methods

		//Centers
		const float centerX();
		const float centerY();
		const float centerZ();
		vec3 center();

		//Midpoints
		float halfWidthX();
		float halfWidthY();
		float halfWidthZ();
		vec3 halfWidth();

		//Min/Max
		T minX() const;
		T maxX() const;
		T minY() const;
		T maxY() const;
		T minZ() const;
		T maxZ() const;

		//Corner functions
		//Splitting functions
		//Set box
};