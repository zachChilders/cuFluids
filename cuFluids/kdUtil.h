#pragma once

typedef enum DIMENSION{ X, Y, Z };

class vec3
{
	public:
		float x;
		float y;
		float z;

		vec3()
		{
			x = 0; y = 0; z = 0;
		}

		vec3(float x, float y, float z)
		{
			x = x; y = y; z = z;
		}

		void operator+(vec3 *b)
		{
			x += b->x;
			y += b->y;
			z += b->z;
		}

		void operator+=(vec3 b)
		{
			x += b.x;
			y += b.y;
			z += b.z;
		}


};
