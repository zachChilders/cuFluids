#pragma once
#include <cmath>
class fluidParticle
{
	friend class Cell;
	private:
		float xPos, yPos, zPos;
		int xVelocity, yVelocity, zVelocity;

	public:
		fluidParticle();
		fluidParticle(float x, float y);
		fluidParticle(float x, float y, float z);
		~fluidParticle();
		void reset();

		//These should be applied in order functionally
		void update(float xVelocity, float yVelocity, float zVelocity);
		
		float* distances;

		void getPositions(float *x, float *y, float *z);
		float getMagnitude();

		//Overload * for matrix multiplication.
};