#include  "stdafx.h"
#include  "particle.h"

//Default
fluidParticle::
	fluidParticle()
	{
		xPos = 0;
		yPos = 0;
		zPos = 0;

		//Allocate our distance vector
		// = to the number of particles
		distances = new float[16 * 16];
	}

//2 Dimensional
fluidParticle::
	fluidParticle(float x, float y)
	{
		xPos = x;
		yPos = y;
		distances = new float[16 * 16];
	}

//3 Dimensional
fluidParticle::
	fluidParticle(float x, float y, float z)
	{
		xPos = x;
		yPos = y;
		zPos = z;
	}

//Nothing to destruct yet.
fluidParticle::
	~fluidParticle()
{
	delete distances;
};

//Resets assuming a square surface.
void fluidParticle::
	reset()
	{
		xPos = -xPos;
		yPos = -yPos;
		zPos = -zPos;
	}

//Navier-Stokes Here

//Weight kernel
void fluidParticle::
	weight()
{

};

void fluidParticle::
	pressure()
{

};

void fluidParticle::
	viscosity()
{

};

void fluidParticle::
	external()
{

};

void fluidParticle::
	gravity()
{
	// (0, 0, -g);
};

//Update the next velocity
void fluidParticle::
	update(fluidParticle up, fluidParticle down, fluidParticle left, fluidParticle right)
	{
		//Navier-Stokes Code
		xPos += xVelocity;
		yPos += yVelocity;
		zPos += zVelocity;
	}

//Get fluid positions
void fluidParticle::
	getPositions(float *x, float *y, float *z)
	{
		*x = xPos;
		*y = yPos;
		*z = zPos;
	}

//Find a magnitude.
float fluidParticle::
	getMagnitude()
	{
		if( !(xVelocity) || !(yVelocity) || !(zVelocity)) {
		
			return (xVelocity * yVelocity * zVelocity) /  (xVelocity + yVelocity + zVelocity);
		}
		return 0;
	}