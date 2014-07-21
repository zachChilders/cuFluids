#include  stdafx.h 
#include  particle.h 

//Default
fluidParticle::
	fluidParticle()
	{
		xPos = 0;
		yPos = 0;
		zPos = 0;
	}

//2 Dimensional
fluidParticle::
	fluidParticle(float x, float y, float viscosity)
	{
		xPos = x;
		yPos = y;
		viscosity = viscosity;
	}

//3 Dimensional
fluidParticle::
	fluidParticle(float x, float y, float z, float viscosity)
	{
		xPos = x;
		yPos = y;
		zPos = z;
		viscosity = viscosity;
	}

//Nothing to destruct yet.
fluidParticle::
	~fluidParticle(){};

//Resets assuming a square surface.
void fluidParticle::
	reset()
	{
		xPos = -xPos;
		yPos = -yPos;
		zPos = -zPos;
	}

//Navier-Stokes Here
void fluidParticle::
	advection(fluidParticle up, fluidParticle down, fluidParticle left, fluidParticle right)
{
	

};

void fluidParticle::
	pressure()
{
};

void fluidParticle::
	external()
{
};

//Update the next velocity
void fluidParticle::
	update(fluidParticle up, fluidParticle down, fluidParticle left, fluidParticle right)
	{
		//Navier-Stokes Code
		advection(up, down, left, right);
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