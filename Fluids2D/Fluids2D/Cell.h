#include "particle.h"

class Cell
{
public:
	Cell();
	~Cell();
	void solve();

private:
	float totalPressure, xPressure, yPressure;
	float totalViscosity, xViscosity, yViscosity;

	float weight;

	fluidParticle *particles;
	float xVel, yVel;
	void getParticles();

	void pressure();
	void external();
	void viscosity();
	void gravity();
	void weight();
};