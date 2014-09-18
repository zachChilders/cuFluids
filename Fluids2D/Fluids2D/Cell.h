#include "particle.h"

class Cell
{
public:
	Cell();
	~Cell();
	void solve();

private:
	fluidParticle *particles;
	float xVel, yVel;
	void getParticles();

	void pressure();
	void external();
	void viscosity();
	void gravity();
	void weight();
};