#pragma once
#include "Cell.h"
#include <time.h>

#define NUM_PARTICLES 1000

class Solver
{

	public:
		Solver();
		Solver(int xLength, int yLength);
		~Solver();
		void solve();
		Cell operator[](int index);
		std::vector<fluidParticle> master;
		void reweight(); //Weights everything properly

	private:
		int width, height;
		Cell *cells;
};
