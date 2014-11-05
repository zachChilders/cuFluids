#pragma once
#include "Cell.h"
#include <iostream>
#include <fstream>
#include <time.h>

#include <vector>

#define NUM_PARTICLES 1000

class Solver
{
	public:
		Solver();
		Solver(int xLength, int yLength);
		~Solver();
		void solve(float eX, float eY);
		Cell operator[](int index);
		std::vector<fluidParticle> master;
		void reweight(); //Weights everything properly
		void print();
		void log();

	private:
		int width, height;
		Cell *cells;
		std::ofstream logFile;
};
