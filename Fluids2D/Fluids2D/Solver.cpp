#include "Solver.h"

Solver::Solver()
{
	srand(time(NULL));

	width = 4;
	height = 4;
	cells = new Cell[width * height];

	//Create a grid of cells
	for (int i = 0; i < width * height; i++)
	{
		cells[i].xPos = i * 10;
		cells[i].yPos = i * 10;
	}

	//Create a bunch of particles.
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		master.push_back(fluidParticle((rand() % width), (rand() % height)));
	}
}

Solver::Solver(int xLength, int yLength)
{
	srand(time(NULL));

	width = xLength;
	height = yLength;
	cells = new Cell[width * height];
	
	//Create a grid of cells
	for (int i = 0; i < width * height; i++)
	{
		cells[i].xPos = i * 10;
		cells[i].yPos = i * 10;
	}

	//Make an oodle of particles.
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		master.push_back(fluidParticle((rand() % width), (rand() % height)));
	}
}

Solver::~Solver()
{
	delete [] cells;
}

void Solver::solve()
{

	for (int i = 0; i < width * height; i++)
	{
		cells[i].solve();
	}
}

Cell Solver::operator[](int index)
{
	return cells[index];
}

void Solver::reweight()
{
	float maxParticles = 0;
	//Find the maximum 
	for (int i = 0; i < width * height; i++)
	{
		if (cells[i].particles.size() > maxParticles)
		{
			maxParticles = cells[i].particles.size();
		}
	}

	for (int i = 0; i < width * height; i++)
	{

	}
}