#include "Solver.h"

Solver::Solver()
{
	srand(time(NULL));

	width = 5;
	height = 5;
	cells = new Cell[width * height];

	//Create a grid of cells
	for (int i = 0; i < width * height; i++)
	{
		cells[i].xPos = i * 10;
		cells[i].yPos = i * 10;
		cells[i].grid = this;
	}
	
	logFile.open("log.txt", std::ios::out);
	////Create a bunch of particles.
	//for (int i = 0; i < NUM_PARTICLES; i++)
	//{
	//	master.push_back( fluidParticle((rand() % width), (rand() % height)));
	//}
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

	/*
	for (int i = 0; i < NUM_PARTICLES; i++)
	{
		master.push_back(fluidParticle((rand() % width), (rand() % height)));
	}*/
}

Solver::~Solver()
{
	logFile.close();
	delete [] cells;
}

void Solver::solve(float eX, float eY)
{

	for (int i = 0; i < width * height; i++)
	{
		cells[i].solve(eX, eY);
	}
}

Cell Solver::operator[](int index)
{
	return cells[index];
}

void Solver::print()
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			std::cout << cells[i + j * width].getMagnitude() << " \t";
		}
		std::cout << std::endl << std::endl;
	}
}

void Solver::log()
{
	logFile << cells[0].getMagnitude() << std::endl;
}