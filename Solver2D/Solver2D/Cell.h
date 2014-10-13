#pragma once

#include "particle.h"
#include "Solver.h"

#include <vector>

class Cell
{
	friend class Solver;

	public:
		static Solver* grid; //This needs to be set in main
		Cell();
		~Cell();
		void solve();
		void setWeight();
		float getMagnitude();

	private:
		int xPos, yPos;

		float totalPressure, xPressure, yPressure;
		float totalViscosity, xViscosity, yViscosity;
		float externalForce;
		
		glm::fvec2 weight;
		std::vector<fluidParticle> particles;
		float xVel, yVel;
		void getParticles();

		void pressure();
		void external(float eX, float eY);
		void viscosity();
		void gravity();
};