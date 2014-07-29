/*****************************************************
Particle Emitter

Zach Childers

Summer 2014

This code is based off of code written by Georg Albrecht of UC Santa Cruz.

/*****************************************************/


#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <vector>

#include "Particle.h"

class System
{
	int systemType; //1 = smoke, 2 = fountain, 3 = fire, 4 = fire with smoke
	glm::vec3 systemPull; //used to store global compounding system pull in x,y,z
	//Particle particles[MAX_PARTICLES]; //initalizes and array of type Particle
	std::vector<Particle> particles;
public:
	System();
	~System();
	
	void createParticles(); //calls createParticle() to initalize all particles in system
	void updateParticles(); //updates particles according to forces being used

	int getNumOfParticles(); // returns the number of particles in the system (legacy)
	
	glm::vec4 getRGBA(int i);
	glm::vec3 getPosition(int i);
	float getScale(int i); //returns scale of particle
	float getDirection(int i); //returns direction of particle for texture rotation
	void modifySystemPull(float x, float y, float z); //used to modify x,y,z pull magnitudes
};