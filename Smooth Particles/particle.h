/*****************************************************
Particle Class

Zach Childers

Summer 2014

/*****************************************************/

#pragma once

#include <glm/glm.hpp>

#define MAX_PARTICLES 10000

class Particle
{
private:
	//These will tie into shaders
	glm::vec4 color; // color values
	glm::vec3 position; // initial onscreen position
public:
	//These are going to be tweaked
	float lifespan, age, scale, direction; // how long the particle will exist for, alpha blending variable; how old is it.
	glm::vec3 movement; // movement vector
	glm::vec3 pull; // compounding directional pull in the x,y,z directions

	//Finalized
	//Constructors
	Particle();
	~Particle();

	//Accessors
	glm::vec4 getRGBA();
	glm::vec3 getPosition();
	void setRGBA(float r, float g, float b, float a);
	void setPos(float x, float y, float z);

	//For Sorting
	bool operator< (Particle *other);
	
}; 