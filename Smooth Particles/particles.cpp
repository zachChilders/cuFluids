/*****************************************************
Particle Class

Zach Childers

Summer 2014

/*****************************************************/

#include "Particle.h"

Particle::Particle()
{
	lifespan = (((rand()%10+1)))/10.0f;

	age = 0.0f;
	scale = 0.25f;
	direction = 0;   
	position.x = ((rand()%2)-(rand()%2));

	position.y = -30;

	position.z = 0; 

	movement.x = (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0035) - (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0035);
	movement.y = ((((((5) * rand()%11) + 3)) * rand()%11) + 7) * 0.015; 
	movement.z = (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0015) - (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0015);
   
	color.x = 0.0f;
	color.y = 0.0f;
	color.z = 1.0f;
	
	pull.x = 0.0f;
	pull.y = 0.0f;
	pull.z = 0.0f;

	

}

Particle::~Particle()
{
};

glm::vec4 Particle::getRGBA()
{
	return color;
}

glm::vec3 Particle::getPosition()
{
	return position;
}

void Particle::setRGBA(float r, float g, float b, float a)
{
	color = glm::vec4(r, g, b, a);

};

void Particle::setPos(float x, float y, float z)
{
	position = glm::vec3(x, y, z);
};

bool Particle::operator<(Particle *other)
{
	return (position.z < other->getPosition().z);

};