#pragma once

#include "Particle.h"

Particle::Particle()
{
	lifespan = (((rand()%10+1)))/10.0f;
	type = 0;

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

glm::vec3 Particle::getRGB()
{
	return color;
}