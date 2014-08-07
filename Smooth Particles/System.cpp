/*****************************************************
Particle Emitter

Zach Childers

Summer 2014

A point that a particle system will emit from. It controls 
the aspects of the particles. 

This code is based off of code written by Georg Albrecht of UC Santa Cruz.


/*****************************************************/

#include "System.h"

System::System()
{
//	//Particle Pool
//	for (int i = 0; i < MAX_PARTICLES; i++)
//	{
//		particlePool[i] = Particle();
//	}
	//Define a single particle to instance from.
	/*GLfloat vertices [20] = { 
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		 1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 0.0f, 1.0f
	};
	
	GLuint particleVBO;
	glGenBuffers(1, &particleVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
	glBufferData(GL_ARRAY_BUFFER, 20 * sizeof(GLfloat), vertices, GL_STREAM_DRAW);*/

	//More stuff here
	//http://open.gl/geometry
	//https://en.wikipedia.org/wiki/Vertex_Buffer_Object#References
	//The idea is to pass this buffer in to a geometry shader.  
	//Ideally this will be able to instance 2.43 Trillion particles from the one buffer
	//It should also be able to apply velocity.
	//

}

System::~System()
{
}

void System::createParticles()
{

	systemPull.y = 0.005;
	systemPull.x = systemPull.z = 0.0f; 

	//This should be more dynamic
	for(int i = 0; i < MAX_PARTICLES; i++)
	{
		//createParticle(&particles[i]);
		//particles.push_back(particlePool[i]);
		particles[i] = Particle();
	}
}
//We're going to replace this with a shader.
void System::updateParticles()
{
   for(int i = 0; i < MAX_PARTICLES; i++)
   {
      particles[i].age = particles[i].age + 0.02;
      
      particles[i].direction = particles[i].direction + ((((((int)(0.5) * rand()%11) + 1)) * rand()%11) + 1);
	  
	  glm::vec3 position = particles[i].getPosition();

	  particles[i].setPos( (position.x + particles[i].movement.x + particles[i].pull.x),
						   (position.y + particles[i].movement.y + particles[i].pull.y),
						   (position.z + particles[i].movement.z + particles[i].pull.z) );
      
      particles[i].pull.x = particles[i].pull.x + systemPull.x;
      particles[i].pull.y = particles[i].pull.y + systemPull.y; // accleration due to gravity
      particles[i].pull.z = particles[i].pull.z + systemPull.z;

         float temp = particles[i].lifespan/particles[i].age;
         if((temp) < 1.75)
         {//red
			particles[i].setRGBA(1.0f, 0.25f, 0.0f, 0.0f);
         }
         else if((temp) < 3.0)
         {//gold
			particles[i].setRGBA(1.0f, 0.9f, 0.0f, 0.0f);
         }
         else if((temp) < 10.0)
         {//yellow
			particles[i].setRGBA(1.0f, 1.0f, 0.0f, 0.0f);
         }
         else
         {// initial light yellow
			particles[i].setRGBA(1.0f, 0.95f, 0.8f, 0.0f);
         }
      
        if (particles[i].age > particles[i].lifespan || position.y > 45 || position.y < -35 || position.x > 80 || position.x < -80)
			particles[i] = Particle();
     
   }
}

glm::vec4 System::getRGBA(int i)
{
	return particles[i].getRGBA();
};

glm::vec3 System::getPosition(int i)
{
	return particles[i].getPosition();
};

int System::getNumOfParticles(void)
{
//   return particles.size();
	return MAX_PARTICLES;
}

float System::getScale(int i)
{
   return particles[i].scale;
}

float System::getDirection(int i)
{
   return particles[i].direction;
}

void System::modifySystemPull(float x, float y, float z)
{
   systemPull.x += x;
   systemPull.y += y;
   systemPull.z += z;
}