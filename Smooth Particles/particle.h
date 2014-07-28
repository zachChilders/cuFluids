
#include <glm/glm.hpp>

#define MAX_PARTICLES 10000

class Particle
{
public:
	int type; //used only for "fire with smoke" to tell between fire and smoke 0=fire, 1=smoke, 2=other
	float lifespan, age, scale, direction; // how long the particle will exist for, alpha blending variable; how old is it.
	glm::vec3 position; // initial onscreen position
	glm::vec3 movement; // movement vector
	glm::vec3 color; // color values
	glm::vec3 pull; // compounding directional pull in the x,y,z directions
	Particle();
	~Particle();
	glm::vec3 getRGB();
}; 