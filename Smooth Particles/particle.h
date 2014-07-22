#pragma once

#include "shaders.h"

#define NUM_PARTICLE_ATTRIBUTES 6
#define MAX_PARTICLES_ON_SCENE 100000

#define PARTICLE_TYPE_GENERATOR 0
#define PARTICLE_TYPE_NORMAL 1


class CParticle
{
	public:
		glm::vec3 position;
		glm::vec3 velocity;
		glm::vec3 color;
		float lifeTime;
		float size;
		int type;
};

class CParticleSystemTransformFeedback
{
	public:
		bool InitializeParticleSystem();

		void RenderParticles();
		void UpdateParticles(float fTimePassed);

		void SetGeneratorProperties(glm::vec3 genPosition, glm::vec3 genVelocityMin, glm::vec3 genVelocityMax,
			glm::vec3 genColor, float genLifeMin, float genLifeMax, float genSize, float every, int numToGenerate);

		void ClearAllParticles();
		bool ReleaseParticleSystem();

		int GetNumParticles();

		void SetMatrices(glm::mat4* matProjection, glm::vec3 vEye, glm::vec3 vView, glm::vec3 vUpVector);

		CParticleSystemTransformFeedback();
	private:
		bool Initialized;
		unsigned int transformFeedbackBuffer;

		unsigned int ParticleBuffer[2];
		unsigned int VAO[2];

		unsigned int query;
		unsigned int texture;

		int curReadBuffer;
		int numParticles;

		glm::mat4 matProjection, matView;
		glm::vec3 vQuad1, vQuad2;

		float elapsedTime;
		float nextGenerationTime;

		glm::vec3 genPosition;
		glm::vec3 genVelocityMin, genVelocityRange;
		glm::vec3 genGravityVector;
		glm::vec3 genColor;

		float genLifeMin, genLifeRange;
		float genSize;

		int numToGenerate;

		CShader shVertexRender, shGeomRender, shFragRender;
		CShader shVertexUpdate,  shGeomUpdate, shFragUpdate;
		CShaderProgram spRenderParticles;
		CShaderProgram spUpdateParticles;


};
