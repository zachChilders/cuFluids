#include "common_header.h"

#include "particle.h"

CParticleSystemTransformFeedback::CParticleSystemTransformFeedback()
{
	Initialized = false;
	curReadBuffer = 0;

}

bool CParticleSystemTransformFeedback::InitializeParticleSystem()

{
	if (Initialized)
	{
		return false;
	}

	const char* varyings[NUM_PARTICLE_ATTRIBUTES] =
	{
		"positionOut",
		"velocityOut",
		"colorOut",
		"lifeTimeOut",
		"sizeOut",
		"typeOut",
	};

	shVertexUpdate.LoadShader("shaders\\particles_update.vert", GL_VERTEX_SHADER);
	shGeomUpdate.LoadShader("shaders\\particles_update.geom", GL_GEOMETRY_SHADER);

	spUpdateParticles.CreateProgram();
	spUpdateParticles.AddShaderToProgram(&shVertexUpdate);
	spUpdateParticles.AddShaderToProgram(&shGeomUpdate);
	for(int i = 0; i < NUM_PARTICLE_ATTRIBUTES; i++)
	{
		glTransformFeedbackVaryings(spUpdateParticles.GetProgramID(), 6, varyings, GL_INTERLEAVED_ATTRIBS);
	}
	spUpdateParticles.LinkProgram();

	shVertexRender.LoadShader("shaders\\particles_render.vert", GL_VERTEX_SHADER);
	shGeomRender.LoadShader("shaders\\particles_render.geom", GL_GEOMETRY_SHADER);
	shFragRender.LoadShader("shaders\\particles_render.frag", GL_FRAGMENT_SHADER);

	spRenderParticles.CreateProgram();

	spRenderParticles.AddShaderToProgram(&shVertexRender);
	spRenderParticles.AddShaderToProgram(&shGeomRender);
	spRenderParticles.AddShaderToProgram(&shFragRender);

	spRenderParticles.LinkProgram();

	glGenTransformFeedbacks(1, &transformFeedbackBuffer);
	glGenQueries(1, &query);
	
	glGenBuffers(2, ParticleBuffer);
	glGenVertexArrays(2, VAO);

	CParticle partInitialization;
	partInitialization.type = PARTICLE_TYPE_GENERATOR;

	for(int i = 0; i < 2; i++)
	{
		glBindVertexArray(VAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, ParticleBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(CParticle) * MAX_PARTICLES_ON_SCENE, NULL,GL_DYNAMIC_DRAW);

		for (int i = 0; i < NUM_PARTICLE_ATTRIBUTES; i++)
		{
			glEnableVertexAttribArray(i);
		}
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)0); // Position
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)12); // Velocity
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)24); // Color
		glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)36); // Lifetime
		glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)40); // Size
		glVertexAttribPointer(5, 1, GL_INT,	  GL_FALSE, sizeof(CParticle), (const GLvoid*)44); // Type

	}
	curReadBuffer = 0;
	numParticles = 1;
	
	Initialized = true;

	return true;
}

float grandf(float min, float add)
{
	float random = float(rand()%(RAND_MAX+1))/float(RAND_MAX);
	return min+add*random;
}


void CParticleSystemTransformFeedback::UpdateParticles(float timePassed)
{
	if (!Initialized)
	{
		return;
	}

	spUpdateParticles.UseProgram();
	glm::vec3 Upload;
	spUpdateParticles.SetUniform("timePassed", timePassed);
	spUpdateParticles.SetUniform("genPosition", genPosition);
	spUpdateParticles.SetUniform("genVelocityMin", genVelocityMin);
	spUpdateParticles.SetUniform("genVelocityRange", genVelocityRange);
	spUpdateParticles.SetUniform("genColor", genColor);
	spUpdateParticles.SetUniform("genGravityVector", genGravityVector);	

	spUpdateParticles.SetUniform("genLifeMin", genLifeMin);
	spUpdateParticles.SetUniform("genLifeRange", genLifeRange);

	spUpdateParticles.SetUniform("genSize", genSize);
	spUpdateParticles.SetUniform("numToGenerate", 0);

	elapsedTime += timePassed;

	if (elapsedTime > nextGenerationTime)
	{
		spUpdateParticles.SetUniform("numToGenerate", numToGenerate);
		elapsedTime -= nextGenerationTime;

		glm::vec3 randomSeed = glm::vec3(grandf(-10.0f, 20.0f), grandf(-10.0f, 20.0f), grandf(-10.0f, 20.0f));
		spUpdateParticles.SetUniform("randomSeed", &randomSeed);
	}

	glEnable(GL_RASTERIZER_DISCARD);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, transformFeedbackBuffer);

	glBindVertexArray(VAO[curReadBuffer]);
	glEnableVertexAttribArray(1);  //This reenables velocity



}