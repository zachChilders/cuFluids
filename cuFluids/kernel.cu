
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "KDTree.h"

#include <algorithm>
#include <Windows.h>
#include "GLutils.h"

#include "controls.hpp"
#include "shader.hpp"
#include "texture.hpp"

#include <cuda_gl_interop.h>

#define NUM_PARTICLES 2

//CUDA_CALLABLE_MEMBER void addKernel(Point3D* x)
//{
//    int i = threadIdx.x;
//	x[i] + 1;
//}

int windowInit();
GLFWwindow* window;

const int MaxParticles = 100000;
Point3D ParticlesContainer[MaxParticles];
int LastUsedParticle = 0;

int FindUnusedParticle(){

	for (int i = LastUsedParticle; i<MaxParticles; i++){
		if (ParticlesContainer[i].life < 0){
			LastUsedParticle = i;
			return i;
		}
	}

	for (int i = 0; i<LastUsedParticle; i++){
		if (ParticlesContainer[i].life < 0){
			LastUsedParticle = i;
			return i;
		}
	}

	return 0; // All particles are taken, override the first one
}

void SortParticles(){
	std::sort(&ParticlesContainer[0], &ParticlesContainer[MaxParticles]);
}

void cudaErrorCheck(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		std::cout << "Failed." << std::endl;
	}
}

int main()
{
	//	KDTree k;

	std::vector<Point3D> v;
	std::cout << "Inserting Elements" << std::endl;
	for (int i = 1; i < NUM_PARTICLES; i++)
	{
		Point3D p = Point3D(i, 0, 0);
		v.push_back(p);
	}

	for (auto b : v)
	{
		std::cout << b << std::endl;
	}
	std::cout << "==========" << std::endl;

	Point3D *devA;

	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&devA, v.size() * sizeof(Point3D));
	cudaErrorCheck(cudaStatus);
	cudaStatus = cudaMemcpy(devA, &v[0], v.size() * sizeof(Point3D), cudaMemcpyHostToDevice);
	cudaErrorCheck(cudaStatus);
	/*
		std::cout << "===============" << std::endl;
		std::cout << "Flattening tree" << std::endl;*/


	//std::vector <Point3D> a = k.flatten();

	windowInit();

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		printf("GlewInit error");
		exit(1);
	}

	glfwSetCursorPos(window, 1024 / 2, 768 / 2);
	glfwSwapInterval(1);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	 //Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	// Dark blue background
	glClearColor(0.f, 0.f, 0.f, 0.0f);

	 //Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	 //Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders("Particle.vert", "Particle.frag");

	// Vertex shader
	GLuint CameraRight_worldspace_ID = glGetUniformLocation(programID, "CameraRight_worldspace");
	GLuint CameraUp_worldspace_ID = glGetUniformLocation(programID, "CameraUp_worldspace");
	GLuint ViewProjMatrixID = glGetUniformLocation(programID, "VP");

	// fragment shader
	GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");
	GLuint lightDir = glGetUniformLocation(programID, "lightDir");


	static GLfloat* g_particule_position_size_data = new GLfloat[MaxParticles * 4];
	static GLubyte* g_particule_color_data = new GLubyte[MaxParticles * 4];

	//Probably not necessary, since we won't be killing particles.
	for (int i = 0; i<MaxParticles; i++){
		ParticlesContainer[i].life = -1.0f;
	}

	/* Generate 10 new particule each millisecond,
	but limit this to 16 ms (60 fps), or if you have 1 long frame (1sec),
	newparticles will be huge and the next frame even longer.*/
	int newparticles = (int)(10000.0);
	if (newparticles >(int)(0.016f*10000.0))
		newparticles = (int)(0.016f*10000.0);

	for (int i = 0; i < 100; i++)
	{
		float zOffset = 0;
		for (int j = 0; j < 100; j++)
		{
			int particleIndex = FindUnusedParticle();
			ParticlesContainer[particleIndex].life = 5.0f; // This particle will live 5 seconds.
			ParticlesContainer[particleIndex].position = glm::vec3(0, 0, -20.0f);

			float spread = 1.5f;
			glm::vec3 maindir = glm::vec3(0.0f, 10.0f, 0.0f);

			//Random direction
			glm::vec3 randomdir = glm::vec3(
				(rand() % 2000 - 1000.0f) / 1000.0f,
				(rand() % 2000 - 1000.0f) / 1000.0f,
				(rand() % 2000 - 1000.0f) / 1000.0f
				);

			ParticlesContainer[particleIndex].velocity = maindir + randomdir*spread;

			ParticlesContainer[particleIndex].size = (rand() % 1000) / 2000.0f + 0.1f;
			ParticlesContainer[particleIndex].position.x += i - (newparticles / 2);
			ParticlesContainer[particleIndex].position.z += zOffset;
		}
		zOffset -= 10;
	}

	 //The VBO containing the 4 vertices of the particles.
	 //Thanks to instancing, they will be shared by all particles.
	static const GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, 1.0f,
		0.5f, -0.5f, 1.0f,
		-0.5f, 0.5f, 1.0f,
		0.5f, 0.5f, 1.0f,
	};
	GLuint billboard_vertex_buffer;
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	 //The VBO containing the positions and sizes of the particles
	GLuint particles_position_buffer;
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

	double lastTime = glfwGetTime();
	do
	{
		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		lastTime = currentTime;

		//MVP should be set - ZC
		computeMatricesFromInputs();
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();

		/* We will need the camera's position in order to sort the particles
		 w.r.t the camera's distance.
		 There should be a getCameraPosition() function in common/controls.cpp, 
		 but this works too.*/
		glm::vec3 CameraPosition(glm::inverse(ViewMatrix)[3]);

		glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;

		
		// update loop, cudaIZE
		int ParticlesCount = 0;
		for (int i = 0; i < MaxParticles; i++){

			Point3D& p = ParticlesContainer[i]; // shortcut

			// Decrease life
			//p.life -= delta;

			// Simulate simple physics : gravity only, no collisions
			p.velocity += glm::vec3(0.0f, -9.81f, 0.0f) * (float)delta * 0.5f;
			p.position += p.velocity * (float)delta;
			
			ParticlesContainer[i].position += glm::vec3(0.0f, 2.0f, 0.0f) * (float)delta;



			// Fill the GPU buffer
			g_particule_position_size_data[4 * ParticlesCount + 0] = p.position.x;
			g_particule_position_size_data[4 * ParticlesCount + 1] = p.position.y;
			g_particule_position_size_data[4 * ParticlesCount + 2] = p.position.z;

			g_particule_position_size_data[4 * ParticlesCount + 3] = p.size;

			ParticlesCount++;
			
		}

		SortParticles();

		glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
		glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
		glBufferSubData(GL_ARRAY_BUFFER, 0, ParticlesCount * sizeof(GLfloat)* 4, g_particule_position_size_data);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// Use our shader
		glUseProgram(programID);

		 //Same as the billboards tutorial
		glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
		glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);

		glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);

		// 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
		glVertexAttribPointer(
			0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
			);

		 //2nd attribute buffer : positions of particles' centers
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size : x + y + z + size => 4
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
			);


		glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
		glVertexAttribDivisor(1, 1); // positions : one per quad (its center)                 -> 1

		glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, ParticlesCount);

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ENTER) != GLFW_PRESS &&
	glfwWindowShouldClose(window) == 0);


	delete[] g_particule_position_size_data;

	// Cleanup VBO and shader
	glDeleteBuffers(1, &particles_position_buffer);
	glDeleteBuffers(1, &billboard_vertex_buffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);


	 //Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}

int windowInit()
{

	if (!glfwInit())
	{
		printf("Failed to init GLFW.\n");
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(720, 640, "Fluids", NULL, NULL);

	if (window == NULL)
	{
		printf("Failed to open GLFW window.\n");
		fprintf(stderr, "Failed to open GLFW window.");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	return 0;
}