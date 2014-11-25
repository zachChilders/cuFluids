
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "KDTree.h"


#include <Windows.h>
#include "GLutils.h"

#include <cuda_gl_interop.h>

//CUDA_CALLABLE_MEMBER void addKernel(Point3D* x)
//{
//    int i = threadIdx.x;
//	x[i] + 1;
//}


int windowInit();
GLFWwindow* window;

void cudaErrorCheck(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		std::cout << "Failed." << std::endl;
	}
}

int main()
{
	cudaGraphicsResource_t cgr;
	
//	KDTree k;

	std::vector<Point3D> v;
	std::cout << "Inserting Elements" << std::endl;
	for (int i = 1; i < 10; i++)
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
	for (auto b : v)
	{
		std::cout << b << std::endl;
	}

	windowInit();

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		printf("GlewInit error");
		exit(1);
	}
	
	do{

			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

		
			glfwSwapBuffers(window);
			glfwPollEvents();

		} while (glfwGetKey(window, GLFW_KEY_ENTER) != GLFW_PRESS &&
				 glfwWindowShouldClose(window) == 0);
	
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

	window = glfwCreateWindow(720, 640, "Particle System", NULL, NULL);

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