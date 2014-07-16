#include<GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <SOIL/SOIL.h>

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

GLFWwindow* window;

void startContext();

// Shader sources
const GLchar* vertexSource =
    "#version 150 core\n"
    "in vec3 position;"
    "in vec3 color;"
    "in vec2 texcoord;"
    "out vec3 Color;"
    "out vec2 Texcoord;"
    "uniform mat4 model;"
    "uniform mat4 view;"
    "uniform mat4 proj;"
	"uniform vec3 overrideColor;"
    "void main() {"
    "   Color = overrideColor * color;"
    "   Texcoord = texcoord;"
    "   gl_Position = proj * view * model * vec4(position, 1.0);"
    "}";

const GLchar* fragmentSource =
    "#version 150 core\n"
    "in vec3 Color;"
    "in vec2 Texcoord;"
    "out vec4 outColor;"
    "uniform sampler2D texKitten;"
    "uniform sampler2D texPuppy;"
    "void main() {"
	"	outColor = vec4(Color, 0.8) * mix(texture(texKitten, Texcoord), texture(texPuppy, Texcoord), 0.5);"
    "}";


int main()
{
	startContext();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthFunc(GL_LESS);
	
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint vbo; //Create a handle for our vertex buffer object
	glGenBuffers(1, &vbo); //Generate 1 buffer

	GLfloat vertices[] = {
        -0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
         0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

        -0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        -0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        -0.5f, -1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,

         0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f, -1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
         0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,

        -0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f, -1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f, -1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        -0.5f, -1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f, -1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,

        -0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
         0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
         0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        -0.5f,  1.35f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
        -0.5f,  1.35f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,

		//FLOOR
        -5.0f, -5.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         5.0f, -5.0f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
         5.0f,  5.0f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
         5.0f,  5.0f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        -5.0f,  5.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        -5.0f, -5.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

	glBindBuffer(GL_ARRAY_BUFFER, vbo); //Upload data to the GPU
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); //memcpy
	
	// Create and compile the vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Create and compile the fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

	
    // Specify the layout of the vertex data
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 
							8 * sizeof(GLfloat), 0);
						  //Var	   Number Type    ??  	    Line total		  offset
    GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 
							8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

	//Texturing
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);

	int width, height;
    unsigned char* image = SOIL_load_image("cat.png", &width, &height, 0, SOIL_LOAD_RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    SOIL_free_image_data(image);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	GLint uniModel = glGetUniformLocation(shaderProgram, "model");

	//Camera matrix
	glm::mat4 view = glm::lookAt(
						glm::vec3(4.0f, 0.0f ,2.0f),
						glm::vec3(0,0,0),
						glm::vec3(0,0,1.0f));
	GLint uniView = glGetUniformLocation(shaderProgram, "view");
	glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

	//Projection matrix.  45 FOV, 4:3 ratio, display range
	glm::mat4 proj = glm::perspective(45.0f, 4.0f / 3.0f, 1.0f, 10.0f);
	GLint uniProj = glGetUniformLocation(shaderProgram, "proj");
	glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

    GLint uniColor = glGetUniformLocation(shaderProgram, "overrideColor");

	do{

		//Clear the screen
		glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 model;
		glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

		//Draw cube
		glDrawArrays(GL_TRIANGLES, 0, 30);

		glEnable(GL_STENCIL_TEST);
		// Draw floor with stencil
        glStencilFunc(GL_ALWAYS, 1, 0xFF); //Stencil
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilMask(0xFF); //Mask
        glDepthMask(GL_FALSE);
        glClear(GL_STENCIL_BUFFER_BIT);       
        glDrawArrays(GL_TRIANGLES, 30, 6); //Actual drawing
        glStencilFunc(GL_EQUAL, 1, 0xFF); //Stencil
        glStencilMask(0x00); //Mask
        glDepthMask(GL_TRUE);


		// Draw cube reflection
		model = glm::scale(glm::translate(model, glm::vec3(0, 0, -1)), glm::vec3(1, 1, -1));
		glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));
		
		glUniform3f(uniColor, 0.3f, 0.3f, 0.3f);
		glDrawArrays(GL_TRIANGLES, 0, 30);
		glUniform3f(uniColor, 1.0f, 1.0f, 1.0f);

        glDisable(GL_STENCIL_TEST);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while( glfwGetKey(window, GLFW_KEY_ENTER ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );

	glDeleteTextures(1, &tex);

	glDeleteProgram(shaderProgram);
	glDeleteShader(fragmentShader);
	glDeleteShader(vertexShader);

	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);

	glfwTerminate();

	return 0;

}

void startContext()
{
	if (!glfwInit())
	{
		printf("Failed to init GLFW.\n");
		fprintf(stderr, "Failed to initialize GLFW\n");
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(1024, 768, "Fluid Renderer", NULL, NULL);

	if (window == NULL)
	{
		printf("Failed to open GLFW window.\n");
		fprintf(stderr, "Failed to open GLFW window.");
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);

	//Initialize GLEW
	glewExperimental=true;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "Failed to initialize GLEW\n");
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
}