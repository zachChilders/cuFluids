#version 330 core

// Interpolated values from the vertex shaders
in vec2 UV;
in vec4 particlecolor;

// Ouput data
out vec4 color;

uniform sampler2D myTextureSampler;


void main(){
	// Output color = color of the texture at the specified UV


	// calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord * 2.0 - vec2(0.5);    
    float mag = dot(N.xy, N.xy);
    if (mag > 2.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(vec3(0, 0, 1), N));

    color = vec4(0,0,1,1) * diffuse;

}