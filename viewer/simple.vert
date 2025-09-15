#version 330 core
layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec4 in_color;

uniform mat4 model;
uniform mat4 model_normal;
uniform mat4 view_normal;
uniform mat4 view;
uniform mat4 proj;
//uniform vec4 lightDir;

out vec4 pv_color;
out vec3 N;
out vec4 P;
out vec4 wPos;

void main() {
    pv_color = in_color;

    N = normalize(mat3(view_normal) * mat3(model_normal) * vNormal);//Norm_cam
    P  = view * model * vec4(vPosition, 1);
    wPos= model*vec4(vPosition,1);
    gl_Position = proj * P;
}
