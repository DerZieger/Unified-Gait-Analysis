#version 330 core

out vec4 out_col;// color

in vec3 N;
in vec4 P;
in vec4 wPos;
in vec4 pv_color;

void main() {
    vec3 viewDir = normalize(-P.xyz);
    vec3 normal = normalize(N);
    out_col=vec4(1,0,0,1);
}
