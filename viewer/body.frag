#version 330 core

out vec4 out_col;// color

in vec3 N;
in vec4 P;
in vec4 pv_color;


uniform int colormap;

float quintic_polynom(float x, float a, float b, float c, float d, float e, float f){
    return a * x * x * x * x * x + b * x * x * x * x + c * x * x * x + d * x * x + e * x + f;
}

vec4 colormap_coolwarm(float x) {
    float r = clamp(quintic_polynom(x, 3.386620678625149, -7.369972920457828, 3.4326128856393847, -0.20287273789508226, 1.2337352193942814, 0.22782212158237447) , 0.0, 1.0);
    float g = clamp(quintic_polynom(x, -4.100613455333197, 12.65040481292699, -14.248873433493298, 4.283163016197883, 1.1570081223589035, 0.31517959448301974) , 0.0, 1.0);
    float b = clamp(quintic_polynom(x, -4.039790277068525, 11.71483408307063, -9.811444974608307, 0.17143642458263936, 1.3566250992767772, 0.759078949535126) , 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main() {
    vec3 viewDir = normalize(-P.xyz);
    vec3 normal = normalize(N);

    vec4 base_color = vec4(0.9,0.9,0.9,1);
    if(colormap==0){
        base_color=mix(vec4(0.9,0.9,0.9,1),vec4(44.0/255.0,162.0/255.0,95.0/255.0,1),pv_color.w);
    }else if(colormap==1){
        base_color = colormap_coolwarm(pv_color.w);
    }else{
        base_color=vec4(pv_color.xyz * max(dot(viewDir, normal), 0),1);
    }
    out_col = base_color;
}
