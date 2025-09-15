import OpenGL.GL as gl
import numpy as np


class Shader:
    """
    A class to encapsulate the creation and management of OpenGL shader programs.
    """

    @staticmethod
    def compile_shader(source: str, shader_type) -> int:
        """
        Compiles a shader from source code.

        Parameters:
        - source (str): The shader source code.
        - shader_type: The type of shader (e.g., gl.GL_VERTEX_SHADER).

        Returns:
        - int: The compiled shader ID.
        """
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)

        # Check for compilation errors
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(shader).decode()
            print(f"Shader compile error: {error}")
            return None

        return shader

    @staticmethod
    def create_shader_program_from_str(vertex_source: str, fragment_source: str) -> int:
        """
        Creates a shader program from vertex and fragment shader source strings.

        Parameters:
        - vertex_source (str): The vertex shader source code.
        - fragment_source (str): The fragment shader source code.

        Returns:
        - int: The created shader program ID.
        """
        vertex_shader = Shader.compile_shader(vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = Shader.compile_shader(fragment_source, gl.GL_FRAGMENT_SHADER)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)

        # Check for linking errors
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            print("Error linking shader program")
            return None

        # Clean up shaders as they are no longer needed after linking
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)

        return program

    def __init__(self, vertex_shader_path: str, fragment_shader_path: str) -> None:
        """
        Initializes a Shader object by reading and compiling shaders from files.

        Parameters:
        - vertex_shader_path (str): File path to the vertex shader source.
        - fragment_shader_path (str): File path to the fragment shader source.
        """
        with open(vertex_shader_path) as f:
            vs = f.read()

        with open(fragment_shader_path) as f:
            fs = f.read()

        self.shID = Shader.create_shader_program_from_str(vs, fs)

    def uniformInt(self, name: str, val: np.int32) -> None:
        """
        Sets an integer uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.int32): The integer value to set.
        """
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniform1i(loc, val)

    def uniformUint(self, name: str, val: np.uint32) -> None:
        """
        Sets an unsigned integer uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.uint32): The unsigned integer value to set.
        """
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniform1ui(loc, val)

    def uniformFloat(self, name: str, val: np.float32) -> None:
        """
        Sets a float uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.float32): The float value to set.
        """
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniform1f(loc, val)

    def uniformVec3(self, name: str, val: np.array) -> None:
        """
        Sets a vec3 uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.array): A vector of size 3.
        """
        val2 = val.flatten()
        assert val2.size == 3
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniform3f(loc, val2[0], val2[1], val2[2])

    def uniformVec4(self, name: str, val: np.array) -> None:
        """
        Sets a vec4 uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.array): A vector of size 4.
        """
        val2 = val.flatten()
        assert val2.size == 4
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniform4f(loc, val2[0], val2[1], val2[2], val2[3])

    def uniformMat3(self, name: str, val: np.matrix) -> None:
        """
        Sets a mat3 uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.matrix): A 3x3 matrix.
        """
        assert val.flatten().size == 9
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniformMatrix3fv(loc, 1, gl.GL_FALSE, val.T.astype(np.float32))

    def uniformMat4(self, name: str, val: np.matrix) -> None:
        """
        Sets a mat4 uniform value in the shader.

        Parameters:
        - name (str): The name of the uniform variable.
        - val (np.matrix): A 4x4 matrix.
        """
        assert val.flatten().size == 16
        self.bind()
        loc = gl.glGetUniformLocation(self.shID, name)
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, val.T.astype(np.float32))

    def bind(self) -> None:
        """Binds the shader program for use."""
        gl.glUseProgram(self.shID)

    def unbind(self) -> None:
        """Unbinds any shader program, setting it to zero."""
        gl.glUseProgram(0)
