import OpenGL.GL as gl
from OpenGL.raw.GL import _types as _cs
import numpy as np
from viewer.shader import Shader
from viewer.camera import Camera


def calcNormals(verts: np.array, faces: np.array) -> np.array:
    """
    Calculate vertex normals for a mesh given vertices and faces.

    Parameters:
    - verts (np.array): Array of vertices of shape (V, 3).
    - faces (np.array): Array of face indices of shape (F, 3).

    Returns:
    - np.array: Normal vectors for each vertex.
    """
    normals = np.zeros_like(verts)
    face_vert = verts[faces]

    # Calculate face normals
    face_normals = np.cross(
        face_vert[::, 1] - face_vert[::, 0], face_vert[::, 2] - face_vert[::, 0]
    )

    # Normalize face normals
    mag = np.sqrt(np.einsum("...i,...i", face_normals, face_normals))
    face_normals[:, 0] /= mag[:]
    face_normals[:, 1] /= mag[:]
    face_normals[:, 2] /= mag[:]

    # Accumulate face normals to vertex normals
    np.add.at(normals, faces.flatten(), np.repeat(face_normals, 3, axis=0))

    # Normalize vertex normals
    mag = np.sqrt(np.einsum("...i,...i", normals, normals))
    normals[:, 0] /= mag[:]
    normals[:, 1] /= mag[:]
    normals[:, 2] /= mag[:]
    return normals


class TriMesh:
    def __init__(
        self,
        vertices: np.array,
        indices: np.array,
        normals: np.array,
        shader: Shader,
        dyn=False,
    ) -> None:
        """
        Initializes a triangular mesh with vertices, indices, normals, and a shader.

        Parameters:
        - vertices (np.array): Vertex positions.
        - indices (np.array): Indices for triangle faces.
        - normals (np.array): Normal vectors for vertices.
        - shader (Shader): Shader object for rendering.
        - dyn (bool): If True, buffers are dynamic (modifiable).
        """
        assert indices.size % 3 == 0
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(3)
        self.ebo = gl.glGenBuffers(1)  # Element buffer for indices

        gl.glBindVertexArray(self.vao)

        # Buffer and attribute setup for vertices
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo[0])
        if dyn:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                vertices.nbytes,
                vertices.flatten(),
                gl.GL_DYNAMIC_DRAW,
            )
        else:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                vertices.nbytes,
                vertices.flatten(),
                gl.GL_STATIC_DRAW,
            )

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # Buffer and attribute setup for normals
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo[1])
        if dyn:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                normals.nbytes,
                normals.flatten(),
                gl.GL_DYNAMIC_DRAW,
            )
        else:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, normals.nbytes, normals.flatten(), gl.GL_STATIC_DRAW
            )

        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        # Buffer setup for vertex colors
        vertcol = np.ones((vertices.flatten().size // 3, 4), dtype=np.float32)
        vertcol[:] = np.array((0.8, 0.8, 0.8, 1), dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo[2])
        if dyn:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                vertcol.nbytes,
                vertcol.flatten(),
                gl.GL_DYNAMIC_DRAW,
            )
        else:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER, vertcol.nbytes, vertcol.flatten(), gl.GL_STATIC_DRAW
            )

        gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)

        # Buffer setup for element indices
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices.flatten(),
            gl.GL_STATIC_DRAW,
        )

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self.numTris = indices.size // 3
        self.numVerts = vertices.flatten().size // 3
        self.shader = shader
        self.dyn = dyn
        self.modelmat = np.identity(4)
        self.verts = vertices
        self.norms = normals
        self.inds = indices

    def render(self, cam: Camera, wireframe: bool = False) -> None:
        """
        Render the mesh.

        Parameters:
        - cam (Camera): The camera object for view, projection matrices.
        - wireframe (bool): If True, render in wireframe mode.
        """
        if wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDisable(gl.GL_CULL_FACE)

        self.shader.bind()
        # Send matrices to shader
        self.shader.uniformMat4("model", self.modelmat)
        self.shader.uniformMat4("view", cam.view)
        self.shader.uniformMat4("proj", cam.proj)
        self.shader.uniformMat4("view_normal", cam.view_normal)
        self.shader.uniformMat4(
            "model_normal", np.transpose(np.linalg.inv(self.modelmat))
        )

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.numTris * 3, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        self.shader.unbind()

        if wireframe:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def setShader(self, shader: Shader) -> None:
        """Set a new shader for the mesh."""
        self.shader = shader

    def updateBuffer(self, buf: int, data: np.array, numVals: int = 3) -> None:
        """Update a specified buffer with new data."""
        assert data.flatten().size // numVals == self.numVerts

        if not self.dyn:
            print("Static mesh: buffer not updated")
            return
        gl.glBindVertexArray(self.vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo[buf])
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, data.nbytes, data.flatten())

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def updateVerts(self, vertices: np.array) -> None:
        """Update vertex positions."""
        self.updateBuffer(0, vertices)

    def updateNorms(self, normal: np.array) -> None:
        """Update vertex normals."""
        self.updateBuffer(1, normal)

    def updateVertCol(self, cols: np.array) -> None:
        """Update vertex colors."""
        self.updateBuffer(2, cols, 4)

    def write_to_file(self, path: str):
        """Write mesh data to a file."""
        with open(path, "w") as file:
            for vert in self.verts[0]:
                file.write(
                    "v {} {} {}\n".format(str(vert[0]), str(vert[1]), str(vert[2]))
                )
            for face in self.inds:
                file.write(
                    "f {}//{} {}//{} {}//{}\n".format(
                        str(face[0] + 1),
                        str(face[0] + 1),
                        str(face[1] + 1),
                        str(face[1] + 1),
                        str(face[2] + 1),
                        str(face[2] + 1),
                    )
                )

    def set_model_translation(self, translation: np.array):
        """Set the model's translation in the world."""
        self.modelmat[0, 3] = translation[0]
        self.modelmat[1, 3] = translation[1]
        self.modelmat[2, 3] = translation[2]


class PointCloud:
    def __init__(self, vertices: np.array, shader: Shader, dyn=False) -> None:
        """
        Initializes a point cloud with vertices and a shader.

        Parameters:
        - vertices (np.array): Point positions.
        - shader (Shader): Shader object for rendering.
        - dyn (bool): If True, buffers are dynamic (modifiable).
        """
        indices = np.arange(vertices.flatten().size // 3, dtype=np.int32)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)

        # Buffer setup for point vertices
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        if dyn:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                vertices.nbytes,
                vertices.flatten(),
                gl.GL_DYNAMIC_DRAW,
            )
        else:
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                vertices.nbytes,
                vertices.flatten(),
                gl.GL_STATIC_DRAW,
            )

        # Buffer setup for element indices
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices.flatten(),
            gl.GL_STATIC_DRAW,
        )

        gl.glVertexAttribPointer(
            0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * vertices.itemsize, None
        )
        gl.glEnableVertexAttribArray(0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self.numVerts: int = vertices.flatten().size // 3
        self.shader: Shader = shader
        self.dyn: bool = dyn
        self.modelmat: np.matrix[4, 4] = np.identity(4)

    def render(self, cam: Camera) -> None:
        """Render the point cloud."""
        self.shader.bind()
        self.shader.uniformMat4("model", self.modelmat)
        self.shader.uniformMat4("view", cam.view)
        self.shader.uniformMat4("proj", cam.proj)
        self.shader.uniformMat4("view_normal", cam.view_normal)
        self.shader.uniformMat4(
            "model_normal", np.transpose(np.linalg.inv(self.modelmat))
        )

        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_POINTS, self.numVerts, gl.GL_UNSIGNED_INT, None)
        gl.glPointSize(5)

        gl.glBindVertexArray(0)
        self.shader.unbind()

    def setShader(self, shader: Shader) -> None:
        """Set a new shader for the point cloud."""
        self.shader = shader

    def updateVerts(self, vertices: np.array) -> None:
        """Update point positions."""
        assert vertices.flatten().size // 3 == self.numVerts

        if not self.dyn:
            print("Static mesh: vertices not updated")
            return
        gl.glBindVertexArray(self.vao)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices.flatten())

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)


class SphereMesh(TriMesh):
    def __init__(
        self,
        center: np.array,
        radius: float,
        sector_count: int,
        stack_count: int,
        shader: Shader,
        dyn: bool = False,
    ):
        """
        Initializes a spherical mesh with the given parameters.

        Parameters:
        - center (np.array): Center position of the sphere.
        - radius (float): Radius of the sphere.
        - sector_count (int): Number of sectors.
        - stack_count (int): Number of stacks.
        - shader (Shader): Shader object for rendering.
        - dyn (bool): If True, buffers are dynamic (modifiable).
        """
        sph = SphereMesh.generateGeometry(
            center, radius, sector_count, stack_count, True
        )
        indices = sph["inds"]
        vertices = sph["verts"]
        normals = sph["norms"]
        assert indices.size % 3 == 0
        super().__init__(vertices, indices, normals, shader, dyn)

        self.numSects = sector_count
        self.numStacks = stack_count
        self.center = center
        self.radius = radius

    def updateSphere(self, center: np.array, radius: float):
        """Update the sphere's geometry based on new center and radius."""
        self.radius = radius
        self.center = center
        sph = SphereMesh.generateGeometry(center, radius, self.numSects, self.numStacks)
        self.updateVerts(sph["verts"])
        self.updateNorms(sph["norms"])

    @staticmethod
    def generateGeometry(
        center: np.array,
        radius: float,
        sector_count: int = 20,
        stack_count: int = 20,
        genIndices: bool = False,
    ):
        """
        Generate the geometry of the sphere.

        Parameters:
        - center (np.array): Center position of the sphere.
        - radius (float): Radius of the sphere.
        - sector_count (int): Number of sectors.
        - stack_count (int): Number of stacks.
        - genIndices (bool): If True, generate indices.

        Returns:
        - dict: Dictionary containing vertices, normals, (optionally) indices.
        """
        sectorAngles = np.linspace(0, 2 * np.pi, sector_count + 1)
        stackAngles = np.linspace(np.pi / 2, -np.pi / 2, stack_count + 1)

        xy = radius * np.cos(stackAngles[:, np.newaxis])
        x = (xy * np.cos(sectorAngles)).flatten()
        y = (xy * np.sin(sectorAngles)).flatten()
        z = (
            radius * np.sin(stackAngles[:, np.newaxis]) * np.ones((1, sector_count + 1))
        ).flatten()  # Match x, y dimensions

        # Normalize vectors for normals
        lengthInv = 1.0 / max(radius, 0.000001)
        normals = np.stack(
            (x * lengthInv, y * lengthInv, z * lengthInv), axis=-1
        )  # Normalized normals

        # Calculate positions including center offset
        positions = np.stack(
            (x + center[0], y + center[1], z + center[2]), axis=-1
        )  # Final positions with center offset

        if genIndices:
            indices = np.zeros(stack_count * sector_count * 6, dtype=np.uint32)
            index_idx = 0
            for i in range(stack_count):
                k1 = i * (sector_count + 1)
                k2 = k1 + sector_count + 1
                for j in range(sector_count):
                    if i != 0:
                        indices[index_idx : index_idx + 3] = [k1, k2, k1 + 1]
                        index_idx += 3

                    if i != (stack_count - 1):
                        indices[index_idx : index_idx + 3] = [k1 + 1, k2, k2 + 1]
                        index_idx += 3
                    k1 += 1
                    k2 += 1

            return {
                "verts": positions.astype(np.float32),
                "norms": normals.astype(np.float32),
                "inds": indices.astype(np.uint32),
            }
        return {
            "verts": positions.astype(np.float32),
            "norms": normals.astype(np.float32),
        }
