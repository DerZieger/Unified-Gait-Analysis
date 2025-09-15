import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
from viewer.camera import Camera
from PIL import Image
from PIL import ImageOps


class Window:
    """
    Class to create an OpenGL window using GLFW and manage rendering with ImGui integration.
    """

    def __init__(
        self, width: int = 800, height: int = 600, title: str = "OpenGL Window"
    ) -> None:
        """
        Initializes the window with OpenGL context.

        Parameters:
        - width (int): Width of the window.
        - height (int): Height of the window.
        - title (str): Title of the window.
        """
        if not glfw.init():
            raise Exception("GLFW cannot be initialized!")

        self.width: int = width
        self.height: int = height
        self.title: str = title

        # Set window hints for OpenGL
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created!")
        glfw.make_context_current(self.window)

        # Setup ImGui context
        imgui.create_context()
        self.imgui_impl = GlfwRenderer(self.window)

        # Set callback functions
        glfw.set_cursor_pos_callback(self.window, self.mouse_input_clb)
        glfw.set_key_callback(self.window, self.key_input_clb)
        glfw.set_framebuffer_size_callback(self.window, self.resize_clb)
        glfw.set_scroll_callback(self.window, self.scroll_clb)

        # Setup camera
        self.camera: Camera = Camera(
            position=[0.0, 0.0, 3.0],
            target=[0.0, 0.0, 0.0],
            up_vector=[0.0, 1.0, 0.0],
            fov_deg=45.0,
            aspect_ratio=width / height,
            near_plane=0.1,
            far_plane=100.0,
        )

        # Initial mouse positions
        self.lastX: float = width / 2
        self.lastY: float = height / 2
        self.firstMouse = True

        # Movement flags
        self.move: list[bool] = [False] * 8

        # Set OpenGL state
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        glfw.swap_interval(1)

        self.camera_speed: float = 0.05
        gl.glClearColor(0.1, 0.1, 0.1, 1)

    def key_input_clb(self, window, key, scancode, action, mode) -> None:
        """
        Key input callback for handling key presses.

        Parameters:
        - window: the window receiving the event
        - key: key being pressed or released
        - scancode: platform-specific scancode of the key
        - action: action (PRESS, RELEASE, or REPEAT)
        - mode: bit field describing which modifier keys are held down
        """
        if imgui.get_io().want_capture_keyboard or imgui.get_io().want_capture_mouse:
            self.imgui_impl.keyboard_callback(window, key, scancode, action, mode)
            for i in range(len(self.move)):
                self.move[i] = False
            return

        # Close window with Escape key
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # Map keys to movement flags
        key_map = {
            glfw.KEY_W: 0,
            glfw.KEY_S: 1,
            glfw.KEY_A: 2,
            glfw.KEY_D: 3,
            glfw.KEY_R: 4,
            glfw.KEY_F: 5,
            glfw.KEY_E: 6,
            glfw.KEY_Q: 7,
        }
        if key in key_map:
            self.move[key_map[key]] = (
                action == glfw.PRESS or glfw.get_key(window, key) == glfw.PRESS
            )

    def mouse_input_clb(self, window, xpos, ypos) -> None:
        """
        Mouse input callback for handling mouse movements.

        Parameters:
        - window: the window receiving the event
        - xpos: x-coordinate of the cursor
        - ypos: y-coordinate of the cursor
        """
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            self.imgui_impl.mouse_callback(window, xpos, ypos)
            return

        if self.firstMouse:
            self.lastX, self.lastY = xpos, ypos
            self.firstMouse = False

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            xoffset = self.lastX - xpos
            yoffset = self.lastY - ypos
            self.lastX, self.lastY = xpos, ypos

            xoffset *= self.camera.rot_speed
            yoffset *= self.camera.rot_speed

            self.camera.yaw(xoffset)
            self.camera.pitch(yoffset)

        self.lastX, self.lastY = xpos, ypos

    def resize_clb(self, window, w, h) -> None:
        """
        Resize callback for handling window size changes.

        Parameters:
        - window: the window being resized
        - w: new width of the window
        - h: new height of the window
        """
        self.height = h
        self.width = w
        gl.glViewport(0, 0, w, h)
        self.camera.aspect_ratio = w / h
        self.camera.update()

    def scroll_clb(self, window, xoff, yoff) -> None:
        """
        Scroll callback for handling scroll input.

        Parameters:
        - window: the window receiving the event
        - xoff: scroll offset along the x-axis
        - yoff: scroll offset along the y-axis
        """
        if imgui.get_io().want_capture_mouse or imgui.get_io().want_capture_keyboard:
            self.imgui_impl.scroll_callback(window, xoff, yoff)
            return
        self.camera_speed = max(
            0.000001, yoff * 0.1 * self.camera_speed + self.camera_speed
        )

    def camUpdate(self) -> None:
        """Update the camera position based on input flags."""
        if self.move[0]:
            self.camera.forward(self.camera_speed)
        if self.move[1]:
            self.camera.backward(self.camera_speed)
        if self.move[2]:
            self.camera.leftward(self.camera_speed)
        if self.move[3]:
            self.camera.rightward(self.camera_speed)
        if self.move[4]:
            self.camera.upward(self.camera_speed)
        if self.move[5]:
            self.camera.downward(self.camera_speed)
        if self.move[6]:
            self.camera.roll(self.camera_speed)
        if self.move[7]:
            self.camera.roll(-self.camera_speed)

    def screenshot(self, filename: str = "screenshot.png") -> None:
        """
        Capture a screenshot of the current window content.

        Parameters:
        - filename (str): path where to save the screenshot.
        """
        import os

        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        data = gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE
        )
        image = Image.frombytes("RGBA", (self.width, self.height), data)
        image = ImageOps.flip(image)
        image.save(filename, "PNG")

    def newFrame(self, skip_swap_buffer=False) -> None:
        """
        Prepare a new frame for rendering.

        Parameters:
        - skip_swap_buffer (bool): if True, skips swapping the buffer.
        """
        imgui.render()
        self.imgui_impl.render(imgui.get_draw_data())
        if not skip_swap_buffer:
            glfw.swap_buffers(self.window)
        self.imgui_impl.process_inputs()
        glfw.poll_events()
        self.camUpdate()
        self.camera.update()
        imgui.new_frame()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)


# Main program execution
if __name__ == "__main__":
    win = Window()
    win.main_loop()
