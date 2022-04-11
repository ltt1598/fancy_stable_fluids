# reference implementation:
# https://github.com/taichi-dev/tai-objc-runtime/blob/master/stable_fluid.py

# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation

import taichi as ti
import numpy as np
import colorsys

ti.init(arch=ti.cpu)

# resolution constants
SIM_RES = 128
RENDER_RES_X = 1280
RENDER_RES_Y = 768
aspect_ratio = float(RENDER_RES_X) / float(RENDER_RES_Y)
SIM_RES_Y = SIM_RES
SIM_RES_X = int(SIM_RES_Y * aspect_ratio)

# sim constants

# numerical solver constants








# taichi fields
pixels = ti.Vector.field(3, ti.f32, shape=(RENDER_RES_X, RENDER_RES_Y))

















# random color gen
def generate_color():
    c = np.array(colorsys.hsv_to_rgb(np.random.random(), 1.0, 1.0))
    c *= 0.5
    return c

# mouse events
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None
        self.mouse_ticks = 0

    def __call__(self, window):
        # [0]: whether mouse is moved
        # [1:3]: normalized delta direction
        # [3:5]: current mouse xy
        # [5:8]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if window.is_pressed(ti.ui.LMB):
            mxy = np.array(window.get_cursor_pos(), dtype=np.float32)

            # change dye color every 6 mouse events
            if self.mouse_ticks > 5:
                self.mouse_ticks = 0
                self.prev_color = generate_color()

            if self.prev_mouse is None: # mouse pressed
                self.mouse_ticks = 0
                self.prev_mouse = mxy
                self.prev_color = generate_color()
            else: # mouse moving
                self.mouse_ticks += 1
                delta_mxy = mxy - self.prev_mouse
                if np.linalg.norm(delta_mxy) > 1e-4:
                    mouse_data[0] = 1
                    mouse_data[1], mouse_data[2] = delta_mxy[0], delta_mxy[1]
                    mouse_data[3], mouse_data[4] = mxy[0], mxy[1]
                    mouse_data[5:8] = self.prev_color
                self.prev_mouse = mxy
                
        else:
            mouse_data[0] = 0
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

def main():
    window = ti.ui.Window('Fancy Stable Fluids', (RENDER_RES_X, RENDER_RES_Y), vsync=True)
    canvas = window.get_canvas()
    md_gen = MouseDataGen()
    paused = False

    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == 'r':
                paused = False
                # reset()
            elif e.key == 'p':
                paused = not paused

        if not paused:
            mouse_data = md_gen(window)
            # step(mouse_data)
            # render()

        canvas.set_image(pixels)
        window.show()

if __name__ == '__main__':
    main()
