# References
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py
# https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/ggui_examples/stable_fluid_ggui.py

import colorsys
import taichi as ti
import numpy as np
import random

ti.init(arch=ti.cpu)

# constants
SCREEN_RES_X = 1920
SCREEN_RES_Y = 1080
texel_size_X = 1.0 / SCREEN_RES_X
texel_size_y = 1.0 / SCREEN_RES_Y

aspect_ratio = float(SCREEN_RES_X) / float(SCREEN_RES_Y)
GRID_RES_Y = 64
GRID_RES_X = int(GRID_RES_Y * aspect_ratio)

grid_2_screen = float(SCREEN_RES_Y) / float(GRID_RES_Y)
screen_2_grid = 1.0 / grid_2_screen

dt = 0.0167
p_jacobi_iters = 20  
f_strength = 1e5
curl_strength = 7.0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
# dye_decay = 1
force_radius = SCREEN_RES_Y / 8.0
paused = False

# taichi fields
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur
# simulation quantities
_velocities = ti.Vector.field(2, float, shape=(GRID_RES_X, GRID_RES_Y))
_new_velocities = ti.Vector.field(2, float, shape=(GRID_RES_X, GRID_RES_Y))
_pressures = ti.field(float, shape=(GRID_RES_X, GRID_RES_Y))
_new_pressures = ti.field(float, shape=(GRID_RES_X, GRID_RES_Y))
velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
velocity_divs = ti.field(float, shape=(GRID_RES_X, GRID_RES_Y))
velocity_curls = ti.field(float, shape=(GRID_RES_X, GRID_RES_Y))

# visualization quantities
_dye_buffer = ti.Vector.field(3, float, shape=(SCREEN_RES_X, SCREEN_RES_Y))
_new_dye_buffer = ti.Vector.field(3, float, shape=(SCREEN_RES_X, SCREEN_RES_Y))
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


# simulation helpers
@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def sample_sim(qf, u, v):
    i = int(u)
    i = max(0, min(GRID_RES_X - 1, i))
    j = int(v)
    j = max(0, min(GRID_RES_Y - 1, j))
    return qf[i, j]

@ti.func
def bilerp_sim(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample_sim(vf, iu, iv)
    b = sample_sim(vf, iu + 1, iv)
    c = sample_sim(vf, iu, iv + 1)
    d = sample_sim(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def sample_screen(qf, u, v):
    i = int(u)
    i = max(0, min(SCREEN_RES_X - 1, i))
    j = int(v)
    j = max(0, min(SCREEN_RES_Y - 1, j))
    return qf[i, j]

@ti.func
def bilerp_screen(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample_screen(vf, iu, iv)
    b = sample_screen(vf, iu + 1, iv)
    c = sample_screen(vf, iu, iv + 1)
    d = sample_screen(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()): # RK-3
    v1 = bilerp_sim(vf, p)
    p -= dt * v1
    return p

# key quantity evaluation
@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample_sim(vf, i - 1, j)
        vr = sample_sim(vf, i + 1, j)
        vb = sample_sim(vf, i, j - 1)
        vt = sample_sim(vf, i, j + 1)
        vc = sample_sim(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == GRID_RES_X - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == GRID_RES_Y - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5

@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample_sim(vf, i - 1, j)
        vr = sample_sim(vf, i + 1, j)
        vb = sample_sim(vf, i, j - 1)
        vt = sample_sim(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


# simulation key steps
@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(),
                  imp_data: ti.ext_arr()):

    omx, omy = imp_data[2], imp_data[3]
    mdir = ti.Vector([imp_data[0], imp_data[1]])

    for i, j in vf:
        dx, dy = ((i + 0.5) * grid_2_screen - omx), ((j + 0.5) * grid_2_screen - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)
        momentum = (mdir * f_strength * factor) * dt * grid_2_screen
        v = vf[i, j]
        vf[i, j] = v + momentum

    for i, j in dyef:
        dx, dy = ((i + 0.5) - omx), ((j + 0.5) - omy)
        d2 = dx * dx + dy * dy        
        dc = dyef[i, j]
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (2 / (SCREEN_RES_Y / 16)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])
        dyef[i, j] = dc

@ti.kernel
def advect_sim(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in qf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp_sim(qf, p)

@ti.kernel
def advect_dye(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in qf:
        p = ti.Vector([i, j]) + 0.5
        p *= screen_2_grid
        p = backtrace(vf, p, dt)
        p *= grid_2_screen
        new_qf[i, j] = bilerp_screen(qf, p) * dye_decay

@ti.kernel
def decay_pressure(pf: ti.template(), pf_decay:ti.f32):
    for i,j in pf:
        pf[i,j] *= pf_decay

@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample_sim(pf, i - 1, j)
        pr = sample_sim(pf, i + 1, j)
        pb = sample_sim(pf, i, j - 1)
        pt = sample_sim(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25

@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample_sim(pf, i - 1, j)
        pr = sample_sim(pf, i + 1, j)
        pb = sample_sim(pf, i, j - 1)
        pt = sample_sim(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])

@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # for visual enhancement. (not physically-based)
    for i, j in vf:
        cl = sample_sim(cf, i - 1, j)
        cr = sample_sim(cf, i + 1, j)
        cb = sample_sim(cf, i, j - 1)
        ct = sample_sim(cf, i, j + 1)
        cc = sample_sim(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb),
                           abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = min(max(vf[i, j] + force * dt, -1e3), 1e3)

# simulation main loop
def step(mouse_data):

    # advect quantities
    advect_sim(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect_dye(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    # apply force
    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)
    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    # pressure solve (jacobi)
    divergence(velocities_pair.cur)
    decay_pressure(pressures_pair.cur, 0.8)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    # update velocity
    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)

# mouse events
class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None
        self.mouse_ticks = 0

    def __call__(self, window):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if window.is_pressed(ti.ui.LMB):
            mxy = np.array(window.get_cursor_pos(), dtype=np.float32)
            mxy[0] *= SCREEN_RES_X
            mxy[1] *= SCREEN_RES_Y

            # change dye color every 5 mouse events
            if self.mouse_ticks < 5:
                self.mouse_ticks += 1
            else:
                self.mouse_ticks = 0
                self.prev_color = colorsys.hsv_to_rgb(random.uniform(0, 1), 0.8, 1.0)

            if self.prev_mouse is None:
                self.prev_mouse = mxy
                self.mouse_ticks = 0
                self.prev_color = colorsys.hsv_to_rgb(random.uniform(0, 1), 0.8, 1.0)
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data

window = ti.ui.Window('Stable Fluid', (SCREEN_RES_X, SCREEN_RES_Y), vsync=True)
canvas = window.get_canvas()
md_gen = MouseDataGen()

while window.running:
    if window.get_event(ti.ui.PRESS):
        e = window.event
        if e.key == ti.ui.ESCAPE:
            break
        elif e.key == 'r':
            paused = False
            reset()
        elif e.key == 's':
            if curl_strength:
                curl_strength = 0
            else:
                curl_strength = 7
        elif e.key == 'p':
            paused = not paused

    if not paused:
        mouse_data = md_gen(window)
        step(mouse_data)

    canvas.set_image(dyes_pair.cur)
    window.show()
