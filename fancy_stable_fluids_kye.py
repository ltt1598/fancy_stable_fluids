# reference implementation:
# https://github.com/taichi-dev/tai-objc-runtime/blob/master/stable_fluid.py

# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation

import taichi as ti
import numpy as np
import colorsys

ti.init(arch=ti.cpu)

SIM_RES = 64
RENDER_RES_X = 1080
RENDER_RES_Y = 1080
aspect_ratio = float(RENDER_RES_X) / float(RENDER_RES_Y)
SIM_RES_Y = SIM_RES # dx = sim_texel_size_y
SIM_RES_X = int(SIM_RES_Y * aspect_ratio)

max_fps = 60
dt = 1.0 / max_fps
p_jacobi_iters = 20
p_jacobi_warm_starting = 0.8

# assert res > 2

@ti.func
def sample_clamp_to_edge(qf, u, v, res_x, res_y):
    i, j = int(u), int(v)
    # clamp to edge
    i = max(0, min(res_x - 1, i))
    j = max(0, min(res_y - 1, j))
    return qf[i, j]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, u, v, res_x, res_y):
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = int(s), int(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample_clamp_to_edge(vf, iu, iv, res_x, res_y)
    b = sample_clamp_to_edge(vf, iu + 1, iv, res_x, res_y)
    c = sample_clamp_to_edge(vf, iu, iv + 1, res_x, res_y)
    d = sample_clamp_to_edge(vf, iu + 1, iv + 1, res_x, res_y)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


@ti.data_oriented
class Texture(object):
    def __init__(self, f, res_x, res_y):
        self.field = f
        self.res_x = res_x
        self.res_y = res_y
        self.texel_size_x = 1.0 / res_x
        self.texel_size_y = 1.0 / res_y

    @staticmethod
    def Scalar(res_x, res_y):
        f = ti.field(ti.f32, shape=(res_x, res_y))
        return Texture(f, res_x, res_y)

    @staticmethod
    def Vector(dim, res_x, res_y):
        f = ti.Vector.field(dim, dtype=ti.f32, shape=(res_x, res_y))
        return Texture(f, res_x, res_y)

    @ti.func
    def sample_sep(self, u, v):
        u *= self.res_x
        v *= self.res_y
        return bilerp(self.field, u, v, self.res_x, self.res_y)

    @ti.func
    def sample(self, uv):
        return self.sample_sep(uv[0], uv[1])

    @ti.func
    def normalize(self, ij):
        # u, v = ij * self.texel_size
        u = ij[0] * self.texel_size_x
        v = ij[1] * self.texel_size_y
        return ti.Vector([u, v])


_velocities = Texture.Vector(2, SIM_RES_X, SIM_RES_Y)
_new_velocities = Texture.Vector(2, SIM_RES_X, SIM_RES_Y)
velocity_divs = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_pressures = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_new_pressures = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_curls = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_vorticities = Texture.Scalar(SIM_RES_X, SIM_RES_Y)

color_buffer = Texture.Vector(3, RENDER_RES_X, RENDER_RES_Y)
_dye_buffer = Texture.Vector(3, RENDER_RES_X, RENDER_RES_Y)
_new_dye_buffer = Texture.Vector(3, RENDER_RES_X, RENDER_RES_Y)


def make_bloom_mipmap():
    cur_res_x = SIM_RES_X
    cur_res_y = SIM_RES_Y
    mm = []
    BLOOM_ITERS = 8
    # while cur_res > 2:
    for _ in range(BLOOM_ITERS):
        cur_res_x = (cur_res_x >> 1)
        cur_res_y = (cur_res_y >> 1)
        if cur_res_x < 4 or cur_res_y < 4:
            break
        mm.append(Texture.Vector(3, cur_res_x, cur_res_y))
    return mm


_bloom_final = Texture.Vector(3, SIM_RES_X, SIM_RES_Y)
_bloom_mipmap = make_bloom_mipmap()

_sunrays = Texture.Scalar(SIM_RES_X, SIM_RES_Y)
_sunrays_scratch = Texture.Scalar(SIM_RES_X, SIM_RES_Y)


class TexPair:
    def __init__(self, cur, nxt):
        assert isinstance(cur, Texture)
        assert isinstance(nxt, Texture)
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),
           dissipation: float):
    for i, j in qf.field:
        uv = qf.normalize(ti.Vector([i, j]) + 0.5)
        vel = vf.sample(uv)
        prev_uv = uv - dt * vel
        q_s = qf.sample(prev_uv)
        decay = 1.0 + dissipation * dt
        new_qf.field[i, j] = q_s / decay


force_radius = 0.1 / 100
inv_force_radius = 1.0 / force_radius


@ti.kernel
def impulse_velocity(
        vf: ti.template(), omx: float, omy: float, fx: float, fy: float):
    for i, j in vf.field:
        u, v = vf.normalize(ti.Vector([i, j]) + 0.5)
        dx, dy = (u - omx), (v - omy)
        d2 = dx * dx + dy * dy
        momentum = ti.exp(-d2 * inv_force_radius) * ti.Vector([fx, fy])
        vel = vf.field[i, j]
        vf.field[i, j] = vel + momentum


dye_radius = 0.1 / 100
inv_dye_radius = 1.0 / dye_radius


@ti.kernel
def impulse_dye(dye: ti.template(), omx: float, omy: float, r: float, g: float,
                b: float):
    for i, j in dye.field:
        u, v = dye.normalize(ti.Vector([i, j]) + 0.5)
        dx, dy = (u - omx), (v - omy)
        d2 = dx * dx + dy * dy
        impulse = ti.exp(-d2 * inv_dye_radius) * ti.Vector([r, g, b])
        col = dye.field[i, j]
        dye.field[i, j] = col + impulse


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in _curls.field:
        res_x = vf.res_x
        res_y = vf.res_y
        vl = sample_clamp_to_edge(vf.field, i - 1, j, res_x, res_y)[1]
        vr = sample_clamp_to_edge(vf.field, i + 1, j, res_x, res_y)[1]
        vb = sample_clamp_to_edge(vf.field, i, j - 1, res_x, res_y)[0]
        vt = sample_clamp_to_edge(vf.field, i, j + 1, res_x, res_y)[0]
        vort = vr - vl - vt + vb
        _curls.field[i, j] = 0.5 * vort


curl_strength = 2.0

@ti.kernel
def vorticity_confinement(vf: ti.template()):
    for i, j in vf.field:
        res_x = vf.res_x
        res_y = vf.res_y
        vl = sample_clamp_to_edge(_curls.field, i - 1, j, res_x, res_y)
        vr = sample_clamp_to_edge(_curls.field, i + 1, j, res_x, res_y)
        vb = sample_clamp_to_edge(_curls.field, i, j - 1, res_x, res_y)
        vt = sample_clamp_to_edge(_curls.field, i, j + 1, res_x, res_y)
        vc = sample_clamp_to_edge(_curls.field, i, j, res_x, res_y)

        force = 0.5 * ti.Vector([abs(vt) - abs(vb),
                                 abs(vr) - abs(vl)]).normalized(1e-3)
        force *= curl_strength * vc
        vel = vf.field[i, j]
        vf.field[i, j] = min(max(vel + force * dt, -1e3), 1e3)


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf.field:
        res_x = vf.res_x
        res_y = vf.res_y
        vl = sample_clamp_to_edge(vf.field, i - 1, j, res_x, res_y)[0]
        vr = sample_clamp_to_edge(vf.field, i + 1, j, res_x, res_y)[0]
        vb = sample_clamp_to_edge(vf.field, i, j - 1, res_x, res_y)[1]
        vt = sample_clamp_to_edge(vf.field, i, j + 1, res_x, res_y)[1]
        vc = sample_clamp_to_edge(vf.field, i, j, res_x, res_y)
        if i == 0:
            vl = -vc[0]
        if i == vf.res_x - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == vf.res_y - 1:
            vt = -vc[1]
        velocity_divs.field[i, j] = 0.5 * (vr - vl + vt - vb)

@ti.kernel
def decay_pressure(pf: ti.template(), pf_decay:ti.f32):
    for i, j in pf.field:
        pf.field[i, j] *= pf_decay

@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf.field:
        res_x = pf.res_x
        res_y = pf.res_y
        pl = sample_clamp_to_edge(pf.field, i - 1, j, res_x, res_y)
        pr = sample_clamp_to_edge(pf.field, i + 1, j, res_x, res_y)
        pb = sample_clamp_to_edge(pf.field, i, j - 1, res_x, res_y)
        pt = sample_clamp_to_edge(pf.field, i, j + 1, res_x, res_y)
        div = velocity_divs.field[i, j]
        new_pf.field[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf.field:
        res_x = vf.res_x
        res_y = vf.res_y
        pl = sample_clamp_to_edge(pf.field, i - 1, j, res_x, res_y)
        pr = sample_clamp_to_edge(pf.field, i + 1, j, res_x, res_y)
        pb = sample_clamp_to_edge(pf.field, i, j - 1, res_x, res_y)
        pt = sample_clamp_to_edge(pf.field, i, j + 1, res_x, res_y)
        vel = sample_clamp_to_edge(vf.field, i, j, res_x, res_y)
        vel -= 0.5 * ti.Vector([pr - pl, pt - pb])
        vf.field[i, j] = vel


@ti.kernel
def fill_color_v2(vf: ti.template()):
    for i, j in vf:
        v = vf.field[i, j]
        color_buffer.field[i, j] = ti.Vector([abs(v[0]), abs(v[1]), 0.25])


@ti.func
def linear_to_gamma(rgb):
    rgb = max(rgb, 0)
    EXP = 0.416666667
    return max(1.055 * ti.pow(rgb, EXP) - 0.055, 0)


@ti.kernel
def fill_color_v3(dye: ti.template()):
    for i, j in color_buffer.field:
        uv = dye.normalize(ti.Vector([i, j]) + 0.5)
        v = dye.sample(uv)
        c = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])

        # sunrays = _sunrays.sample(uv)
        # c *= sunrays

        # bloom = _bloom_final.sample(uv) * 0.25
        # bloom *= sunrays
        # bloom = linear_to_gamma(bloom)
        # c += bloom

        color_buffer.field[i, j] = c


def run_impulse_kernels(mouse_data):
    f_strength = 6000.0

    normed_mxy = mouse_data[2:4]
    force = mouse_data[0:2] * f_strength * dt
    impulse_velocity(velocities_pair.cur, float(normed_mxy[0]), float(normed_mxy[1]),
                     float(force[0]), float(force[1]))

    rgb = mouse_data[4:7]
    impulse_dye(dyes_pair.cur, float(normed_mxy[0]), float(normed_mxy[1]), float(rgb[0]),
                float(rgb[1]), float(rgb[2]))


BLOOM_THRESHOLD = 0.6
BLOOM_SOFT_KNEE = 0.7
BLOOM_KNEE = BLOOM_THRESHOLD * BLOOM_SOFT_KNEE + 0.0001
BLOOM_CURVE_X = BLOOM_THRESHOLD - BLOOM_KNEE
BLOOM_CURVE_Y = BLOOM_KNEE * 2
BLOOM_CURVE_Z = 0.25 / BLOOM_KNEE


@ti.kernel
def bloom_prefilter(qf: ti.template()):
    # assuming qf is a dye field
    for i, j in _bloom_final.field:
        uv = _bloom_final.normalize(ti.Vector([i, j]) + 0.5)
        # vi, vj = int(i * ratio), int(j * ratio)
        # coord = ti.Vector([i, j]) + 0.5 - dt * vf[vi, vj] / ratio
        c = qf.sample(uv)
        br = max(c[0], max(c[1], c[2]))
        rq = min(max(br - BLOOM_CURVE_X, 0), BLOOM_CURVE_Y)
        rq = BLOOM_CURVE_Z * rq * rq
        c *= max(rq, br - BLOOM_THRESHOLD) / max(br, 0.0001)
        _bloom_final.field[i, j] = c


@ti.kernel
def bloom_fwd_blur(src: ti.template(), dst: ti.template()):
    for i, j in dst.field:
        u, v = dst.normalize(ti.Vector([i, j]) + 0.5)
        texel_sz_x = dst.texel_size_x
        texel_sz_y = dst.texel_size_y
        c = ti.Vector([0.0, 0.0, 0.0])
        c += src.sample_sep(u - texel_sz_x, v)
        c += src.sample_sep(u + texel_sz_x, v)
        c += src.sample_sep(u, v - texel_sz_y)
        c += src.sample_sep(u, v + texel_sz_y)
        # c = src.sample(v)
        dst.field[i, j] = c * 0.25


@ti.kernel
def bloom_inv_blur(src: ti.template(), dst: ti.template()):
    for i, j in dst.field:
        u, v = dst.normalize(ti.Vector([i, j]) + 0.5)
        texel_sz_x = dst.texel_size_x
        texel_sz_y = dst.texel_size_y
        c = ti.Vector([0.0, 0.0, 0.0])
        c += src.sample_sep(u - texel_sz_x, v)
        c += src.sample_sep(u + texel_sz_x, v)
        c += src.sample_sep(u, v - texel_sz_y)
        c += src.sample_sep(u, v + texel_sz_y)
        dst.field[i, j] += c * 0.25


def apply_bloom(qf):
    bloom_prefilter(qf)
    last = _bloom_final
    for bm in _bloom_mipmap:
        bloom_fwd_blur(last, bm)
        last = bm
    for i in reversed(range(len(_bloom_mipmap) - 1)):
        bm = _bloom_mipmap[i]
        bloom_inv_blur(last, bm)
        last = bm
    bloom_inv_blur(last, _bloom_final)


@ti.kernel
def k_sunrays_mask(dye_r: ti.template()):
    for i, j in _sunrays_scratch.field:
        uv = _sunrays_scratch.normalize(ti.Vector([i, j]) + 0.5)
        r, g, b = dye_r.sample(uv)
        br = max(r, max(g, b))
        a = 1.0 - min(max(br * 20.0, 0), 0.8)
        _sunrays_scratch.field[i, j] = a


SUNRAYS_DENSITY = 0.3
SUNRAYS_DECAY = 0.95
SUNRAYS_EXPOSURE = 0.7
SUNRAYS_ITERATIONS = 16


@ti.kernel
def k_sunrays():
    for i, j in _sunrays.field:
        cur_coord = _sunrays_scratch.normalize(ti.Vector([i, j]) + 0.5)
        dir = cur_coord - 0.5
        dir *= (SUNRAYS_DENSITY / SUNRAYS_ITERATIONS)
        illumination_decay = 1.0
        total_color = _sunrays_scratch.field[i, j]
        for _ in range(SUNRAYS_ITERATIONS):
            cur_coord -= dir
            col = _sunrays_scratch.sample(cur_coord)
            total_color += col * illumination_decay
            illumination_decay *= SUNRAYS_DECAY
        _sunrays.field[i, j] = total_color * SUNRAYS_EXPOSURE


def apply_sunrays(qf):
    k_sunrays_mask(qf)
    k_sunrays()


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt, 0.0)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, 0.8)
    velocities_pair.swap()
    dyes_pair.swap()

    run_impulse_kernels(mouse_data)

    if curl_strength:
        vorticity(velocities_pair.cur)
        vorticity_confinement(velocities_pair.cur)

    divergence(velocities_pair.cur)
    decay_pressure(pressures_pair.cur, p_jacobi_warm_starting)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    apply_bloom(dyes_pair.cur)
    apply_sunrays(dyes_pair.cur)
    fill_color_v3(dyes_pair.cur)


def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

# random color gen
def generate_color():
    c = np.array(colorsys.hsv_to_rgb(np.random.random(), 1.0, 1.0))
    c *= 0.5
    return c

class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: delta direction  (not normalized)
        # [2:4]: current mouse xy (normalized)
        # [4:7]: color
        # [7]: mouse moved
        mouse_data = np.array([0] * 9, dtype=np.float32)
        if gui.is_pressed(ti.ui.LMB):
            mxy = vec2_npf32(gui.get_cursor_pos())
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                self.prev_color = generate_color()
            else:
                mdir = mxy - self.prev_mouse
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                mouse_data[7] = True
                self.prev_mouse = mxy
        else:
            mouse_data[7] = False
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.field.fill(ti.Vector([0, 0]))
    pressures_pair.cur.field.fill(0.0)
    dyes_pair.cur.field.fill(ti.Vector([0, 0, 0]))
    color_buffer.field.fill(ti.Vector([0, 0, 0]))


def main():
    gui = ti.ui.Window('Fancy Stable Fluids', (RENDER_RES_X, RENDER_RES_Y), vsync=True)
    canvas = gui.get_canvas()
    md_gen = MouseDataGen()
    paused = False
    while True:
        while gui.get_event(ti.ui.PRESS):
            e = gui.event
            if e.key == ti.ui.ESCAPE:
                exit(0)
            elif e.key == 'r':
                paused = False
                reset()
            elif e.key == 'p':
                paused = not paused

        if not paused:
            mouse_data = md_gen(gui)
            step(mouse_data)

        canvas.set_image(color_buffer.field)
        gui.show()


if __name__ == '__main__':
    main()