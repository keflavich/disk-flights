"""Numba kernels shared by the scattering Monte Carlo and the ray marcher.

Geometry: disk frame, z = disk axis, lengths in AU, the star at the origin. Model parameter
vector P = (rho0, r0, h0, beta, p, rmin, rmax) as DiskModel.params. Tables on uniform grids
in ln r, mu = |z| / r (the model is mirror-symmetric about z = 0) and azimuth phi.

Non-axisymmetric models multiply the axisymmetric density by delta(R, phi) (a surface-density
perturbation, e.g. a planet's gap and spirals): rho = rho_axi(R, z) exp(LD[0](ln R, phi)), with LD
tabulated on nodes ln R = PT[0] + i PT[1], phi = PT[2] + j PT[3] (periodic in phi); PT[4] = 0
switches the perturbation off (axisymmetric models). LD[1] is the planet's excess E over the
gap level (linear, in units of rho_axi; LD[0] is then ln(delta - E)), which PT[5] > 0
concentrates towards the midplane (see density3). Tables with a phi axis of
length 1 are axisymmetric.
"""
import numpy as np
from numba import njit


@njit(inline="always", cache=True)
def density(R, z, P):
    r = np.sqrt(R * R + z * z)
    if r < P[5] or r > P[6] or R <= 0.0:
        return 0.0
    h = P[2] * (R / P[1]) ** P[3]
    q = z / h
    return P[0] * (P[1] / R) ** (P[3] - P[4]) * np.exp(-0.5 * q * q)


@njit(inline="always", cache=True)
def pert_log(x, y, PT, LD):
    """(ln(delta - E), clump excess E) at (x, y), from LD[0] and LD[1]; 0 outside the tabulated
    radii or when PT[4] == 0."""
    if PT[4] == 0.0:
        return 0.0, 0.0
    R = np.sqrt(x * x + y * y)
    if R <= 0.0:
        return 0.0, 0.0
    fi = (np.log(R) - PT[0]) / PT[1]
    nr, nphi = LD.shape[1], LD.shape[2]
    if fi < 0.0 or fi > nr - 1:
        return 0.0, 0.0
    i = min(int(fi), nr - 2)
    ti = fi - i
    fj = (np.arctan2(y, x) - PT[2]) / PT[3]
    fj = fj - nphi * np.floor(fj / nphi)
    j = int(fj)
    tj = fj - j
    j = j % nphi
    j1 = (j + 1) % nphi
    a = (1 - ti) * (1 - tj)
    b = (1 - ti) * tj
    c = ti * (1 - tj)
    e = ti * tj
    return (a * LD[0, i, j] + b * LD[0, i, j1] + c * LD[0, i + 1, j] + e * LD[0, i + 1, j1],
            a * LD[1, i, j] + b * LD[1, i, j1] + c * LD[1, i + 1, j] + e * LD[1, i + 1, j1])


@njit(inline="always", cache=True)
def density3(x, y, z, P, PT, LD):
    """rho_axi * delta; with PT[5] = clump_h > 0 the planet's excess over the gap level (LD[1])
    gets a Gaussian vertical profile of scale PT[5] h(R) at fixed surface density
    (diskplanet.perturb): rho_axi * ((delta - E) + E f)."""
    R = np.sqrt(x * x + y * y)
    d = density(R, z, P)
    if d > 0.0 and PT[4] != 0.0:
        lt, e = pert_log(x, y, PT, LD)
        if PT[5] > 0.0 and e != 0.0:
            k = PT[5]
            q = z / (P[2] * (R / P[1]) ** P[3])
            f = np.exp(-0.5 * q * q * (1.0 / (k * k) - 1.0)) / k
            d *= np.exp(lt) + e * f
        else:
            d *= np.exp(lt)
    return d


@njit(inline="always", cache=True)
def phi_weights(phi0, dphi, n, phi):
    """(j, j + 1 mod n, fraction) for periodic linear interpolation between nodes phi0 + j dphi;
    (0, 0, 0) when n == 1."""
    if n == 1:
        return 0, 0, 0.0
    f = (phi - phi0) / dphi
    f = f - n * np.floor(f / n)
    j = int(f)
    t = f - j
    j = j % n
    return j, (j + 1) % n, t


@njit(inline="always", cache=True)
def node_interp3(tab, x0, dx, y0, dy, phi0, dphi, x, y, phi):
    """node_interp on tab[iphi, ix, iy], linear and periodic in phi (tab.shape[0] == 1:
    axisymmetric)."""
    j0, j1, t = phi_weights(phi0, dphi, tab.shape[0], phi)
    a = node_interp(tab[j0], x0, dx, y0, dy, x, y)
    if t == 0.0:
        return a
    return (1 - t) * a + t * node_interp(tab[j1], x0, dx, y0, dy, x, y)


@njit(inline="always", cache=True)
def geo_step(R, z, r, P, frac):
    """Spatial step that resolves the vertical structure (frac of a scale height), with long
    strides through the empty regions: the inner cavity, outside rmax, far above the disk."""
    if r > P[6]:
        return max(r - P[6], 1e-3 * r)
    if r < P[5]:
        return max(P[5] - r, 1e-3 * P[5])
    h = P[2] * (max(R, 1e-9) / P[1]) ** P[3]
    s = frac * h
    az = abs(z)
    if az > 4.0 * h:
        s = max(s, 0.5 * (az - 3.0 * h))
    return min(s, 0.25 * r)


@njit(inline="always", cache=True)
def node_interp(tab, x0, dx, y0, dy, x, y):
    """Bilinear interpolation on a node table tab[ix, iy] with uniform axes, clamped."""
    fx = (x - x0) / dx
    fy = (y - y0) / dy
    nx, ny = tab.shape[0], tab.shape[1]
    if fx <= 0.0:
        ix, tx = 0, 0.0
    elif fx >= nx - 1:
        ix, tx = nx - 2, 1.0
    else:
        ix = int(fx)
        tx = fx - ix
    if fy <= 0.0:
        iy, ty = 0, 0.0
    elif fy >= ny - 1:
        iy, ty = ny - 2, 1.0
    else:
        iy = int(fy)
        ty = fy - iy
    return ((1 - tx) * ((1 - ty) * tab[ix, iy] + ty * tab[ix, iy + 1])
            + tx * ((1 - ty) * tab[ix + 1, iy] + ty * tab[ix + 1, iy + 1]))


@njit(inline="always", cache=True)
def cell_weights(x0, dx, n, x):
    """(index, fraction) for linear interpolation between cell centres x0 + (i + 0.5) dx."""
    f = (x - x0) / dx - 0.5
    if f <= 0.0:
        return 0, 0.0
    if f >= n - 1:
        return n - 2, 1.0
    i = int(f)
    return i, f - i


@njit(inline="always", cache=True)
def scatter_dir(ux, uy, uz, cost, phi):
    sint = np.sqrt(max(0.0, 1.0 - cost * cost))
    cp, sp = np.cos(phi), np.sin(phi)
    if abs(uz) > 0.99999:
        return sint * cp, sint * sp, cost * (1.0 if uz > 0 else -1.0)
    t = np.sqrt(1.0 - uz * uz)
    nx = sint * (ux * uz * cp - uy * sp) / t + ux * cost
    ny = sint * (uy * uz * cp + ux * sp) / t + uy * cost
    nz = -sint * cp * t + uz * cost
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / n, ny / n, nz / n
