"""Camera-ray integration through the axisymmetric disk (numba).

Along each ray (camera at c, unit direction d, disk frame, AU) the radiance per bin is
  I_b = sum over steps of S_b (e^-tau_b(s0) - e^-tau_b(s1)),  tau_b = k_b N(s), N = int rho ds
with the source function at the step midpoint
  S_b = albedo_b [ F*_b e^{-k_b N*(r, mu)} Phi_b(cos Theta) / 4pi + J_b + 3 g_b H_b . n ]_+
        + (1 - albedo_b) B_b(T)
where F*_b = L_b / 4pi r^2 is the unattenuated stellar flux, N*(r, mu, phi) the radial column
from the star, Theta the angle between the stellar ray and the direction toward the camera
(n = -d) and J, H the Monte Carlo scattered field (mcscat). The column N along the ray is
returned so the caller can attenuate the background sky and add the star (point source) itself.

Tables carry a leading phi axis (length 1 for axisymmetric models): ncol, temp (nphi_t, n_r,
n_mu) on nodes phi = tphi0 + j tdphi; J, HR, HZ, HP (nb, nphi_j, nr_j, nmu_j) on cells centred
at phi = -pi + (j + 1/2) 2pi / nphi_j. The density perturbation is (PT, LD), see kernels.
"""
import numpy as np
from numba import njit, prange

from .kernels import density3, geo_step, node_interp3, cell_weights, phi_weights

HC_K_NM = 1.4387769e7  # hc/k in nm K
TWO_HC2 = 2 * 6.62607015e-34 * 2.99792458e8 ** 2  # W m^2 / sr


@njit(inline="always")
def _cells4(T, b, j0, j1, tj, ia, ta, ib, tb):
    """Interpolate T[b, phi, r, mu] between cell centres (bilinear in r, mu; linear in phi)."""
    a = ((1 - ta) * ((1 - tb) * T[b, j0, ia, ib] + tb * T[b, j0, ia, ib + 1])
         + ta * ((1 - tb) * T[b, j0, ia + 1, ib] + tb * T[b, j0, ia + 1, ib + 1]))
    if tj == 0.0:
        return a
    c = ((1 - ta) * ((1 - tb) * T[b, j1, ia, ib] + tb * T[b, j1, ia, ib + 1])
         + ta * ((1 - tb) * T[b, j1, ia + 1, ib] + tb * T[b, j1, ia + 1, ib + 1]))
    return (1 - tj) * a + tj * c


@njit(parallel=True, fastmath=True)
def render_rays(cam, dirs, P, PT, LD, k, alb, g, phase, lfac, lam_nm,
                lnr0, dlnr, mu0, dmu, tphi0, tdphi, ncol, temp,
                elnr0, edlnr, emu_n, J, HR, HZ, HP, frac, dtau_max, tau_stop):
    npx = dirs.shape[0]
    nb = k.shape[0]
    nph = phase.shape[1]
    nphi_j = J.shape[1]
    nr_j = J.shape[2]
    jdphi = 2 * np.pi / nphi_j
    jphi0 = -np.pi + 0.5 * jdphi
    img = np.zeros((npx, nb), np.float32)
    colo = np.zeros(npx)
    rout = P[6] * 1.02
    kmin = k.min()
    att_live = np.exp(-tau_stop)
    for ip in prange(npx):
        dx, dy, dz = dirs[ip, 0], dirs[ip, 1], dirs[ip, 2]
        x, y, z = cam[0], cam[1], cam[2]
        N = 0.0
        att = np.ones(nb)
        acc = np.zeros(nb)
        for _step in range(200000):
            R = np.sqrt(x * x + y * y)
            r = np.sqrt(R * R + z * z)
            if r > rout and x * dx + y * dy + z * dz >= 0:
                break
            if kmin * N > tau_stop:
                break
            # the optical-depth step limit follows the most opaque band that is still visible
            # (bands already attenuated below e^-tau_stop no longer contribute)
            kact = 0.0
            for b in range(nb):
                if att[b] > att_live and k[b] > kact:
                    kact = k[b]
            s = geo_step(R, z, r, P, frac)
            xm, ym, zm = x + 0.5 * s * dx, y + 0.5 * s * dy, z + 0.5 * s * dz
            Rm = np.sqrt(xm * xm + ym * ym)
            rho = density3(xm, ym, zm, P, PT, LD)
            if rho > 0 and kact * rho * s > dtau_max:
                s = dtau_max / (kact * rho)
                xm, ym, zm = x + 0.5 * s * dx, y + 0.5 * s * dy, z + 0.5 * s * dz
                Rm = np.sqrt(xm * xm + ym * ym)
                rho = density3(xm, ym, zm, P, PT, LD)
            if rho > 0:
                dN = rho * s
                rm = np.sqrt(Rm * Rm + zm * zm)
                lr = np.log(rm)
                mum = abs(zm) / rm
                phm = np.arctan2(ym, xm)
                nst = node_interp3(ncol, lnr0, dlnr, mu0, dmu, tphi0, tdphi, lr, mum, phm)
                tm = node_interp3(temp, lnr0, dlnr, mu0, dmu, tphi0, tdphi, lr, mum, phm)
                cth = -(xm * dx + ym * dy + zm * dz) / rm
                fp = (cth + 1.0) * 0.5 * (nph - 1)
                jp = min(max(int(fp), 0), nph - 2)
                tp = min(max(fp - jp, 0.0), 1.0)
                ia, ta = cell_weights(elnr0, edlnr, nr_j, lr)
                ib, tb = cell_weights(0.0, 1.0 / emu_n, emu_n, mum)
                j0, j1, tj = phi_weights(jphi0, jdphi, nphi_j, phm)
                zs = 1.0 if zm >= 0 else -1.0
                nR = -(xm * dx + ym * dy) / Rm if Rm > 0 else 0.0
                nP = -(xm * dy - ym * dx) / Rm if Rm > 0 else 0.0
                nZ = -dz * zs
                inv_r2 = 1.0 / (rm * rm)
                for b in range(nb):
                    if att[b] < att_live:
                        continue
                    a1 = att[b] * np.exp(-k[b] * dN)
                    src = 0.0
                    tst = k[b] * nst
                    if tst < 40.0:
                        ph = (1 - tp) * phase[b, jp] + tp * phase[b, jp + 1]
                        src += lfac[b] * inv_r2 * np.exp(-tst) * ph
                    if ia >= 0:
                        jj = _cells4(J, b, j0, j1, tj, ia, ta, ib, tb)
                        hr = _cells4(HR, b, j0, j1, tj, ia, ta, ib, tb)
                        hz = _cells4(HZ, b, j0, j1, tj, ia, ta, ib, tb)
                        hp = _cells4(HP, b, j0, j1, tj, ia, ta, ib, tb)
                        ms = jj + 3.0 * g[b] * (hr * nR + hz * nZ + hp * nP)
                        if ms > 0:
                            src += ms
                    src *= alb[b]
                    if tm > 0.0:
                        xp = HC_K_NM / (lam_nm[b] * tm)
                        if xp < 600.0:
                            src += (1 - alb[b]) * TWO_HC2 / (lam_nm[b] * 1e-9) ** 5 / np.expm1(xp) * 1e-9
                    acc[b] += src * (att[b] - a1)
                    att[b] = a1
                N += dN
            x, y, z = x + s * dx, y + s * dy, z + s * dz
        for b in range(nb):
            img[ip, b] = acc[b]
        colo[ip] = N
    return img, colo


@njit(cache=True)
def column_segment(a, b, P, PT, LD, frac):
    """int rho ds (g cm^-3 AU) on the straight segment from point a to point b (AU), midpoint
    rule with the geometric step of the ray marcher (a fraction of the local scale height)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    if L <= 0.0:
        return 0.0
    dx, dy, dz = dx / L, dy / L, dz / L
    x, y, z = a[0], a[1], a[2]
    t = 0.0
    N = 0.0
    while t < L:
        R = np.sqrt(x * x + y * y)
        r = np.sqrt(R * R + z * z)
        s = min(geo_step(R, z, r, P, frac), L - t)
        s = max(s, 1e-9)
        N += density3(x + 0.5 * s * dx, y + 0.5 * s * dy, z + 0.5 * s * dz, P, PT, LD) * s
        x, y, z = x + s * dx, y + s * dy, z + s * dz
        t += s
    return N
