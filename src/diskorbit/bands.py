"""Non-contiguous broad bands (IR / mm) with the SpectralGrid interface the renderer uses
(wl, dwl, n, edges-free subsample, bin_index), plus the false-colour band sets."""
from dataclasses import dataclass

import numpy as np

# band centres (micron); each band spans centre * (1 -+ HALF_WIDTH)
IR_BANDS_UM = (1.25, 1.65, 2.2, 4.5, 10.0, 24.0, 70.0, 160.0, 1300.0)
HALF_WIDTH = 0.1

# false-colour sets: (blue, green, red) band centres in micron
COLOUR_SETS = {
    "nir": (1.25, 1.65, 2.2),
    "mir": (4.5, 10.0, 24.0),
    "fir": (24.0, 70.0, 160.0),
}
MM_BAND_UM = 1300.0


@dataclass(frozen=True)
class BandGrid:
    lo: np.ndarray  # (n,) nm
    hi: np.ndarray  # (n,) nm

    @property
    def wl(self):
        return np.sqrt(self.lo * self.hi)

    @property
    def dwl(self):
        return self.hi - self.lo

    @property
    def n(self):
        return len(self.lo)

    def bin_index(self, wavelength_nm):
        i = np.nonzero((self.lo <= wavelength_nm) & (wavelength_nm < self.hi))[0]
        return int(i[0]) if len(i) else -1

    def subsample(self, per_bin=21):
        """(n, per_bin) wavelengths log-uniformly spanning each band."""
        f = (np.arange(per_bin) + 0.5) / per_bin
        return np.exp(np.log(self.lo)[:, None] + f[None, :] * np.log(self.hi / self.lo)[:, None])


def ir_grid(bands_um=IR_BANDS_UM, half_width=HALF_WIDTH):
    c = np.asarray(bands_um, float) * 1000.0
    return BandGrid(c * (1 - half_width), c * (1 + half_width))


def band_index(grid, um):
    return int(np.argmin(np.abs(grid.wl - um * 1000.0)))
