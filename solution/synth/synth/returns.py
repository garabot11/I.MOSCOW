"""Return model: detection probability, range noise / quantisation and intensity.

Detection is a logistic in ln S with S = rho * cos(theta) * f / r^2 (rho Lambertian reflectivity, theta incidence
angle, f fraction of the beam footprint on the surface, r range):

    P(S) = 1 / (1 + exp(-(k * (ln S - ln S90) + ln 9)))       S90 = 0.1 / R10^2

so that 10 % reflectivity at the rated range R10 (Hesai 200 m, Falcon 250 m, datasheets) gives P = 0.9.  The
steepness k is fitted to the real per-ring no-return fraction (``tools/calibrate_returns.py``).  Returns beyond
``max_range_m`` never occur (Hesai: no real return beyond ~209 m).  Range noise is Gaussian (sigma) followed by
quantisation (Hesai 4 mm measured, Falcon 5 mm datasheet).  Intensity = scale * rho * cos(theta)^g * exp(noise),
rounded to integers >= 1; scale and noise are fitted to real per-surface histograms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ReturnModel:
    range_10pct_m: float = 200.0
    steepness: float = 3.0
    max_range_m: float = 209.0
    min_range_m: float = 0.5
    sigma_m: float = 0.02
    quant_m: float = 0.004
    detect_cos_exp: float = 1.0       # S uses cos(theta)^detect_cos_exp (1 = Lambertian; real walls return at grazing incidence)
    dropout: float = 0.0              # random loss of otherwise detected returns
    intensity_scale: float = 100.0
    intensity_cos_exp: float = 1.0
    intensity_log_sigma: float = 0.25
    intensity_range_ref_m: float = 0.0   # > 0: intensity *= min(1, (ref / r)^intensity_range_exp)
    intensity_range_exp: float = 1.0
    intensity_max: float = 255.0

    @classmethod
    def hesai(cls, **kw):
        # calibration v1 against roundT_squareT_pressureGate_squareT (tools/realism_hesai.py, results/realism_hesai.json)
        base = dict(detect_cos_exp=0.45, dropout=0.025, intensity_cos_exp=0.15, intensity_log_sigma=0.45,
                    intensity_range_ref_m=60.0, intensity_range_exp=1.3)
        base.update(kw)
        return cls(**base)

    @classmethod
    def falcon(cls, **kw):
        # datasheet: 250 m at 10 % (POD > 90 %), 1.5-500 m, precision 2 cm, resolution 0.5 cm; detection and
        # intensity shape as calibrated for Hesai, no intensity roll-off with range (tools/realism_falcon.py)
        base = dict(range_10pct_m=250.0, max_range_m=500.0, min_range_m=1.5, sigma_m=0.02, quant_m=0.005,
                    detect_cos_exp=0.45, dropout=0.02, intensity_cos_exp=0.15, intensity_log_sigma=0.45)
        base.update(kw)
        return cls(**base)

    def to_dict(self):
        return asdict(self)

    def p_detect(self, rho, cos_t, r, frac=1.0):
        rho = np.asarray(rho, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        r = np.where(np.isfinite(r), np.maximum(r, 1e-3), 1e9)
        S = np.maximum(rho * np.clip(cos_t, 0.0, 1.0) ** self.detect_cos_exp * frac, 1e-12) / (r * r)
        S90 = 0.1 / self.range_10pct_m ** 2
        z = self.steepness * (np.log(S) - np.log(S90)) + np.log(9.0)
        p = (1.0 - self.dropout) / (1.0 + np.exp(-np.clip(z, -50, 50)))
        return np.where((r <= self.max_range_m) & (r >= self.min_range_m), p, 0.0)

    def detect(self, rng, rho, cos_t, r, frac=1.0):
        p = self.p_detect(rho, cos_t, r, frac)
        return rng.random(np.shape(p)) < p

    def measure_range(self, rng, r):
        r = np.asarray(r, dtype=np.float64)
        noisy = r + rng.normal(0.0, self.sigma_m, r.shape)
        return np.round(noisy / self.quant_m) * self.quant_m

    def intensity(self, rng, rho, cos_t, r=None):
        rho = np.asarray(rho, dtype=np.float64)
        base = self.intensity_scale * rho * np.clip(cos_t, 0.0, 1.0) ** self.intensity_cos_exp
        if self.intensity_range_ref_m > 0 and r is not None:
            rr = np.maximum(np.asarray(r, dtype=np.float64), 1e-3)
            base = base * np.minimum(1.0, (self.intensity_range_ref_m / rr) ** self.intensity_range_exp)
        val = base * np.exp(rng.normal(0.0, self.intensity_log_sigma, np.shape(base)))
        return np.clip(np.round(val), 1.0, self.intensity_max).astype(np.float32)
