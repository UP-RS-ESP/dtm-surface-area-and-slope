import numpy as np
from scipy.integrate import dblquad
from tqdm import trange


def synthetic_landscape(x, y, landscape_type, amplitude):
    if landscape_type == "hill":
        dtm = np.exp((x * x + y * y) / -100000)
    elif landscape_type == "sincos":
        dtm = (
            np.cos(np.pi * x / 100) * np.sin(np.pi * y / 100)
            + np.cos(np.pi * x / 10) * np.sin(np.pi * y / 10) / 10
        )
    else:
        raise ValueError('Unsupported landscape type. Use "sincos" or "hill".')
    return dtm * amplitude


def synthetic_landscape_slope(x, y, landscape_type, amplitude):
    if landscape_type == "hill":
        dtm = amplitude * np.exp((x * x + y * y) / -100000)
        dx = 2 * x * dtm / -100000
        dy = 2 * y * dtm / -100000
    elif landscape_type == "sincos":
        dx = (
            -amplitude
            * np.pi
            / 100
            * (
                np.sin(np.pi * x / 100) * np.sin(np.pi * y / 100)
                + np.sin(np.pi * x / 10) * np.sin(np.pi * y / 10)
            )
        )
        dy = (
            amplitude
            * np.pi
            / 100
            * (
                np.cos(np.pi * x / 100) * np.cos(np.pi * y / 100)
                + np.cos(np.pi * x / 10) * np.cos(np.pi * y / 10)
            )
        )
    else:
        raise ValueError('Unsupported landscape type. Use "sincos" or "hill".')
    return np.arctan(np.sqrt(dx * dx + dy * dy))


def surface_area_tin(dtm, resolution):
    def _heronsa(a, b, c):
        s = (a + b + c) / 2.0
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    dzx = dtm[:, 1:] - dtm[:, :-1]
    dzy = dtm[1:, :] - dtm[:-1, :]
    dzd = dtm[1:, 1:] - dtm[:-1, :-1]
    dzb = dtm[:-1, 1:] - dtm[1:, :-1]
    dzx = np.sqrt(resolution * resolution + dzx * dzx) / 2.0
    dzy = np.sqrt(resolution * resolution + dzy * dzy) / 2.0
    dzd = np.sqrt(2 * resolution * resolution + dzd * dzd) / 2.0
    dzb = np.sqrt(2 * resolution * resolution + dzb * dzb) / 2.0
    a = np.zeros((dtm.shape[0] - 2, dtm.shape[1] - 2))
    a += _heronsa(dzx[1:-1, :-1], dzy[:-1, :-2], dzd[:-1, :-1])
    a += _heronsa(dzx[:-2, :-1], dzy[:-1, 1:-1], dzd[:-1, :-1])
    a += _heronsa(dzx[:-2, 1:], dzy[:-1, 1:-1], dzb[:-1, 1:])
    a += _heronsa(dzx[1:-1, 1:], dzy[:-1, 2:], dzb[:-1, 1:])
    a += _heronsa(dzx[1:-1, :-1], dzy[1:, :-2], dzb[1:, :-1])
    a += _heronsa(dzx[2:, :-1], dzy[1:, 1:-1], dzb[1:, :-1])
    a += _heronsa(dzx[2:, 1:], dzy[1:, 1:-1], dzd[1:, 1:])
    a += _heronsa(dzx[1:-1, 1:], dzy[1:, 2:], dzd[1:, 1:])
    return np.pad(a, 1, constant_values=np.nan) / resolution / resolution


def surface_area_cos(dtm, resolution):
    dy, dx = np.gradient(dtm, resolution)
    cos = np.cos(np.arctan(np.sqrt(dx * dx + dy * dy)))
    a = np.ones((dtm.shape[0], dtm.shape[1]))
    a *= resolution * resolution
    return a / cos


def synthetic_landscape_surface_area(x, y, landscape_type, amplitude):
    if landscape_type == "hill":

        def da(x, y, amplitude):
            dtm = amplitude * np.exp((x * x + y * y) / -100000)
            dx = 2 * x * dtm / -100000
            dy = 2 * y * dtm / -100000
            return np.sqrt(1 + dx * dx + dy * dy)

    elif landscape_type == "sincos":

        def da(x, y, amplitude):
            dx = (
                -amplitude
                * np.pi
                / 100
                * (
                    np.sin(np.pi * x / 100) * np.sin(np.pi * y / 100)
                    + np.sin(np.pi * x / 10) * np.sin(np.pi * y / 10)
                )
            )
            dy = (
                amplitude
                * np.pi
                / 100
                * (
                    np.cos(np.pi * x / 100) * np.cos(np.pi * y / 100)
                    + np.cos(np.pi * x / 10) * np.cos(np.pi * y / 10)
                )
            )
            return np.sqrt(1 + dx * dx + dy * dy)

    else:
        raise ValueError('Unsupported landscape type. Use "sincos" or "hill".')

    # numerical integration for each pixel
    a = np.zeros(x.shape)
    w = y[1, 0] - y[0, 0]
    v = w / 2
    for i in trange(x.shape[0], ncols=70):
        for k in range(x.shape[1]):
            xik, yik = x[i, k], y[i, k]
            aik, err = dblquad(
                da,
                xik - v,
                xik + v,
                lambda x: yik - v,
                lambda x: yik + v,
                args=(amplitude,),
            )
            a[i, k] = aik
    return a
