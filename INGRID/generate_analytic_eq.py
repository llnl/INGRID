import numpy as np
import freeqdsk

def psi_analytic(x, y, x0, y0, a0=0.1):

    dist_sq = (x - x0) ** 2 + (y - y0) ** 2
    dist = np.sqrt(dist_sq)
    a0_sq = a0**2
    res = np.zeros(x.shape)
    res = np.log(dist)
    res[np.where(dist_sq < a0_sq)] = 0.5 * (
        dist_sq[np.where(dist_sq < a0_sq)] / a0_sq - 1.0
    ) + np.log(a0)
    return res


def psi_analytic3(x: np.ndarray, y: np.ndarray, p1: tuple, p2: tuple, p3: tuple, i1: float, i2: float, i3: float):
    """Generate analytic psi from three currents

    :param x: x-coorindates
    :param y: y-coordinates
    :param p1: position of current filament 1
    :param p2: position of current filament 2
    :param p3: position of current filament 2
    :param i1: current through filament 1
    :param i2: current through filament 2
    :param i3: current through filament 3
    :return: psi
    """
    psi = i1 * psi_analytic(x, y, p1[0], p1[1])
    psi += i2 * psi_analytic(x, y, p2[0], p2[1])
    psi += i3 * psi_analytic(x, y, p3[0], p3[1])

    return psi


