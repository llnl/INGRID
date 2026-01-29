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


def psi_analytic3(x, y, p1, p2, p3, i1, i2, i3):
    """_summary_

    :param x: _description_
    :param y: _description_
    :param p1: _description_
    :param p2: _description_
    :param p3: _description_
    :param i1: _description_
    :param i2: _description_
    :param i3: _description_
    :return: _description_
    """
    psi = i1 * psi_analytic(x, y, p1[0], p1[1])
    psi += i2 * psi_analytic(x, y, p2[0], p2[1])
    psi += i3 * psi_analytic(x, y, p3[0], p3[1])

    return psi


