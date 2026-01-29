import numpy as np
import freeqdsk
import contourpy
from matplotlib.patches import Polygon

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

def find_psi_boundaries(r, z, r_grid, z_grid, rmagx, zmagx, psi, n_candidates = 1000):
    """Attempt to find psi at the magnetic axis and the primary separatrix for the generated psi.
    The approach is to loop through values of psi (in order) from psi at the magnetic axis to 
    max(psi). Next we identify the spatial of coordinates contours at that value of psi. The 
    criteria for identifying the separatrix is that it is the last value of psi for which a) a 
    contour encloses the magnetic axis and b) it is a closed loop. 

    :param r: r-coordinates
    :param z: z-coordintaes
    :param r_grid: r-grid
    :param z_grid: z-grid
    :param rmagx: r-coordinate of magx
    :param zmagx: z-coordinate of magx
    :param psi: psi
    :return: psi_magx, psi_sepx
    """
    ix_magx = abs(r - rmagx).argmin()
    iy_magx = abs(z - zmagx).argmin()
    psi_magx = psi[ix_magx,iy_magx]
    candidate_psis = np.linspace(psi_magx+0.01*(psi.max()-psi_magx), psi.max(), n_candidates)

    cg = contourpy.contour_generator(x=r_grid, y=z_grid, z=psi)

    psi_sepx = None
    for candidate_psi in candidate_psis:
        cg = contourpy.contour_generator(x=r_grid, y=z_grid, z=psi)
        l = cg.lines(candidate_psi)

        contours_closed = []
        contains_magx = []

        # Check for closed contours
        for cl in l:
            contours_closed.append(all(cl[0] == cl[-1]))

        if any(contours_closed):
            # Check whether the magnetic axis is within each contour
            for i, cl in enumerate(l):
                if contours_closed[i] is True:
                    sepx_candidate = Polygon(np.array([cl[:,0], cl[:,1]]).transpose(),
                        closed=True,
                        facecolor="white",
                        edgecolor="none"
                        )
                    contains_magx.append(sepx_candidate.contains_point(((rmagx, zmagx))))    

        if any(contours_closed) and any(contains_magx):
            continue
        else:
            psi_sepx =  candidate_psi
            break
    
    if psi_sepx is None:
        psi_sepx = candidate_psis[-1]

    return psi_magx, psi_sepx
