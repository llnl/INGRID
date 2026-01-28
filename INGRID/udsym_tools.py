import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve


class PsinExtender:
    """Class for extending psin from the lower half of a magnetic equilibrium to the upper half by matching field lines on the outer/inner side."""

    def __init__(self, eqdsk_data: dict, r_magx: float = None, z_magx: float = None):
        """Initialise

        :param eqdsk_data: Magnetic equilibrium data
        """
        self.eqdsk = eqdsk_data
        self.r = self.eqdsk["r_grid"]
        self.z = self.eqdsk["z_grid"]
        self.psin = (self.eqdsk["psi"] - self.eqdsk["simagx"]) / (self.eqdsk["sibdry"] - self.eqdsk["simagx"])
        psin_shape = self.psin.shape
        self.num_x = psin_shape[0]
        self.num_y = psin_shape[1]
        ix_magx, iy_magx = np.unravel_index(self.eqdsk["psi"].argmin(), self.eqdsk["psi"].shape)
        self.ix_magx = ix_magx 
        self.iy_magx = iy_magx + 1
        self.r_magx = self.r[self.ix_magx,0]
        self.z_magx = self.z[0,self.iy_magx]

    def _get_monotonic_section(self, arr: np.ndarray) -> int:
        """Find the monotonically increasing region of an input array which starts from the first element

        :param arr: Input array
        :return: Index of the last element on the monotonically increasing region
        """
        for i in range(1, len(arr)):
            if arr[i] < arr[i - 1]:
                max_idx = i
                break
            else:
                max_idx = i + 1
        return max_idx

    def extend_psi(self):
        """Extend psi from the middle of the domain into the upper half by tracing contours of psin from the inner to the outer side"""
        r_modified = self.r.copy()
        z_modified = self.z.copy()
        psin_modified = self.psin.copy()

        num_x = psin_modified.shape[0]
        num_y = psin_modified.shape[1]

        r_mid, psin_mid = self._get_mid_arrays(r_modified, psin_modified, self.iy_magx)

        self._interpfunc_r_psin_in, self._interpfunc_r_psin_out = (
            self._get_interp_funcs_r_psin(r_mid, psin_mid, self.ix_magx)
        )
        self._interpfunc_psin_r_in, self._interpfunc_psin_r_out = (
            self._get_interp_funcs_psin_r(r_mid, psin_mid, self.ix_magx)
        )

        # Fill the upper half of psin array from psin_2D
        for ix in range(num_x):
            for jy in range(self.iy_magx, num_y):
                rval = r_modified[ix, jy]
                zval = z_modified[ix, jy]
                rhoval = np.sqrt((rval - self.r_magx) ** 2 + (zval - self.z_magx) ** 2)
                thetaval = np.arctan2(zval - self.z_magx, rval - self.r_magx)
                thetaval = np.pi - thetaval  # -using zero at imid, pi at omid
                psin_guess = np.clip(
                    self._psin_2d_simple(rhoval, thetaval),
                    np.min(psin_modified),
                    np.max(psin_modified),
                )
                psin_modified[ix, jy] = psin_guess
                # psinnval = self._psin_2d(rhoval, thetaval, psin_guess)
                # psin_modified[ix, jy] = psinnval

        # Save the modified arrays
        self.r_modified = r_modified
        self.z_modified = z_modified
        self.psin_modified = psin_modified
        
        self._update_eqdsk_data()
    
    def _update_eqdsk_data(self):
        """Apply changes to R, Z and psi in the input eqdsk data"""
        self.eqdsk["r_grid"] = self.r_modified
        self.eqdsk["z_grid"] = self.z_modified
        self.eqdsk["psi"] = self.psin_modified * (self.eqdsk["sibdry"] - self.eqdsk["simagx"]) + self.eqdsk["simagx"]

    def _get_mid_arrays(self, r: np.ndarray, psin: np.ndarray, idx_magx: int) -> np.ndarray:
        """Get arrays of R and psin in the middle of the Z domain

        :param r: R coordinate array
        :param psin: psin values array
        :return: Arrays (R, psin)
        """

        r_mid = r[:, idx_magx]
        psin_mid = psin[:, idx_magx]

        return r_mid, psin_mid

    def _get_outer_r_psin(
        self, r_mid: np.ndarray, psin_mid: np.ndarray, idx_magx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get R and psin arrays from the magnetic axis to the outer midplane

        :param r_mid: Array of R values along the middle of the Z domain
        :param psin_mid: Array of psin values along the middle of the Z domain
        :param idx_magx: Index of the middle of the Z domain
        :return: Arrays (R, psin)
        """
        r_out = r_mid[idx_magx : self.num_x - 1] - r_mid[idx_magx]
        psin_out = psin_mid[idx_magx : self.num_x - 1] - psin_mid[idx_magx]
        return r_out, psin_out

    def _get_inner_r_psin(
        self, r_mid: np.ndarray, psin_mid: np.ndarray, idx_magx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get R and psin arrays from the magnetic axis to the inner midplane (arrays will be reversed so psin is increasing)

        :param r_mid: Array of R values along the middle of the Z domain
        :param psin_mid: Array of psin values along the middle of the Z domain
        :param idx_magx: Index of the middle of the Z domain
        :return: Arrays (R, psin)
        """
        r_in = -(r_mid[0 : idx_magx + 1] - r_mid[idx_magx])
        psin_in = psin_mid[0 : idx_magx + 1] - psin_mid[idx_magx]
        psin_in = psin_in[::-1]  # reverse order, so it should (monotonically) increase
        r_in = r_in[::-1]
        return r_in, psin_in

    def _get_interp_funcs_r_psin(
        self, r_mid: np.ndarray, psin_mid: np.ndarray, idx_magx: int
    ) -> tuple[interpolate.CubicSpline, interpolate.CubicSpline]:
        """Get functions which interpolate R as a function of psin, separately for the inner and outer sides

        :param r_mid: Array of R values along the middle of the Z domain
        :param psin_mid: Array of psin values along the middle of the Z domain
        :param idx_magx: Index of the middle of the Z domain
        :return: Two interpolation functions: (inner side, outer side)
        """

        # Outer midplane part rho(psin)
        r_out, psin_out = self._get_outer_r_psin(r_mid, psin_mid, idx_magx)
        interpfunc_out = interpolate.CubicSpline(psin_out, r_out)

        # Inner midplane part rho(psin)
        r_in, psin_in = self._get_inner_r_psin(r_mid, psin_mid, idx_magx)
        max_idx = self._get_monotonic_section(psin_in)
        interpfunc_in = interpolate.CubicSpline(psin_in[:max_idx], r_in[:max_idx])

        return interpfunc_in, interpfunc_out

    def _get_interp_funcs_psin_r(
        self, r_mid: np.ndarray, psin_mid: np.ndarray, idx_magx: int
    ) -> tuple[interpolate.CubicSpline, interpolate.CubicSpline]:
        """Get functions which interpolate psin as a function of R, separately for the inner and outer sides

        :param r_mid: Array of R values along the middle of the Z domain
        :param psin_mid: Array of psin values along the middle of the Z domain
        :param idx_magx: Index of the middle of the Z domain
        :return: Two interpolation functions: (inner side, outer side)
        """

        # Outer midplane part rho(psin)
        r_out, psin_out = self._get_outer_r_psin(r_mid, psin_mid, idx_magx)
        interpfunc_out = interpolate.CubicSpline(r_out, psin_out)

        # Inner midplane part rho(psin)
        r_in, psin_in = self._get_inner_r_psin(r_mid, psin_mid, idx_magx)
        interpfunc_in = interpolate.CubicSpline(r_in, psin_in)

        return interpfunc_in, interpfunc_out

    def _rho_2d(self, psin: float, theta: float) -> float:
        """Find coordinate rho at a given location (psin, theta)

        :param psin: psin
        :param theta: Theta coordinate centred on magnetic axis
        :return: Rho
        """
        # Define rho(psin,theta) in the upper plane
        rho_in = self._interpfunc_r_psin_in(psin)
        rho_out = self._interpfunc_r_psin_out(psin)
        res = rho_in + (rho_out - rho_in) * (theta / np.pi)
        return res

    def _eqn(self, psin: float, *data: list[float]) -> float:
        """Equation to be solved to find psin at a given location (rho, theta)

        :param psin: psin
        :return: Equation result
        """
        rho, theta = data
        res = self._rho_2d(psin, theta) - rho
        return res

    def _psin_2d(self, rho: float, theta: float, psin_0: float) -> float:
        """Solve for psin the non-linear equation, rho_2d(psin, theta) = rho

        :param rho: Radial coordinate centred on magnetic axis
        :param theta: Theta coordinated centred on magnetic axis
        :return: Value of psin at coords (rho, theta)
        """
        data = (rho, theta)
        psin = fsolve(self._eqn, psin_0, args=data)
        return psin

    def _psin_2d_simple(self, rhoval, thetaval):
        """Solve for psin by interpolating between psin at the same rho either side of the midplane

        :param rho: Radial coordinate centred on magnetic axis
        :param theta: Theta coordinated centred on magnetic axis
        :return: Value of psin at coords (rho, theta)
        """
        psin_in = self._interpfunc_psin_r_in(rhoval)
        psin_out = self._interpfunc_psin_r_out(rhoval)
        psinnval = (psin_in * (np.pi - thetaval) + psin_out * thetaval) / np.pi
        return psinnval
