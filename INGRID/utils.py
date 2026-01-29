#!/usr/bin/env pythonu
# -*- coding: utf-8 -*-
"""Helper classes for the ingrid module and topologies package.

This module contains classes `IngridUtils` and `TopologyUtils`. These classes
encapsulate much of the critical methods for handling file I/O, topology analysis,
generating patch maps, and generating grids.

"""
from __future__ import print_function, division, absolute_import
import freeqdsk
import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except:
    pass
import matplotlib.pyplot as plt
import inspect
from scipy.optimize import root

from pathlib import Path
from freeqdsk import geqdsk

from INGRID.interpol import EfitData
from INGRID.line_tracing import LineTracing
from INGRID.geometry import Point, Line, Patch, orientation_between
from INGRID.udsym_tools import PsinExtender
from INGRID.generate_analytic_eq import psi_analytic3, find_psi_boundaries

class IngridUtils:
    """
    The base class for :class:`ingrid.Ingrid` that handles backend management of key
    Ingrid capabilities. This class can be directly utilized by advanced users
    and developers of the Ingrid code.

    Class `IngridUtils` encapsulates implementation details of file I/O,
    keyword parsing, sorting of geometry, and managing of other helper classes
    such as :class:`line_tracing.LineTracing` and :class:`interpol.EfitData`.

    Parameters
    ----------
    settings : optional
        Dictionary representation of the settings file.
        The dictionary is obtained via a YAML dump while operating in gui
        mode. Providing no dictionary will populate the Ingrid object
        with a default value settings attribute. Any missing entries
        provided by a user will be populated with default values by Ingrid.
        (Refer to the Ingrid "Settings File Format" for a complete descri
        ption of the settings dictionary)

    **kwargs :
        Keyword arguments for processing of input data.
        (See method 'ProcessKeywords' in class 'IngridUtils' for a
        list of supported keywords)


    Attributes
    ----------
    InputFile : str
        Path to the parameter file.

    config : str
        The configuration of the topology.

    settings : dict
        Core settings dictionary containing all data used for generating patches
        and grids.

    settings_lookup : dict
        Top level entries of the parameter file and convenience attribute for
        later accessing of entries.

    default_values_lookup : dict
        General structure of YAML settings file in dictionary form.

    default_grid_settings : dict
        Default entries for key `grid_settings` in the YAML settings file.

    default_integrator_settings : dict
        Default entries for key `integrator_settings` in the YAML settings file.

    default_target_plate_settings : dict
        Default entries for key `target_plates` in the YAML settings file.

    default_limiter_settings : dict
        Default entries for key `limiter` in the YAML settings file.

    default_patch_data_settings : dict
        Default entries for key `patch_data` in the YAML settings file.

    default_DEBUG_settings : dict
        Default entries for key `DEBUG` in the YAML settings file.

    PlateData : dict
        Dictionary containing Line objects that correspond to target plates
        loaded by the user.

    PsiUNorm : EfitData
        `EfitData` class object containing unormalized
        psi data from provided neqdsk file in settings.

    LimiterData : Line
        Line class object representing tokamak limiter.

    magx : tuple
        (R, Z) coordinates of magnetic-axis (float entries).

    xpt1 : tuple
        (R, Z) coordinates of primary x-point (float entries).

    xpt2 : tuple
        (R, Z) coordinates of secondary x-point (float entries).

    LineTracer : LineTracing
        `LineTracing` instance that for topology analysis and poloidal/radial tracing.

    """

    def __init__(self, settings: dict = {}, **kwargs):
        self.InputFile = None
        self.config = None
        self.settings_lookup = [
            "dir_settings",
            "grid_settings",
            "integrator_settings",
            "target_plates",
            "DEBUG",
        ]
        self.SetDefaultSettings()
        self.PopulateSettings(settings)
        self.ProcessKeywords(**kwargs)

    def get_config(self) -> str:
        """
        Get the configuration obtained during analysis of x-points.

        Parameters
        ----------


        Returns
        -------
            A string identifying the configuration contained within LineTracing attribute `LineTracer`.
        """
        return self.LineTracer.config

    def SetDefaultSettings(self) -> None:
        """
        Set all default values that will populate the ``settings`` dict.

        This instantiates the following entries within the settings file:

        - 'grid_settings'
        - 'integrator_settings'
        - 'target_plates'
        - 'limiter'
        - 'patch_data'
        - 'analytic_equilibrium_generation'
        - 'DEBUG'

        Additional entries may be added here as development continues.
        """

        self.default_grid_settings = {
            "num_xpt": 1,
            "nlevs": 30,
            "view_mode": "filled",
            "psi_1": 0.0,
            "psi_core": 0.0,
            "psi_pf_1": 0.0,
            "psi_pf_2": 0.0,
            "psi_1": 0.0,
            "psi_2": 0.0,
            "rmagx": 0.0,
            "zmagx": 0.0,
            "rxpt": 0.0,
            "zxpt": 0.0,
            "rxpt2": 0.0,
            "zxpt2": 0.0,
            "guard_cell_eps": 1.0e-3,
            "grid_generation": {
                "np_default": 2,
                "nr_default": 2,
                "poloidal_f_default": "x, x",
                "radial_f_default": "x, x",
                "distortion_correction": {
                    "all": {
                        "theta_min": 80.0,
                        "theta_max": 120.0,
                        "resolution": 1000,
                        "active": False,
                    },
                },
            },
            "patch_generation": {
                "core_split_point_ratio": 0.5,
                "pf_split_point_ratio": 0.5,
                "strike_pt_loc": "limiter",
                "rmagx_shift": 0.0,
                "zmagx_shift": 0.0,
                "magx_tilt_1": 0.0,
                "magx_tilt_2": 0.0,
                "use_xpt1_W": False,
                "use_xpt1_E": False,
                "use_xpt1_N": False,
                "use_xpt1_S": False,
                "use_xpt2_W": False,
                "use_xpt2_E": False,
                "use_xpt2_N": False,
                "use_xpt2_S": False,
                "xpt1_W_tilt": -0.785398,  # All values of pi / 4 radians.
                "xpt1_E_tilt": 0.785398,
                "xpt1_N_tilt": 1.570796,
                "xpt1_S_tilt": -1.570796,
                "xpt2_W_tilt": -0.785398,
                "xpt2_E_tilt": 0.785398,
                "xpt2_N_tilt": 1.570796,
                "xpt2_S_tilt": -1.570796,
            },
            "up_down_symmetry": False,
        }

        self.default_integrator_settings = {
            "dt": 0.01,
            "eps": 5e-5,
            "first_step": 1e-5,
            "step_ratio": 0.02,
            "tol": 5e-3,
            "max_step": 0.064,
        }

        self.default_target_plate_settings = {
            "plate_E1": {"file": "", "rshift": 0.0, "zshift": 0.0},
            "plate_E2": {"file": "", "rshift": 0.0, "zshift": 0.0},
            "plate_W1": {"file": "", "rshift": 0.0, "zshift": 0.0},
            "plate_W2": {"file": "", "rshift": 0.0, "zshift": 0.0},
        }

        self.default_limiter_settings = {
            "file": "",
            "use_efit_bounds": False,
            "efit_buffer_r": 1e-2,
            "efit_buffer_z": 1e-2,
            "rshift": 0.0,
            "zshift": 0.0,
        }

        self.default_patch_data_settings = {
            "file": "",
            "use_file": False,
            "preferences": {"new_file": False, "new_fname": ""},
        }

        self.default_DEBUG_settings = {
            "visual": {
                "find_NSEW": False,
                "patch_map": False,
                "subgrid": False,
                "gridue": False,
                "SF_analysis": False,
            },
            "verbose": {
                "patch_generation": False,
                "grid_generation": False,
                "SF_analysis": False,
            },
        }

        self.default_dir_settings = {
            "eqdsk": ".",
            "limiter": ".",
            "patch_data": ".",
            "target_plates": ".",
        }

        self.analytic_equilibrium_generation_settings = {
            "use": False,
            "nx": 128,
            "ny": 128,
            "r_i1": 1.0,
            "z_i1": 0.0,
            "r_i2": 0.6,
            "z_i2": -1.5,
            "r_i3": 1.4,
            "z_i3": -1.5,
            "bcentr": -1.0,
            "i1": 0.40,
            "i2": 0.12,
            "i3": 0.16,
        }

        self.default_values_lookup = {
            "eqdsk": "",
            "dir_settings": self.default_dir_settings,
            "grid_settings": self.default_grid_settings,
            "integrator_settings": self.default_integrator_settings,
            "target_plates": self.default_target_plate_settings,
            "limiter": self.default_limiter_settings,
            "patch_data": self.default_patch_data_settings,
            "analytic_equilibrium_generation": self.analytic_equilibrium_generation_settings,
            "DEBUG": self.default_DEBUG_settings,
        }

        self.PlateData = {
            "plate_W1": {},
            "plate_E1": {},
            "plate_W2": {},
            "plate_E2": {},
        }

    def ProcessKeywords(self, **kwargs) -> None:
        """
        Process kwargs and set all file paths accordingly.
        """
        for k, v in kwargs.items():
            if k == "InputFile" or k == "yaml":
                print("# Processing Input File:", v)
                self.InputFile = v
                self.PopulateSettings(self.ReadYamlFile(v))
                continue
            if k == "W1TargetFile" or k == "w1":
                self.settings["target_plates"]["plate_W1"]["file"] = v
                continue
            if k == "E1TargetFile" or k == "e1":
                self.settings["target_plates"]["plate_E1"]["file"] = v
                continue
            if k == "E2TargetFile" or k == "e2":
                self.settings["target_plates"]["plate_E2"]["file"] = v
                continue
            if k == "W2TargetFile" or k == "w2":
                self.settings["target_plates"]["plate_W2"]["file"] = v
                continue
            if k in ["LimiterFile", "limiter", "wall"]:
                self.settings["limiter"]["file"] = v
            if k == "EqFile" or k == "eq":
                self.settings["eqdsk"] = v
                continue
            print('Keyword "' + k + '" unknown and ignored...')

    def PopulateSettings(self, settings: dict, verbose: bool = True) -> None:
        """
        Populate a settings dict with any missing entries that INGRID may need.

        This should be used to screen for any illegal parameter file entries
        and to ensure clean data entry.

        Parameters
        ----------
        settings : dict
            Dictionary object conforming to structure of ``settings`` dictionary

        verbose : bool, optional
            Print full output to terminal. Defaults to False.

        """

        def _check_settings_input(input_settings, comparison):
            raise_assertion = False
            items = []
            for k in input_settings.keys():
                if comparison.get(k) is None:
                    items.append(k)
                    raise_assertion = True

            if raise_assertion is True:
                raise ValueError(
                    f"Invalid entries {items} in provided settings file f{self.InputFile}. Remove invalid entries listed."
                )

        def _get_default_values(item, sub_item=None, attribute=None):
            """
            Helper function for processing the YAML file dump.
            Determines if the entry within the YAML file dump is valid and currently
            recognized by INGRID.
            """
            try:
                default_values = self.default_values_lookup[item]
            except KeyError:
                print(
                    f"Key "
                    "{item}"
                    " not recognized... Add default values to source code for support."
                )
                return None

            if item and sub_item and attribute:
                return self.default_values_lookup[item][sub_item][attribute]
            elif item and sub_item:
                return self.default_values_lookup[item][sub_item]
            elif item:
                return self.default_values_lookup[item]

        _check_settings_input(
            input_settings=settings, comparison=self.default_values_lookup
        )

        # First level entries within YAML file dump.
        for item in self.default_values_lookup.keys():

            try:
                # Check to see if the item is in the YAML dump.
                settings[item]
            except KeyError:
                settings[item] = _get_default_values(item)
                if verbose:
                    print('Could not find "{}" in YAML file.'.format(item))
                    print(
                        'Populated "{}" with default value of "{}".\n'.format(
                            item, settings[item]
                        )
                    )
                continue

            if item in ["eqdsk"]:
                continue

            _check_settings_input(
                input_settings=settings[item],
                comparison=self.default_values_lookup[item],
            )

            # Second level entries within YAML file dump.

            for sub_item in self.default_values_lookup[item].keys():
                try:
                    settings[item][sub_item]
                except KeyError:
                    settings[item][sub_item] = _get_default_values(item, sub_item)
                    if verbose:
                        print(
                            'Could not find "{}/{}" in YAML file.'.format(
                                item, sub_item
                            )
                        )
                        print(
                            'Populated "{}/{}" with default value of "{}".\n'.format(
                                item, sub_item, settings[item][sub_item]
                            )
                        )
                    continue

                if item in ["grid_settings", "target_plates"] and sub_item in [
                    "patch_generation",
                    "grid_generation",
                    "plate_E1",
                    "plate_W1",
                    "plate_E2",
                    "plate_W2",
                ]:
                    for plate_attribute in self.default_values_lookup[item][
                        sub_item
                    ].keys():
                        try:
                            if settings[item][sub_item][plate_attribute] is None:
                                settings[item][sub_item][plate_attribute] = (
                                    _get_default_values(item, sub_item, plate_attribute)
                                )
                        except:
                            settings[item][sub_item][plate_attribute] = (
                                _get_default_values(item, sub_item, plate_attribute)
                            )

        # Update references to YAML entries.
        self.settings = settings
        self.grid_settings = settings["grid_settings"]
        self.integrator_settings = settings["integrator_settings"]
        self.target_plates = settings["target_plates"]
        self.analytic_equilibrium_generation_settings = settings["analytic_equilibrium_generation"]
        self.DEBUG = settings["DEBUG"]
        self.ProcessPaths()

    def ProcessPaths(self) -> None:
        """
        Update settings by pre-pending path entries to all file entries.
        """
        for k in self.default_dir_settings.keys():
            path_obj = Path(self.settings["dir_settings"][k])
            if path_obj.is_dir() is False:
                v = self.settings["dir_settings"][k]
                raise ValueError(
                    f"# Error processing directory provided for entry '{k}'. Entry '{v}' is not a valid directory."
                )

            if k == "eqdsk":
                self.settings["eqdsk"] = str(
                    (path_obj / self.settings["eqdsk"]).absolute()
                )
                continue
            if k == "limiter":
                self.settings["limiter"]["file"] = str(
                    (path_obj / self.settings["limiter"]["file"]).absolute()
                )
                continue
            if k == "target_plates":
                self.settings["target_plates"]["plate_W1"]["file"] = str(
                    (
                        path_obj / self.settings["target_plates"]["plate_W1"]["file"]
                    ).absolute()
                )
                self.settings["target_plates"]["plate_E1"]["file"] = str(
                    (
                        path_obj / self.settings["target_plates"]["plate_E1"]["file"]
                    ).absolute()
                )
                self.settings["target_plates"]["plate_W2"]["file"] = str(
                    (
                        path_obj / self.settings["target_plates"]["plate_W2"]["file"]
                    ).absolute()
                )
                self.settings["target_plates"]["plate_E2"]["file"] = str(
                    (
                        path_obj / self.settings["target_plates"]["plate_E2"]["file"]
                    ).absolute()
                )
                continue
            if k == "patch_data":
                self.settings["patch_data"]["file"] = str(
                    (path_obj / self.settings["patch_data"]["file"]).absolute()
                )
                self.settings["patch_data"]["preferences"]["new_fname"] = str(
                    (
                        path_obj
                        / self.settings["patch_data"]["preferences"]["new_fname"]
                    ).absolute()
                )
                continue

    def GenerateAnalyticEquilibrium(self, generate_eq_settings: dict) -> freeqdsk.geqdsk.GeqdskDataDict:
        """Use the settings provided in the input file to generate an analytic snowflake equilibrium using a simple three-current model

        :param generate_eq_settings: Settings for analytic equilibrium generation
        :return: geqdsk dictionary
        """
        # User-specified parameters
        nx = generate_eq_settings["nx"] 
        ny = generate_eq_settings["ny"] 
        p1 = (generate_eq_settings["r_i1"],generate_eq_settings["z_i1"])
        p2 = (generate_eq_settings["r_i2"],generate_eq_settings["z_i2"])
        p3 = (generate_eq_settings["r_i3"],generate_eq_settings["z_i3"])
        bcentr = generate_eq_settings["bcentr"] 
        i1 =  generate_eq_settings["i1"] 
        i2 =  generate_eq_settings["i2"] 
        i3 =  generate_eq_settings["i3"] 

        # Derived parameters
        rleft = min([p1[0], p2[0], p3[0]]) - (max([p1[0], p2[0], p3[0]]) - min([p1[0], p2[0], p3[0]]))
        rdim = 3*(max([p1[0], p2[0], p3[0]]) - min([p1[0], p2[0], p3[0]]))
        zmid = p1[1]
        zdim = 2.5*(zmid-min([p1[1], p2[1], p3[1]]))
        rmagx = p1[0]
        zmagx = p1[1]
        rcentr = rmagx

        # Generate coordinates
        rmin = rleft
        rmax = rmin + rdim
        zmin = zmid - 0.5 * zdim
        zmax = zmin + zdim
        r = np.linspace(rmin,rmax, nx)
        z = np.linspace(zmin,zmax, ny)
        r_grid, z_grid = np.meshgrid(r, z)
        r_grid = r_grid.T
        z_grid = z_grid.T
        rlim = np.array([rmin, rmax, rmax, rmin])
        zlim = np.array([zmax, zmax, zmin, zmin])

        # Generate psi
        psi = psi_analytic3(r_grid, z_grid, p1, p2, p3, i1, i2, i3)

        try:
            psi_magx, psi_sepx = find_psi_boundaries(r, z, r_grid, z_grid, rmagx, zmagx, psi)
        except:
            psi_magx = None 
            psi_sepx = None

        # Generate the geqdsk dictionary
        # Generate the geqdsk file
        geqdsk_data = freeqdsk.geqdsk.GeqdskDataDict(
            nx=nx, 
            ny=ny, 
            rdim=rdim, 
            zdim=zdim, 
            rleft=rleft, 
            zmid=zmid, 
            rmagx=rmagx,
            zmagx=zmagx,
            r_grid=r_grid, 
            z_grid=z_grid,
            psi=psi,
            bcentr=bcentr,
            rlim=rlim,
            zlim=zlim,
            rcentr=rcentr,
            simagx=psi_magx,
            sibdry=psi_sepx,
        )

        return geqdsk_data

    
    def LoadGEQDSK(self, geqdsk_path: str, up_down_symmetry: bool = False, generate_eq_settings: dict = {"use": False}) -> None:
        """
        Python class to read the psi data in from an ascii file.

        Saves the boundary information and generates an EfitData instance.
        """

        if generate_eq_settings["use"] is True:
            geqdsk_data = self.GenerateAnalyticEquilibrium(generate_eq_settings)
        else:
            with open(geqdsk_path, "r") as f:
                geqdsk_data = geqdsk.read(f)
                if not isinstance(geqdsk_data,dict):
                    geqdsk_data = geqdsk_data.__dict__
        
        if up_down_symmetry:
            # Chop off the top half of the domain and join lines of constant psi either side of the midplane  
            pe = PsinExtender(geqdsk_data)
            pe.extend_psi()

        #
        # Extract quantities needed to initialize EfitData class
        #
        nx = geqdsk_data["nx"]
        ny = geqdsk_data["ny"]
        rdim = geqdsk_data["rdim"]
        zdim = geqdsk_data["zdim"]
        zmid = geqdsk_data["zmid"]
        rleft = geqdsk_data["rleft"]
        rcenter = geqdsk_data["rcentr"]
        bcenter = geqdsk_data["bcentr"]
        rmagx = geqdsk_data["rmagx"]
        zmagx = geqdsk_data["zmagx"]
        rlimiter = geqdsk_data["rlim"]
        zlimiter = geqdsk_data["zlim"]
        psi = geqdsk_data["psi"]

        #
        # Derived values
        #
        rmin = rleft
        rmax = rmin + rdim
        zmin = zmid - 0.5 * zdim
        zmax = zmin + zdim

        #
        # Reproduce efit grid
        #
        self.PsiUNorm = EfitData(
            rmin=rmin,
            rmax=rmax,
            nr=nx,
            zmin=zmin,
            zmax=zmax,
            nz=ny,
            rcenter=rcenter,
            bcenter=bcenter,
            rlimiter=rlimiter,
            zlimiter=zlimiter,
            rmagx=rmagx,
            zmagx=zmagx,
            name="Efit Data",
            parent=self,
        )

        self.PsiUNorm.init_bivariate_spline(
            self.PsiUNorm.r[:, 0], self.PsiUNorm.z[0, :], psi
        )

        if (
            self.settings["grid_settings"]["rmagx"] is None
            or self.settings["grid_settings"]["zmagx"] is None
        ):
            (
                self.settings["grid_settings"]["rmagx"],
                self.settings["grid_settings"]["zmagx"],
            ) = (self.PsiUNorm.rmagx, self.PsiUNorm.zmagx)

        #
        # Save the stored data in the object
        #
        self.geqdsk_data = geqdsk_data
        #
        # Update the eqdsk file referenced in settings to that of the loaded data
        #
        self.settings["eqdsk"] = geqdsk_path

    def ParseTxtCoordinates(
        self, fpath: str, rshift: float = 0.0, zshift: float = 0.0
    ) -> tuple:
        """
        Extract the (R,Z) coordinates from a .txt file.

        Files of types .txt must conform to the following format:

        .. math::

            r_0, z_0

            r_1, z_1

            r_2, z_2

            .....

            r_n, z_n


        Put otherwise, r and z values are differentiated by a comma and
        each coordinate must appear on a new line withing the file.

        If a line starts with the character '#', it will be skipped.


        Parameters
        ----------
        fpath : str
            The path to the text file containing (R, Z) coordinate entries.

        rshift: float, optional
            Applies a translation to the R coordinate of the text file entries.

        zshift : float, optional
            Applies a translation to the Z coordinate of the text file entries.


        Returns
        -------
            A 2-tuple (R, Z) with list entries containing R and Z data respectively.


        Raises
        ------
            IOError
                If error occurs while reading text file.
            ValueError
                If fpath string provided leads to an invalid file.
        """

        R, Z = [], []

        if Path(fpath).is_file() and Path(fpath).suffix in [".txt"]:
            try:
                with open(fpath) as f:

                    for line in f:

                        if line.startswith("#"):
                            # move along to the next iteration
                            # this simulates a comment
                            continue
                        # we deal with separator ',' or ' '
                        point = (
                            line.replace("D", "e")
                            .replace("(", "")
                            .replace(")", "")
                            .strip()
                        )
                        if point.count(",") > 0:
                            x = float(point.split(",")[0])
                            y = float(point.split(",")[1])
                        else:

                            x = float(point.split(" ")[0])
                            y = float(point.split(" ")[1])

                        R.append(x + rshift)
                        Z.append(y + zshift)
            except:
                raise IOError(
                    f"# Error occured when reading data from file {fpath}:\t 'open(fpath)' error"
                )

        else:
            raise ValueError(
                f"# Error occur when reading data from file {fpath}:\t file does not exist or is not of extension '*.txt'"
            )

        return R, Z

    def SetLimiter(
        self,
        fpath: str = "",
        coordinates: list = [],
        rshift: float = 0.0,
        zshift: float = 0.0,
    ) -> None:
        """
        Instantiate the class Line object that represents the tokamak limiter.

        This method accepts either coordinates or a valid file path to
        coordinate data.

        If `fpath` and `coordinates` are at their default values, then the
        EFIT data will be searched for it's default limiter values.

        Parameters
        ----------
        fpath : str, optional
            A file path to a '.txt' or '.npy' file containing (R, Z) data.

        coordinates : list, optional
            A list with two entries containing R and Z entries respectively.

        rshift : float, optional
            Apply a translation to the R coordinate of the limiter entries.

        zshift : float, optional
            Apply a translation to the Z coordinate of the limiter entries.

        """

        if rshift is None:
            rshift = self.settings["limiter"]["rshift"]
        if zshift is None:
            zshift = self.settings["limiter"]["zshift"]

        use_efit_bounds = self.settings["limiter"]["use_efit_bounds"]

        #
        # TODO: Refactor the codebase to leverage numpy arrays....
        #
        coordinates = np.array(coordinates).T

        #
        # If we have no coordinates or fpath provided, we need to
        # leverage efit boundaries. Overrides any user settings.
        #
        if coordinates.shape[0] == 0 and fpath == "":
            use_efit_bounds = True

        if fpath not in ["", "."]:
            try:
                print("# Processing file for limiter data : {}".format(fpath))
                self.geqdsk_data["rlim"], self.geqdsk_data["rlim"] = (
                    self.ParseTxtCoordinates(fpath)
                )
            except:
                raise ValueError(f"# Error in method 'SetLimiter' with fpath={fpath}")

        #
        # List of coordinates provided for initialization
        #
        elif len(coordinates) > 0:
            self.geqdsk_data["rlim"] = coordinates[:, 0] + rshift
            self.geqdsk_data["zlim"] = coordinates[:, 1] + zshift

        #
        # Empty list of coordinates falls back on using eqdsk limiter settings
        #
        else:
            self.LoadGEQDSK(geqdsk_path=self.settings["eqdsk"], up_down_symmetry=self.settings["grid_settings"]["up_down_symmetry"],generate_eq_settings=self.settings["analytic_equilibrium_generation"])
            self.geqdsk_data["rlim"] += rshift
            self.geqdsk_data["zlim"] += zshift

        if use_efit_bounds:

            #
            # The bounding box starts with using the computational domain bounds.
            # Buffer values shrink the bounding box to allow us to avoid stepping
            # outside the domain during integration/line-tracing
            #
            rbuff = self.settings["limiter"]["efit_buffer_r"]
            zbuff = self.settings["limiter"]["efit_buffer_z"]

            #
            # Define the bounding box vertices as:
            # - LL : Lower left
            # - LR : Lower right
            # - UL : Upper left
            # - UR : Upper right
            #
            LL = np.array([self.PsiUNorm.rmin + rbuff, self.PsiUNorm.zmin + zbuff])
            LR = np.array([self.PsiUNorm.rmax - rbuff, self.PsiUNorm.zmin + zbuff])
            UL = np.array([self.PsiUNorm.rmin + rbuff, self.PsiUNorm.zmax - zbuff])
            UR = np.array([self.PsiUNorm.rmax - rbuff, self.PsiUNorm.zmax - zbuff])

            #
            # Define the simple limiter path and translate by shift values
            #
            limiter_path = np.vstack([LL, LR, UR, UL, LL])
            self.geqdsk_data["rlim"] = limiter_path[:, 0] + rshift
            self.geqdsk_data["zlim"] = limiter_path[:, 1] + zshift

        self.LimiterData = Line(
            [Point(p) for p in zip(self.geqdsk_data["rlim"], self.geqdsk_data["zlim"])]
        )

    def SetTargetPlate(
        self, settings: dict, rshift: float = 0.0, zshift: float = 0.0
    ) -> None:
        """
        Initialize a target plate Line object.

        This method can initialize target plates:

        - `plate_W1`
        - `plate_E1`
        - `plate_W2`
        - `plate_E2`

        with explicit (R, Z) coordinates.

        Parameters
        ----------
        settings : dict
            Argument dict specifying which plate to define and with what
            (R,Z). See notes for more details.

        rshift : float, optional
            Translate the (R, Z) coordinates by a float value.

        zshift : float, optional
            Translate the (R, Z) coordinates by a float value.

        Notes
        -----
        Parameter ``settings`` is in the following form: ``{plate_name: rz_data}``

        where ``plate_name`` is a valid key string (see table below), and
        ``rz_data`` is an iterable structure with two iterable entries
        containing R and Z entries (see examples for usage).

        Most users should use the Ingrid class method SetGeometry
        rather than interface directly with the IngridUtils method
        since SetGeometry calls this IngridUtils method after processing
        the user input.

        Valid plate keys are as follows:

        ======== =====================
          Plate   Accepted Keys (str)
        -------- ---------------------
        Plate W1 ``plate_W1``, ``W1``
        Plate E1 ``plate_E1``, ``E1``
        Plate W2 ``plate_W2``, ``W2``
        plate_E2 ``plate_E2``, ``E2``
        ======== =====================

        Example
        -------
        Defining target plate `plate_W1` with coordinates [(1,2), (2, 3), (3, 4)]

        >>> MyIG = IngridUtils()
        >>> r_entries = [1, 2, 3]
        >>> z_entries = [2, 3, 4]
        >>> rz_data = (r_entries, z_entries)
        >>> MyIG.SetTargetPlate({'plate_W1': rz_data})

        """

        for _k, _v in settings.items():
            k = _k
            v = _v
            break

        if k.lower() in ["w1", "westplate1", "plate_w1"]:
            k = "plate_W1"
        elif k.lower() in ["e1", "eastplate1", "plate_e1"]:
            k = "plate_E1"
        elif k.lower() in ["w2", "westplate2", "plate_w2"]:
            k = "plate_W2"
        elif k.lower() in ["e2", "eastplate2", "plate_e2"]:
            k = "plate_E2"
        else:
            raise ValueError(f"# Invalid key '{k}' provided for 'SetTargetPlate'")

        print(f"# Setting coordinates for '{k}'")

        R, Z = v

        # Make sure coordinates are unique
        data = np.array([c for c in zip(R, Z)])
        a, index = np.unique(data, return_index=True, axis=0)
        index.sort()
        self.PlateData[k] = Line(
            [Point(x + rshift, y + zshift) for x, y in data[index]]
        )

    def OrderTargetPlate(self, plate_key: str) -> None:
        """
        Ensures the target plate points are oriented **clockwise** around the
        magnetic axis (per INGRID convention).

        Parameters
        ----------
        plate_key : str
            The key corresponding to target plate Line object to sort.

        Notes
        -----
        Ordering of target plates is **crucial** when using target plates for
        creating a patch map.

        Valid plate keys are as follows:

        ======== =====================
          Plate   Accepted Keys (str)
        -------- ---------------------
        Plate W1 ``plate_W1``, ``W1``
        Plate E1 ``plate_E1``, ``E1``
        Plate W2 ``plate_W2``, ``W2``
        Plate E2 ``plate_E2``, ``E2``
        ======== =====================
        """

        k = plate_key

        if not isinstance(self.PlateData.get(k), Line):
            raise ValueError(
                f"# Plate '{k}' is not loaded as a Line object. Make sure 'SetGeometry({{'{k}' : ... }})' has been called."
            )

        plate = self.PlateData[k]

        start = plate.p[0]
        end = plate.p[-1]

        # Endpoints are on same vertical line
        if start.x == end.x:
            # If end point is above start point
            if start.y < end.y:
                # Return target plate as is
                ordered_plate = plate.copy()
            else:
                # Flip plate orientation
                ordered_plate = plate.reverse_copy()

        # Endpoints are on same horizontal line
        elif start.y == end.y:
            # If start point is right of end point
            if start.x > end.x:
                # Return strike plate as is
                ordered_plate = plate.copy()
            else:
                # Flip plate orientation
                ordered_plate = plate.reverse_copy()

        else:

            # Check the angle between the plate endpoints with
            # tail fixed on the magnetic axis
            v_reference = np.array(
                [
                    self.settings["grid_settings"]["rmagx"],
                    self.settings["grid_settings"]["zmagx"],
                ]
            )

            v_start = np.array([start.x, start.y])
            v_end = np.array([end.x, end.y])

            if orientation_between(v_start, v_end, v_reference) <= 0:
                ordered_plate = plate.copy()
            else:
                ordered_plate = plate.reverse_copy()

            # Gather raw data
            self.PlateData[k] = ordered_plate

    def OrderTargetPlates(self) -> None:
        """
        Convenience method for ordering target plate Point objects.
        """
        for k in self.PlateData.keys():
            if type(self.PlateData[k]) == Line:
                self.OrderTargetPlate(k)

    def OrderLimiter(self) -> None:
        """
        Ensures the limiter points are oriented **clockwise** around the
        magnetic axis (per INGRID convention).

        This method requires the limiter geometry to have been defined
        as well as the magnetic axis to be refined.

        Parameters
        ----------

        Notes
        -----
        Ordering of limiter is **crucial** when using limiter for
        creating a patch map. This occurs for all cases with two
        x-points.
        """

        limiter = self.LimiterData.fluff_copy(100)

        rmid = (self.PsiUNorm.rmin + self.PsiUNorm.rmax) / 2
        zmid = (self.PsiUNorm.zmin + self.PsiUNorm.zmax) / 2

        lookup = {}
        point_list = []
        for p in limiter.p:
            try:
                lookup[(p.x, p.y)]
            except:
                if (p.x >= rmid) and (p.y >= zmid):
                    lookup[(p.x, p.y)] = p
                    point_list.append(p)

        quadrant_boundary = Line(point_list)

        ordered = False
        start = quadrant_boundary.p[0]
        for p in reversed(quadrant_boundary.p):
            end = p
            if end.coor != start.coor:
                break

        # Endpoints are on same vertical line
        if start.x == end.x:
            # If end point is above start point
            if start.y < end.y:
                # Return target plate as is
                ordered = True

        # Endpoints are on same horizontal line
        elif start.y == end.y:
            # If start point is left of end point
            if start.x < end.x:
                # Return strike plate as is
                ordered = True

        else:

            # Check the angle between the plate endpoints with
            # tail fixed on the magnetic axis
            v_reference = np.array(
                [
                    self.settings["grid_settings"]["rmagx"],
                    self.settings["grid_settings"]["zmagx"],
                ]
            )

            v_start = np.array([start.x, start.y])
            v_end = np.array([end.x, end.y])

            if orientation_between(v_start, v_end, v_reference) <= 0:
                ordered = True

        if ordered is False:
            self.LimiterData = self.LimiterData.reverse_copy()

    def FindMagAxis(self, r: float, z: float) -> None:
        """
        Refine the entries and assign to the magnetic-axis in settings.

        Parameters
        ----------
        r : float
            R coordinate of magnetic-axis guess.

        z : float
            Z coordinate of magnetic-axis guess.
        """
        sol = root(self.PsiUNorm.Gradient, [r, z])
        self.settings["grid_settings"]["rmagx"] = sol.x[0]
        self.settings["grid_settings"]["zmagx"] = sol.x[1]
        self.magx = (sol.x[0], sol.x[1])

    def FindXPoint(self, r: float, z: float) -> None:
        """
        Refine the entries and assign to the primary x-point in settings.

        Parameters
        ----------
        r : float
            R coordinate of primary x-point guess.

        z : float
            Z coordinate of primary x-point guess.
        """
        sol = root(self.PsiUNorm.Gradient, [r, z])
        self.settings["grid_settings"]["rxpt"] = sol.x[0]
        self.settings["grid_settings"]["zxpt"] = sol.x[1]
        self.xpt1 = (sol.x[0], sol.x[1])

    def FindXPoint2(self, r: float, z: float) -> None:
        """
        Refine the entries and assign to the secondary x-point in settings.

        Parameters
        ----------
        r : float
            R coordinate of secondary x-point guess.

        z : float
            Z coordinate of secondary x-point guess.
        """
        sol = root(self.PsiUNorm.Gradient, [r, z])
        self.settings["grid_settings"]["rxpt2"] = sol.x[0]
        self.settings["grid_settings"]["zxpt2"] = sol.x[1]
        self.xpt2 = (sol.x[0], sol.x[1])

    def _find_roots(self, tk_controller=None):
        """Displays a plot, and has the user click on an approximate
        zero point. Uses a root finder to adjust to the more exact point.
        Right click to disable.
        Parameter:
            - tk_controller : Tk object
            A reference to the root TK object (for usage in gui mode).
        """
        self._root_finder = RootFinder(self.PsiUNorm, controller=tk_controller)

    def _toggle_root_finder(self):
        """
        Activates or deactivates the root finder ability. Enables
        the user to save the location where they last clicked.
        """
        self._root_finder.toggle_root_finding()

    def _find_psi_lines(self, tk_controller=None):
        self._psi_finder = RootFinder(
            self.PsiNorm, mode="psi_finder", controller=tk_controller
        )

    def GetMagxData(self) -> tuple:
        """
        Return the magnetic-axis (r,z) coordinates and associated
        un-normalized psi value.

        Parameters
        ----------

        Returns
        -------
            A 3-tuple of r-z coordinates and scalar psi value
        """
        return (
            self.magx[0],
            self.magx[1],
            self.PsiUNorm.get_psi(self.magx[0], self.magx[1]),
        )

    def GetXptData(self) -> dict:
        """
        Return all x-point (r,z) coordinates and associated
        un-normalized psi values.

        Parameters
        ----------

        Returns
        -------
            A dict containing an (r, z, psi) entry for each x-point
        """
        xpt_info = {}
        if hasattr(self, "xpt1"):
            xpt_info["xpt1"] = (
                self.xpt1[0],
                self.xpt1[1],
                self.PsiUNorm.get_psi(self.xpt1[0], self.xpt1[1]),
            )
        if hasattr(self, "xpt2"):
            xpt_info["xpt2"] = (
                self.xpt2[0],
                self.xpt2[1],
                self.PsiUNorm.get_psi(self.xpt2[0], self.xpt2[1]),
            )
        return xpt_info

    def PrepLineTracing(self):
        """
        Initializes the line tracing class for the construction
        of the grid.
        """

        self.LineTracer = LineTracing(self.PsiNorm, self.settings)

    def GetPatchTagMap(self, config: str) -> dict:
        """
        Get the Patch-Tag mapping for a particular configuration.

        This mapping is used to identify patch names with a particular
        patch tag code.

        Parameters
        ----------
        config : str
            The configuration to get the patch tag map for.

        Returns
        -------
            A dictionary containing the tag to patch name mappings.
        """
        PatchTagMap = {}
        TempLabels = ["A", "B", "C", "D", "E", "F"]
        for label in TempLabels:
            for i in range(1, 3):
                PatchTagMap[label + str(i)] = label + str(i)
        else:
            PatchTagMap = {}
            TempLabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
            for label in TempLabels:
                for i in range(1, 4):
                    PatchTagMap[label + str(i)] = label + str(i)

        # Make it bijective.
        PatchNameMap = {}
        for tag, name in PatchTagMap.items():
            PatchNameMap[name] = tag
        return {**PatchTagMap, **PatchNameMap}

    def ClassifyTopology(self, visual=False) -> None:
        """
        Determine the topology that is being operated on.

        Inspects the ``settings['grid_settings']['num_xpt']`` entry to
        determine which classification scheme to employ.

        Parameters
        ----------
        visual : bool, optional
            Flag for activating the analysis of x-points in a visual DEBUG mode.
            Default is False.

        Raises
        ------
        ValueError
            If user specifies ``settings['grid_settings']['num_xpt']``
            with value other than 1 (int) or 2 (int).
        """
        print("")
        print("# Begin classification....")
        print("")

        if self.settings["grid_settings"]["num_xpt"] == 1:
            self.LineTracer.SNL_find_NSEW(self.xpt1, self.magx, visual)

        elif self.settings["grid_settings"]["num_xpt"] == 2:
            # Check if limiter contains magx, xpt1,and xpt2
            from matplotlib.patches import Polygon

            limiter = Polygon(
                np.column_stack(self.LimiterData.points()).T,
                fill=True,
                closed=True,
                color="red",
                label="Limiter",
            )

            missing_items = []
            if (limiter.get_path().contains_point(self.magx)) is False:
                missing_items.insert(0, "magx")
            if (limiter.get_path().contains_point(self.xpt1)) is False:
                missing_items.insert(0, "xpt1")
            if (limiter.get_path().contains_point(self.xpt2)) is False:
                missing_items.insert(0, "xpt2")
            if len(missing_items) == 0:
                self.LineTracer.DNL_find_NSEW(self.xpt1, self.xpt2, self.magx, visual)
            else:
                raise ValueError(
                    f"# Topological item(s) {missing_items} not contained in the limiter geometry provided. Check coordinates provided are correct and/or edit the limiter geometry."
                )
        else:
            raise ValueError(
                f"# No support available for 'num_xpt' value of {self.settings['grid_settings']['num_xpt']}"
            )

        self.config = self.LineTracer.config

    def WriteGridueSNL(self, gridue_settings: dict, fname: str = "gridue") -> bool:
        """
        Write a gridue file for a single-null configuration.

        Parameters
        ----------
        gridue_settings : dict
            A dictionary containing grid data to be written to the gridue file.

        fname : str, optional
            The file name/path to save the gridue file to.
            Defaults to 'gridue'.

        Returns
        -------
            True if file was written with no errors
        """

        def format_header(gridue):
            header_items = ["nxm", "nym", "ixpt1", "ixpt2", "iyseptrx1"]
            header = ""
            for item in header_items:
                header += "{}".format(gridue[item]).rjust(4)

            header += "\n"
            return header

        def format_body(data):

            delim_val = 0
            delim_char = ""
            body = ""

            for n in range(5):
                for j in range(len(data[0])):
                    for i in range(len(data)):
                        delim_val += 1
                        val = (
                            np.format_float_scientific(
                                data[i][j][n], precision=15, unique=False
                            )
                            .rjust(23)
                            .replace("e", "D")
                        )
                        if delim_val == 3:
                            delim_val = 0
                            delim_char = "\n"
                        body += val + delim_char
                        delim_char = ""

            if delim_val % 3 != 0:
                body += "\n"

            return body

        f = open(fname, mode="w")
        f.write(format_header(gridue_settings) + "\n")

        body_items = ["rm", "zm", "psi", "br", "bz", "bpol", "bphi", "b"]
        for item in body_items:
            f.write(format_body(gridue_settings[item]) + "\n")

        runidg = "iogridue"
        f.write(runidg + "\n")

        f.close()

        return True

    def WriteGridueDNL(self, gridue_settings: dict, fname: str = "gridue") -> bool:
        """
        Write a gridue file for a double-null configuration.

        Parameters
        ----------
        gridue_settings : dict
            A dictionary containing grid data to be written to the gridue file.

        fname : str, optional
            The file name/path to save the gridue file to.
            Defaults to 'gridue'.

        Returns
        -------
            True if file was written with no errors
        """

        def format_header(gridue):
            header_rows = [
                ["nxm", "nym"],
                ["iyseparatrix1", "iyseparatrix2"],
                ["ix_plate1", "ix_cut1", "_FILLER_", "ix_cut2", "ix_plate2"],
                ["iyseparatrix3", "iyseparatrix4"],
                ["ix_plate3", "ix_cut3", "_FILLER_", "ix_cut4", "ix_plate4"],
            ]

            header = ""
            for header_items in header_rows:
                for item in header_items:
                    header += "{}".format(gridue[item]).rjust(4)
                header += "\n"
            return header

        def format_body(data):

            delim_val = 0
            delim_char = ""
            body = ""

            for n in range(5):
                for j in range(len(data[0])):
                    for i in range(len(data)):
                        delim_val += 1
                        val = (
                            np.format_float_scientific(
                                data[i][j][n], precision=15, unique=False
                            )
                            .rjust(23)
                            .replace("e", "D")
                        )
                        if delim_val == 3:
                            delim_val = 0
                            delim_char = "\n"
                        body += val + delim_char
                        delim_char = ""

            if delim_val % 3 != 0:
                body += "\n"

            return body

        f = open(fname, mode="w")
        f.write(format_header(gridue_settings) + "\n")

        body_items = ["rm", "zm", "psi", "br", "bz", "bpol", "bphi", "b"]
        for item in body_items:
            f.write(format_body(gridue_settings[item]) + "\n")

        runidg = "iogridue"
        f.write(runidg + "\n")

        f.close()
        return True

    def ReconstructPatches(self, raw_patch_list: list) -> dict:
        """
        Reconstruct a Patch objects from a saved file.

        This method takes in an Ingrid formatted .npy file that
        contains the information needed to reconstruct a patch map
        from a past INGRID session.

        Parameters
        ----------
        fname : str
            The file path to the patch data file obtained after a call to
            Ingrid class method SavePatches

        Returns
        -------
            A dict of reconstructed Patch objects.
        """
        patches = {}

        for raw_patch in raw_patch_list:
            patch_data, cell_data, patch_settings = raw_patch
            NR = patch_data[0]
            NZ = patch_data[1]
            ER = patch_data[2]
            EZ = patch_data[3]
            SR = patch_data[4]
            SZ = patch_data[5]
            WR = patch_data[6]
            WZ = patch_data[7]

            N = Line([Point(p) for p in zip(NR, NZ)])
            E = Line([Point(p) for p in zip(ER, EZ)])
            S = Line([Point(p) for p in zip(SR, SZ)])
            W = Line([Point(p) for p in zip(WR, WZ)])

            patch = Patch(
                [N, E, S, W],
                patch_name=patch_settings["patch_name"],
                plate_patch=patch_settings["plate_patch"],
                plate_location=patch_settings["plate_location"],
                PatchTagMap=patch_settings["PatchTagMap"],
            )

            patches[patch.patch_name] = patch

        patches = {k: v for k, v in patches.items()}

        return patches

    def CheckPatches(self, verbose: bool = False) -> None:
        """
        Check that Patch objects adjacent to target plates are monotonic in psi.

        This method is a wrapper for the CheckPatches method specific to the
        current topology being operated on.

        Parameters
        ----------
        verbose : bool, optional
            Flag for printing full output to terminal. Defaults to False.
        """
        self.CurrentTopology.CheckPatches(verbose=verbose)

    def PrepGridue(self, guard_cell_eps=1e-3) -> None:
        """
        Prepare the gridue for writing.

        This method calls topology specific implementations of methods that
        concatenate the Patch object subgrids into a global grid.
        """
        self.CurrentTopology.SetupPatchMatrix()
        self.CurrentTopology.concat_grid(guard_cell_eps=guard_cell_eps)
        self.CurrentTopology.set_gridue()

    @classmethod
    def _CheckOverlapCells(Grid, Verbose=False):
        from shapely.geometry import Polygon

        r = Grid["rm"]
        z = Grid["zm"]
        idx = [1, 2, 4, 3]
        p = []
        pinfo = []
        Nx = len(r)
        Ny = len(r[0])
        # polygon = [[Polygon([(x,y) for (x,y) in zip(r[i,j,idx],z[i,j,idx]]) for y in range(0,Ny)] for i range(0,Nx)]

        for i in range(Nx):
            for j in range(Ny):
                c = [(r[i, j, idxx], z[i, j, idxx]) for idxx in idx]
                if Verbose:
                    print(c)
                p.append(Polygon(c))
                pinfo.append((i, j))
        ListIntersect = []
        for p1, pinfo1 in zip(p, pinfo):
            for p2, pinfo2 in zip(p, pinfo):
                if (
                    p1.intersects(p2)
                    and np.sum(abs(np.array(pinfo1) - np.array(pinfo2))) > 2
                ):
                    ListIntersect.append((pinfo1, pinfo2))
                    print("p1:{} and p2:{} intersect!".format(pinfo1, pinfo2))
        return ListIntersect


class TopologyUtils:
    """
    The base class for all INGRID topologies.

    Encapsulates key methods generating patch maps, visualizing data,
    generating grids, and exporting of grids.

    These methods are to be interfaced by the child :class:`ingrid.Ingrid` or
    can be used by advanced users of the code.

    Attributes
    ----------
    parent : Ingrid
        Ingrid object the topology is being managed by.

    settings : dict
        Core settings dict of parent.

    CurrentListPatch : dict
        Lookup dict of Patches that have been refined (populated during Patch refinement).

    ConnexionMap : dict
        A mapping describing how Patch objects are connected to each other (see notes).

    CorrectDistortion : dict
        The settings to be used for correcting grid shearing.

    config : str
        Configuration of topology.

    PlateData : dict
        Parent target plate data dict.

    PatchTagMap : dict
        A dictionary containing the tag to patch name mappings for this topology.

    LineTracer : LineTracing
        Parent LineTracing instance.

    PsiUNorm : EfitData
        Parent PsiUNorm instance.

    PsiNorm : EfitData
        Parent PsiNorm instance.

    """

    def __init__(self, Ingrid_obj: object, config: str):
        self.parent = Ingrid_obj
        self.config = config
        self.settings = Ingrid_obj.settings
        self.PlateData = Ingrid_obj.PlateData
        self.PatchTagMap = self.parent.GetPatchTagMap(config=config)
        self.LineTracer = Ingrid_obj.LineTracer
        self.PsiUNorm = Ingrid_obj.PsiUNorm
        self.PsiNorm = Ingrid_obj.PsiNorm
        self.CurrentListPatch = {}
        self.ConnexionMap = {}
        self.Verbose = False
        self.GetDistortionCorrectionSettings()

    def RefreshSettings(self):
        self.settings = self.parent.settings

    def OrderPatches(self):
        pass

    def patch_diagram(self, fig: object = None, ax: object = None, up_down_symmetry: bool = False, zmagx: float = 0.0) -> None:
        """
        Generate the patch diagram for a given configuration.

        Parameters
        ----------
        fig : object, optional
            Matplotlib figure to show the Patch map on.

        ax : object, optional
            Matplotlib axes to plot the Patch map on.
        up_down_symmetry: bool, optional (defaults to False)
            Up/down symmetry switch
        zmagx: float, optional (defaults to 0.0)
            Z position of magnetic axis (used in setting y-axis limits for the plot)

        """

        colors = {
            "A": "red",
            "B": "blue",
            "C": "navajowhite",
            "D": "firebrick",
            "E": "magenta",
            "F": "olivedrab",
            "G": "darkorange",
            "H": "yellow",
            "I": "navy",
        }
        alpha = {"3": 1.0, "2": 0.75, "1": 0.5}

        f = fig if fig else plt.figure("INGRID Patch Map", figsize=(6, 10))
        f.subplots_adjust(bottom=0.2)
        a = ax if ax else f.subplots(1, 1)
        a.set_xlim([self.PsiUNorm.rmin, self.PsiUNorm.rmax])
        if up_down_symmetry:
            a.set_ylim(self.PsiUNorm.zmin, zmagx + 0.03*(self.PsiUNorm.zmax-self.PsiUNorm.zmin))
        else:
            a.set_ylim([self.PsiUNorm.zmin, self.PsiUNorm.zmax])
        a.set_aspect("equal", adjustable="box")

        a.set_xlabel("R")
        a.set_ylabel("Z")
        a.set_title(f"{self.config} Patch Diagram")

        for i, patch in enumerate(self.patches.values()):
            patch.plot_border(color="black", ax=a)
            patch.fill(
                colors[patch.get_tag()[0]], ax=a, alpha=alpha[patch.get_tag()[-1]]
            )
            patch.color = colors[patch.get_tag()[0]]
        handles, labels = a.get_legend_handles_labels()
        lookup = {label: handle for label, handle in zip(labels, handles)}
        a.legend(
            handles=[handle for handle in lookup.values()],
            labels=[label for label in lookup.keys()],
            bbox_to_anchor=(1.25, 1.0),
            loc="upper right",
            ncol=1,
        )
        f.show()

    def grid_diagram(self, fig: object = None, ax: object = None, up_down_symmetry: bool = False, zmagx: float = 0.0) -> None:
        """
        Generates the grid diagram for a given configuration.

        Parameters
        ----------
        fig : object, optional
            Matplotlib figure to show the grid diagram on.

        ax : object, optional
            Matplotlib axes to plot the grid diagram on.
        up_down_symmetry: bool, optional (defaults to False)
            Up/down symmetry switch
        zmagx: float, optional (defaults to 0.0)
            Z position of magnetic axis (used in setting y-axis limits for the plot)

        """
        if fig is None:
            fig = plt.figure("INGRID Grid", figsize=(6, 10))

        if ax is None:
            ax = fig.gca()

        plt.figure(fig.number)
        fig.subplots_adjust(bottom=0.2)
        ax.set_xlim(self.PsiUNorm.rmin, self.PsiUNorm.rmax)
        if up_down_symmetry:
            ax.set_ylim(self.PsiUNorm.zmin, zmagx + 0.03*(self.PsiUNorm.zmax-self.PsiUNorm.zmin))
        else:
            ax.set_ylim(self.PsiUNorm.zmin, self.PsiUNorm.zmax)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        ax.set_title(f"{self.config} Subgrid")

        for patch in self.patches.values():
            patch.plot_subgrid(fig=fig, ax=ax)
            print(f"# Plotting subgrid {patch.patch_name}")

        fig.show()

    def _animate_grid(self):

        try:
            plt.close("INGRID: Debug")
        except:
            pass
        plt.figure("INGRID: Debug", figsize=(6, 10))
        plt.xlim(self.PsiUNorm.rmin, self.PsiUNorm.rmax)
        plt.ylim(self.PsiUNorm.zmin, self.PsiUNorm.zmax)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.title("visualize gridue")

        k = [1, 2, 4, 3, 1]

        for i in range(len(self.rm)):
            for j in range(len(self.rm[0])):
                plt.plot(self.rm[i][j][k], self.zm[i][j][k])
                plt.pause(0.01)

    def get_config(self) -> str:
        """
        Return the configuration string stored in the TopologyUtils class

        Parameters
        ----------

        Returns
        -------
            A string indicating the topology
        """
        return self.config

    def get_func(self, func: str) -> "function":
        """
        Create a function from a string input.

        Will be used to generate a poloidal or radial transformation
        function.

        Parameters
        ----------
        func : str
            An expression to generate a function from.

        Returns
        -------
            A function generated from the str input.

        Examples
        --------
        When calling method ``get_func`` the user must provide a **string**
        with the following general format:

        .. math::

            x, f(x)

        That is, the dependent variable and expression to evaluate are
        separated with a comma character.

        The `Sympy` library supports most expressions that can be generated with
        Python natively. See the `Sympy` documentation for advanced features.

        Defining a function representing f(x) = x ^ 2:

        >>> func = 'x, x ** 2'
        >>> f = MyTopologyUtils.get_func(func)
        >>> f(np.array([0, 1, 2, 3]))
        array([0, 1, 4, 9])

        Defining a function representing f(x) = exp(x)

        >>> func = 'x, exp(x)'
        >>> f = MyTopologyUtils.get_func(func)
        >>> f(np.array([0, 1, 2, 3]))
        array([ 1.        ,  2.71828183,  7.3890561 , 20.08553692])

        """

        def make_sympy_func(var, expression):
            import sympy as sp

            _f = sp.lambdify(var, expression, "numpy")
            return _f

        f_str_raw = func

        f_str_raw = f_str_raw.replace(" ", "")
        delim = f_str_raw.index(",")

        var = f_str_raw[0:delim]
        expression = f_str_raw[delim + 1 :]

        func = make_sympy_func(var, expression)
        # TODO: Check Normalization of the function to 1
        return func

    def CheckFunction(self, expression: str, Verbose: bool = False) -> bool:
        """
        Check if a str is in the correct format for method ``get_func``

        Parameters
        ----------
        expression : str
            Expression to check.

        Verbose : bool, optional
            Print full output to terminal. Default to False

        Returns
        -------
            True if expression is valid. False otherwise.
        """
        ExpValid = False
        try:
            com = "lambda {}:{}".format(
                expression.split(",")[0], expression.split(",")[1]
            )
            if Verbose:
                print(com)
            eval(com)
            ExpValid = True
        except:
            raise ValueError(
                'Unable to parse expression entry "{}".'.format(expression)
            )
        return ExpValid

    def GetFunctions(self, Patch: Patch, Verbose: bool = False) -> tuple:
        """
        Get the poloidal and radial transformations for a Patch.

        Poloidal and radial transformations affect more than a single Patch
        in the index space. This method ensures that the same transformations
        are applied to dependent Patches (e.g. radial transformation T is
        applied to all Patch objects in the same radial level).

        Parameters
        ----------
        Patch : Patch
            The Patch to get the functions for.

        Verbose : bool, optional
            Print all output to terminal. Default to False.

        Returns
        -------
            2-tuple containing functions for radial and poloidal direction respectively.
        """

        poloidal_tag, radial_tag = Patch.get_tag()
        p_f = "poloidal_f_" + poloidal_tag
        r_f = "radial_f_" + radial_tag

        try:
            _poloidal_f = self.settings["grid_settings"]["grid_generation"][p_f]
            valid_function = self.CheckFunction(_poloidal_f, Verbose)
            if valid_function:
                _poloidal_f = self.get_func(_poloidal_f)
            else:
                raise ValueError(
                    "# Invalid function entry. Applying default poloidal function."
                )
        except:
            _poloidal_f = self.settings["grid_settings"]["grid_generation"][
                "poloidal_f_default"
            ]
            valid_function = self.CheckFunction(_poloidal_f, Verbose)
            if valid_function:
                _poloidal_f = self.get_func(_poloidal_f)
            else:
                _poloidal_f = lambda x: x

        try:

            # Adding CORE radial_f support for SNL cases via entry 'radial_f_3'
            if (
                self.config in ["USN", "LSN"]
                and self.settings["grid_settings"]["grid_generation"].get("radial_f_3")
                is not None
                and poloidal_tag + radial_tag in ["B1", "C1", "D1", "E1"]
            ):
                _radial_f = self.settings["grid_settings"]["grid_generation"][
                    "radial_f_3"
                ]
            else:
                _radial_f = self.settings["grid_settings"]["grid_generation"][r_f]
            valid_function = self.CheckFunction(_radial_f, Verbose)
            if valid_function:
                _radial_f = self.get_func(_radial_f)
            else:
                raise ValueError(
                    "# Invalid function entry. Applying default radial function."
                )
        except:
            _radial_f = self.settings["grid_settings"]["grid_generation"][
                "radial_f_default"
            ]
            valid_function = self.CheckFunction(_radial_f, Verbose)
            if valid_function:
                _radial_f = self.get_func(_radial_f)
            else:
                _radial_f = lambda x: x

        return (_radial_f, _poloidal_f)

    def GetNpoints(self, Patch: Patch) -> tuple:
        """
        Get the number of poloidal and radial grid cells to be generated for a Patch.

        Because of index space dependence and formatting, this method ensures adjacent
        patches in the index space are matching in dimensions.

        Parameters
        ----------
        Patch : Patch
            Patch object to get the np and nr values for.

        Returns
        -------
            A 2-tuple containing the number of radial and poloidal cells to generate, respectively.
        """

        poloidal_tag, radial_tag = Patch.get_tag()
        np_tag = "np_" + poloidal_tag
        nr_tag = "nr_" + radial_tag
        try:
            np_cells = self.settings["grid_settings"]["grid_generation"][np_tag]
        except:
            np_cells = self.settings["grid_settings"]["grid_generation"]["np_default"]

        try:
            nr_cells = self.settings["grid_settings"]["grid_generation"][nr_tag]
        except:
            nr_cells = self.settings["grid_settings"]["grid_generation"]["nr_default"]

        return (nr_cells, np_cells)

    def GetDistortionCorrectionSettings(self) -> dict:
        """
        Get settings associated with the ``CorrectDistortion`` capability.

        Parameters
        ----------

        Returns
        -------
            A dictionary containing ``CorrectDistortion`` settings.
        """
        if (
            self.settings["grid_settings"]["grid_generation"].get(
                "distortion_correction"
            )
            is not None
        ):
            CD = self.settings["grid_settings"]["grid_generation"][
                "distortion_correction"
            ]
        else:
            CD = {}

        self.distortion_correction = CD
        return CD

    def construct_grid(
        self,
        np_cells: int = 1,
        nr_cells: int = 1,
        Verbose: bool = False,
        ShowVertices: bool = False,
        RestartScratch: bool = False,
        ListPatches: str = "all",
    ) -> None:
        """
        Construct a grid by refining a Patch map.

        This method gathers transformation and dimension information to apply
        to each Patch. In addition, this applies any ``CorrectDistortion``
        settings the user may want to apply.

        **Assumes a Patch map has been generated or that Patches have been
        loaded** (equivalent).

        Parameters
        ----------
        np_cells : int, optional
            Number of poloidal cells to create in the local Patch grid.
            Defaults to 1.
        nr_cells : int, optional
            Number of radial cells to create in the local Patch grid.
            Defaults to 1.
        Verbose : bool, optional
            Print all output to terminal.
            Defaults to False
        ShowVertices : bool, optional
            Emphasize spline vertices on grid figure with bold markers.
            Defaults to False.
        RestartScratch : bool, optional
            Flag for repeating the Patch refinement process for an already refined Patch.
            Defaults to False.
        ListPatches : str, optional
            Specify which Patches to generate a grid for (see notes).
            Defaults to 'all'.

        Examples
        --------
        Parameter ``ListPatches`` can be used to specify which Patches to refine
        into a grid. The default value of `all` instructs TopologyUtils to refine
        the entire Patch map. To specify Patches, the user is to provide a list
        containing **names** of Patches to refine.

        Refining only the outer-most psi boundary (SOL) for a lower single-null
        configuration:

        >>> patch_names = ['A2', 'B2', 'C2', 'D2', 'E2', 'F2']
        >>> MyTopologyUtils.construct_grid(np_cells=3, nr_cells=3, ListPatches=patch_names)

        Refining only the inboard and outboard psi boundary for an unbalanced
        double-null configuration:

        >>> patch_names = ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3']
        >>> MyTopologyUtils.construct_grid(np_cells=3, nr_cells=3, ListPatches=patch_names)

        Notes
        -----
        The Patch refinement process is often order-dependent. This is to ensure
        alignment of grid with minimal editing.

        Because of this, parameters such as ``ListPatches`` is suggested to be used by
        a user or developer who is sure of what they are doing.
        """

        # Straighten up East and West segments of our patches,
        # Plot borders and fill patches.
        if Verbose:
            print("Construct Grid")
        try:
            visual = self.settings["DEBUG"]["visual"]["subgrid"]
        except:
            visual = False
        try:
            verbose = self.settings["DEBUG"]["verbose"]["grid_generation"]
        except:
            verbose = False

        verbose = Verbose or verbose

        self.GetDistortionCorrectionSettings()

        print(">>> Patches:", [k for k in self.patches.keys()])
        if RestartScratch:
            self.CurrentListPatch = {}

        for name, patch in self.patches.items():

            if self.distortion_correction.get(name) is not None:
                patch.distortion_correction = self.distortion_correction.get(name)
            elif self.distortion_correction.get(patch.get_tag()) is not None:
                patch.distortion_correction = self.distortion_correction.get(
                    patch.get_tag()
                )
            elif self.distortion_correction.get("all") is not None:
                patch.distortion_correction = self.distortion_correction.get("all")
            else:
                patch.distortion_correction = {"Active": False}
            if (ListPatches == "all" and patch not in self.CurrentListPatch) or (
                ListPatches != "all" and name in ListPatches
            ):
                self.SetPatchBoundaryPoints(patch, verbose)
                (nr_cells, np_cells) = self.GetNpoints(patch)
                (_radial_f, _poloidal_f) = self.GetFunctions(patch)
                print(f">>> Making subgrid in patch {name}:")
                print(f"    np = {np_cells}, nr = {nr_cells}")
                print(f"    fp = {inspect.getsource(_poloidal_f)}")
                print(f"    fr = {inspect.getsource(_radial_f)}", end="")
                patch.RemoveDuplicatePoints()
                patch.make_subgrid(
                    self,
                    np_cells,
                    nr_cells,
                    _poloidal_f=_poloidal_f,
                    _radial_f=_radial_f,
                    verbose=verbose,
                    visual=visual,
                    ShowVertices=ShowVertices,
                )

                self.CurrentListPatch[name] = patch
                print(f"    {name} subgrid complete.\n\n")
        self.AdjustGrid()

    def SetPatchBoundaryPoints(self, Patch: Patch, verbose: bool = False) -> None:
        """
        Set the Patch ``BoundaryPoints`` dict based off TopologyUtils ``ConnexionMap``.

        Parameters
        ----------
        Patch : Patch
            The Patch to set the boundary points for.
        verbose: bool
            Print full output to terminal.

        Notes
        -----
        The ``ConnexionMap`` represents the layout of adjacent patches and will
        lookup what is adjacent to the Patch parameter being operated on.
        """
        Patch.TerminatesLoop = False
        if self.ConnexionMap.get(Patch.get_tag()) is not None:
            if verbose:
                print("Find connexion map for patch {}".format(Patch.patch_name))
            for v in self.ConnexionMap.get(Patch.get_tag()).items():
                Boundary, AdjacentPatch = v
                Patch.BoundaryPoints[Boundary] = self.GetBoundaryPoints(AdjacentPatch)
                if verbose:
                    print("Find Boundaries points for {}".format(Patch.patch_name))
            if self.ConnexionMap.get(Patch.get_tag()).get("E") is not None:
                Patch.TerminatesLoop = True

    def GetBoundaryPoints(self, AdjacentPatchInfo: tuple) -> list:
        """
        Get the points along a boundary for a particular Patch.

        Parameters
        ----------
        AdjacentPatchInfo : tuple
            A 2-tuple containing the tag of the Patch to obtain boundary points
            `from` (str value), and a character indicating which boundary to
            access ('N', 'S', 'E', 'W').

        Returns
        -------
            A list containing the Points along the specified boundary for a Patch.
        """
        if AdjacentPatchInfo is not None:
            PatchTag = AdjacentPatchInfo[0]
            Boundary = AdjacentPatchInfo[1]
            for patch in self.patches.values():
                if patch.get_tag() == PatchTag:
                    if Boundary == "S":
                        return patch.S_vertices
                    elif Boundary == "N":
                        return patch.N_vertices
                    elif Boundary == "E":
                        return patch.E_vertices
                    elif Boundary == "W":
                        return patch.W_vertices
        return None

    def CheckPatches(self, verbose: bool = False) -> None:
        """
        Convenience method for calling the Patch class method ``CheckPatch``.

        Checks to make sure Patch objects stored in the TopologyUtils ``patches``
        dictionary are monotonic in psi along strike geometry (if applicable).

        Parameters
        ----------
        verbose : bool, optional
            Print full output to terminal. Defaults to False.
        """
        for name, patch in self.patches.items():
            if patch.plate_patch:
                print(" # Checking patch: ", name)
                patch.CheckPatch(self)

    def SetupPatchMatrix(self) -> list:
        """
        Instantiate the list representation of the Patch layout in index space.

        Method ``concat_grid`` uses this structure when combining refined Patches
        into a global grid.

        Parameters
        ----------

        Returns
        -------
            The 'PatchMatrix' list
        """
        p = self.patches

        if self.config in ["LSN", "USN"]:
            if self.settings["grid_settings"]["up_down_symmetry"]:
                self.patch_matrix = [
                    [[None], [None], [None], [None], [None], [None], [None], [None]],
                    [
                        [None],
                        p["A2"],
                        p["B2"],
                        [None],
                        [None],
                        p["E2"],
                        p["F2"],
                        [None],
                    ],
                    [
                        [None],
                        p["A1"],
                        p["B1"],
                        [None],
                        [None],
                        p["E1"],
                        p["F1"],
                        [None],
                    ],
                    [[None], [None], [None], [None], [None], [None], [None], [None]],
                ]
            else:
                self.patch_matrix = [
                    [[None], [None], [None], [None], [None], [None], [None], [None]],
                    [
                        [None],
                        p["A2"],
                        p["B2"],
                        p["C2"],
                        p["D2"],
                        p["E2"],
                        p["F2"],
                        [None],
                    ],
                    [
                        [None],
                        p["A1"],
                        p["B1"],
                        p["C1"],
                        p["D1"],
                        p["E1"],
                        p["F1"],
                        [None],
                    ],
                    [[None], [None], [None], [None], [None], [None], [None], [None]],
                ]

        elif self.config in ["SF45", "SF75", "SF105", "SF135"]:
            if self.settings["grid_settings"]["up_down_symmetry"]:
                self.patch_matrix = [
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                    [
                        [None],
                        p["A3"],
                        p["B3"],
                        [None],
                        [None],
                        p["E3"],
                        p["F3"],
                        p["G3"],
                        [None],
                        [None],
                        p["H3"],
                        p["I3"],
                        [None],
                    ],
                    [
                        [None],
                        p["A2"],
                        p["B2"],
                        [None],
                        [None],
                        p["E2"],
                        p["F2"],
                        p["G2"],
                        [None],
                        [None],
                        p["H2"],
                        p["I2"],
                        [None],
                    ],
                    [
                        [None],
                        p["A1"],
                        p["B1"],
                        [None],
                        [None],
                        p["E1"],
                        p["F1"],
                        p["G1"],
                        [None],
                        [None],
                        p["H1"],
                        p["I1"],
                        [None],
                    ],
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                ]
            else:
                self.patch_matrix = [
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                    [
                        [None],
                        p["A3"],
                        p["B3"],
                        p["C3"],
                        p["D3"],
                        p["E3"],
                        p["F3"],
                        p["G3"],
                        [None],
                        [None],
                        p["H3"],
                        p["I3"],
                        [None],
                    ],
                    [
                        [None],
                        p["A2"],
                        p["B2"],
                        p["C2"],
                        p["D2"],
                        p["E2"],
                        p["F2"],
                        p["G2"],
                        [None],
                        [None],
                        p["H2"],
                        p["I2"],
                        [None],
                    ],
                    [
                        [None],
                        p["A1"],
                        p["B1"],
                        p["C1"],
                        p["D1"],
                        p["E1"],
                        p["F1"],
                        p["G1"],
                        [None],
                        [None],
                        p["H1"],
                        p["I1"],
                        [None],
                    ],
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                ]
        elif self.config in ["SF15", "SF165"]:
            if self.settings["grid_settings"]["up_down_symmetry"]:
                self.patch_matrix = [
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                    [
                        [None],
                        p["A3"],
                        p["B3"],
                        [None],
                        [None],
                        p["E3"],
                        p["F3"],
                        [None],
                        [None],
                        p["G3"],
                        p["H3"],
                        p["I3"],
                        [None],
                    ],
                    [
                        [None],
                        p["A2"],
                        p["B2"],
                        [None],
                        [None],
                        p["E2"],
                        p["F2"],
                        [None],
                        [None],
                        p["G2"],
                        p["H2"],
                        p["I2"],
                        [None],
                    ],
                    [
                        [None],
                        p["A1"],
                        p["B1"],
                        [None],
                        [None],
                        p["E1"],
                        p["F1"],
                        [None],
                        [None],
                        p["G1"],
                        p["H1"],
                        p["I1"],
                        [None],
                    ],
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                ]
            else:
                self.patch_matrix = [
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                    [
                        [None],
                        p["A3"],
                        p["B3"],
                        p["C3"],
                        p["D3"],
                        p["E3"],
                        p["F3"],
                        [None],
                        [None],
                        p["G3"],
                        p["H3"],
                        p["I3"],
                        [None],
                    ],
                    [
                        [None],
                        p["A2"],
                        p["B2"],
                        p["C2"],
                        p["D2"],
                        p["E2"],
                        p["F2"],
                        [None],
                        [None],
                        p["G2"],
                        p["H2"],
                        p["I2"],
                        [None],
                    ],
                    [
                        [None],
                        p["A1"],
                        p["B1"],
                        p["C1"],
                        p["D1"],
                        p["E1"],
                        p["F1"],
                        [None],
                        [None],
                        p["G1"],
                        p["H1"],
                        p["I1"],
                        [None],
                    ],
                    [
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                        [None],
                    ],
                ]

        elif self.config in ["UDN"]:
            self.patch_matrix = [
                [
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                ],
                [
                    [None],
                    p["A3"],
                    p["B3"],
                    p["C3"],
                    p["D3"],
                    [None],
                    [None],
                    p["E3"],
                    p["F3"],
                    p["G3"],
                    p["H3"],
                    [None],
                ],
                [
                    [None],
                    p["A2"],
                    p["B2"],
                    p["C2"],
                    p["D2"],
                    [None],
                    [None],
                    p["E2"],
                    p["F2"],
                    p["G2"],
                    p["H2"],
                    [None],
                ],
                [
                    [None],
                    p["A1"],
                    p["B1"],
                    p["C1"],
                    p["D1"],
                    [None],
                    [None],
                    p["E1"],
                    p["F1"],
                    p["G1"],
                    p["H1"],
                    [None],
                ],
                [
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                    [None],
                ],
            ]

        return self.patch_matrix

    def concat_grid(self, guard_cell_eps: float = 1e-3) -> None:
        """
        Concatenate a refined Patch map into a global grid.

        This method take grid data and represents it into Fortran
        formatted arrays that will be written to gridue.

        Adding of guard cells is done in this method as well.

        Parameters
        ----------
        guard_cell_eps : float, optional
            Determines the size of guard cell padding.
        """

        patch_matrix = self.patch_matrix

        for patch in self.patches.values():
            patch.npol = len(patch.cell_grid[0]) + 1
            patch.nrad = len(patch.cell_grid) + 1

        if self.parent.settings["grid_settings"]["up_down_symmetry"] is True:

            if self.parent.settings["grid_settings"]["num_xpt"] == 1:
                self.rm, self.zm = self._concat_two_patch_regions(
                    patch_matrix, pindex1=3, pindex2=5, pindex3=7, rindex=3
                )
            elif self.parent.settings["grid_settings"]["num_xpt"] == 2:
                if self.config in ["SF45", "SF75", "SF105", "SF135"]:
                    self.rm, self.zm = self._concat_three_patch_regions(
                        patch_matrix,
                        pindex1=3,
                        pindex2=5,
                        pindex3=8,
                        pindex4=10,
                        pindex5=12,
                        rindex=4,
                    )
                elif self.config in ["SF15", "SF165"]:
                    self.rm, self.zm = self._concat_three_patch_regions(
                        patch_matrix,
                        pindex1=3,
                        pindex2=5,
                        pindex3=7,
                        pindex4=9,
                        pindex5=12,
                        rindex=4,
                    )
                elif self.config in ["UDN"]:
                    # Not yet implemented
                    pass

        elif self.parent.settings["grid_settings"]["up_down_symmetry"] is False:
            if self.parent.settings["grid_settings"]["num_xpt"] == 1:
                self.rm, self.zm = self._concat_one_patch_region(patch_matrix)
            elif self.parent.settings["grid_settings"]["num_xpt"] == 2:
                if self.config in ["SF45", "SF75", "SF105", "SF135"]:
                    self.rm, self.zm = self._concat_two_patch_regions(
                        patch_matrix, pindex1=8, pindex2=10, pindex3=12, rindex=4
                    )
                elif self.config in ["SF15", "SF165"]:
                    self.rm, self.zm = self._concat_two_patch_regions(
                        patch_matrix, pindex1=7, pindex2=9, pindex3=12, rindex=4
                    )
                elif self.config in ["UDN"]:
                    self.rm, self.zm = self._concat_two_patch_regions(
                        patch_matrix, pindex1=5, pindex2=7, pindex3=11, rindex=4
                    )

    def _add_guardc(self, cell_map, ixlb, ixrb, nxpt=1, eps=1e-3):
        """
        Add guard cells to the cell array.
        """

        def set_guard(cell_map, ix, iy, eps, boundary):
            # Note: 'USN' and 'right' is really just 'LSN' and 'left' settings.
            # TODO: Edit the code to reflect this at some point so the next reader is not overwhelmed.
            if boundary == "left":
                ixn = ix + 1
                iyn = iy
                cell_map[ix][iy][1] = cell_map[ixn][iyn][1] + eps * (
                    cell_map[ixn][iyn][1] - cell_map[ixn][iyn][2]
                )
                cell_map[ix][iy][2] = cell_map[ixn][iyn][1]
                cell_map[ix][iy][3] = cell_map[ixn][iyn][3] + eps * (
                    cell_map[ixn][iyn][3] - cell_map[ixn][iyn][4]
                )
                cell_map[ix][iy][4] = cell_map[ixn][iyn][3]
                cell_map[ix][iy][0] = 0.25 * (
                    cell_map[ix][iy][1]
                    + cell_map[ix][iy][2]
                    + cell_map[ix][iy][3]
                    + cell_map[ix][iy][4]
                )

            elif boundary == "right":
                ixn = ix - 1
                iyn = iy
                cell_map[ix][iy][2] = cell_map[ixn][iyn][2] + eps * (
                    cell_map[ixn][iyn][2] - cell_map[ixn][iyn][1]
                )
                cell_map[ix][iy][1] = cell_map[ixn][iyn][2]
                cell_map[ix][iy][4] = cell_map[ixn][iyn][4] + eps * (
                    cell_map[ixn][iyn][4] - cell_map[ixn][iyn][3]
                )
                cell_map[ix][iy][3] = cell_map[ixn][iyn][4]
                cell_map[ix][iy][0] = 0.25 * (
                    cell_map[ix][iy][1]
                    + cell_map[ix][iy][2]
                    + cell_map[ix][iy][3]
                    + cell_map[ix][iy][4]
                )

            elif boundary == "bottom":
                ixn = ix
                iyn = iy + 1
                cell_map[ix][iy][1] = cell_map[ixn][iyn][1] + eps * (
                    cell_map[ixn][iyn][1] - cell_map[ixn][iyn][3]
                )
                cell_map[ix][iy][3] = cell_map[ixn][iyn][1]
                cell_map[ix][iy][2] = cell_map[ixn][iyn][2] + eps * (
                    cell_map[ixn][iyn][2] - cell_map[ixn][iyn][4]
                )
                cell_map[ix][iy][4] = cell_map[ixn][iyn][2]
                cell_map[ix][iy][0] = 0.25 * (
                    cell_map[ix][iy][1]
                    + cell_map[ix][iy][2]
                    + cell_map[ix][iy][3]
                    + cell_map[ix][iy][4]
                )
            elif boundary == "top":
                ixn = ix
                iyn = iy - 1
                cell_map[ix][iy][3] = cell_map[ixn][iyn][3] + eps * (
                    cell_map[ixn][iyn][3] - cell_map[ixn][iyn][1]
                )
                cell_map[ix][iy][1] = cell_map[ixn][iyn][3]
                cell_map[ix][iy][4] = cell_map[ixn][iyn][4] + eps * (
                    cell_map[ixn][iyn][4] - cell_map[ixn][iyn][2]
                )
                cell_map[ix][iy][2] = cell_map[ixn][iyn][4]
                cell_map[ix][iy][0] = 0.25 * (
                    cell_map[ix][iy][1]
                    + cell_map[ix][iy][2]
                    + cell_map[ix][iy][3]
                    + cell_map[ix][iy][4]
                )

            return cell_map

        np = len(cell_map) - 2
        nr = len(cell_map[0]) - 2

        for iy in range(1, nr + 1):
            ix = ixlb
            cell_map = set_guard(cell_map, ix, iy, eps, boundary="left")
            ix = ixrb + 1
            cell_map = set_guard(cell_map, ix, iy, eps, boundary="right")

        for ix in range(np + 2):
            iy = 0
            cell_map = set_guard(cell_map, ix, iy, eps, boundary="bottom")
            iy = nr + 1
            cell_map = set_guard(cell_map, ix, iy, eps, boundary="top")

        return cell_map

    def _concat_one_patch_region(self, patch_matrix: list):
        """Concatenate the grid cells from all patches grouped as a single patch region

        Parameters
        ----------
        patch_matrix: list
            Matrix containing all patches including boundary regions

        Returns
        -------
            rm, zm
        """
        np_total = int(np.sum([patch.npol - 1 for patch in patch_matrix[1][1:-1]])) + 2
        nr_total = int(np.sum([patch[1].nrad - 1 for patch in patch_matrix[1:3]])) + 2

        rm = np.zeros((np_total, nr_total, 5), order="F")
        zm = np.zeros((np_total, nr_total, 5), order="F")

        ixcell = 0
        jycell = 0

        # Iterate over all the patches in our SNL configuration (we exclude guard cells denoted by '[None]')
        for ixp in range(1, 7):

            nr_sum = 0
            for jyp in range(1, 3):
                # Point to the current patch we are operating on.
                local_patch = patch_matrix[jyp][ixp]
                nr_sum += local_patch.nrad - 1

                # Access the grid that is contained within this local_patch.
                # ixl - number of poloidal cells in the patch.
                for ixl in range(len(local_patch.cell_grid[0])):
                    # jyl - number of radial cells in the patch
                    for jyl in range(len(local_patch.cell_grid)):

                        ixcell = (
                            int(
                                np.sum(
                                    [
                                        patch.npol - 1
                                        for patch in patch_matrix[1][1 : ixp + 1]
                                    ]
                                )
                            )
                            - len(local_patch.cell_grid[0])
                            + ixl
                            + 1
                        )

                        jycell = nr_sum - (local_patch.nrad - 1) + jyl + 1

                        ind = 0
                        for coor in ["CENTER", "SW", "SE", "NW", "NE"]:
                            rm[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].x
                            )
                            zm[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].y
                            )
                            ind += 1

        # Flip indices into gridue format.
        for i in range(len(rm)):
            rm[i] = rm[i][::-1]
        for i in range(len(zm)):
            zm[i] = zm[i][::-1]

        # Add guard cells to the concatenated grid.
        ixrb = len(rm) - 2
        ixlb = 0
        rm = self._add_guardc(rm, ixlb, ixrb)
        zm = self._add_guardc(zm, ixlb, ixrb)

        try:
            debug = self.settings["DEBUG"]["visual"]["gridue"]
        except:
            debug = False

        if debug:
            self._animate_grid()

        return rm, zm

    def _concat_two_patch_regions(
        self, patch_matrix: list, pindex1: int, pindex2: int, pindex3: int, rindex: int
    ):
        """Concatenate the grid cells from all patches grouped as two separate patch regions

        Parameters
        ----------
        patch_matrix: list
            Matrix containing all patches including boundary regions
        pindex1: int
            Poloidal patch index for the end of the first patch region
        pindex2: int
            Poloidal patch index for the start of the second patch region
        pindex3: int
            Poloidal patch index for the end of the second patch region
        rindex: int
            Number of radial patches

        Returns
        -------
            rm, zm
        """

        # Total number of poloidal indices in all subgrids.
        np_total1 = (
            int(np.sum([patch.npol - 1 for patch in patch_matrix[1][1:pindex1]])) + 2
        )

        # Total number of radial indices in all subgrids.
        nr_total1 = (
            int(np.sum([patch[1].nrad - 1 for patch in patch_matrix[1:rindex]])) + 2
        )

        # Total number of poloidal indices in all subgrids.
        np_total2 = (
            int(np.sum([patch.npol - 1 for patch in patch_matrix[1][pindex2:pindex3]]))
            + 2
        )

        # Total number of radial indices in all subgrids.
        nr_total2 = (
            int(np.sum([patch[pindex2].nrad - 1 for patch in patch_matrix[1:rindex]]))
            + 2
        )

        rm1 = np.zeros((np_total1, nr_total1, 5), order="F")
        zm1 = np.zeros((np_total1, nr_total1, 5), order="F")
        rm2 = np.zeros((np_total2, nr_total2, 5), order="F")
        zm2 = np.zeros((np_total2, nr_total2, 5), order="F")

        ixcell = 0
        jycell = 0

        # Iterate over all the patches in our DNL configuration (we exclude guard cells denoted by '[None]')
        for ixp in range(1, pindex1):

            nr_sum = 0
            for jyp in range(1, 4):
                # Point to the current patch we are operating on.
                local_patch = patch_matrix[jyp][ixp]

                if local_patch == [None]:
                    continue

                nr_sum += local_patch.nrad - 1

                # Access the grid that is contained within this local_patch.
                # ixl - number of poloidal cells in the patch.
                for ixl in range(len(local_patch.cell_grid[0])):
                    # jyl - number of radial cells in the patch
                    for jyl in range(len(local_patch.cell_grid)):

                        ixcell = (
                            int(
                                np.sum(
                                    [
                                        patch.npol - 1
                                        for patch in patch_matrix[1][1 : ixp + 1]
                                    ]
                                )
                            )
                            - len(local_patch.cell_grid[0])
                            + ixl
                            + 1
                        )

                        jycell = nr_sum - (local_patch.nrad - 1) + jyl + 1

                        ind = 0
                        for coor in ["CENTER", "SW", "SE", "NW", "NE"]:
                            rm1[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].x
                            )
                            zm1[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].y
                            )
                            ind += 1

        ixcell = 0
        jycell = 0

        for ixp in range(pindex2, pindex3):

            nr_sum = 0
            for jyp in range(1, 4):
                # Point to the current patch we are operating on.
                local_patch = patch_matrix[jyp][ixp]

                if local_patch == [None]:
                    continue

                nr_sum += local_patch.nrad - 1

                # Access the grid that is contained within this local_patch.
                # ixl - number of poloidal cells in the patch.
                for ixl in range(len(local_patch.cell_grid[0])):
                    # jyl - number of radial cells in the patch
                    for jyl in range(len(local_patch.cell_grid)):

                        ixcell = (
                            int(
                                np.sum(
                                    [
                                        patch.npol - 1
                                        for patch in patch_matrix[1][pindex2 : ixp + 1]
                                    ]
                                )
                            )
                            - len(local_patch.cell_grid[0])
                            + ixl
                            + 1
                        )

                        jycell = nr_sum - (local_patch.nrad - 1) + jyl + 1

                        ind = 0
                        for coor in ["CENTER", "SW", "SE", "NW", "NE"]:
                            rm2[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].x
                            )
                            zm2[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].y
                            )
                            ind += 1

        # Flip indices into gridue format.
        for i in range(len(rm1)):
            rm1[i] = rm1[i][::-1]
        for i in range(len(zm1)):
            zm1[i] = zm1[i][::-1]
        for i in range(len(rm2)):
            rm2[i] = rm2[i][::-1]
        for i in range(len(zm2)):
            zm2[i] = zm2[i][::-1]

        # Add guard cells to the concatenated grid.
        ixrb1 = len(rm1) - 2
        ixlb1 = 0
        ixrb2 = len(rm2) - 2
        ixlb2 = 0

        rm1 = self._add_guardc(rm1, ixlb1, ixrb1)
        zm1 = self._add_guardc(zm1, ixlb1, ixrb1)
        rm2 = self._add_guardc(rm2, ixlb2, ixrb2)
        zm2 = self._add_guardc(zm2, ixlb2, ixrb2)

        rm = np.concatenate((rm1, rm2))
        zm = np.concatenate((zm1, zm2))

        try:
            debug = self.yaml["DEBUG"]["visual"]["gridue"]
        except:
            debug = False

        if debug:
            self._animate_grid()

        return rm, zm

    def _concat_three_patch_regions(
        self,
        patch_matrix: list,
        pindex1: int,
        pindex2: int,
        pindex3: int,
        pindex4: int,
        pindex5: int,
        rindex: int,
    ):
        """Concatenate the grid cells from all patches grouped as three separate patch regions

        Parameters
        ----------
        patch_matrix: list
            Matrix containing all patches including boundary regions
        pindex1: int
            Poloidal patch index for the end of the first patch region
        pindex2: int
            Poloidal patch index for the start of the second patch region
        pindex3: int
            Poloidal patch index for the end of the second patch region
        pindex4: int
            Poloidal patch index for the start of the third patch region
        pindex5: int
            Poloidal patch index for the end of the third patch region
        rindex: int
            Number of radial patches

        Returns
        -------
            rm, zm
        """

        # Total number of poloidal indices in all subgrids.
        np_total1 = (
            int(np.sum([patch.npol - 1 for patch in patch_matrix[1][1:pindex1]])) + 2
        )

        # Total number of radial indices in all subgrids.
        nr_total1 = (
            int(np.sum([patch[1].nrad - 1 for patch in patch_matrix[1:rindex]])) + 2
        )

        # Total number of poloidal indices in all subgrids.
        np_total2 = (
            int(np.sum([patch.npol - 1 for patch in patch_matrix[1][pindex2:pindex3]]))
            + 2
        )

        # Total number of radial indices in all subgrids.
        nr_total2 = (
            int(np.sum([patch[pindex2].nrad - 1 for patch in patch_matrix[1:rindex]]))
            + 2
        )

        # Total number of poloidal indices in all subgrids.
        np_total3 = (
            int(np.sum([patch.npol - 1 for patch in patch_matrix[1][pindex4:pindex5]]))
            + 2
        )

        # Total number of radial indices in all subgrids.
        nr_total3 = (
            int(np.sum([patch[pindex4].nrad - 1 for patch in patch_matrix[1:rindex]]))
            + 2
        )

        rm1 = np.zeros((np_total1, nr_total1, 5), order="F")
        zm1 = np.zeros((np_total1, nr_total1, 5), order="F")
        rm2 = np.zeros((np_total2, nr_total2, 5), order="F")
        zm2 = np.zeros((np_total2, nr_total2, 5), order="F")
        rm3 = np.zeros((np_total3, nr_total3, 5), order="F")
        zm3 = np.zeros((np_total3, nr_total3, 5), order="F")

        ixcell = 0
        jycell = 0

        # Iterate over all the patches in our DNL configuration (we exclude guard cells denoted by '[None]')
        for ixp in range(1, pindex1):

            nr_sum = 0
            for jyp in range(1, rindex):
                # Point to the current patch we are operating on.
                local_patch = patch_matrix[jyp][ixp]

                if local_patch == [None]:
                    continue

                nr_sum += local_patch.nrad - 1

                # Access the grid that is contained within this local_patch.
                # ixl - number of poloidal cells in the patch.
                for ixl in range(len(local_patch.cell_grid[0])):
                    # jyl - number of radial cells in the patch
                    for jyl in range(len(local_patch.cell_grid)):

                        ixcell = (
                            int(
                                np.sum(
                                    [
                                        patch.npol - 1
                                        for patch in patch_matrix[1][1 : ixp + 1]
                                    ]
                                )
                            )
                            - len(local_patch.cell_grid[0])
                            + ixl
                            + 1
                        )

                        jycell = nr_sum - (local_patch.nrad - 1) + jyl + 1

                        ind = 0
                        for coor in ["CENTER", "SW", "SE", "NW", "NE"]:
                            rm1[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].x
                            )
                            zm1[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].y
                            )
                            ind += 1

        ixcell = 0
        jycell = 0

        for ixp in range(pindex2, pindex3):

            nr_sum = 0
            for jyp in range(1, rindex):
                # Point to the current patch we are operating on.
                local_patch = patch_matrix[jyp][ixp]

                if local_patch == [None]:
                    continue

                nr_sum += local_patch.nrad - 1

                # Access the grid that is contained within this local_patch.
                # ixl - number of poloidal cells in the patch.
                for ixl in range(len(local_patch.cell_grid[0])):
                    # jyl - number of radial cells in the patch
                    for jyl in range(len(local_patch.cell_grid)):

                        ixcell = (
                            int(
                                np.sum(
                                    [
                                        patch.npol - 1
                                        for patch in patch_matrix[1][pindex2 : ixp + 1]
                                    ]
                                )
                            )
                            - len(local_patch.cell_grid[0])
                            + ixl
                            + 1
                        )

                        jycell = nr_sum - (local_patch.nrad - 1) + jyl + 1

                        ind = 0
                        for coor in ["CENTER", "SW", "SE", "NW", "NE"]:
                            rm2[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].x
                            )
                            zm2[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].y
                            )
                            ind += 1

        ixcell = 0
        jycell = 0

        for ixp in range(pindex4, pindex5):

            nr_sum = 0
            for jyp in range(1, rindex):
                # Point to the current patch we are operating on.
                local_patch = patch_matrix[jyp][ixp]

                if local_patch == [None]:
                    continue

                nr_sum += local_patch.nrad - 1

                # Access the grid that is contained within this local_patch.
                # ixl - number of poloidal cells in the patch.
                for ixl in range(len(local_patch.cell_grid[0])):
                    # jyl - number of radial cells in the patch
                    for jyl in range(len(local_patch.cell_grid)):

                        ixcell = (
                            int(
                                np.sum(
                                    [
                                        patch.npol - 1
                                        for patch in patch_matrix[1][pindex4 : ixp + 1]
                                    ]
                                )
                            )
                            - len(local_patch.cell_grid[0])
                            + ixl
                            + 1
                        )

                        jycell = nr_sum - (local_patch.nrad - 1) + jyl + 1

                        ind = 0
                        for coor in ["CENTER", "SW", "SE", "NW", "NE"]:
                            rm3[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].x
                            )
                            zm3[ixcell][jycell][ind] = (
                                local_patch.cell_grid[jyl][ixl].vertices[coor].y
                            )
                            ind += 1

        # Flip indices into gridue format.
        for i in range(len(rm1)):
            rm1[i] = rm1[i][::-1]
        for i in range(len(zm1)):
            zm1[i] = zm1[i][::-1]
        for i in range(len(rm2)):
            rm2[i] = rm2[i][::-1]
        for i in range(len(zm2)):
            zm2[i] = zm2[i][::-1]
        for i in range(len(rm3)):
            rm3[i] = rm3[i][::-1]
        for i in range(len(zm3)):
            zm3[i] = zm3[i][::-1]

        # Add guard cells to the concatenated grid.
        ixrb1 = len(rm1) - 2
        ixlb1 = 0
        ixrb2 = len(rm2) - 2
        ixlb2 = 0
        ixrb3 = len(rm3) - 2
        ixlb3 = 0

        rm1 = self._add_guardc(rm1, ixlb1, ixrb1)
        zm1 = self._add_guardc(zm1, ixlb1, ixrb1)
        rm2 = self._add_guardc(rm2, ixlb2, ixrb2)
        zm2 = self._add_guardc(zm2, ixlb2, ixrb2)
        rm3 = self._add_guardc(rm3, ixlb3, ixrb3)
        zm3 = self._add_guardc(zm3, ixlb3, ixrb3)

        rm = np.concatenate((rm1, rm2, rm3))
        zm = np.concatenate((zm1, zm2, zm3))

        try:
            debug = self.yaml["DEBUG"]["visual"]["gridue"]
        except:
            debug = False

        if debug:
            self._animate_grid()

        return rm, zm
