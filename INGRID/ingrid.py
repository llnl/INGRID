#!/usr/bin/env pythonu
# -*- coding: utf-8 -*-
"""Ingrid module for interfacing all grid generator capabilities.

This module contains the Ingrid class that drives all code functionality.

The Ingrid class is to encapsulate all functionality and allow the user to
easily take advantage of advanced features of the code.

"""
from __future__ import print_function, division, absolute_import
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    pass
import matplotlib.pyplot as plt
import pathlib
import functools
from scipy.optimize import root, minimize

import yaml as yml
import os
from pathlib import Path
from time import time, perf_counter

from INGRID.interpol import EfitData
from INGRID.utils import IngridUtils
from INGRID.topologies.snl import SNL
from INGRID.topologies.sf15 import SF15
from INGRID.topologies.sf45 import SF45
from INGRID.topologies.sf75 import SF75
from INGRID.topologies.sf105 import SF105
from INGRID.topologies.sf135 import SF135
from INGRID.topologies.sf165 import SF165
from INGRID.topologies.udn import UDN
from INGRID.geometry import Point, Line
#/Users/sruiz/Dev/PhD/Ingrid_latest/INGRID/INGRID/gridue-to-bout/gridue_to_bout.py
from INGRID.gridue_converter.gridue_to_bout import Convert_grids

def QuickStart() -> None:
    """
    Start Ingrid in gui mode with default settings.

    Rather than providing data to the class constructor, a user can opt to
    start Ingrid immediately in it's gui form. This is useful for users who
    are not familiar with the Ingrid class and it's capabiliites. Advanced
    users may still find QuickStart useful, but would also use the code in
    self-authored scripts.

    """
    QuickGrid = Ingrid()
    QuickGrid.StartGUI()


class Ingrid(IngridUtils):
    """
    The driver class of the grid generator. A user will be able to load
    all experimental data, create patch maps, create grids, and export
    grids with the methods available. The GUI version of the Ingrid code
    is also accessed exclusively through this class.

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
    settings : dict
        Dictionary representation of the settings file.

    PlateData : dict
        Dictionary mapping target plates to their
        corresponding class 'Line' objects.

    LimiterData : Line
        Line class object representing tokamak limiter.

    magx : tuple
        (R, Z) coordinates of magnetic-axis (float entries).

    xpt1 : tuple
        (R, Z) coordinates of primary x-point (float entries).

    xpt2 : tuple
        (R, Z) coordinates of secondary x-point (float entries).

    PsiNorm : EfitData
        'EfitData' class object containing normalized
        psi data calculated from class attribute 'PsiUNorm'.

    CurrentTopology : Topology
        Tokamak magnetic configuration type currently being operated on.
        Class 'Topology' acts as thebase class for all magnetic
        configurations.

    """

    def __init__(self, settings: dict = {}, **kwargs):
        IngridUtils.__init__(self, settings, **kwargs)
        self.PrintSummaryInput()

    def _timer(func):
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = perf_counter()
            value = func(*args, **kwargs)
            end_time = perf_counter()
            run_time = end_time - start_time
            print(f"--> Finished {func.__name__!r} in {run_time:.6f}s...")
            return value
        return wrapper_timer

    def StartGUI(self, test_initialization: bool = False) -> None:
        """
        Start GUI for Ingrid.

        Will assume usage on a machine with tk GUI capabilities.
        No prior settings file is required as the user will be
        prompted with an option to generate a new file.

        Parameters
        ----------
        test_initialization : optional
            Flag to trigger a TkInitializationSuccess exception when
            entering the method calling tk.mainloop(). Suggests a successful
            initialization of the IngridGUI class on the tk side of things.
        """
        from INGRID.gui.ingrid_gui import IngridGUI
        self.IngridWindow = IngridGUI(IngridSession=self)
        self.IngridWindow.Run(test_initialization=test_initialization)

    def RefreshSettings(self):
        try:
            self.CurrentTopology.RefreshSettings()
        except:
            pass

    def SaveSettingsFile(self, fname: str = '', settings: dict = {}) -> 'Path':
        """
        Save a new settings .yml file.

        Parameters
        ----------
        fname : optional
            Name of new settings '.yml' file.
            If default value of '', then Ingrid will generate
            a '.yml' file named 'INGRID_Session' appended with
            a timestamp.

        settings : optional
            Ingrid settings dictionary to dump into
            the '.yml' file. Defaults to empty dict which produces a
            template settings file.

        Returns
        -------
            A Path instance representing the saved YAML file.
        """

        if fname == '':
            fname = 'INGRID_Session' + str(int(time())) + '.yml'

        fpath = Path(fname)

        if fpath.suffix == '':
            fname += '.yml'
        elif fpath.suffix != '.yml':
            fname = fname[:-4] + '.yml'

        self.PopulateSettings(settings)

        with open(fname, 'w') as f:
            yml.dump(self.settings, f)

        return fpath

    def PrintSummaryInput(self) -> None:
        """
        Print a summary of the currently loaded data files

        Will print relevant EQDSK, patch data files, target plate files,
        and limiter files.
        """
        print('')
        print('Equilibrium File:', self.settings['eqdsk'])

        if (self.settings['patch_data']['use_file']) \
            and (Path(self.settings['patch_data']['file']).is_file()) \
                and (Path(self.settings['patch_data']['file']).suffix == '.npy'):

            print('Patch data-file:')
            print(' # Patch data-file:', self.settings['patch_data']['file'])

        if self.settings['grid_settings']['patch_generation']['strike_pt_loc'] == 'limiter':
            print('Limiter File:')

            if self.settings['limiter']['file'] == '' or Path(self.settings['limiter']['file']).is_file() is False:

                s = 'in eqdsk file.' if self.settings['limiter']['use_efit_bounds'] is False else 'from efit bounds.'
                print(' # Limiter:', 'Using eqdsk (RLIM, ZLIM) coordinates ' + s)
            else:
                print(' # Limiter:', self.settings['limiter']['file'])
        else:
            print('Target Files:')
            print(' # W1:', self.settings['target_plates']['plate_W1']['file'])
            print(' # E1:', self.settings['target_plates']['plate_E1']['file'])
            print(' # E2:', self.settings['target_plates']['plate_E2']['file'])
            print(' # W2:', self.settings['target_plates']['plate_W2']['file'])
        print('')

    def PrintSummaryParams(self) -> None:
        """
        Print a summary of key settings values.
        """
        print('')
        print('Summary:')
        print(' # Number of x-points:', self.settings['grid_settings']['num_xpt'])
        print(' # Using strike geometry:', self.settings['grid_settings']['patch_generation']['strike_pt_loc'])
        print(' # Use patch data-file:', self.settings['patch_data']['use_file'])
        print('')
        self.PrintSummaryInput()

    def SetGeometry(self, geo_items: dict, rshift: float = 0.0, zshift: float = 0.0) -> None:
        """
        Define and load the tokamak strike geometry into the Ingrid object.
        Allows the user to set target plate data and/or limiter data that
        will be used to generate a patch map.

        Parameters
        ----------
        geo_items : dict
            Argument dict specifying which item(s) to set and by what way.

        rshift : float, optional
            Translate 'geo_items' to coordinate R',
            with R' = R + rshift. Will override all 'rshift' provided entries
            in 'geo_items'.

        zshift : float, optional
            Translate 'geo_items' to coordinate Z',
            with Z' = Z + zshift. Will override all 'zshift' provided entries
            in 'geo_items'.

        Notes
        -----
        Multiple geometry items can be set at once, but the following
        key-value format must be obeyed for any number of entries:

        ``{geo_key: geo_item}``

        All keys for 'geo_items' are type 'str'. Accepted key
        values are as follows:

        ======== =====================
        Geometry   Accepted Keys (str)
        -------- ---------------------
        Plate W1 ``plate_W1``, ``W1``
        Plate E1 ``plate_E1``, ``W1``
        Plate W2 ``plate_W2``, ``W1``
        Plate E2 ``plate_E2``, ``W1``
        Limiter  ``limiter``, ``wall``
        ======== =====================

        **The above keys are NOT case sensitive.**

        Corresponding key values can vary in data type depending
        on means of setting geometry. The types and their usage
        are as follows:

        'str': Path to '.txt' file containing coordinate data or
        to Ingrid generated '.npy' file (obtained via methods

        'SaveLimiterData' and 'SaveTargetPlateData').
        Note: When setting Limiter geometry, the user can provide the
        value 'default' to set limiter data to that which is contained
        in the loaded neqdsk file.

        'list', 'tuple': Iterables provided as values must be
        of length == 2. The first entry corresponds to R coordinate
        information, and the second entry corresponds to Z coordinate
        information. This information can be in the form of a list or
        NumPy array with shape == (N,).

        'dict': One can map to a dictionary taking on a variety of
        formats depending on need.

        Setting geometry with explicit RZ coordinates:
        {'R': R_data, 'Z': z_data}
        Where R_data and Z_data are in the form of a list or NumPy
        array with shape == (N,) as above.

        Setting geometry with data from external file:
        {'file': str}

        Notes: The above dict option support keys 'rshift' and 'zshift'
        for the user to provide transformations to individual geometry
        items (see examples).

        Because the core 'settings' attribute contained by the
        Ingrid class contains dict structures itself, the user can also
        provide settings['limiter'] and settings['target_plates'][k]
        to method 'SetGeometry' (where k corresponds to a plate key).


        Examples
        --------
        Setting default limiter data contained in loaded neqdsk
        file:

        >>> MyIG = Ingrid()
        >>> MyIG.SetGeometry({'limiter': 'default'})


        Setting target plate 'E1' with numpy array:

        >>> MyIG = Ingrid()
        >>> MyIG.SetGeometry({'E1': 'E1_data.npy'})


        Setting limiter data with Ingrid '.npy' file:

        >>> MyIG = Ingrid()
        >>> MyIG.SetGeometry({'limiter': 'LimiterData.npy'})


        Setting both target plates 'W1' and 'E1' with '.txt'
        and '.npy' files while only applying rshift and zshift
        to target plate 'E1'
        (Note the dict structure used for specifying 'E1'):

        >>> MyIG = Ingrid()
        >>> geometry_dict = {
        ... 'W1': 'W1_data.txt',
        ... 'E1': {
        ... 'file': 'E1_data.npy',
        ... 'rshift': 1.23, 'zshift': 3.14
        ... }
        ... }
        >>> MyIG.SetGeometry(geometry_dict)


        Setting plate 'W2' with user provided NumPy array
        for RZ coordinates:

        >>> MyIG = Ingrid()
        >>> R_data = my_numpy_array_1
        >>> Z_data = my_numpy_array_2
        >>> RZ_dict = {'R': my_numpy_array_1, 'Z': my_numpy_array_2}
        >>> MyIG.SetGeometry({'plate_W1': RZ_dict})

        Raises
        ------
        ValueError
            If file path provided does not lead to actual file.
        ValueError
            If file provided is not of suffix '.txt' or '.npy'.
        ValueError
            If invalid 'geo_items' key provided.
        ValueError
            If 'geo_items' dict value contains invalid key.
        ValueError
            If value associated with 'geo_items' key is not of
                supported data type.
        """

        def _parse_v_for_file_values(k_str, v_dict):
            if v_dict.get('file') not in [None, '', '.']:

                if Path(v['file']).is_file() is False:
                    raise ValueError(f"# File '{v_dict['file']}' provided in settings dict corresponding to '{k_str}' does not exist.")

                if Path(v_dict['file']).suffix == '.txt':
                    x_ret_val, y_ret_val = self.ParseTxtCoordinates(v_dict['file'])
                elif Path(v_dict['file']).suffix == '.npy':
                    x_ret_val, y_ret_val = np.load(v_dict['file'])
                else:
                    raise ValueError(f"# Invalid file extension '{Path(v_dict['file']).suffix}' used in SetGeometry.")

                return (x_ret_val, y_ret_val)

        def _parse_v_for_xy_coordinates(v_dict):
            x_flag = False
            y_flag = False
            if v_dict.get('x') is not None:
                x_ret_val = v_dict['x']
                x_flag = True
            if v_dict.get('X') is not None:
                x_ret_val = v_dict['X']
                x_flag = True
            if v_dict.get('r') is not None:
                x_ret_val = v_dict['r']
                x_flag = True
            if v_dict.get('R') is not None:
                x_ret_val = v_dict['R']
                x_flag = True
            if v_dict.get('y') is not None:
                y_ret_val = v_dict['y']
                y_flag = True
            if v_dict.get('Y') is not None:
                y_ret_val = v_dict['Y']
                y_flag = True
            if v_dict.get('z') is not None:
                y_ret_val = v_dict['z']
                y_flag = True
            if v_dict.get('Z') is not None:
                y_ret_val = v_dict['Z']
                y_flag = True

            if x_flag is True and y_flag is True:
                return (x_ret_val, y_ret_val)
            else:
                return None

        for k, v in geo_items.items():

            TargetPlate = False
            Limiter = False
            DefaultEQ = False

            if k.lower() in ['w1', 'westplate1', 'plate_w1']:
                TargetPlate = True
                k = 'plate_W1'
            elif k.lower() in ['e1', 'eastplate1', 'plate_e1']:
                TargetPlate = True
                k = 'plate_E1'
            elif k.lower() in ['w2', 'westplate2', 'plate_w2']:
                TargetPlate = True
                k = 'plate_W2'
            elif k.lower() in ['e2', 'eastplate2', 'plate_e2']:
                TargetPlate = True
                k = 'plate_E2'
            elif k.lower() in ['limiter', 'wall']:
                Limiter = True
                k = 'limiter'
            else:
                raise ValueError(f"# Invalid key '{k}' used in SetGeometry.")

            if type(v) == str:  # Path to file.
                if Limiter and v.lower() in ['default', 'eq']:
                    DefaultEQ = True
                elif Path(v).suffix == '.txt':
                    x, y = self.ParseTxtCoordinates(v)
                elif Path(v).suffix == '.npy':
                    x, y = np.load(v)
                else:
                    raise ValueError(f"# Invalid file extension '{Path(v).suffix}' used in SetGeometry.")

            elif type(v) == dict:  # Dict of settings.

                xy_result = _parse_v_for_xy_coordinates(v)

                if xy_result is None:  # Dict structure with settings file keys.
                    if TargetPlate:
                        if v.get('file') not in ['', '.', None] and Path(v['file']).is_file() is not False:  # Attempt to read file.
                            x, y = _parse_v_for_file_values(k, v)
                        else:
                            raise ValueError(f"# No data source for target plate '{k}' identified.")
                    elif Limiter:
                        if v.get('file') not in ['', '.', None] and Path(v['file']).is_file() is not False:  # Attempt to read file.
                            x, y = _parse_v_for_file_values(k, v)
                        else:
                            DefaultEQ = True
                elif type(xy_result) == tuple:
                    x, y = xy_result

                else:
                    raise ValueError(f"# Invalid key '{k}' was used within dic in SetGeometry.")

                if v.get('rshift') is not None and rshift == 0.0:
                    rshift = v['rshift']
                if v.get('zshift') is not None and zshift == 0.0:
                    zshift = v['zshift']

            elif type(v) in [list, tuple]:
                x, y = v

            else:
                raise ValueError(f"# Invalid data type {type(v)} was used for setting geometry.")

            if TargetPlate:
                self.SetTargetPlate({k: [x, y]}, rshift=rshift, zshift=zshift)
            elif Limiter and not DefaultEQ:
                self.SetLimiter(coordinates=[x, y], rshift=rshift, zshift=zshift)
            elif Limiter and DefaultEQ:
                self.SetLimiter(rshift=rshift, zshift=zshift)
            else:
                raise ValueError(f"# Invalid key '{k}' was used for setting geometry.")

    def SetTargetPlates(self) -> None:
        """
        Define target plate geometries based off of ``settings`` dict specifications.

        This is a convenience method that calls SetGeometry.
        """
        for plate in self.settings['target_plates']:
            try:
                self.SetGeometry({plate: self.settings['target_plates'][plate]['file']},
                                 rshift=self.settings['target_plates'][plate]['rshift'],
                                 zshift=self.settings['target_plates'][plate]['zshift'])
            except:
                continue

    def SaveGeometryData(self, geo_items: dict, timestamp: bool = False) -> None:
        """
        Save strike geometry Line object RZ coordinates as '.npy' file.

        Geometry data files created with this method can be used in the INGRID
        parameter file.


        Parameters
        ----------
        geo_items : dict
            A dictionary specifying which target plate to save and the
            file name/path. Said dictionary takes the following form:

            ``{geo_name: data_fname, ... }``

            Where both geo_name and data_fname are of type str. Multiple
            data_fname files can be saved at once in a manner similar to
            method 'SetGeometry' (see documentation for 'SetGeometry').

        timestamp : bool, optional
            Append a time stamp to the end of the files.

        Raises
        ------
        ValueError
            If 'geo_items' is not type 'dict'.
        ValueError
            If invalid 'geo_items' key provided.
        ValueError
            If data_fname entry is not of type 'str'.
        ValueError
            If data_fname entry is an empty string.
        ValueError
            If requested strike geometry Line to save has no data.
        """

        if type(settings) is dict:

            if timestamp is True:  # Get same timestamp for all files.
                f_tail = '_' + str(int(time()))
            else:
                f_tail = ''

            for k, v in settings.items():
                if k.lower() in ['w1', 'westplate1', 'plate_w1']:
                    k = 'plate_W1'
                elif k.lower() in ['e1', 'eastplate1', 'plate_e1']:
                    k = 'plate_E1'
                elif k.lower() in ['w2', 'westplate2', 'plate_w2']:
                    k = 'plate_W2'
                elif k.lower() in ['e2', 'eastplate2', 'plate_e2']:
                    k = 'plate_E2'
                elif k.lower() in ['limiter', 'wall']:
                    k = 'limiter'
                else:
                    raise ValueError(f"# Invalid key '{k}' provided for 'SaveGeometryData'")

                if type(v) is not str:
                    raise ValueError(f"# Value {v} associated with key '{k}' must be of type str")

                fpath = Path(v)

                if fpath.name.strip() == '':
                    raise ValueError(f"# Cannot save '{k}' to empty file name.")

                fname = fpath.fname

                if k == 'limiter':
                    geo = self.LimiterData
                else:
                    geo = self.PlateData[k]

                if type(geo) is Line:
                    R = np.array(geo.xval)
                    Z = np.array(geo.yval)
                else:
                    raise ValueError(f"# No Line object has been instantiated for {k}")

                fname += f_tail  # Add on f_tail value from start.

                np.save(fname, np.array([R, Z]))
                print(f"# Saved '{k}' plate data to file " + {fname} + ".npy")

        else:
            raise ValueError(f"# Argument 'geo_items' must be of type dict.")

    def LoadGeometryData(self, geo_items: dict) -> None:
        """
        Load strike geometry RZ coordinates from external file.

        Said external file must be of type '.txt' or a properly formatted
        '.npy' file.

        Parameters
        ----------
        geo_items : dict
            The argument dict specifying which strike geometry to load data into.

        Notes
        -----
        ``geo_items`` is of the form: ``{geo_name: data_fname, ... }``

        where both ``geo_name`` and ``data_fname`` are of type str.

        Multiple ``data_fname`` files can be loaded at once in a manner similar
        to method :meth:`~SetGeometry`.

        Raises
        ------
        ValueError
            If 'geo_items' is not type 'dict'.
        ValueError
            If invalid 'geo_items' key provided.
        ValueError
            If data_fname entry is not of type 'str'.
        ValueError
            If data_fname is not a file.
        ValueError
            If data_fname file is not of format '.txt' or '.npy'

        """
        if type(geo_items) is dict:

            for k, v in geo_items.keys():
                if k.lower() in ['w1', 'westplate1', 'plate_w1']:
                    k = 'plate_W1'
                elif k.lower() in ['e1', 'eastplate1', 'plate_e1']:
                    k = 'plate_E1'
                elif k.lower() in ['w2', 'westplate2', 'plate_w2']:
                    k = 'plate_W2'
                elif k.lower() in ['e2', 'eastplate2', 'plate_e2']:
                    k = 'plate_E2'
                elif k.lower() in ['limiter', 'wall']:
                    k = 'limiter'
                else:
                    raise ValueError(f"# Invalid key '{k}' provided for 'LoadGeometryData'")

                if type(v) is not str:
                    raise ValueError(f"# fname associated with key '{k}' must be of type {type('')}. Received type {type(v)}")

                fpath = Path(v)

                if fpath.is_file() is False:
                    raise ValueError(f"# fname associated with key '{k}' does not exist.")

                suffix = fpath.suffix
                fname = fpath.name

                if suffix == '.npy':
                    R, Z = np.load(fname)
                elif suffix == '.txt':
                    R, Z = self.ParseTxtCoordinates(fname)
                else:
                    raise ValueError(f"# File type '{suffix}' is not supported (requires '.txt', '.npy')")

                if k == 'limiter':
                    destination = self.LimiterData
                else:
                    destination = self.PlateData[k]

                if R[0] is None or Z[0] is None:
                    destination = None
                else:
                    destination = Line([Point(P) for P in zip(R, Z)])

                print(f"# Loaded '{k}' data from file " + fname)

        else:
            raise ValueError(f"# LoadGeometryData input must be of type dict with format {{GEO_KEY : FNAME}}")

    def ClearLegend(self, ax) -> None:
        """
        Safely remove the legend form the normalized psi data.
        """
        if ax.get_legend() is not None:
            [line.remove() for line in ax.get_legend().get_lines()]

    def RemovePlotLine(self, label: str, ax: object = None) -> None:
        if ax is None:
            ax = plt.gca()
        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == label]
        except:
            pass

    def RemovePlotPoint(self, label: str, ax: object = None) -> None:
        if ax is None:
            ax = plt.gca()
        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == label]
        except:
            pass

    def RemovePlotPatch(self, label: str, ax: object = None) -> None:
        if ax is None:
            ax = plt.gca()
        try:
            [ax_patch.remove() for ax_patch in ax.patches if ax_patch.get_label() == label]
        except:
            pass

    def PlotStrikeGeometry(self, ax: object = None) -> None:
        """
        Plot all strike geometry to be used for drawing the Patch Map.

        Checks the central INGRID ``settings`` attribute for whether
        ``settings['patch_generation']['strike_pt_loc']`` is ``True`` or if
        ``settings['grid_settings']['num_xpt']`` is equal to ``2`` in order
        to determine whether or not to plot the limiter geometry.

        Otherwise only any loaded target plates will be plotted.
        """

        if ax is None:
            ax = plt.gca()

        # In case previous plot still contains limiter when not needed.
        self.RemovePlotLine(label='limiter', ax=ax)
        for plate_label in self.PlateData.keys():
            self.RemovePlotLine(label=plate_label, ax=ax)

        if self.settings['grid_settings']['num_xpt'] == 2:
            self.PlotLimiter(ax=ax)

        if self.settings['grid_settings']['patch_generation']['strike_pt_loc'] == 'limiter':
            self.PlotLimiter(ax=ax)
        else:
            self.PlotTargetPlates(ax=ax)

    def PlotTargetPlates(self, ax: object = None) -> None:
        """
        Plot all PlateData and remove outdated plate line artists.
        """
        plate_colors = ['blue', 'orange', 'firebrick', 'green']

        if ax is None:
            ax = plt.gca()
        for k, c in zip([key for key in self.PlateData.keys()], plate_colors):
            self.PlotTargetPlate(plate_key=k, color=c, ax=ax)

    def PlotTargetPlate(self, plate_key: str, color: str = 'red', ax: object = None) -> None:
        """
        Plot a target plate corresponding to a plate key.

        Parameters
        ----------
        plate_key: str
            An Ingrid supported target plate key (see method
            SetGeometry for supported plate keys)

        color : str, optional
            Color to provide to matplotlib
        """
        if ax is None:
            ax = plt.gca()

        if plate_key in [k for k in self.PlateData.keys()]:
            self.RemovePlotLine(label=plate_key, ax=ax)
            if type(self.PlateData[plate_key]) is Line:
                print(f"# Plotting plate '{plate_key}'")
                self.PlateData[plate_key].plot(label=plate_key, color=color)
            else:
                pass
        else:
            raise ValueError(f"# Value '{plate_key}' is not an Ingrid supported target plate key.")

    def PlotLimiter(self, ax: object = None) -> None:
        """
        Plot limiter geometry.
        """
        if ax is None:
            ax = plt.gca()
        self.RemovePlotLine(label='limiter', ax=ax)
        self.LimiterData.plot(color='dodgerblue', label='limiter')

    def PlotPsiUNorm(self) -> None:
        """
        Plot unnormalized psi data.
        """
        try:
            if self.PsiUNorm.ax.__dict__['collections'] is not None:
                self.PsiUNorm.ax.collections = []
                plt.draw()
        except:
            pass
        self.PsiUNorm.plot_data(self.settings['grid_settings']['nlevs'], 
                                up_down_symmetry=self.settings["grid_settings"]["up_down_symmetry"], 
                                zmagx=self.magx[1] + self.settings["grid_settings"]["patch_generation"]["zmagx_shift"])

    def PlotPsiNorm(self, view_mode: str = 'filled') -> None:
        """
        Plot normalized psi data.
        """
        try:
            if self.PsiNormAx.__dict__['collections'] is not None:
                self.PsiNormAx.collections = []
                plt.draw()
        except:
            pass

        self.PsiNorm.plot_data(nlevs=self.settings['grid_settings']['nlevs'], 
                               fig=self._PsiNormFig,
                               ax=self.PsiNormAx, 
                               view_mode=view_mode, 
                               up_down_symmetry=self.settings["grid_settings"]["up_down_symmetry"], 
                               zmagx=self.magx[1] + self.settings["grid_settings"]["patch_generation"]["zmagx_shift"])

    def PlotPsiNormBounds(self) -> None:
        """
        Plot contour lines associated with psi boundary values provided.

        This method extracts psi values from the 'settings' dict and plots the
        psi level. In addition to the psi values in 'settings', the primary
        and, if applicable, secondary separatrix are plotted as well.

        If the user is operating on a single null configuration, the psi values
        plotted are 'psi_1', 'psi_core', 'psi_pf_1'.

        If the user is operating on a case with two x-points, the psi values
        are the same as above but with 'psi_1', 'psi_2', and
        'psi_pf_2' also included in the plot.
        """

        nxpt = self.settings['grid_settings']['num_xpt']
        if nxpt == 1:
            self.contour_colour_dic = {'psi_1': 'lime',
                   'psi_core': 'cyan',
                   'psi_pf_1': 'white'}
            num_psi_levels = 4
        elif nxpt == 2:
            self.contour_colour_dic = {'psi_core': 'cyan',
                   'psi_1': 'lime',
                   'psi_2': 'fuchsia',
                   'psi_pf_1': 'white',
                   'psi_pf_2': 'yellow'}
            num_psi_levels = 7

        for k, c in self.contour_colour_dic.items():
            self.PsiNorm.PlotLevel(self.settings['grid_settings'][k], color=self.contour_colour_dic[k], label=k)

        self.PsiNorm.PlotLevel(1.0, color='red', label='Primary Separatrix')
        if nxpt == 2:
            self.PsiNorm.PlotLevel(
                self.PsiNorm.get_psi(self.xpt2[0], self.xpt2[1]), color='blue', label='Secondary Separatrix')

        handles, labels = self.PsiNorm.ax.get_legend_handles_labels()
        contour_count = 0
        contour_colours = list(self.contour_colour_dic.values()) + ["red", "blue"]
        for i in range(len(handles)):
            if isinstance(handles[i], matplotlib.collections.PathCollection):
                handles[i] = matplotlib.lines.Line2D([0],[0],color=contour_colours[contour_count])
                contour_count+=1
        lookup = {label: handle for label, handle in zip(labels, handles)}
        try:
            self.PsiNorm.fig.legends[0].remove()
        except:
            pass
        self.PsiNorm.fig.legend(handles=[handle for handle in lookup.values()], labels=[label for label in lookup.keys()],
                               bbox_to_anchor=(0.5, 1), loc='upper center',
                               ncol=min((len([label for label in lookup.keys()]) + num_psi_levels) // 3, 5), facecolor="gray", framealpha=0.2)

    def PlotPsiNormMagReference(self, ax: object = None) -> None:
        """
        Plot a marker on the magnetic axis and all x-points of interest.
        """

        if ax is None:
            ax = plt.gca()

        (x, y) = self.magx
        x += self.settings['grid_settings']['patch_generation']['rmagx_shift']
        y += self.settings['grid_settings']['patch_generation']['zmagx_shift']
        self.RemovePlotPoint(label='magx', ax=ax)
        ax.plot(x, y, '+', color='yellow', ms=15, linewidth=5, label='magx', zorder=9999)

        (x, y) = self.xpt1
        self.RemovePlotPoint(label='xpt1', ax=ax)
        ax.plot(x, y, '+', color='orange', ms=15, linewidth=5, label='xpt1')

        self.RemovePlotPoint(label='xpt2', ax=ax)
        if self.settings['grid_settings']['num_xpt'] == 2:
            try:
                (x, y) = self.xpt2
                ax.plot(x, y, '+', color='red', ms=15, linewidth=5, label='xpt2')
            except:
                pass

    def PlotTopologyAnalysis(self) -> None:
        """
        Shade the private flux, core, and show where the secondary x-point
        travels.

        This method can be used to interpret which type of configuration the
        user is handling.
        """

        # Clean up the figure.
        self.RemovePlotPatch(label='Core')
        self.RemovePlotPatch(label='PF_1')
        self.RemovePlotLine(label='RegionLineCut')

        self.PsiNorm.ax.add_patch(self.LineTracer.RegionPolygon['Core'])
        self.PsiNorm.ax.add_patch(self.LineTracer.RegionPolygon['PF_1'])
        self.LineTracer.RegionLineCut.plot(color='white', label='RegionLineCut')

    def PlotPsiLevel(self, efit_psi: object, level: float, Label: str = '') -> None:
        """
        Plot a contour corresponding to a psi level.

        Parameters
        ----------
        efit_psi : EfitData
            The 'EfitData' object to get the psi data from.

        level : float
            The psi value to plot.

        Label : str, optional
            Label to provide to matplotlib.pyplot.contour.
        """
        plt.contour(efit_psi.r, efit_psi.z, efit_psi.v, [level], colors='red', label=Label)

    def PlotMidplane(self, ax: object = None) -> None:
        """
        Plot midplane line through magnetic axis with any applied
        transformations specified in settings.

        This method can be used to inspect the effects of 'magx_tilt_1',
        'magx_tilt_2', 'rmagx_shift', and 'zmagx_shift'.
        """
        try:
            magx_tilt_1 = self.settings['grid_settings']['patch_generation']['magx_tilt_1']
        except KeyError:
            magx_tilt_1 = 0.0
        try:
            magx_tilt_2 = self.settings['grid_settings']['patch_generation']['magx_tilt_2']
        except KeyError:
            magx_tilt_2 = 0.0

        if ax is None:
            ax = plt.gca()

        R = self.settings['grid_settings']['rmagx'] + self.settings['grid_settings']['patch_generation']['rmagx_shift']
        Z = self.settings['grid_settings']['zmagx'] + self.settings['grid_settings']['patch_generation']['zmagx_shift']
        magx = Point(np.array([R, Z]))

        # Generate Horizontal Mid-Plane lines
        LHS_Point = Point(magx.x - 1e6 * np.cos(magx_tilt_1), magx.y - 1e6 * np.sin(magx_tilt_1))
        RHS_Point = Point(magx.x, magx.y)
        midline_1 = Line([LHS_Point, RHS_Point])

        LHS_Point = Point(magx.x, magx.y)
        RHS_Point = Point(magx.x + 1e6 * np.cos(magx_tilt_2), magx.y + 1e6 * np.sin(magx_tilt_2))
        midline_2 = Line([LHS_Point, RHS_Point])

        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'midline_1']
        except:
            pass
        midline_1.plot(color='darkkhaki', label='midline_1')

        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'midline_2']
        except:
            pass
        midline_2.plot(color='lightpink', label='midline_2')

    def PlotEastWestXpt1Ref(self, ax: object = None) -> None:
        """
        Plot midplane line through magnetic axis with any applied
        transformations specified in settings.

        This method can be used to inspect the effects of 'magx_tilt_1',
        'magx_tilt_2', 'rmagx_shift', and 'zmagx_shift'.
        """
        try:
            magx_tilt_1 = self.settings['grid_settings']['patch_generation']['xpt1_W_tilt']
        except KeyError:
            magx_tilt_1 = 0.0
        try:
            magx_tilt_2 = self.settings['grid_settings']['patch_generation']['xpt1_E_tilt']
        except KeyError:
            magx_tilt_2 = 0.0
        try:
            magx_tilt_3 = self.settings['grid_settings']['patch_generation']['xpt1_S_tilt']
        except KeyError:
            magx_tilt_3 = 0.0
        try:
            magx_tilt_4 = self.settings['grid_settings']['patch_generation']['xpt1_N_tilt']
        except KeyError:
            magx_tilt_4 = 0.0

        if ax is None:
            ax = plt.gca()

        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt1_ref_E']
        except:
            pass
        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt1_ref_W']
        except:
            pass

        if self.settings['grid_settings']['patch_generation']['use_xpt1_W'] is True:
            R = self.settings['grid_settings']['rxpt']
            Z = self.settings['grid_settings']['zxpt']
            xpt1 = Point(np.array([R, Z]))

        # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt1.x - 1e6 * np.cos(magx_tilt_1), xpt1.y - 1e6 * np.sin(magx_tilt_1))
            RHS_Point = Point(xpt1.x, xpt1.y)
            xpt1_midline_ref_E = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt1_ref_W']
            except:
                pass
            xpt1_midline_ref_E.plot(color='darkkhaki', label='xpt1_ref_W')

        if self.settings['grid_settings']['patch_generation']['use_xpt1_E'] is True:
            R = self.settings['grid_settings']['rxpt']
            Z = self.settings['grid_settings']['zxpt']
            xpt1 = Point(np.array([R, Z]))

        # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt1.x, xpt1.y)
            RHS_Point = Point(xpt1.x + 1e6 * np.cos(magx_tilt_2), xpt1.y + 1e6 * np.sin(magx_tilt_2))
            xpt1_midline_ref_W = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt1_ref_E']
            except:
                pass
            xpt1_midline_ref_W.plot(color='lightpink', label='xpt1_ref_E')

        if self.settings['grid_settings']['patch_generation']['use_xpt1_S'] is True:
            R = self.settings['grid_settings']['rxpt']
            Z = self.settings['grid_settings']['zxpt']
            xpt1 = Point(np.array([R, Z]))

            # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt1.x, xpt1.y)
            RHS_Point = Point(xpt1.x + 1e6 * np.cos(magx_tilt_3), xpt1.y + 1e6 * np.sin(magx_tilt_3))
            xpt1_midline_ref_S = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt1_ref_S']
            except:
                pass
            xpt1_midline_ref_S.plot(color='teal', label='xpt1_ref_S')

        if self.settings['grid_settings']['patch_generation']['use_xpt1_N'] is True:
            R = self.settings['grid_settings']['rxpt']
            Z = self.settings['grid_settings']['zxpt']
            xpt1 = Point(np.array([R, Z]))

            # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt1.x, xpt1.y)
            RHS_Point = Point(xpt1.x + 1e6 * np.cos(magx_tilt_4), xpt1.y + 1e6 * np.sin(magx_tilt_4))
            xpt1_midline_ref_N = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt1_ref_N']
            except:
                pass
            xpt1_midline_ref_N.plot(color='darkturquoise', label='xpt1_ref_N')

    def PlotEastWestXpt2Ref(self, ax: object = None) -> None:
        """
        Plot midplane line through magnetic axis with any applied
        transformations specified in settings.

        This method can be used to inspect the effects of 'magx_tilt_1',
        'magx_tilt_2', 'rmagx_shift', and 'zmagx_shift'.
        """
        try:
            magx_tilt_1 = self.settings['grid_settings']['patch_generation']['xpt2_W_tilt']
        except KeyError:
            magx_tilt_1 = 0.0
        try:
            magx_tilt_2 = self.settings['grid_settings']['patch_generation']['xpt2_E_tilt']
        except KeyError:
            magx_tilt_2 = 0.0
        try:
            magx_tilt_3 = self.settings['grid_settings']['patch_generation']['xpt2_N_tilt']
        except KeyError:
            magx_tilt_3 = 0.0
        try:
            magx_tilt_4 = self.settings['grid_settings']['patch_generation']['xpt2_S_tilt']
        except KeyError:
            magx_tilt_4 = 0.0

        if ax is None:
            ax = plt.gca()

        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt2_ref_E']
        except:
            pass
        try:
            [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt2_ref_W']
        except:
            pass

        if self.settings['grid_settings']['patch_generation']['use_xpt2_W'] is True:
            R = self.settings['grid_settings']['rxpt2']
            Z = self.settings['grid_settings']['zxpt2']
            xpt2 = Point(np.array([R, Z]))

        # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt2.x - 1e6 * np.cos(magx_tilt_1), xpt2.y - 1e6 * np.sin(magx_tilt_1))
            RHS_Point = Point(xpt2.x, xpt2.y)
            xpt2_midline_ref_E = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt2_ref_W']
            except:
                pass
            xpt2_midline_ref_E.plot(color='darkkhaki', label='xpt2_ref_W')

        if self.settings['grid_settings']['patch_generation']['use_xpt2_E'] is True:
            R = self.settings['grid_settings']['rxpt2']
            Z = self.settings['grid_settings']['zxpt2']
            xpt2 = Point(np.array([R, Z]))

        # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt2.x, xpt2.y)
            RHS_Point = Point(xpt2.x + 1e6 * np.cos(magx_tilt_2), xpt2.y + 1e6 * np.sin(magx_tilt_2))
            xpt2_midline_ref_W = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt2_ref_E']
            except:
                pass
            xpt2_midline_ref_W.plot(color='lightpink', label='xpt2_ref_E')

        if self.settings['grid_settings']['patch_generation']['use_xpt2_N'] is True:
            R = self.settings['grid_settings']['rxpt2']
            Z = self.settings['grid_settings']['zxpt2']
            xpt2 = Point(np.array([R, Z]))

            # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt2.x, xpt2.y)
            RHS_Point = Point(xpt2.x + 1e6 * np.cos(magx_tilt_3), xpt2.y + 1e6 * np.sin(magx_tilt_3))
            xpt2_midline_ref_N = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt2_ref_N']
            except:
                pass
            xpt2_midline_ref_N.plot(color='darkturquoise', label='xpt2_ref_N')
        
        if self.settings['grid_settings']['patch_generation']['use_xpt2_S'] is True:
            R = self.settings['grid_settings']['rxpt2']
            Z = self.settings['grid_settings']['zxpt2']
            xpt2 = Point(np.array([R, Z]))

            # Generate Horizontal Mid-Plane lines
            LHS_Point = Point(xpt2.x, xpt2.y)
            RHS_Point = Point(xpt2.x + 1e6 * np.cos(magx_tilt_4), xpt2.y + 1e6 * np.sin(magx_tilt_4))
            xpt2_midline_ref_S = Line([LHS_Point, RHS_Point])

            try:
                [ax_line.remove() for ax_line in ax.lines if ax_line.get_label() == 'xpt2_ref_S']
            except:
                pass
            xpt2_midline_ref_S.plot(color='teal', label='xpt2_ref_S')

    def PlotPatches(self) -> None:
        """
        Plot the patch map that was generated with method 'CreatePatches'
        """
        try:
            plt.close(self._PatchFig)
        except:
            pass
        self._PatchFig = plt.figure('INGRID: ' + self.CurrentTopology.config + ' Patches', figsize=(6, 10))
        self.PatchAx = self._PatchFig.add_subplot(111)
        self.CurrentTopology.patch_diagram(fig=self._PatchFig, 
                                           ax=self.PatchAx, 
                                           up_down_symmetry=self.settings["grid_settings"]["up_down_symmetry"], 
                                           zmagx=self.magx[1] + self.settings["grid_settings"]["patch_generation"]["zmagx_shift"])
        self.PlotStrikeGeometry(ax=self.PatchAx)
        if self.settings['grid_settings']['patch_generation']['strike_pt_loc'] == 'target_plates':
            self.RemovePlotLine(label='limiter', ax=self.PatchAx)

    def PlotGrid(self) -> None:
        """
        Plot the grid that was generated with method 'ConstructGrid'.
        """
        try:
            plt.close(self._SubgridFig)
        except:
            pass
        self._SubgridFig = plt.figure('INGRID: ' + self.CurrentTopology.config + ' Grid', figsize=(6, 10))
        self._SubgridAx = self._SubgridFig.add_subplot(111)
        self.CurrentTopology.grid_diagram(fig=self._SubgridFig, 
                                          ax=self._SubgridAx, 
                                          up_down_symmetry=self.settings["grid_settings"]["up_down_symmetry"], 
                                          zmagx=self.magx[1] + self.settings["grid_settings"]["patch_generation"]["zmagx_shift"])

    def PlotSubgrid(self) -> None:
        """
        Alias for method `PlotGrid`. See `PlotGrid` for documentation.
        """
        self.PlotGrid()

    def AutoRefineMagAxis(self) -> None:
        """
        Refine magnetic axis RZ coordinates.

        Will apply a root_finder method to 'rmagx' and 'zmagx' float values
        stored in attribute "settings['grid_settings']".
        """
        self.FindMagAxis(self.settings['grid_settings']['rmagx'], self.settings['grid_settings']['zmagx'])

    def AutoRefineXPoint(self) -> None:
        """
        Refine primary x-point RZ coordinates.

        Will apply a root_finder method to 'rxpt' and 'zxpt' float values
        stored in attribute "settings['grid_settings']".
        """
        self.FindXPoint(self.settings['grid_settings']['rxpt'], self.settings['grid_settings']['zxpt'])

    def AutoRefineXPoint2(self) -> None:
        """
        Refine secondary x-point RZ coordinates.

        Will apply a root_finder method to 'rxpt2' and 'zxpt2' float values
        stored in attribute "settings['grid_settings']".
        """
        self.FindXPoint2(self.settings['grid_settings']['rxpt2'], self.settings['grid_settings']['zxpt2'])

    def SetMagReference(self) -> None:
        """
        Set the appropriate reference points in the domain. Namely the
        magnetic-axis, primary x-point, and (if applicable), secondary x-point.
        """
        self.magx = (self.settings['grid_settings']['rmagx'], self.settings['grid_settings']['zmagx'])
        self.xpt1 = (self.settings['grid_settings']['rxpt'], self.settings['grid_settings']['zxpt'])

        if self.settings['grid_settings']['num_xpt'] == 2:
            self.xpt2 = (self.settings['grid_settings']['rxpt2'], self.settings['grid_settings']['zxpt2'])

    def CalcPsiNorm(self) -> None:
        """
        Normalize psi data to a refined magnetic axis and primary x-point
        """

        # use the same bounds as the efit data
        self.PsiNorm = EfitData(self.PsiUNorm.rmin, self.PsiUNorm.rmax,
                                self.PsiUNorm.nr, self.PsiUNorm.zmin,
                                self.PsiUNorm.zmax, self.PsiUNorm.nz,
                                self.PsiUNorm.rcenter, self.PsiUNorm.bcenter,
                                self.PsiUNorm.rlimiter, self.PsiUNorm.zlimiter,
                                self.PsiUNorm.rmagx, self.PsiUNorm.zmagx,
                                name='Normalized Efit Data', parent=self)
        psi = self.PsiUNorm.v
        psi_magx = self.PsiUNorm.get_psi(self.magx[0], self.magx[1])
        psi_xpt1 = self.PsiUNorm.get_psi(self.xpt1[0], self.xpt1[1])
        psinorm = (psi - np.full_like(psi, psi_magx)) / (psi_xpt1 - psi_magx)

        self.PsiNorm.init_bivariate_spline(self.PsiNorm.r[:, 0], 
                                           self.PsiNorm.z[0, :], 
                                           psinorm)

        self._PsiNormFig = plt.figure('INGRID: ' + self.PsiNorm.name, figsize=(8, 10))
        self.PsiNormAx = self._PsiNormFig.add_subplot(111)

    def AnalyzeTopology(self) -> None:
        """
        Perform analysis on normalized psi to determine magnetic topolgy.
        """

        try:
            visual = self.settings['DEBUG']['visual']['find_NSEW']
        except:
            visual = False

        self.PrepLineTracing()
        self.OrderLimiter()
        self.OrderTargetPlates()
        self.ClassifyTopology(visual=visual)

        print('')
        print('# Identified {} configuration.'.format(self.config))
        print('')

        self.SetTopology(self.config)

    def SetTopology(self, topology: str) -> None:
        """
        Initialize the current topology to a particular magnetic topology.

        Parameters
        ----------
        topology : str
            String literal corresponding to which magnetic topology
            to initialize.

            Values can be:
                'LSN': Lower Single Null
                'USN': Upper Single Null
                'UDN': Unbalanced Double Null
                'SF15': Snowflake-15
                'SF45': Snowflake-45
                'SF75': Snowflake-75
                'SF105': Snowflake-105
                'SF135': Snowflake-135
                'SF165': Snowflake-165

        Raises
        ------
        ValueError
            If user provided unrecognized string entry.
        """
        if topology in ['LSN', 'USN']:
            ingrid_topology = SNL(self, topology)

        elif topology == 'UDN':
            ingrid_topology = UDN(self, topology)

        elif topology == 'SF15':
            ingrid_topology = SF15(self, topology)

        elif topology == 'SF45':
            ingrid_topology = SF45(self, topology)

        elif topology == 'SF75':
            ingrid_topology = SF75(self, topology)

        elif topology == 'SF105':
            ingrid_topology = SF105(self, topology)

        elif topology == 'SF135':
            ingrid_topology = SF135(self, topology)

        elif topology == 'SF165':
            ingrid_topology = SF165(self, topology)

        else:
            raise ValueError(f"# Invalid entry: no Ingrid topology {topology} exists.")

        self.CurrentTopology = ingrid_topology

    def StartSetup(self, **kwargs) -> None:
        """
        A collection of essential tasks before generating a patch map from
        scratch.

        The user should ensure that the 'settings' dict is populated with
        the correct paths to relevant neqdsk data and geometry files.

        The user should ensure 'settings[''grid_settings''][''num_xpt'] is
        set with the correct integer value.
        """

        if self.settings['grid_settings']['num_xpt'] == 1:
            topology = 'SNL'
        elif self.settings['grid_settings']['num_xpt'] == 2:
            topology = 'DNL'
        else:
            v_error_str = '# Invalid number of x-points.' \
                + f'(User provided {self.settings["grid_settings"]["num_xpt"]}.' \
                + 'Must be <= 2).'
            raise ValueError(v_error_str)

        self.LoadGEQDSK(self.settings['eqdsk'], 
                        self.settings["grid_settings"]["up_down_symmetry"], 
                        self.settings["grid_settings"]["remove_upper_divertor"],
                        self.settings["analytic_equilibrium_generation"])
        self.AutoRefineMagAxis()
        self.AutoRefineXPoint()
        if topology == 'DNL':
            self.AutoRefineXPoint2()
        if hasattr(self, 'LimiterData'):
            self.LimiterData = None
        self.SetGeometry({'limiter': self.settings['limiter']})
        if hasattr(self, 'PlateData'):
            self.PlateData = {k: {} for k in self.PlateData.keys()}
        self.SetTargetPlates()
        self.SetMagReference()
        self.CalcPsiNorm()
        self.PrepLineTracing()

    def ShowSetup(self, view_mode: str = 'filled') -> None:
        """
        Show Ingrid setup that a patch map will be generated from.

        This method plots normalized psi data, psi boundaries, strike geometry,
        and midplane lines through the magnetic axis.
        """
        try:
            ax = plt.figure('INGRID: Normalized Efit Data').axes[0]
            self.ClearLegend(ax=ax)
            self.RemovePlotPatch(ax=ax, label='Core')
            self.RemovePlotPatch(ax=ax, label='PF_1')
            self.RemovePlotLine(ax=ax, label='RegionLineCut')
        except:
            pass
        self.PlotPsiNorm(view_mode=view_mode)
        self.PlotPsiNormMagReference()
        self.PlotStrikeGeometry(ax=self.PsiNormAx)
        self.PlotMidplane(ax=self.PsiNormAx)
        self.PlotEastWestXpt1Ref(ax=self.PsiNormAx)
        if self.settings['grid_settings']['num_xpt'] == 2:
            self.PlotEastWestXpt2Ref(ax=self.PsiNormAx)
        self.PlotPsiNormBounds()
        self.PrintSummaryParams()

    @_timer
    def ConstructPatches(self) -> None:
        """
        Create a patch map that can be refined into a grid.

        This method assumes the user has either loaded patch data
        from a previous Ingrid session (see 'LoadPatches'), or that
        the user has already successfully called method 'AnalyzeTopology'.

        Should the user want to automatically enable patch saving, the user
        should set the entry

        'settings[''patch_data''][''preferences''][''new_file'']'

        with a value of 'True'.

        """
        self.OrderLimiter()
        self.OrderTargetPlates()
        self.PrepLineTracing()
        self.CurrentTopology.construct_patches()
        self.CurrentTopology.GroupPatches()
        self.CheckPatches()

        if self.settings['patch_data']['preferences']['new_file']:
            self.SavePatches(self.settings['patch_data']['preferences']['new_fname'])

    def CreatePatches(self) -> None:
        """
        An alias for `ConstructPatches`. See `ConstructPatches`
        for more details.

        """
        self.ConstructPatches()

    def SavePatches(self, fname: str = '') -> None:
        """
        Save patches as '.npy' file for later reconstruction in Ingrid.

        This file contains the information required to reconstruct patches
        at a later time and bypass the line_tracing.

        Parameters
        ----------
        fname : str, optional
            Name of file/location for patch data.
        """
        if fname in ['', None]:
            fname = self.CurrentTopology.config + '_patches_' + str(int(time()))

        data = {}
        data['patch_data'] = [patch.as_np() for patch in self.CurrentTopology.patches.values()]
        data['topo_data'] = {}
        data['topo_data']['config'] = self.CurrentTopology.config
        data['topo_data']['magx'] = self.GetMagxData()
        for k, v in self.GetXptData().items():
            data['topo_data'][k] = v
        data['NSEW_data'] = self.CurrentTopology.LineTracer.NSEW_lookup

        np.save(fname, np.array([data]))
        if Path(fname).suffix == '.npy':
            tail = ''
        else:
            tail = '.npy'
        print(f"# Saved patch data for file {Path(fname).name + tail}")

    def LoadPatches(self, fname: str = '') -> None:
        """
        Load patches stored in an Ingrid generated '.npy' file.

        Parameters
        ----------
        fname : str, optional
            Path to patch data.
                If no fname is provided to method 'LoadPatches', Ingrid code will check the settings
                'dict' for a file under entry ``settings['patch_data']['file']``

        """

        if type(fname) is not str:
            raise ValueError('# User did not provide a string to patch data.')

        if fname.strip() == '':  # Check if settings contains patch data.
            fname = self.settings['patch_data']['file']
        data = np.load(fname, allow_pickle=True)[0]  # np.array containing dict

        if type(data) is dict:
            self.PopulateSettings(Ingrid.ReadYamlFile(self.InputFile))
            self.StartSetup()
            self.LineTracer.NSEW_lookup = data['NSEW_data']
            self.SetTopology(data['topo_data']['config'])
            self.CurrentTopology.patches = self.ReconstructPatches(data['patch_data'])
            self.CurrentTopology.OrderPatches()
            self.CurrentTopology.SetupPatchMatrix()
            self.CheckPatches()

    def ApplyUpDownSymmetry(self) -> None:
        if self.settings["grid_settings"]["up_down_symmetry"]:
            if self.CurrentTopology.config == "LSN":
                del self.CurrentTopology.patches["C1"]
                del self.CurrentTopology.patches["C2"]
                del self.CurrentTopology.patches["D1"]
                del self.CurrentTopology.patches["D2"]
            elif self.CurrentTopology.config in ["SF15", "SF75"]:
                del self.CurrentTopology.patches["C1"]
                del self.CurrentTopology.patches["C2"]
                del self.CurrentTopology.patches["C3"]
                del self.CurrentTopology.patches["D1"]
                del self.CurrentTopology.patches["D2"]
                del self.CurrentTopology.patches["D3"]
            else:
                #TODO: Implement up/down symmetry for other geometries
                raise Exception("Up/down symmetry not yet implemented for config " + self.CurrentTopology.config)

    @_timer
    def ConstructGrid(self, NewFig: bool = True, ShowVertices: bool = False) -> None:
        """
        Refine a generated patch map into a grid for exporting.

        Parameters
        ----------
        NewFig : bool, optional
            Plot the created grid on a new figure.

        ShowVertices : bool, optional
            Plot vertices in refined grid with bolded markers.

        ListPatches : bool, optional
            Generate a grid for a particular patch.
            Requires the correct patch name associated with the 'Patch'
            object.

            Warning: Grid generation is order dependent. Specifying a
            particular patch to generate a grid would only be done in
            rare cases and require the user to know dependencies for
            the particular patch.

        """
        self.CurrentTopology.construct_grid()

    def CreateSubgrid(self, NewFig: bool = True, ShowVertices: bool = False) -> None:
        """
        Alias for `ConstructGrid`. See `ConstructGrid` for documentation.

        """
        self.ConstructGrid(NewFig, ShowVertices)

    def ExportBOUTgrid(self, gridue_file, bout_grid_name: str = 'bout_from_in.grd.nc', plotting = True, verbose = True, ignore_checks = False):
        """
        Export a BOUT grid file for the created grid.

        Parameters
        ----------
        gridue_file : str, mandatory
            Name of gridue file to convert to BOUT grid.

        bout_grid_name : str, optional
            Name of BOUT grid file to save.

        plotting : bool, optional
            If True, plot the gridue file before conversion.

        verbose : bool, optional
            If True, print verbose output during conversion.

        ignore_checks : bool, optional
            If True, ignore checks for gridue file format and structure.

        """
        bout_grid_name = gridue_file + "_" + bout_grid_name
        Convert_grids(gridue_file,bout_grid_name,plotting, verbose, ignore_checks)

    def ExportGridue(self, fname: str = 'gridue', guard_cell_eps=1e-3) -> None:
        """
        Export a gridue file for the created grid.

        Parameters
        ----------
        fname : str, optional
            Name of gridue file to save.

        """
        self.PrepGridue(guard_cell_eps=guard_cell_eps)

        if type(self.CurrentTopology) in [SNL]:
            if self.WriteGridueSNL(self.CurrentTopology.gridue_settings, fname):
                print(f"# Successfully saved gridue file as '{fname}'")
                self.ExportBOUTgrid(fname, 'bout_from_in.grd.nc', plotting = True, verbose = True, ignore_checks = False)
        elif type(self.CurrentTopology) in [SF15, SF45, SF75, SF105, SF135, SF165, UDN, CDN]:
            if self.WriteGridueDNL(self.CurrentTopology.gridue_settings, fname):
                print(f"# Successfully saved gridue file as '{fname}'")
                self.ExportBOUTgrid(fname, 'bout_from_in.grd.nc', plotting = True, verbose = True, ignore_checks = False)

    @staticmethod
    def ImportGridue(fname: str = 'gridue') -> dict:
        """
        Import UEDGE grid file as dictionary.

        Parameters
        ----------
        fname : str, optional
            Path/file name to gridue formatted file.

        Returns
        -------
            A dict containing header and body information from the gridue file.

        """
        try:
            f = open(fname, mode='r')
            Values = [int(x) for x in next(f).split()]
            
            # Very crude method of checking what format the gridue is in
            if len(Values) > 2:
                HeaderItems = ['nxm', 'nym', 'ixpt1', 'ixpt2', 'iyseptrx1']
                gridue_settings = dict(zip(HeaderItems, Values))
                next(f)
            else:
                gridue_settings = {}
                header_rows = [
                    ['nxm', 'nym'],
                    ['iyseparatrix1', 'iyseparatrix2'],
                    ['ix_plate1', 'ix_cut1', '_FILLER_', 'ix_cut2', 'ix_plate2'],
                    ['iyseparatrix3', 'iyseparatrix4'],
                    ['ix_plate3', 'ix_cut3', '_FILLER_', 'ix_cut4', 'ix_plate4']
                ]
                for row in header_rows:
                    gridue_settings = {**gridue_settings, **dict(zip(row, Values))}
                    Values = [int(x) for x in next(f).split()]

            BodyItems = ['rm', 'zm', 'psi', 'br', 'bz', 'bpol', 'bphi', 'b']
            Str = {i: [] for i in BodyItems}
            k = iter(Str.keys())
            Key = next(k)
            for line in f:
                if line == 'iogridue\n':
                    continue
                if line == '\n':
                    try:
                        Key = next(k)
                    except:
                        continue
                    print(Key)
                else:
                    Str[Key].append(line)
            f.close()
            nx = gridue_settings['nxm'] + 2
            ny = gridue_settings['nym'] + 2
            for k, v in Str.items():
                L = (''.join(v).replace('\n', '').replace('D', 'e')).split()
                _l = iter(L)
                vv = next(_l)

                data_ = np.zeros((nx, ny, 5))
                for n in range(5):
                    for j in range(ny):
                        for i in range(nx):

                            data_[i][j][n] = float(vv)

                            try:
                                vv = next(_l)
                            except:
                                continue
                gridue_settings[k] = data_
            return gridue_settings
        except Exception as e:
            print(repr(e))

    @staticmethod
    def PlotGridue(GridueParams: dict, edgecolor='black', ax: object = None):
        """
        Plot UEDGE grid from 'dict' obtained from method 'ImportGridue'

        Parameters
        ----------
        GridueParams : dict
            Gridue header and body information as a dictionary.
            (See method ImportGridue)

        edgecolor : str, optional
            Color of grid.

        ax : object, optional
            Matplotlib axes to plot on.

        """
        r = GridueParams['rm']
        z = GridueParams['zm']
        Nx = len(r)
        Ny = len(r[0])
        patches = []
        plt.figure(figsize=(6, 10))
        if ax is None:
            ax = plt.gca()
        idx = [np.array([1, 2, 4, 3, 1])]
        for i in range(Nx):
            for j in range(Ny):
                p = matplotlib.patches.Polygon(np.concatenate((r[i][j][idx], z[i][j][idx])).reshape(2, 5).T, fill=False, closed=True, edgecolor=edgecolor)
                ax.add_patch(p)  # draw the contours
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('R')
        ax.set_ylabel('Z')
        plt.ylim(z.min(), z.max())
        plt.xlim(r.min(), r.max())
        plt.show()

    @staticmethod
    def ReadYamlFile(FileName: str) -> dict:
        """
        Read a yaml file and return a dictionary

        Parameters
        ----------
        FileName : str
            Path/file name of '.yml' parameter file represented as dictionary.

        Returns
        -------
            Settings file represented as a dictionary

        Raises
        ------
            IOError: If error occurs while loading yml file.

        """
        File = pathlib.Path(FileName).expanduser()
        if File.exists() and File.is_file():
            try:
                with open(FileName, 'r') as f:
                    ymlDict = yml.load(f, Loader=yml.Loader)
                return ymlDict
            except:
                raise IOError('Cannot load the yml file: ' + FileName)

        else:
            print('Current Working Directory:' + os.getcwd())
            raise IOError('Cannot find the file: ' + FileName)
