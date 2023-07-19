r"""Direct access to TAU simulation data.

The DRL (Deutsches Luft- und Raumfahrtzentrum) TAU_ code saves
snapshots in the NetCFD format. The :class:`TAUDataloader` is a
wrapper around the NetCFD Python bindings to simplify the access
to snapshot data.

.. _TAU: https://www.dlr.de/as/desktopdefault.aspx/tabid-395/526_read-694/

"""
# standard library packages
from os.path import join, split
from glob import glob
from typing import List, Dict, Tuple, Union, Set
# third party packages
from netCDF4 import Dataset
import torch as pt
import numpy as np
# flowtorch packages
from flowtorch import DEFAULT_DTYPE
from .dataloader import Dataloader
from .tau_dataloader import TAUConfig
from .utils import check_list_or_str, check_and_standardize_path

SURF_SOLUTION_NAME = ".surface.pval.unsteady_"
PSOLUTION_POSTFIX = ".domain_"
PMESH_NAME = "domain_{:s}_grid_1"
PVERTEX_KEY = "pcoord"
PWEIGHT_KEY = "pvolume"
PADD_POINTS_KEY = "addpoint_idx"
PGLOBAL_ID_KEY = "globalidx"
VERTEX_KEYS = ("points_xc", "points_yc", "points_zc")
WEIGHT_KEY = "volume"

COMMENT_CHAR = "#"
CONFIG_SEP = ":"
SOLUTION_PREFIX_KEY = "solution_prefix"
GRID_FILE_KEY = "primary_grid"
GRID_PREFIX_KEY = "grid_prefix"
N_DOMAINS_KEY = "n_domains"

class TAUSurfaceDataloader(Dataloader):
    """Load TAU simulation data.

    The loader is currently limited to read:
    - mesh vertices, serial 
    - interal surface solution, serial

    Examples

    >>> from os.path import join
    >>> from flowtorch import DATASETS
    >>> from flowtorch.data import TAUSurfaceDataloader
    >>> path = DATASETS["tau_surface_import"]
    >>> loader = TAUDataloader(join(path, "tau_euler.para"))
    >>> times = loader.write_times
    >>> fields = loader.field_names[times[0]]
    >>> fields
    ['cp', 'x_velocity', 'y_velocity', ...]
    >>> cp = loader.load_snapshot("cp", times)

    """

    def __init__(self, parameter_file: str, distributed: bool = False,
                 dtype: str = DEFAULT_DTYPE):
        """Create loader instance from TAU parameter file.

        :param parameter_file: path to TAU simulation parameter file
        :type parameter_file: str
        :param distributed: True if mesh and solution are distributed in domain
            files; defaults to False
        :type distributed: bool, optional
        :param dtype: tensor type, defaults to DEFAULT_DTYPE
        :type dtype: str, optional
        """
        self._para = TAUConfig(parameter_file)
        self._distributed = distributed
        self._dtype = dtype
        self._time_iter = self._decompose_file_name()
        self._para._parse_bmap()
        self._mesh_data = None
        self._zone_names = None
        self._zone = self.zone_names[0]
        self._global_id_of_zone_points = {}

    def _decompose_file_name(self) -> Dict[str, str]:
        """Extract write time and iteration from file name.

        :raises FileNotFoundError: if no solution files are found
        :return: dictionary with write times as keys and the corresponding
            iterations as values
        :rtype: Dict[str, str]
        """
        base = join(self._para.path, self._para.config[SOLUTION_PREFIX_KEY])
        base += SURF_SOLUTION_NAME
        suffix = f"{PSOLUTION_POSTFIX}0" if self._distributed else "e???"
        files = glob(f"{base}i=*t=*{suffix}")
        if len(files) < 1:
            raise FileNotFoundError(
                f"Could not find solution files in {self._sol_path}/")
        time_iter = {}
        split_at = PSOLUTION_POSTFIX if self._distributed else " "
        for f in files:
            t = f.split("t=")[-1].split(split_at)[0]
            i = f.split("i=")[-1].split("_t=")[0]
            time_iter[t] = i
        return time_iter

    def _file_name(self, time: str, suffix: str = "") -> str:
        """Create solution file name from write time.

        :param time: snapshot write time
        :type time: str
        :param suffix: suffix to append to the file name; used for decomposed
            simulations
        :type suffix: str, optional
        :return: name of solution file
        :rtype: str
        """
        itr = self._time_iter[time]
        path = join(self._para.path, self._para.config[SOLUTION_PREFIX_KEY])
        return f"{path}{SURF_SOLUTION_NAME}i={itr}_t={time}{suffix}"
        
    def _load_domain_mesh_data(self, pid: str) -> pt.Tensor:
        """Load vertices and volumes for a single processor domain.

        :param pid: domain id
        :type pid: str
        :return: tensor of size n_points x 4, where n_points is the number
            of unique cells in the domain, and the 4 columns contain the
            coordinates of the vertices (x, y, z) and the cell volumes
        :rtype: pt.Tensor
        """
        print("self._load_domain_mesh_data() not implemented yet!")
        return 0

    def _load_mesh_data(self):
        """Load mesh vertices and global_id for the current zone.

        The mesh data is saved as class member `_mesh_data`. The tensor has the
        dimension n_points x 4; the first three columns correspond to the x/y/z
        coordinates. The 4th column is usually filled with ones for the weights.
        """
        if self._distributed:
            n = self._para.config[N_DOMAINS_KEY]
            self._mesh_data = pt.cat(
                [self._load_domain_mesh_data(str(pid)) for pid in range(n)],
                dim=0
            )
        else:
            path = join(self._para.path, self._para.config[GRID_FILE_KEY])
            # Get surface-mesh information from the netcdf4 mesh file
            with Dataset(path) as data:
                # boundary_markers : list associating the marker values with surface element index positions
                # length : number of surface elements in the mesh
                # example:   boundary_markers = [1,1,1,2,2,3,3,3]
                #            First three surface elements in the mesh belong to marker 1, 
                #            elements 4 and 5 belong to marker 2 and so on.
                boundary_markers = pt.tensor(data.variables["boundarymarker_of_surfaces"][:], dtype=int)
                try:
                    # surface_tris : list containing the definition of surface triangles
                    # length : number of triangles in the mesh x 3
                    # example:   surface_tris = [[5,2,1],
                    #                            [1,4,6],
                    #                            [2,3,5]]
                    #            First triangle is defined by points 5, 2 and 1.
                    #            Second triangle is defined by points 1, 4, and 6 and so on.
                    surface_tris = pt.tensor(data.variables["points_of_surfacetriangles"][:], dtype=int)
                except KeyError:
                    pass
                try:
                    # surface_quads : list containing the definition of surface quadrilaterals
                    # length : number of quads in the mesh x 4
                    # example: see surface_tris
                    surface_quads = pt.tensor(data.variables["points_of_surfacequadrilaterals"][:], dtype=int)
                except KeyError:
                    pass

            # Define the marker id based on the zone
            for zone_name, zone_markers in self._para._bmap.items():
                indices_of_marker = np.where(np.isin(boundary_markers, zone_markers))
                # Extract the global ID's of selected points (global ID is the index position of a point in the mesh file):
                # TODO: This is very cumbersome. See above.
                if surface_quads is not None and surface_tris is not None:
                    # Expand surface_tris by 4th entry with 'nan' so the tensors of tris and quads can be added together
                    dummy_tensor = pt.zeros(size=(len(surface_tris),1))
                    dummy_tensor[dummy_tensor==0] = float('nan')
                    surface_tris_expanded = pt.cat((surface_tris,dummy_tensor),1)
                    # join tensors of tris and quads
                    points_of_tris_and_quads = pt.cat((surface_tris_expanded, surface_quads))
                    # Extract indices of surface points in the mesh file
                    global_id_of_marker_points = np.unique(points_of_tris_and_quads[indices_of_marker].flatten())
                    # Drop nan and convert to int again
                    global_id_of_marker_points = global_id_of_marker_points[~np.isnan(global_id_of_marker_points)].astype(int)
                elif surface_tris is not None:
                    # if only triangles are present in the mesh the extraction is easy
                    global_id_of_marker_points = np.unique(surface_tris[indices_of_marker].flatten())
                elif surface_quads is not None:
                    # if only quads are present in the mesh the extraction is easy
                    global_id_of_marker_points = np.unique(surface_quads[indices_of_marker].flatten())
                else:
                    print("Error loading surface data. No triangles or quadrilaterals found in the mesh: {}".format(path))
                self._global_id_of_zone_points[zone_name] = global_id_of_marker_points

    def _load_single_snapshot(self, field_name: str, time: str) -> pt.Tensor:
        """Load a single snapshot of a single field from the netCDF4 file(s).

        :param field_name: name of the field
        :type field_name: str
        :param time: snapshot write time
        :type time: str
        :return: tensor holding the field values
        :rtype: pt.Tensor
        """
        if self._distributed:
            print("Loading of distributed snapshots is not implemented!")
        else:
            path = self._file_name(time)
            with Dataset(path) as data:
                # Get the global ids from the surface datafile.
                global_id = data.variables["global_id"]
                # Extract the index positions of the points of the current zone in the surface datafile
                index_position_in_surf_data = np.isin(global_id, self._global_id_of_zone_points[self._zone])
                # Get the data
                field = pt.tensor(
                    data.variables[field_name][index_position_in_surf_data], dtype=self._dtype)
        return field

    def load_snapshot(self, field_name: Union[List[str], str],
                      time: Union[List[str], str]) -> Union[List[pt.Tensor], pt.Tensor]:
        check_list_or_str(field_name, "field_name")
        check_list_or_str(time, "time")

        # load multiple fields
        if isinstance(field_name, list):
            if isinstance(time, list):
                return [
                    pt.stack([self._load_single_snapshot(field, t)
                              for t in time], dim=-1)
                    for field in field_name
                ]
            else:
                return [
                    self._load_single_snapshot(field, time) for field in field_name
                ]
        # load single field
        else:
            if isinstance(time, list):
                return pt.stack(
                    [self._load_single_snapshot(field_name, t) for t in time],
                    dim=0
                )
            else:
                return self._load_single_snapshot(field_name, time)

    @property
    def write_times(self) -> List[str]:
        return sorted(list(self._time_iter.keys()), key=float)

    @property
    def field_names(self) -> Dict[str, List[str]]:
        """Find available fields in solution files.

        Available fields are determined by matching the number of
        weights with the length of datasets in the available
        solution files; for distributed cases, the fields are only
        determined based on *domain_0*.

        :return: dictionary with time as key and list of
            available solution fields as value
        :rtype: Dict[str, List[str]]
        """
        suffix = ""
        self._field_names = {}
        for time in self.write_times:
            self._field_names[time] = []
            with Dataset(self._file_name(time, suffix)) as data:
                for key in data.variables.keys():
                    # Dirty workaround: Give all keys and let the user be smart. Works okay for the example dataset.
                    self._field_names[time].append(key)
        return self._field_names

    @property
    def vertices(self) -> pt.Tensor:
        # Use _load_mesh_data to fill the dict if not already done.
        if not self._global_id_of_zone_points:
            self._load_mesh_data()
        path = join(self._para.path, self._para.config[GRID_FILE_KEY])
        with Dataset(path) as data:
            # Get the vertices of the points of the current zone
            vertices = pt.stack(
                    [pt.tensor(data[key][self._global_id_of_zone_points[self._zone]], dtype=self._dtype)
                     for key in VERTEX_KEYS],
                    dim=-1
                )
        return vertices

    @property
    def weights(self) -> pt.Tensor:
        # This would require an additional opening of the mesh file. Workaround?
        weights = pt.ones(self.vertices.shape[0], dtype=self._dtype)
        return weights

    @property
    def zone_names(self) -> List[str]:
        """Names of available blocks/zones.

        :return: block/zone names
        :rtype: List[str]
        """
        if self._zone_names is None:
            self._zone_names = list(self._para._bmap.keys())
        return self._zone_names

    @property
    def zone(self) -> str:
        """Currently selected block/zone.

        :return: block/zone name
        :rtype: str
        """
        return self._zone

    @zone.setter
    def zone(self, value: str):
        """Select active block/zone.

        The selected block remains unchanged if an invalid
        block name is passed

        :param value: name of block to select
        :type value: str
        """
        if value in self.zone_names:
            self._zone = value
        else:
            print(f"{value} not found. Available zones are:")
            print(self.zone_names)
