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
    >>> from flowtorch.data import TAUDataloader
    >>> path = DATASETS["tau_backward_facing_step"]
    >>> loader = TAUDataloader(join(path, "simulation.para"))
    >>> times = loader.write_times
    >>> fields = loader.field_names[times[0]]
    >>> fields
    ['density', 'x_velocity', 'y_velocity', ...]
    >>> density = loader.load_snapshot("density", times)

    To load distributed simulation data, set `distributed=True`
    >>> path = DATASETS["tau_cylinder_2D"]
    >>> loader = TAUDataloader(join(path, "simulation.para"), distributed=True)
    >>> vertices = loader.vertices

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
        print("self._load_domain_mesh_data() not implemented!")
        return 0

    def _load_mesh_data(self):
        """Load mesh vertices and global_id for the current zone.

        The mesh data is saved as class member `_mesh_data`. The tensor has the
        dimension n_points x 3; the first three columns correspond to the x/y/z
        coordinates.
        """
        if self._distributed:
            n = self._para.config[N_DOMAINS_KEY]
            self._mesh_data = pt.cat(
                [self._load_domain_mesh_data(str(pid)) for pid in range(n)],
                dim=0
            )
        else:
            path = join(self._para.path, self._para.config[GRID_FILE_KEY])

            with Dataset(path) as data:
                # Get info from the netcdf4 mesh file
                # TODO: The try/except statement has to be fixed! In case of only tris or only quads
                #       in the mesh the calculation of 'global_id_of_surface_points' will fail!
                boundary_markers = pt.tensor(data.variables["boundarymarker_of_surfaces"][:], dtype=int)
                try:
                    surface_tris = pt.tensor(data.variables["points_of_surfacetriangles"][:], dtype=int)
                except KeyError:
                    pass
                try:
                    surface_quads = pt.tensor(data.variables["points_of_surfacequadrilaterals"][:], dtype=int)
                except KeyError:
                    pass
                # Make sure to load only surface data and not the entire mesh as these vectors get big!
                global_id_of_surface_points = np.unique(pt.cat((surface_tris.flatten(), surface_quads.flatten())))
                vertices = pt.stack(
                    [pt.tensor(data[key][global_id_of_surface_points], dtype=self._dtype)
                     for key in VERTEX_KEYS],
                    dim=-1
                )
                if WEIGHT_KEY in data.variables.keys():
                    weights = pt.tensor(
                        data.variables[WEIGHT_KEY][global_id_of_surface_points], dtype=self._dtype)
                else:
                    print(
                        f"Warning: could not find cell volumes in file {path}")
                    weights = pt.ones(vertices.shape[0], dtype=self._dtype)

            # Define the marker id based on the zone
            # TODO: Zone selection based on names and translation to marker ID's. Therefore we have to parse the bmap
            #       section of the tau para file or the bmap file itself.
            for zone in self.zone_names:
                zone_marker_id = int(zone)
                indices_of_marker = np.where((boundary_markers == zone_marker_id))
                # Extract the global ID's of selected points:
                if surface_quads is not None and surface_tris is not None:
                    # Expand surface_tris by 4th entry with 'nan' so the tensor can be added together
                    dummy_tensor = pt.zeros(size=(len(surface_tris),1))
                    dummy_tensor[dummy_tensor==0] = float('nan')
                    surface_tris_expanded = pt.cat((surface_tris,dummy_tensor),1)
                    # join tensors of tris and quads
                    points_of_tris_and_quads = pt.cat((surface_tris_expanded, surface_quads))
                    # Extract global ID's of unique points
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
                self._global_id_of_zone_points[zone] = global_id_of_marker_points

            self._mesh_data = pt.cat((vertices, weights.unsqueeze(-1)), dim=-1)

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
            field = []
            for pid in range(self._para.config[N_DOMAINS_KEY]):
                path = self._file_name(time, f".domain_{pid}")
                with Dataset(path) as data:
                    field.append(
                        pt.tensor(
                            data.variables[field_name][:], dtype=self._dtype)
                    )
            return pt.cat(field, dim=0)
        else:
            path = self._file_name(time)
            with Dataset(path) as data:
                field = pt.tensor(
                    data.variables[field_name][self._global_id_of_zone_points[self._zone]], dtype=self._dtype)
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
                    dim=-1
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
        self._field_names = {}
        if self._distributed:
            n_points = self._load_domain_mesh_data("0").shape[0]
            suffix = ".domain_0"
        else:
            n_points = self._mesh_data.shape[0]
            suffix = ""
        # This does currently not work as self._mesh_data also contains the farfield
        # for which not data is written out and therefore the length of the vectors do not match!
        # What's the idea behind this anyway?
        for time in self.write_times:
            self._field_names[time] = []
            with Dataset(self._file_name(time, suffix)) as data:
                for key in data.variables.keys():
                    if data[key].shape[0] == n_points:
                        self._field_names[time].append(key)
        return self._field_names

    @property
    def vertices(self) -> pt.Tensor:
        if self._mesh_data is None:
            self._load_mesh_data()
        # I do not understand it, but it works this way!?
        return self._mesh_data[:, :3][self._global_id_of_zone_points[self._zone]]

    @property
    def weights(self) -> pt.Tensor:
        if self._mesh_data is None:
            self._load_mesh_data()
        return self._mesh_data[:, 3][self._global_id_of_zone_points[self._zone]]

    @property
    def zone_names(self) -> List[str]:
        """Names of available blocks/zones.

        Currently only extracts the marker ID's from the the grid file.
        TODO: Extract names from para file and match them to the marker ID's.

        :return: block/zone names
        :rtype: List[str]
        """
        if self._zone_names is None:
            mesh_file = join(self._para.path, self._para.config[GRID_FILE_KEY])
            with Dataset(mesh_file) as data:
                self._zone_names = [ 
                    str(m) for m in data.variables['marker'][:] 
                ]
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