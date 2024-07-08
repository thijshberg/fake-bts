import math
import multiprocessing as mp
import multiprocessing.managers as mpm
import os
import pickle
import random
import xml.etree.ElementTree as ET
from math import floor
import itertools as itools
import subprocess

import bezier.curve as bz
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Point
from scipy.cluster.vq import kmeans, whiten
import re
from dataclasses import dataclass
from numpy.linalg import norm
from matplotlib.patches import Rectangle

from .baseStation import BaseStation, Attacker, ATTACKER_TAC, ATTACKER_ECI
from .util import *

Point.__add__ = add_point
"""
Code for making a map. Many parts of this are depracted or have fallen out of use.
"""


@dataclass
class Obstruction:
    location: Point
    strength: float
    dim_x: float
    dim_y: float
    height: float

    def __init__(self, location, strength, dim_x, dim_y=None, height=None):
        self.location = location
        self.strength = strength
        self.dim_x = dim_x
        self.dim_y = dim_y if (dim_y is not None) else dim_x
        self.height = height if (height is not None) else 50.0

    def asdict(self) -> dict:
        return {'x': self.location.x, 'y': self.location.y, 'strength': self.strength, 'dim_x': self.dim_x, 'dim_y': self.dim_y, 'height': self.height}

    @classmethod
    def fromdict(cls, d: dict):
        return cls(Point(d['x'], d['y']), d['strength'], d['dim_x'], d['dim_y'], d['height'])

    def to_rectangle(self, **kwargs) -> Rectangle:
        return Rectangle((self.location.x-self.dim_x, self.location.y-self.dim_y), 2*self.dim_x, 2*self.dim_y, **kwargs)


class ConnectionMap():
    """
    Note that the initialization here will not produce a usable map, as it does not provide generate a strength_map; instead use the create_map and generate_map methods.
    The file argument allow the data to be read from the disk. This is more efficient for the proxy setup.
    """
    encoding = {}
    obstructions: Sequence[Obstruction] = []
    disabled_stations = []

    def __init__(self, bounds, subdivision, obstructions, mode='empty', map_arr=None, file=None, file_type='pickle', cmap=None, obstruction_strength="random.uniform(0.5,1)", distance_mode=None):
        self.tac = 1000
        if file:
            if file_type == 'pickle':
                with open(file, 'rb') as f:
                    cmap, map_size, nr_stations, subdivision = pickle.load(f)
                self.map_bounds = cmap.map_bounds
                self.mode = cmap.mode
                self.grid_size = cmap.grid_size
                self.grid = cmap.grid
                self.stations = cmap.stations
                self.strength_map = cmap.strength_map
                self.subdivision = cmap.subdivision
                self.obstructions = cmap.obstructions
                self.map_arr = cmap.map_arr
                self.map_bool = cmap.map_bool
                self.plmn_ids = cmap.plmn_ids
                return
            elif file_type == 'raytraced':
                self.strength_map = np.fromfile(file)
            else:
                print("Unknown file type", file_type)
                return
        elif cmap:
            self.map_bounds = cmap.map_bounds
            self.mode = cmap.mode
            self.grid_size = cmap.grid_size
            self.grid = cmap.grid
            self.stations = cmap.stations
            self.strength_map = cmap.strength_map
            self.subdivision = cmap.subdivision
            self.map_arr = cmap.map_arr
            self.map_bool = cmap.map_bool
            self.plmn_ids = cmap.plmn_ids
            return
        self.map_bounds = bounds
        self.mode = mode
        if distance_mode is None:
            self.distance_mode = mode
        self.map_arr = None
        self.map_bool = None
        if mode == 'premade':
            if not map_arr is None:
                self.map_arr = map_arr
                self.grid_size = 1
                self.grid = []
            else:
                raise Exception("No map given!")
        else:
            self.grid_size = math.ceil(math.sqrt(obstructions))
            self.grid = [Point(math.floor((i+1/2)*bounds/self.grid_size), math.floor((j+1/2)*bounds/self.grid_size))
                         for i in range(self.grid_size) for j in range(self.grid_size)]
        self.stations = list()
        self.plmn_ids = list()
        #self.strength_map = None
        self.subdivision = subdivision
        self.stored_map = None
        self.attacker_id = -1

    def calculate_map_bool(self):
        # Users will check all surrounding points, so this could include points which are just out of bounds. We also set the leftmost column to 0 so users will not query column -1
        self.map_bool = np.zeros(
            (self.map_bounds+1, self.map_bounds+1), dtype=bool)
        for x, y in self.map_arr:
            self.map_bool[x, y] = True

    def closest_obstruction(self, p):
        return self.obstructions[np.argmin([p.distance(ob.location) for ob in self.obstructions])]

    def random_suitable_point(self, n=1, method='uniform',  seed=None):
        """
        Finds a random point which is not in one of the defined obstructions

        """
        random.seed(seed)
        if self.mode == 'blocks':
            return self.random_suitable_point_blocks(n, method)
        if self.mode == 'empty':
            return self.random_suitable_point_empty(n, method)
        if self.mode == 'premade':
            return random.sample(self.map_arr.tolist(), k=n)

    def random_suitable_point_blocks(self, n=1, method='all_points'):
        coords = np.linspace(0, self.map_bounds, self.map_bounds//20)
        choice_grid = np.dstack(np.meshgrid(coords, coords)).reshape(-1, 2)
        dim_x = self.obstructions[0].dim_x
        dim_y = self.obstructions[0].dim_y
        directions = np.array(
            [[dim_x+1, dim_y+1], [dim_x+1, -dim_y-1], [-dim_x-1, dim_y+1], [-dim_x-1, -dim_y-1]])
        possible_points = [np.array(
            [ob.location.x, ob.location.y]) + d for ob in self.obstructions for d in directions]
        if method == 'walkable':
            return [Point(*p) for p in random.sample(possible_points, n)]
        for grid_point in self.grid:
            if min(grid_point.distance(ob.location) for ob in self.obstructions) > dim_x:
                possible_points.append(
                    np.array([grid_point.x, grid_point.y]))
        if not possible_points:
            Exception("cmap does not have any possible points!")
        possible_points = np.round(np.array(possible_points))
        if method == 'all_points':
            return [Point(*p) for p in possible_points]
        if method == 'street_points':
            side = np.arange(0, self.map_bounds)
            all_points = cartesian_product(side, side)
            ob_side = np.int16(self.obstructions[0].dim_x)
            for ob in self.obstructions:
                x_filtered = np.abs(all_points[:, 0]-ob.location.x) < ob_side
                y_filtered = np.abs(all_points[:, 1]-ob.location.y) < ob_side
                all_points = all_points[~(x_filtered & y_filtered)]
            if n:
                rando = np.random.default_rng()
                return rando.choice(all_points, n)
            return all_points

        if method == 'random':
            # all grid points for which both coordinates are not too far from the edge
            return [Point(*p) for p in random.sample(list(possible_points), k=n)]
        if method == 'uniform':
            # if placing towers in this way, make sure there are less towers than (self.grid_size)**2
            res = []
            points, _ = kmeans(choice_grid, n)
            for point in points:
                distances = np.array([norm(point-x) for x in possible_points])
                loc = Point(*possible_points[np.argmin(distances)])
                res.append(loc)
                possible_points = np.array([p for p in possible_points if (abs(
                    p[0] - loc.x) > self.map_bounds/self.grid_size/2) and (abs(p[1] - loc.y) > self.map_bounds/self.grid_size/2)])
            return res

    def random_suitable_point_empty(self, n=1, method='random'):
        if method == 'uniform' or method == 'walkable':
            coords = np.linspace(0, self.map_bounds, self.map_bounds//10)
            choice_grid = np.dstack(np.meshgrid(coords, coords)).reshape(-1, 2)
            points, _ = kmeans(choice_grid, n)
            return [Point(round(a[0]), round(a[1]))
                    for a in points]
        if method == 'random':
            # If we get the randoms like this then if we run the creation with more towers the first will line up
            randoms = random.sample(
                range(self.map_bounds//10, 9*self.map_bounds//10), 2*n)
            randoms_x = randoms[::2]  # every second entry starting at 0
            randoms_y = randoms[1::2]  # every second entry starting at 1
            return [Point(round(a[0]), round(a[1]))
                    for a in zip(randoms_x, randoms_y)]

    def strength_at(self, x, y, plmn_id=0, frequency=0, inter_freq=False):
        if self.mode == 'premade':  # even if self.map_bounds is the same as self.subdivision, the normal version gives slightly off results
            return [rec[:3] for rec in self.strength_map[x][y]]
        x_index = min(
            max(0, floor(x/self.map_bounds * self.subdivision)), self.subdivision-1)
        y_index = min(
            max(0, floor(y/self.map_bounds * self.subdivision)), self.subdivision-1)
        enc_mask = list(self.encoding.copy().items())
        if plmn_id:
            ...  # TODO???
        if frequency:
            enc_mask = list(
                filter(lambda x: x[0][1] == frequency, enc_mask))
        for disabled_station in self.disabled_stations:
            enc_mask = list(
                filter(lambda x: x[0][0] != disabled_station, enc_mask))
        try:
            signal_strengths = self.strength_map[(x_index, y_index)][[
                snd(x) for x in enc_mask]]
        except IndexError:
            error_logging(
                f'Index error: {self.strength_map[x_index,y_index]}')
            signal_strengths = [(0, 0)]
        signals_by_eci = itools.groupby(
            zip(enc_mask, signal_strengths), lambda item: item[0][0][0])
        return [[(*rec[0][0], rec[1]) for rec in group] for eci, group in signals_by_eci]

    def nr_attackers(self):
        return len(list(filter(lambda s: isinstance(s, Attacker), self.stations)))

    def store_map(self):
        self.stored_map = self.strength_map

    def get_stored_attack(self, filepath):
        with open(filepath, 'rb') as f:
            att_station, att_strength_data = pickle.load(f)
        self.stations.append(att_station)
        self.strength_map = att_strength_data
        self.attacker_id = att_station.eci

    def remove_attacker(self):
        self.remove_station(self.attacker_id)

    def retrieve_map(self):
        self.strength_map = self.stored_map

    def set_strength_map(self, strength_map):
        self.strength_map = strength_map

    def append_station(self, station):
        self.stations.append(station)

    def remove_station(self, station_id):
        self.stations = delete_on(
            self.stations, lambda stat: stat.eci == station_id)

    def get_map(self):
        return self.map_bool

    def _callmethod(self, fname, args=(), kwargs={}):
        return self.__getattribute__(fname)(*args, **kwargs)

    #########################################################################################
    # Metric computation
    #########################################################################################

    @staticmethod
    def _max_strength_metric(method):
        def inner_metric(self, plmn_id=None):
            def has_plmn_id(x):
                if plmn_id is None:
                    return True
                else:
                    return plmn_id in x[3]
            return method(np.array([
                max((snd(x) for x in ifilter(has_plmn_id, rec)), default=-140)
                for column in self.strength_map
                for rec in column
            ]))
        return inner_metric

    # Due to pickle limitations this cannot be done point-free
    def average_max_strength(self, plmn_id=None):
        return (self._max_strength_metric(np.average))(self, plmn_id=plmn_id)

    def variance_max_strength(self, plmn_id=None):
        return (self._max_strength_metric(np.var))(self, plmn_id=plmn_id)

    def median_max_strength(self, plmn_id=None):
        return (self._max_strength_metric(np.median))(self, plmn_id=plmn_id)

    def minimum_max_strength(self, plmn_id=None):
        return (self._max_strength_metric(np.amin))(self, plmn_id=plmn_id)

    @staticmethod
    def _count_strength_metric(method):
        def inner_metric(self, threshold, plmn_id=None):
            def has_prop(x):
                if plmn_id is None:
                    return x[1] > threshold
                else:
                    return (plmn_id in x[3]) and (x[1] > threshold)
            return method(np.array([
                ilen(ifilter(has_prop, rec))
                for column in self.strength_map
                for rec in column
            ]))
        return inner_metric

    def average_above_threshold(self, threshold, plmn_id=None):
        return (self._count_strength_metric(np.average))(self, threshold, plmn_id=plmn_id)

    def variance_above_threshold(self, threshold, plmn_id=None):
        return (self._count_strength_metric(np.var))(self, threshold, plmn_id=plmn_id)

    def median_above_threshold(self, threshold, plmn_id=None):
        return (self._count_strength_metric(np.median))(self, threshold, plmn_id=plmn_id)

    def percentage_above_threshold(self, threshold, plmn_id=None):
        return (self._count_strength_metric(lambda a: np.sum(a > 0)))(self, threshold, plmn_id=plmn_id)/self.subdivision**2

#########################################################################################
# Proxy setup
#########################################################################################


class NonCachedConnectionMap(ConnectionMap):
    def __init__(self, bounds, subdivision, obstructions=1, mode='empty', distance_mode=None, map_arr=None):
        self.tac = 1000
        self.map_bounds = bounds
        self.map_arr = None
        self.map_bool = None
        self.stations = list()
        self.plmn_ids = list()
        self.obstructions = list()
        self.attacker_id = -1
        self.subdivision = subdivision
        self.mode = mode
        self.distance_mode = distance_mode if distance_mode else self.mode
        self.map_arr = map_arr
        if self.mode == 'premade':
            self.calculate_map_bool()
        self.grid_size = math.ceil(math.sqrt(obstructions))
        self.grid = [Point(math.floor((i+1/2)*bounds/self.grid_size), math.floor((j+1/2)*bounds/self.grid_size))
                     for i in range(self.grid_size) for j in range(self.grid_size)]

    def strength_at(self, x, y, mode=None, frequency=0, plmn_id=0):
        if mode is None:
            _mode = self.distance_mode
        else:
            _mode = mode
        if not frequency:
            return list(filter(bool, [[(s.eci, *r) for r in s.get_signal_strength(Point(x, y), mode=_mode)] for s in self.stations]))
        else:
            return list(filter(bool, ([(s.eci, *r) for r in s.get_signal_strength(Point(x, y), mode=_mode) if r[0] == frequency] for s in self.stations)))

    def _callmethod(self, fname, args=(), kwargs={}):
        return self.__getattribute__(fname)(*args, **kwargs)


#########################################################################################
# Proxy setup
#########################################################################################

class ConnectionMapManager(mpm.BaseManager):
    pass


class ConnectionMapProxy(mpm.NamespaceProxy):
    _exposed_ = ('remove_attacker', 'get_stored_attack', 'store_map', 'retrieve_map', 'get_map', 'random_suitable_point', 'strength_at',
                 'add_stations', 'set_strength_map', 'add_station', 'is_in_obstruction', 'append_station', 'remove_station', '__getattribute__')

    def random_suitable_point(self, seed=None):
        if seed:
            return self._callmethod('random_suitable_point', kwds={'seed': seed})
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('random_suitable_point')


#########################################################################################
# Premade map creation
#########################################################################################

def premade_part(streetname, root, min_nodes):
    street_points = []
    for way in root.iter('way'):
        if any(tag.attrib['v'] == streetname for tag in way.iter('tag') if tag.attrib['k'] == 'name'):
            for nd in way.findall('nd'):
                node_id = nd.attrib['ref']
                for node in root.findall('node'):
                    if node.attrib['id'] == node_id:
                        p = [float(node.attrib['lon']),
                             float(node.attrib['lat'])]
                        street_points.append(p)
    street_points.sort(key=fst)
    unzipped_points = list(zip(*street_points))
    if len(unzipped_points) == 0:
        error_logging(f'Found no data for {streetname}!')
    x_data = np.array(unzipped_points[0])
    if len(x_data) > min_nodes:
        x_min = np.amin(x_data)
        x_max = np.amax(x_data)
        bz_curve = bz.Curve.from_nodes(np.array(street_points).T)
        return street_points, (x_min, x_max, bz_curve)


def create_premade_map(config, save=False, cache=False, frequencies=[]):
    filename = config['filename']
    tree = ET.parse(filename)
    root = tree.getroot()
    streetnames = config.get('streetnames', [])
    names_to_remove = config.get('names_to_remove', [])
    output_folder = config['folder']
    streetname_filter_s = config.get('streetname_identifiers', [])
    streetname_filter = re.compile(f'({"|".join(streetname_filter_s)})+')
    other_filter = re.compile(config.get('other_filter', '^Nonsense'))
    if not streetnames:
        for way in root.iter('way'):
            for tag in way.iter('tag'):
                if tag.attrib['k'] == 'name':
                    if (streetname := tag.attrib['v']) not in streetnames and streetname_filter.search(streetname) and not other_filter.search(streetname):
                        streetnames.append(streetname)
    for name in names_to_remove:
        try:
            streetnames.remove(name)
        except Exception:
            pass
    if config.get('print_streetnames', False):
        print(streetnames)
    bts_locs = config.get('bts_locs', [])
    bts_locs = [list(RDtoGPS(p[0], p[1])) + p[2:] for p in bts_locs]
    bezier_data = []
    points = []
    min_nodes = config.get('min_nodes', 3)
    if (nr_processes := config.get('nr_processes', False)):
        with mp.Pool(nr_processes) as pool:
            data = pool.starmap(premade_part, itools.product(
                streetnames, itools.repeat(root), itools.repeat(min_nodes)), chunksize=10)
        zipped_data = zip(*data)
        points = next(zipped_data)
        bezier_data = next(zipped_data)
    else:
        for streetname in streetnames:  # There are some streets which appear multiple times, so we have to go by streetname
            street_points = []
            for way in root.iter('way'):
                if any(tag.attrib['v'] == streetname for tag in way.iter('tag') if tag.attrib['k'] == 'name'):
                    for nd in way.findall('nd'):
                        node_id = nd.attrib['ref']
                        for node in root.findall('node'):
                            if node.attrib['id'] == node_id:
                                p = [float(node.attrib['lon']),
                                     float(node.attrib['lat'])]
                                street_points.append(p)
            street_points.sort(key=lambda x: x[0])
            points += street_points
            unzipped_points = list(zip(*street_points))
            if len(unzipped_points) == 0:
                print(f'Found no data for {streetname}!')
                exit(0)
            x_data = np.array(unzipped_points[0])
            if len(x_data) > min_nodes:
                x_min = np.amin(x_data)
                x_max = np.amax(x_data)
                bz_curve = bz.Curve.from_nodes(np.array(street_points).T)
                bezier_data.append((x_min, x_max, bz_curve))
    arr = np.array(points)
    min_x = np.amin(arr[:, 0])
    max_x = np.amax(arr[:, 0])
    max_y = np.amax(arr[:, 1])
    min_y = np.amin(arr[:, 1])
    y_conv = 111320
    x_conv = 40075 * math.cos(min_y/360)
    map_bounds = max(math.ceil((max_x - min_x) * x_conv),
                     math.ceil((max_y - min_y) * y_conv))

    def reformat(p):
        return [(p[0]-min_x) * x_conv, (p[1]-min_y) * y_conv]

    def round_point(p):
        return [round(p[0]), round(p[1])]
    t_range = np.arange(0.0, 1.0, 0.001)
    map_arr = np.array([
        round_point(reformat(p))
        for rec in bezier_data
        for p in rec[2].evaluate_multi(t_range).T
    ], dtype=int)
    station_locs = (round_point(reformat(p[:2][::-1])) for p in bts_locs)
    map_arr = np.unique(map_arr, axis=0)
    perturbations = config.get('perturbations', 0)

    def generate_freqlist():
        while True:
            res = []
            while not res:
                for freq, factor in frequencies.items():
                    r = random.random()
                    if r < factor:
                        res.append(freq)
            yield res
    try:
        last_strength = signal_strengths[-1]
    except:
        last_strength = 20
    signal_strengths = itools.chain(
        iter(signal_strengths), itools.repeat(last_strength))
    if perturbations:
        map_arr = perturbate(map_arr, map_bounds, n=perturbations)
    if cache:
        cmap = ConnectionMap(map_bounds, map_bounds, 0,
                             mode='premade', map_arr=map_arr)
        cmap.stations = [BaseStation(i, bts_locs[i][2], position=Point(position), plmn_ids=[
            bts_locs[i][3]]) for i, position in enumerate(station_locs)]
        strength_map = [[None for _ in range(0, map_bounds+1)]
                        for _ in range(0, map_bounds+1)]
        for p in map_arr:
            try:
                strength_map[p[0]][p[1]] = [(i, stat.get_signal_strength(Point(p[0], p[1]), mode=config.get(
                    'method', 'hata')), stat.tac, stat.plmn_ids) for i, stat in enumerate(cmap.stations)]
            except IndexError:
                pass
                print(f'{p[0]},{p[1]}')
        cmap.strength_map = strength_map
        cmap.calculate_map_bool()
        cmap.grid = []
        cmap.plmn_ids = uniques((p[3] for p in bts_locs))
        try:
            os.mkdir(os.path.join('Maps', output_folder))
        except:
            pass
        if save:
            with open(os.path.join('Maps', output_folder, 'map'), 'wb') as file:
                pickle.dump((cmap, map_bounds, len(
                    bts_locs), map_bounds), file)
    else:
        cmap = NonCachedConnectionMap(
            map_bounds, map_bounds, mode='premade', map_arr=map_arr, distance_mode='hata')
        stations = [BaseStation(i, cmap.tac, position=loc, cmap=cmap, noise=noise, frequencies=freq, signal_strength=sig_strength, max_users=max_users) for i, loc, freq, sig_strength, max_users in zip(
            range(no_stations), station_locs, generate_freqlist(), signal_strengths, max_users_list)]
        cmap.stations = stations
    return cmap


def perturbate(points_arr, map_bounds, n=1):
    res = points_arr
    for i in range(n):
        up = points_arr[points_arr[:, 1] < map_bounds-i] + np.array([0, i])
        down = points_arr[points_arr[:, 1] > i] + np.array([0, -i])
        right = points_arr[points_arr[:, 0] < map_bounds-i] + np.array([i, 0])
        left = points_arr[points_arr[:, 0] > i] + np.array([-i, 0])
        res = np.unique(np.concatenate((res, up, down, left, right)), axis=0)
    return res

###############################################################
# Map creation
###############################################################


def _sim_map_part(j, map_size, subdivision, stations, distance_mode):
    """
    generate one column of the strenght map for a list of stations
    """
    def g(i, j):
        s = ((s.eci, s.get_signal_strength(Point(map_size/subdivision * (j+1/2), map_size /
                                                 subdivision * (i+1/2)), mode=distance_mode), s.tac, s.plmn_ids) for s in stations)
        return [rec for rec in s if rec[1] > -140]
    return [g(i, j) for i in range(subdivision)]


def generate_freqlist(frequencies: dict):
    while True:
        res = []
        while not res:
            for freq, factor in frequencies.items():
                r = random.random()
                if r < factor:
                    res.append(freq)
        yield res


def generate_map(no_stations,
                 map_size,
                 subdivision,
                 obstructions,
                 plmn_nr=1,
                 mode='default',
                 frequencies={},
                 distance_mode='default',
                 method='uniform',
                 max_plmn_distance=2000,
                 max_processes=1,
                 obstruction_strength='5',
                 seed=None,
                 attack={},
                 noise=5,
                 cache=True,
                 signal_strengths=[],
                 nr_tracking_areas=1,
                 max_users_list=[],
                 add_outsiders=True):
    """
    generate a map with obstructions, stations and corresponding map of connection strength according to subdivision
    """
    multiprocess = max_processes > 1
    random.seed(seed)
    if cache:
        cmap = ConnectionMap(map_size, subdivision, obstructions,
                             mode=mode, obstruction_strength=obstruction_strength)
    else:
        cmap = NonCachedConnectionMap(
            map_size, subdivision, obstructions=obstructions, mode=mode, distance_mode=distance_mode)
    if obstructions:
        locations = random.sample(cmap.grid, k=obstructions)
        cmap.obstructions = [Obstruction(
            location=loc,
            dim_x=0.4*map_size/cmap.grid_size,
            strength=eval(obstruction_strength),
            height=random.random()*25.0+5.0
        ) for loc in locations]
    cmap_ = cmap if mode == 'blocks' else None

    try:
        last_strength = signal_strengths[-1]
    except:
        last_strength = 20
    signal_strengths = itools.chain(
        iter(signal_strengths), itools.repeat(last_strength))
    max_users_list = itools.chain(max_users_list, itools.repeat(float('inf')))
    signal_strengths = iter(signal_strengths)
    stations = [BaseStation(i, cmap.tac, position=loc, cmap=cmap_, noise=noise, frequencies=freq, signal_strength=sig_strength, max_users=max_users) for i, loc, freq, sig_strength, max_users in zip(
        range(no_stations), cmap.random_suitable_point(n=no_stations, method=method, seed=seed), generate_freqlist(frequencies), signal_strengths, max_users_list)]
    if mode != 'empty' and add_outsiders:
        for direction in (0, 1):
            stations.append(BaseStation(no_stations+direction, cmap.tac, position=Point(cmap.map_bounds//2, -cmap.map_bounds//2+2*direction *
                            cmap.map_bounds), cmap=cmap_, noise=noise, frequencies=list(frequencies.keys()), signal_strength=next(signal_strengths)))
            stations.append(BaseStation(no_stations+2+direction, cmap.tac, position=Point(-cmap.map_bounds//2+2*direction*cmap.map_bounds,
                            cmap.map_bounds//2), cmap=cmap_, noise=noise, frequencies=list(frequencies.keys()), signal_strength=next(signal_strengths)))
    if mode != 'empty':
        for station in stations:
            ob = cmap.closest_obstruction(station.position)
            if ob.location.distance(station.position) > cmap.grid_size/4:
                station.height = 10.0
            else:
                station.height = ob.height+10.0
    if attack:
        if attack.get('all_positions'):
            add_all_attackers(cmap, stations, list(frequencies.keys()))
        if nr_attackers := attack.get('positions'):
            add_random_attackers(cmap, stations,nr_attackers, list(frequencies.keys()))
        else:
            cmap.stations = stations
            position = Attacker.find_position(cmap, mode='maximum_distance')
            frequencies = [attack.get('frequency')] if attack.get(
                'frequency') else list(frequencies.keys())
            stat_positions = np.array([[s.position.x, s.position.y, s.eci]
                                       for s in stations])
            if not len(stat_positions):
                error_logging(
                    'There is no station with the attacker frequency!')
                error_logging('Appointed frequencies:', {
                    s.identifier: s.frequencies for s in stations})
            attack_eci = int(
                min(stat_positions, key=lambda rec: np.linalg.norm(rec[:2]-position))[2])
            attacker = Attacker(attack_eci, identifier=ATTACKER_ECI,
                                position=position, frequencies=frequencies, tac=ATTACKER_TAC)
            stations.append(attacker)
    if nr_tracking_areas > 1:
        locs = np.array([[stat.x, stat.y] for stat in stations])
        clusters = kmeans(locs, nr_tracking_areas)
        for stat, label in zip(stations, clusters):
            stat.tac = int(label)
    if method == 'uniform' and plmn_nr > 1:
        def distance_for_plmn(position, plmn_id):
            return min((s.position.distance(position) for s in stations if plmn_id in s.plmn_ids))
        for plmn_id in range(plmn_nr):
            # Procedure: pick a random station for this plmn, and while there are stations further than the specified distance add one of those at random
            first_station = random.choice(stations)
            # just doing append does some trippy things, as the plmn_ids lists all point to the same list apparently.
            first_station.plmn_ids = first_station.plmn_ids + [plmn_id]
            while True:
                choices = [s for s in stations if (not plmn_id in s.plmn_ids) and (
                    distance_for_plmn(s.position, plmn_id) > max_plmn_distance)]
                if not choices:
                    break
                next_station = random.choice(choices)
                next_station.plmn_ids = next_station.plmn_ids + [plmn_id]
        stations = [s for s in stations if s.plmn_ids]
    else:
        for s in stations:
            s.plmn_ids = list(range(plmn_nr))
    cmap.stations = stations
    cmap.plmn_ids = list(range(plmn_nr))
    return cmap


def add_all_attackers(cmap, stations, frequencies):
    choices = cmap.random_suitable_point(method='all_points')

    def edge_distance_check(p):
        return min(p.x, cmap.map_bounds-p.x, p.y, cmap.map_bounds-p.y) > cmap.map_bounds//10

    def min_dist(p):
        return min(p.distance(s.position) for s in stations)
    choices = list(filter(lambda p: edge_distance_check(p)
                   and min_dist(p) > 50, choices))
    for i, position in enumerate(choices):
        stations.append(Attacker(ATTACKER_ECI, identifier=ATTACKER_ECI+i, position=position,
                        frequencies=frequencies, tac=ATTACKER_TAC))


def add_random_attackers(cmap, stations,nr_attackers, frequencies):
    choices = cmap.random_suitable_point(
        method='street_points', n=nr_attackers)
    for i, position in enumerate(choices):
        stations.append(Attacker(ATTACKER_ECI, identifier=ATTACKER_ECI+i, position=Point(*position),
                        frequencies=frequencies, tac=ATTACKER_TAC))


def create_map(config, save=True, attack=None):
    if filebase := config.get('raytrace_filebase'):
        folder = config.get('raytrace_folder', '')
        check_file = os.path.join(folder, filebase+'_mconf.json')
        if not os.path.exists(check_file):
            print('[MAP] does not exist:', check_file)
            create_raytraced_map(config, filebase, folder)
        if redo_map := config.get('redo_map'):
            print('[MAP] map redo forced by config')
            create_raytraced_map(config, filebase, folder)
        with open(check_file, 'r') as f:
            old_conf = json.load(f)
        if redo_map != False:
            for key, item in config.items():
                if key in ['frequencies', 'redo_map']:
                    continue
                if old_item := old_conf.get(key):
                    if item != old_item:
                        print('[MAP] changed option:', key)
                        create_raytraced_map(config, filebase, folder)
                        break
                elif old_item is None:
                    print('[MAP] new option:', key)
                    create_raytraced_map(config, filebase, folder)
                    break
        return load_raytraced_map(config, attack=attack)
    else:
        return create_map_inner(config, save)


def create_map_inner(config, save=True):
    mode = config.get('mode', 'empty')
    folder_name = config.get('folder', 'tmp')
    map_size = config.get('map_size')
    nr_stations = config.get('nr_stations')
    subdivision = config.get('subdivision', map_size)
    frequencies = {int(key): item for key, item in config.get(
        'frequencies', {1000: 1}).items()}
    if mode == 'premade' and 'filename' in config:
        with open(config['filename'], 'r') as f:
            premade_config = json.load(f)
        cmap = create_premade_map(
            premade_config, cache=config.get('cache', True), save=save)
    else:
        cmap = generate_map(
            nr_stations,
            map_size,
            subdivision,
            config.get('obstructions', 0),
            mode=mode,
            plmn_nr=config.get('plmn_nr', 1),
            frequencies=frequencies,
            attack=config.get('attack', {}),
            distance_mode=config.get('distance_mode', 'empty'),
            method=config.get('method', 'uniform'),
            max_plmn_distance=config.get('max_plmn_distance', 1000),
            seed=config.get('seed'),
            max_processes=config.get('max_processes', 1),
            obstruction_strength=config.get(
                'obstruction_strength', 'random.uniform(0.1,0.5)'),
            noise=config.get('noise', 5),
            cache=config.get('cache', True),
            signal_strengths=config.get('signal_strengths', []),
            max_users_list=config.get('max_users_cell', []),
            nr_tracking_areas=config.get('nr_tracking_areas', 1),
            add_outsiders=config.get('add_outsiders', True)
        )
    if save:
        print('Now writing map to disk')
        filepath = os.path.join('Maps', folder_name)
        try:
            os.mkdir(filepath)
        except:
            pass
        print(filepath)
        with open(os.path.join(filepath, 'map'), 'wb') as f:
            pickle.dump((cmap, map_size, nr_stations, subdivision), f)
    return cmap


def load_raytrace_results(file, bounds):
    data = np.fromfile(file, dtype=np.float32)
    arr = np.array([[0.0]*bounds]*bounds)
    for i in range(bounds):
        arr[i] = data[bounds*i:bounds*(i+1)]
    return arr


RAYTRACER_PATH = "raytracer"


def create_raytraced_map(conf, filebase=None, folder=None, nr_threads=None):
    cmap_noncached = create_map_inner(conf, save=False)
    rt_config = get_cmap_sigtracer_conf(
        cmap_noncached, nr_probes=conf.get('nr_probes', 1_000_000))
    if not filebase:
        filebase = conf.get('raytrace_filebase', filebase)
    if not filebase:
        error_logging("No filebase given!")
        return
    if not folder:
        folder = conf.get('raytrace_folder')
    if folder:
        if not os.path.isdir(folder):
            os.mkdir(folder)
        filebase = os.path.join(folder, filebase)
    conf_file = filebase+"_config.json"
    with open(conf_file, 'w') as f:
        json.dump(rt_config, f, indent=2)
    sys.stdout.flush()
    cmd = f'{RAYTRACER_PATH} {conf_file} {filebase}'
    env = os.environ.copy()
    if nr_threads:
        env['RAYON_NUM_THREADS'] = str(nr_threads)
    proc = subprocess.Popen([cmd],
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    proc.wait()
    for line in proc.stderr.readlines():
        print('[RAYTRACER ERROR]', str(line))
    for line in proc.stdout.readlines():
        print('[RAYTRACER]', str(line))
    map_data = {}
    for s in cmap_noncached.stations:
        for freq in s.frequencies:
            map_data[(s.identifier, freq)] = load_raytrace_results(
                f'{filebase}_{int(s.identifier)}_{freq}.data', cmap_noncached.map_bounds)
    fo_1, fo_2 = itools.tee((s.identifier, freq)
                            for s in cmap_noncached.stations for freq in s.frequencies)
    encoding = {(s, freq): i
                for i, (s, freq) in enumerate(fo_1)
                }
    data = np.dstack([map_data[(s, f)] for (s, f) in fo_2])
    with open(filebase+'_ob.json', 'w') as f:
        json.dump([ob.asdict()
                  for ob in cmap_noncached.obstructions], f, indent=2)
    with open(filebase+'_stat.json', 'w') as f:
        json.dump([station.asdict()
                  for station in cmap_noncached.stations], f, indent=2)
    data.tofile(filebase+'.data')
    with open(filebase+'_enc.json', 'w') as f:
        json.dump([[*k, i] for k, i in encoding.items()], f, indent=2)
    with open(filebase+'_mconf.json', 'w') as f:
        json.dump(conf, f, indent=2)
    return

def load_raytraced_map(conf, attack=None):
    rt_file_root = conf.get('raytrace_filebase')
    if not rt_file_root:
        error_logging('No file root given!')
        return None
    if folder := conf.get('raytrace_folder'):
        rt_file_root = os.path.join(folder, rt_file_root)
    with open(rt_file_root+'_ob.json', 'r') as f:
        obstructions = list(map(Obstruction.fromdict, json.load(f)))
    with open(rt_file_root+'_stat.json', 'r') as f:
        stations = list(map(BaseStation.fromdict, json.load(f)))
    with open(rt_file_root+'_enc.json', 'r') as f:
        encoding = {(identifier, freq): i for (
            identifier, freq, i) in json.load(f)}
    big_map=False
    if len(stations) > 100:
        big_map=True
        cmap = ConnectionMap(conf['map_size'], conf['map_size'], conf['obstructions'],
                            mode='blocks')
    else:
        cmap = ConnectionMap(conf['map_size'], conf['map_size'], conf['obstructions'],
                            mode='blocks', file=rt_file_root+'.data', file_type='raytraced')
    cmap.obstructions = obstructions
    # The frequencies generated from the config can be a subset of all available frequencies
    frequencies = list(zip(filter(lambda s: not isinstance(
        s, Attacker), stations), generate_freqlist(conf['frequencies'])))
    # return stations
    if attack:
        attackers = list(filter(lambda s: isinstance(s, Attacker), stations))
        if not attackers:
            print("ERROR: no attackers!")
            raise(Exception())
        attacker = attackers[attack.get('attack_id', 0)]
        frequencies.append((attacker, [attack['frequency']]))
        stations = list(filter(lambda s: not isinstance(
            s, Attacker) or s == attacker, stations))
    encoding_target = [(s.identifier, f)
                       for s, freq in frequencies for f in freq]
    encoding_mask = []
    for (i, f) in encoding_target:
        if (i, f) not in encoding:
            error_logging(
                "Generated id-frequency is not in the generated map:", (i, f))
            raise Exception("Map encoding error")
        encoding_mask.append(encoding[(i, f)])
    new_encoding = {k: i for i, k in enumerate(
        sorted(encoding_target, key=lambda x: encoding[x]))}
    cmap.encoding = new_encoding
    if big_map:
        map_data = {}
        for identifier,freq in new_encoding:
            map_data[(identifier, freq)] = load_raytrace_results(
                f'{rt_file_root}_{int(identifier)}_{freq}.data', cmap.map_bounds)
        cmap.strength_map = np.dstack(list(map_data.values()))
    else:
        cmap.strength_map = cmap.strength_map.reshape(
            (conf['map_size'], conf['map_size'], len(encoding)))[:, :, encoding_mask]
    for i, (_, g) in enumerate(itools.groupby(new_encoding.keys(), lambda k: k[0])):
        stations[i].frequencies = list(map(snd, g))
    if attack and (sig_diff := attack.get('signal_strength') - attacker.signal_strength) != 0:
        cmap.strength_map[:, :, new_encoding[(
            attacker.identifier, attack['frequency'])]] += sig_diff
    if (nr_tracking_areas := conf.get('nr_tracking_areas', 1)) > 1:
        locs = np.array([[stat.x, stat.y] for stat in stations if 0 < stat.position.x <
                        cmap.map_bounds and 0 < stat.position.y < cmap.map_bounds])
        clusters, _ = kmeans(locs, nr_tracking_areas)
        def cluster_dist(p):
            return np.array([np.linalg.norm(p-cluster) for cluster in clusters])
        for stat in stations:
            if not isinstance(stat, Attacker):
                stat.tac = int(np.argmin(cluster_dist(
                    [stat.position.x, stat.position.y])))
    cmap.stations = stations
    return cmap


def get_cmap_sigtracer_conf(cmap: ConnectionMap, nr_probes=1_000_000):
    res = {
        "width": cmap.map_bounds,
        "height": cmap.map_bounds,
        "samples_per_pixel": 128,
        "nr_probes": nr_probes,
        "max_depth": 50,
        "camera": {
            "look_from": {
                "x": -20.0,
                "y": 100.0,
                "z": 50.0
            },
            "look_at": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "vup": {
                "x": 0.0,
                "y": 1.0,
                "z": 0.0
            },
            "vfov": 50.0,
            "aspect": 1.0
        }, "objects": [{
            "origin": {
                "x": float(cmap.map_bounds/2),
                "y": 0.0,
                "z": float(cmap.map_bounds/2)
            },
            "id": 0,
            "dim_x": float(cmap.map_bounds/2),
            "dim_z": float(cmap.map_bounds/2),
            "dim_y": 0.0,
            "material": {
                "Lambertian": {
                    "albedo": [
                        0.5,
                        0.5,
                        0.5
                    ]
                }
            }
        }]
    }

    for station in cmap.stations:
        for freq in station.frequencies:
            res['objects'].append({
                "origin": {
                    "x": station.x,
                    "y": station.height,
                    "z": station.y
                },
                "dim_x": 1.0,
                "dim_y": 1.0,
                "dim_z": 1.0,
                "id": station.identifier,
                "material": {
                    "Light": {
                        "color": [
                            1.0,
                            1.0,
                            1.0
                        ],
                        "strength": station.signal_strength,
                        "beams": station.nr_beams,
                        "frequency": freq
                    }
                }
            })
    for obstruction in cmap.obstructions:
        res['objects'].append(
            {
                "origin": {
                    "x": obstruction.location.x,
                    "y": obstruction.height/2,
                    "z": obstruction.location.y
                },
                "dim_x": obstruction.dim_x,
                "dim_y": obstruction.height/2,
                "dim_z": obstruction.dim_y,
                "id": -1,
                "material": {
                    "Metal": {
                        "albedo": [
                            0.5,
                            0.5,
                            0.5
                        ],
                        "fuzz": 0.5,
                        "dampening": -10*math.log10(obstruction.strength)
                    }
                }
            })
    return res
