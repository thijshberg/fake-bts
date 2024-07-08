# %%
import dataclasses
import itertools as itools
import json
import os.path
import re
import sys
from statistics import mean
from copy import deepcopy
from typing import Sequence, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from functools import partial
import subprocess

import seaborn as sns
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Point

import numpy as np
import Model.util as util
from experiment_setup import ExperimentSetup, MeasurementError, experiment_main, experiment_core
from generate_configs import FrequencyConfig, FrequencyConfigRange
from Model.active_detection import plot_detection_results
from Model.user import UserHistory
from Model.connection_map import ConnectionMap,get_cmap_sigtracer_conf, create_map,Obstruction, load_raytraced_map,create_raytraced_map
from Model.util import debug, arbitrary_keywords, ilen, fst,plot_points
from Model.baseStation import BaseStation,Attacker
from tqdm import tqdm
from scipy.optimize import minimize
from Model.cppspeedup import cost_free, grad_free, dist
from Model.measurementReport import ReportLogger

"""
This file contains the code for running an experiment with the simulation.
It also contains the experiment instances that were run to create the data for the plots in the paper.
"""
# %%

freqs_constant = [FrequencyConfigRange(1000,
                                       prevalence=0.5,
                                       cell_reselection_priority_intra=5,
                                       q_rx_lev_min_intra=-80),
                  FrequencyConfigRange(2000,
                                       prevalence=0.5,
                                       cell_reselection_priority_intra=3,
                                       q_rx_lev_min_intra=-80)]


OUTPUT_TEST = "scratch/outputs/test_output"
IDLE_BASE_OUTPUT = 'scratch/outputs/idle_base_output'
CONF_BASE = {
    "meas_setup": "basic1.json",
    "att_meas_setup": "att_basic.json",
    "nr_users": 1,
    "max_time": 30,
    "output_folder": IDLE_BASE_OUTPUT,
    "sib_filename": "sib.json",
    "ping": True,
    "paging_level": 0.1,
    "seed": 107,
    "max_processes": 1,
    "network_children": 0,
    "max_steps": 10000,
    'paging_check_time': 5,
    "verbosity": {
        "log_user": 0,
        "print_users_to_page": False,
        "attack_messages": True,
        "reveal_network_records": 0,
        "print_status": False
    },
    "user": {
        "behaviour2_time": 100,
    },
    "synchronize": True,
    "paging_time": [180, 360],
    "debug": False,
    "sync_waiting": 0
}
MAP_CONF_BASE = {
    'map_size': 2000,
    'nr_stations': 5,
    'signal_strengths': [10]
}


OUTPUT_IDLE = "scratch/outputs/idle_output"
CONF_IDLE = {
    "meas_setup": "basic1.json",
    "att_meas_setup": "att_basic.json",
    "nr_users": 1,
    "max_time": 30,
    "output_folder": OUTPUT_IDLE,
    "sib_filename": "sib.json",
    "ping": True,
    "paging_level": 0,
    "seed": 107,
    "max_processes": 1,
    "network_children": 0,
    "max_steps": 10000,
    'paging_check_time': 5,
    "verbosity": {
        "log_user": 0,
        "print_users_to_page": False,
        "attack_messages": True,
        "reveal_network_records": 0,
        "print_status": False
    },
    "user": {
        "behaviour2_time": 100,
    },
    "synchronize": True,
    "paging_time": [180, 360],
    "debug": False,
    "sync_waiting": 0
}
CONF_IDLE_BASE = CONF_IDLE | {
    'output_folder': 'scratch/outputs/idle_base_output'}
MAP_CONF_IDLE = {
    'map_size': 2000,
    'nr_stations': 5,
    'signal_strengths': itools.repeat(20)
}

MAP_CONF_HATA = {
    'map_size': 2000,
    'nr_stations': 5,
    'signal_strengths': itools.repeat(20),
    'distance_mode': 'hata',
    'mode': 'empty'
}


OUTPUT_CONNECTED = "scratch/outputs/connected_output"
OUTPUT_REJECTING = "scratch/outputs/connected_rejecting"
OUTPUT_REJECTING_HATA = "scratch/outputs/connected_rejecting_hata"
OUTPUT_CONNECTED_BASE = "scratch/outputs/connected_base_output"
OUTPUT_SIGSTRENGTH = "scratch/outputs/connected_sigstrength"
OUTPUT_DETECTION = "scratch/outputs/detection"

CONF_ACTIVE = {
    "meas_setup": "basic1.json",
    "att_meas_setup": "att_basic.json",
    "nr_users": 1,
    "max_time": 120,
    "output_folder": OUTPUT_CONNECTED,
    "sib_filename": "sib.json",
    "ping": True,
    "paging_level": 0.2,
    "seed": 107,
    "network_children": 0,
    "max_steps": 100000,
    'paging_check_time': 5,
    "verbosity": {
        "log_user": 0,
        "print_users_to_page": False,
        "attack_messages": True,
        "reveal_network_records": 0,
        "print_status": False
    },
    "user": {
        "behaviour2_time": 50,
    },
    "synchronize": True,
    "paging_time": [360, 720],
    "debug": False,
    "sync_waiting": 0
}
CONF_SHORT = CONF_ACTIVE | {
    'max_steps': 2000,
    'max_time':3600,
}
CONF_ACTIVE_BASE = {
    "meas_setup": "basic1.json",
    "att_meas_setup": "att_basic.json",
    "nr_users": 1,
    "max_time": 120,
    "output_folder": OUTPUT_CONNECTED,
    "sib_filename": "sib.json",
    "ping": True,
    "paging_level": 0.2,
    "seed": 107,
    "max_processes": 1,
    "network_children": 0,
    "max_steps": 100000,
    'paging_check_time': 5,
    "verbosity": {
        "log_user": 0,
        "print_users_to_page": False,
        "attack_messages": True,
        "reveal_network_records": 0,
        "print_status": False
    },
    "user": {
        "behaviour2_time": 50,
    },
    "synchronize": True,
    "paging_time": [360, 720],
    "debug": False,
    "sync_waiting": 0
}
MAP_CONF_ACTIVE = {
    'map_size': 2000,
    'nr_stations': 5
}
MAP_RANDOM = {
    'map_size': 2000,
    'nr_stations': 5,
    'method': 'random'
}

# %%
STDOUT = sys.stdout


@dataclass
class ExperimentRecipe():
    base_conf: dict
    map_conf: dict
    frequencies: Sequence[FrequencyConfigRange]
    conf_variations: dict = field(default_factory=lambda: {'': [0]})
    q_hyst_range: Sequence[int] = field(default_factory=lambda: [0])
    n_iter: int = 1
    seed: int = 0
    add_attacker_after: int = 0
    meas_config = None
    prefix: str = 'out'
    attacker_guess_reps: int = 1
    nr_processes: int = 0
    user_path: list = None

    def run(self,redo_map=False):
        # pylint: disable=unexpected-keyword-arg
        util.listen()
        for f in os.listdir('stacktraces'):
            os.remove(os.path.join('stacktraces',f))
        total_experiments = experiment_main(
            **self.__dict__|{'map_conf':self.map_conf|{'redo_map':redo_map}}, total=self.compute_total_iterations())
        sys.stdout = STDOUT
        print('Finished running', total_experiments, 'experiments!')
        print(self.base_conf['output_folder'])
    
    def iterator(self):
        return _folder_iterator(self.base_conf['output_folder'], f'{self.prefix}.*')

    def compute_stats(self):
        if self.nr_processes > 0:
            return compute_stats_mp(self.nr_processes)(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())
        return compute_stats(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())

    def compute_ratios(self):
        if self.nr_processes > 0:
            return compute_ratios_mp(self.nr_processes)(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())
        return compute_ratios(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())

    def compute_location_guess(self):
        if self.nr_processes > 0:
            return compute_location_guess_mp(self.nr_processes)(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())
        return compute_location_guess(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())

    def compute_signal_data(self):
        if self.nr_processes > 0:
            return compute_signal_data_mp(self.nr_processes)(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())
        return compute_signal_data(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())

    def plot_detection(self):
        return _plot_detection(self.base_conf['output_folder'], f'{self.prefix}.*')

    def id_detection_data(self):
        return id_detection_data(self.base_conf['output_folder'], f'{self.prefix}.*', total=self.compute_total_iterations())

    def compute_total_iterations(self):
        total = 1
        for _, item in self.conf_variations.items():
            if isinstance(item, Sequence):
                total *= len(item)
        for f in self.frequencies:
            for _, item in f.__dict__.items():
                if isinstance(item, Sequence):
                    total *= len(item)
        if self.conf_variations.get('attack',[{}])[0].get('position') == 'all':
            stations = json.load(open(os.path.join(self.map_conf['raytrace_folder'],self.map_conf['raytrace_filebase']+'_stat.json')))
            nr_attackers = sum(map(lambda s: s['eci'] == 100,stations))
            total *= nr_attackers
        total *= len(self.q_hyst_range)
        total *= self.n_iter
        return total


def _plot_detection(folder, pattern):
    results = []
    for abs_folder in _folder_iterator(folder, pattern):
        results.append(
            list(map(fst, plot_detection_results(abs_folder, plot=False))))
    results = list(zip(*results))
    print(results)
    avg_results = [(res[0][0], mean((x[2] for x in res))) for res in results]
    print(avg_results)
    plt.plot(*zip(*avg_results), 'b')


def _id_detection_data_inner(abs_folder):
    detection_time = ExperimentSetup.id_detector_output(abs_folder)
    with open(os.path.join(abs_folder, 'conf'), 'r') as f:
        data = json.load(f)
        data['id_detection_time'] = detection_time
    with open(os.path.join(abs_folder, 'conf'), 'w') as f:
        json.dump(data, fp=f, indent=2)


def _compute_ratios_inner(abs_folder):
    ratios = ExperimentSetup.calculate_ratios(abs_folder)
    with open(os.path.join(abs_folder, 'conf'), 'r') as f:
        data = json.load(f)
        data['ratios'] = ratios
    with open(os.path.join(abs_folder, 'conf'), 'w') as f:
        json.dump(data, fp=f, indent=2)


def _compute_stats_inner(abs_folder: str):
    try:
        res = ExperimentSetup.user_attack_ratio(abs_folder)
        with open(os.path.join(abs_folder, 'conf'), 'r') as f:
            data = json.load(f)
            data['stats'] = res
        with open(os.path.join(abs_folder, 'conf'), 'w') as f:
            json.dump(data, fp=f, indent=2)
    except FileNotFoundError:
        print('Could not find file in', abs_folder, file=sys.__stdout__)


def _compute_location_guess_inner(abs_folder):
    res = ExperimentSetup.guess_attacker_location(abs_folder, reps=100)
    with open(os.path.join(abs_folder, 'conf'), 'r') as f:
        data = json.load(f)
        data['attacker_guess'] = res
    with open(os.path.join(abs_folder, 'conf'), 'w') as f:
        json.dump(data, fp=f, indent=2)


def _compute_signal_data_inner(abs_folder):
    res = ExperimentSetup.calculate_signal_data(abs_folder)
    with open(os.path.join(abs_folder, 'conf'), 'r') as f:
        data = json.load(f)
        data['signal'] = res
    with open(os.path.join(abs_folder, 'conf'), 'w') as f:
        json.dump(data, fp=f, indent=2)


def compute_variant(func):
    def compute_inner(folder: str, pattern: str, total):
        for abs_folder in tqdm(_folder_iterator(folder, pattern), total=total):
            func(abs_folder)
    return compute_inner


def compute_variant_mp(func):
    def take_nr_processes(nr_processes=1):
        def compute_inner(folder: str, pattern: str, total):
            if nr_processes:
                with mp.Pool(nr_processes) as pool:
                    return list(tqdm(pool.imap(func, _folder_iterator(folder, pattern)), total=total))
        return compute_inner
    return take_nr_processes


compute_stats = compute_variant(_compute_stats_inner)
compute_stats_mp = compute_variant_mp(_compute_stats_inner)
compute_ratios = compute_variant(_compute_ratios_inner)
compute_ratios_mp = compute_variant_mp(_compute_ratios_inner)
compute_location_guess = compute_variant(_compute_location_guess_inner)
compute_location_guess_mp = compute_variant_mp(_compute_location_guess_inner)
compute_signal_data = compute_variant(_compute_signal_data_inner)
compute_signal_data_mp = compute_variant_mp(_compute_signal_data_inner)
id_detection_data = compute_variant(_id_detection_data_inner)


def _folder_iterator(folder, pattern):
    pat = re.compile(pattern)
    for subfolder in os.listdir(folder):
        abs_folder = os.path.join(folder, subfolder)
        if os.path.isdir(abs_folder) and pat.match(subfolder):
            yield abs_folder

# %%

################################################################
# Testing recipes
################################################################


idle_baseline_test = ExperimentRecipe(
    base_conf=CONF_IDLE_BASE,
    map_conf=MAP_CONF_IDLE,
    frequencies=freqs_constant,
)
conn_baseline_test = ExperimentRecipe(
    base_conf=CONF_ACTIVE_BASE,
    map_conf=MAP_CONF_ACTIVE,
    frequencies=freqs_constant,
)
idle_test = ExperimentRecipe(
    base_conf=CONF_IDLE,
    map_conf=MAP_CONF_IDLE,
    frequencies=freqs_constant,
)
conn_test = ExperimentRecipe(
    base_conf=CONF_ACTIVE,
    map_conf=MAP_CONF_ACTIVE,
    frequencies=freqs_constant,
)


freqs_test = [FrequencyConfigRange([1000, 2000, 3000],
                                   prevalence=1,
                                   cell_reselection_priority_intra=5,
                                   q_rx_lev_min_intra=-90),
              FrequencyConfigRange(2000,
                                   prevalence=0.0001,
                                   cell_reselection_priority_intra=7,
                                   q_rx_lev_min_intra=-90),
              ]
test_1 = ExperimentRecipe(
    base_conf=CONF_IDLE | {'output_folder': OUTPUT_TEST, 'attack': {
        'frequency': 2000, 'signal_strength': 10}, 'debug': True},
    map_conf=MAP_CONF_IDLE,
    frequencies=freqs_test,
    q_hyst_range=[0],
    prefix='prio',
    conf_variations={'seed': [0], 'rejecting': [1, 0]}
)

################################################################
# Recipes
################################################################

freqs = [FrequencyConfigRange(1000,
                              prevalence=0.5,
                              cell_reselection_priority_intra=5,
                              q_rx_lev_min_intra=-80),
         FrequencyConfigRange(2000,
                              prevalence=0.5,
                              cell_reselection_priority_intra=[4, 6],
                              q_rx_lev_min_intra=-80)]

idle_baseline_frequencies_A = ExperimentRecipe(
    base_conf=CONF_IDLE_BASE,
    conf_variations={'seed': list(range(10, 20))},
    map_conf=MAP_CONF_IDLE,
    frequencies=freqs,
    q_hyst_range=[3]
)


idle_test = ExperimentRecipe(
    base_conf=CONF_IDLE | {
        'output_folder': 'scratch/outputs/test_output', 'attack': True},
    map_conf=MAP_CONF_IDLE,
    conf_variations={'seed': list(range(10))},
    frequencies=freqs,
    prefix='idle_qhyst',
    q_hyst_range=range(10)
)

conn_base = ExperimentRecipe(
    base_conf=CONF_ACTIVE_BASE | {
        'output_folder': 'scratch/outputs/test_output', 'attack': False},
    map_conf=MAP_CONF_IDLE,
    frequencies=freqs_constant,
    conf_variations={'paging_level': [r*0.1 for r in range(1, 10)]},
    prefix='y',
    q_hyst_range=[5],
)
conn_A = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {'attack': True,
                             'output_folder': 'scratch/outputs/test_output'},
    map_conf=MAP_CONF_IDLE,
    conf_variations={'map_seed': list(range(20))},
    prefix='x',
    frequencies=freqs,
)
freqs_B = [FrequencyConfigRange(1000,
                                prevalence=0.5,
                                cell_reselection_priority_intra=3,
                                q_rx_lev_min_intra=-80),
           FrequencyConfigRange(2000,
                                prevalence=0.5,
                                cell_reselection_priority_intra=5,
                                q_rx_lev_min_intra=-80),
           FrequencyConfigRange(1500,
                                prevalence=0,
                                cell_reselection_priority_intra=list(
                                    range(2, 7)),
                                q_rx_lev_min_intra=-80)]
freqs_C = [FrequencyConfigRange(1000,
                                prevalence=0.5,
                                cell_reselection_priority_intra=3,
                                q_rx_lev_min_intra=-80),
           FrequencyConfigRange(2000,
                                prevalence=0.5,
                                cell_reselection_priority_intra=list(
                                    range(2, 7)),
                                q_rx_lev_min_intra=-80)]
freqs_D = [FrequencyConfigRange(1000,
                                prevalence=0.5,
                                cell_reselection_priority_intra=3,
                                q_rx_lev_min_intra=-80),
           FrequencyConfigRange(2000,
                                prevalence=0.5,
                                cell_reselection_priority_intra=5,
                                q_rx_lev_min_intra=-80)]
conn_B = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {'output_folder': OUTPUT_CONNECTED},
    map_conf=MAP_CONF_IDLE,
    conf_variations={'map_seed': list(range(5)), 'seed': list(range(5)), 'attack': [
        {'frequency': 1500, 'signal_strength': i} for i in range(-10, 30, 2)]},
    prefix='priority',
    frequencies=freqs_B,
)
conn_BR = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {
        'output_folder': OUTPUT_REJECTING, 'rejecting': True},
    map_conf=MAP_CONF_IDLE,
    conf_variations={'map_seed': list(range(5)), 'seed': list(range(5)), 'attack': [
        {'frequency': 1500, 'signal_strength': i} for i in range(-10, 30, 2)]},
    prefix='priority',
    frequencies=freqs_B,
    nr_processes=20
)
conn_M = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {'attack': {
        'frequency': 1000, 'signal_strength': 20}, 'output_folder': OUTPUT_TEST, 'max_steps': 3000},
    map_conf=MAP_CONF_IDLE,
    q_hyst_range=list(range(2)),
    conf_variations={'seed': list(range(2))},
    prefix='multi',
    frequencies=freqs,
    nr_processes=10

)
conn_CR = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {
        'output_folder': OUTPUT_REJECTING, 'rejecting': True},
    map_conf=MAP_CONF_IDLE,
    q_hyst_range=list(range(2, 10)),
    conf_variations={'map_seed': list(range(5)), 'seed': list(range(5)), 'attack': [
        {'frequency': f, 'signal_strength': i} for i in list(range(-10, 30, 2)) for f in [1000, 2000]]},
    prefix='samefreq',
    frequencies=freqs_D,
    nr_processes=20
)
conn_test = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {'output_folder': OUTPUT_TEST, 'rejecting': True,
                             'max_steps': 10000, 'attack': {'frequency': 2000, 'signal_strength': 40}},
    map_conf=MAP_CONF_HATA,
    conf_variations={'': ['']},
    prefix='hata',
    frequencies=freqs_D,
)

conn_base = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {'output_folder': OUTPUT_CONNECTED_BASE},
    map_conf=MAP_CONF_IDLE,
    q_hyst_range=list(range(1, 10)),
    conf_variations={'map_seed': list(range(10)), 'seed': list(range(10))},
    prefix='samefreq',
    frequencies=freqs_B,
    nr_processes=20
)
conn_BRH = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {
        'output_folder': OUTPUT_REJECTING_HATA, 'rejecting': True},
    map_conf=MAP_CONF_HATA,
    conf_variations={'map_seed': list(range(5)), 'seed': list(range(5)), 'attack': [
        {'frequency': 1500, 'signal_strength': i} for i in range(-10, 30, 2)]},
    prefix='priority',
    frequencies=freqs_B,
    nr_processes=20
)
conn_CRH = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {
        'output_folder': OUTPUT_REJECTING_HATA, 'rejecting': True},
    map_conf=MAP_CONF_HATA,
    q_hyst_range=list(range(2, 10)),
    conf_variations={'map_seed': list(range(5)), 'seed': list(range(5)), 'attack': [
        {'frequency': f, 'signal_strength': i} for i in list(range(-10, 30, 2)) for f in [1000, 2000]]},
    prefix='samefreq',
    frequencies=freqs_D,
    nr_processes=20
)
conn_CR = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {
        'output_folder': OUTPUT_REJECTING, 'rejecting': True},
    map_conf=MAP_CONF_IDLE,
    q_hyst_range=list(range(2, 10)),
    conf_variations={'map_seed': list(range(5)), 'seed': list(range(5)), 'attack': [
        {'frequency': f, 'signal_strength': i} for i in list(range(-10, 30, 2)) for f in [1000, 2000]]},
    prefix='samefreq',
    frequencies=freqs_D,
    nr_processes=20
)

nr_users = 200
freqs_single = [FrequencyConfigRange(1000,
                                     prevalence=1,
                                     cell_reselection_priority_intra=5,
                                     q_rx_lev_min_intra=-90),
                ]
freqs_duo = [FrequencyConfigRange(1000,
                                  prevalence=0.5,
                                  cell_reselection_priority_intra=5,
                                  q_rx_lev_min_intra=-90,
                                  # max_users=10
                                  ),
             FrequencyConfigRange(2000,
                                  prevalence=0.5,
                                  cell_reselection_priority_intra=6,
                                  q_rx_lev_min_intra=-90
                                  ),
             ]

conn_strength = ExperimentRecipe(
    base_conf=CONF_ACTIVE | {
        'output_folder': OUTPUT_REJECTING, 'rejecting': True},
    map_conf=MAP_CONF_IDLE,
    q_hyst_range=[5],
    conf_variations={'map_seed': list(range(10)), 'seed': list(range(10)), 'attack': [
        {'frequency': 1000, 'signal_strength': i} for i in list(range(-10, 40))]},
    prefix='sig',
    frequencies=freqs_single,
    nr_processes=20
)

CONF_DETECTION = {
    "meas_setup": "basic1.json",
    "att_meas_setup": "att_basic.json",
    "nr_users": 200,
    "max_time": 120,
    "output_folder": OUTPUT_DETECTION,
    "sib_filename": "sib.json",
    "ping": True,
    "paging_level": 0.2,
    "seed": 107,
    "max_processes": 10,
    "network_children": 0,
    'children_buffer_size': 20,
    "max_steps": 4800,
    'paging_check_time': 5,
    'rejecting': True,
    "verbosity": {
        "log_user": 0,
        "print_users_to_page": True,
        "attack_messages": True,
        "reveal_network_records": 0,
        "print_status": False
    },
    "user": {
        "behaviour2_time": 50,
    },
    "synchronize": True,
    "paging_time": [360, 720],
    "debug": False,
    "sync_waiting": 0,
    "cell_reselection_waiting_time": 5,
    'idle_check_time': 120,
    'auto_connect_time': 5,
    'detectors': [{
        'type': 'ta_update',
        'window_size': 100,
        'activate_after': 2000,
        'wait_time': 100,
        'threshold': 1,
        'filename': 'detect'
    }, {
        'type': 'id',
        'window_size': 100,
        'activate_after': 2000,
        'wait_time': 100,
        'threshold': 1,
        'filename': 'detect1'
    }],
    'attack': {
        'frequency': 1000,
        'signal_strength': 30
    },
}

MAP_CONF_DETECTION = {
    'map_size': 2000,
    'nr_stations': 5,
    'signal_strengths': itools.repeat(10),
    # 'max_users_cell':[nr_users/4]*4,
    'nr_tracking_areas': 2
}
MAP_CONF_BUSY = {
    'map_size': 2000,
    'nr_stations': 10,
    'distance_mode': 'hata',
    'signal_strengths': itools.repeat(10),
    'nr_tracking_areas': 2
}

MAP_CONF_SPARSE = {
    'map_size': 3000,
    'nr_stations': 5,
    'signal_strengths': itools.repeat(5),
    # 'max_users_cell':[nr_users/4]*4,
    'nr_tracking_areas': 2
}

MAP_CONF_NIJMEGEN = {
    'filename': 'map_configs/nijmegen_center_1.json',
    'signal_strengths': itools.repeat(10),
    'max_users_cell': [nr_users/4]*4,
    'nr_tracking_areas': 3
}
detection_test = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_SPARSE,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(20)), 'max_processes': [20], 'nr_users': [1000], 'loc_update_waiting_time': list(range(10, 121, 10))+[15, 25],
                     'detectors': [[{
                         'type': 'loc_update',
                         'window_size': 100,
                         'activate_after': 2000,
                         'wait_time': 100,
                         'threshold': 1,
                         'filename': 'loc_updates'
                     }]],
                     },
    nr_processes=5,
    prefix='loc_update_',
    frequencies=freqs_duo,
    add_attacker_after=2424
)
active_test = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_DETECTION,
    q_hyst_range=[5],
    conf_variations={'seed': [1], 'max_processes': [5], 'nr_users': [
        100], 'max_steps': [3600], 'tau_behavior': [{'attack': 'dummy', 'normal': True}]},
    prefix='test',
    frequencies=freqs_duo,
    add_attacker_after=2424
)
str_test = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_BUSY,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(1)), 'max_processes': [20], 'nr_users': [500], 'max_steps': [7200], 'tau_behavior': [{'attack': 'dummy', 'normal': True}],
                     'attack': [{'frequency': 1000, 'signal_strength': s} for s in range(0, 40, 3)],
                     'detectors': [[{
                         'type': 'meas_report',
                         'window_size': 100,
                         'activate_after': 2000,
                         'wait_time': 100,
                         'threshold': 1,
                         'filename': 'meas_reports'
                     }]]},
    prefix='final_str_test_',
    frequencies=freqs_duo,
    add_attacker_after=3600
)
loc_update_test = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_DETECTION,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(1)), 'max_processes': [10], 'nr_users': [500], 'max_steps': [3000], 'tau_behavior': [{'attack': 'dummy', 'normal': True}],
                     'loc_update_waiting_time': [10],
                     'detectors': [[{
                         'type': 'loc_update',
                         'window_size': 100,
                         'activate_after': 2000,
                         'wait_time': 100,
                         'threshold': 1,
                         'filename': 'loc_updates'
                     }]],
                     'rejecting': [True],
                     'attacker_keep': [True],
                     'attack': [{
                         'frequency': 1000,
                         'signal_strength': 30
                     }]},
    prefix='loc_test_s',
    frequencies=freqs_duo,
    add_attacker_after=1337
)

messenger_shoots_back_test = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_DETECTION,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(1)), 'max_processes': [10], 'nr_users': [50], 'max_steps': [3600],
                     'tau_behavior': [{'attack': [(13.6, 'dummy'), (86.4, False)],
                                       'normal':[(83.7, True), (16.3, False)]}]},
    prefix='msb_test_',
    frequencies=freqs_duo,
    add_attacker_after=5800
)
messenger_shoots_back_comparison = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_DETECTION,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(20)), 'max_processes': [20], 'nr_users': [500], 'max_steps': [4800],
                     'tau_behavior': [
        {'attack': [(t, 'dummy'), (100-t, False)],
         'normal':[(83.7, True), (16.3, False)]}
        for t in np.linspace(13.6, 100, 50)
    ]},
    prefix='msb_fully_',
    frequencies=freqs_duo,
    add_attacker_after=2400,
    nr_processes=15
)
messenger_shoots_back_comparison2 = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_DETECTION,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(20)), 'max_processes': [20], 'nr_users': [1000], 'max_steps': [7200], 'tau_behavior': [
        {'attack': [(60, 'dummy'), (40, False)], 'normal':[
            (83.7, True), (16.3, False)]},
        {'attack': [(70, 'dummy'), (30, False)], 'normal':[
            (83.7, True), (16.3, False)]},
        {'attack': [(80, 'dummy'), (20, False)], 'normal':[
            (83.7, True), (16.3, False)]},
        {'attack': [(90, 'dummy'), (10, False)], 'normal':[
            (83.7, True), (16.3, False)]},
        {'attack': [(100, 'dummy')], 'normal':[(83.7, True), (16.3, False)]},
    ]},
    prefix='msb_full_h',
    frequencies=freqs_duo,
    add_attacker_after=3600,
)


detection_1 = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_DETECTION,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(10)),
                     'nr_users': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800, 1000],
                     'detectors': [[
                         {
                             'type': 'full_id_dist',
                             'window_size': 100,
                             'activate_after': 2000,
                             'wait_time': 100,
                             'threshold': 1,
                             'filename': 'detect'
                         }, {
                             'type': 'registration',
                             'window_size': 100,
                             'activate_after': 2000,
                             'wait_time': 100,
                             'threshold': 1,
                             'filename': 'detect1'
                         },
                         {
                             'type': 'dist',
                             'window_size': 100,
                             'activate_after': 2000,
                             'wait_time': 100,
                             'threshold': 1,
                             'filename': 'detect2'
                         }, {
                             'type': 'id_dist',
                             'window_size': 100,
                             'activate_after': 2000,
                             'wait_time': 100,
                             'threshold': 1,
                             'filename': 'detect3'
                         }
                     ]]},
    prefix='detect1_',
    frequencies=freqs_duo,
    add_attacker_after=2424,
    nr_processes=10
)

MAP_CONF_PLOT = {
    'map_size': 2000,
    'nr_stations': 3,
    'signal_strengths': itools.repeat(10),
    # 'max_users_cell':[nr_users/4]*4,
    'nr_tracking_areas': 1
}
path_plot_exp = ExperimentRecipe(
    base_conf=CONF_DETECTION,
    map_conf=MAP_CONF_PLOT,
    q_hyst_range=[5],
    conf_variations={'': [0], 'max_steps': [100000],
                     'map_seed': [4], 'paging_level': [0]},
    prefix='path_',
    frequencies=freqs_single,
    user_path=[(1800, 1600), (1800, 1800), (200, 1800), (200, 1500), (1000,
                                                                      1500), (1000, 500), (200, 500), (200, 200), (1800, 200), (1800, 400)],
    add_attacker_after=1000000
)
#%%
RT_MAP_CONF = {
    'map_size': 1000,
    'nr_stations':3,
    'obstructions':20,
    'signal_strengths': [20]*25,
    'cache':True,
    'mode':'blocks',
    'method':'uniform',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_small',
    'attack': {
        'frequency': 1000,
        'signal_strength': 30
    }
}
RT_MAP_5K = {
    'map_size': 5000,
    'nr_stations':20,
    'obstructions':367,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'method':'uniform',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_5k',
    'attack': {
        'frequency': 1000,
        'signal_strength': 10
    }
}
RT_MAP_2K = {
    'map_size': 2000,
    'nr_stations':5,
    'obstructions':85,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'frequencies':{1000:1,2000:1},
    'method':'uniform',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_2k',
    'attack': {
        'signal_strength': 10
    }
}
RT_MAP_2K_SPARSE = {
    'map_size': 2000,
    'nr_stations':3,
    'obstructions':85,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'frequencies':{1000:1,2000:1},
    'obstruction_strength':'random.uniform(0.3,0.7)',
    'method':'uniform',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_2k_sparse',
    'attack': {
        'signal_strength': 10
    }
}
RT_MAP_2K_SPARSE_2 = {
    'map_size': 2000,
    'nr_stations':3,
    'obstructions':85,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'frequencies':{1000:1,2000:1},
    'obstruction_strength':'random.uniform(0.3,0.7)',
    'method':'uniform',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_2k_sparse_2',
    'attack': {
        'signal_strength': 10
    }
}
RT_MAP_2K_1500ATT = {
    'map_size': 2000,
    'nr_stations':3,
    'obstructions':85,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'frequencies':{1000:1,2000:1},
    'obstruction_strength':'random.uniform(0.3,0.7)',
    'method':'uniform',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_2k_1500att',
    'attack': {
        'signal_strength': 10,
        'frequency':1500
    }
}
RT_MAP_2K_SPARSE_ATTACKER = {
    'map_size': 2000,
    'nr_stations':3,
    'obstructions':85,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'frequencies':{1000:1,2000:1},
    'method':'uniform',
    'obstruction_strength':'random.uniform(0.3,0.7)',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_2k_sparse_test',
    'attack': {
        'signal_strength': 10,
        'all_positions':True
    }
}
RT_MAP_2K_ATTACKER = {
    'map_size': 2000,
    'nr_stations':3,
    'obstructions':85,
    'signal_strengths': [30]*21,
    'cache':True,
    'mode':'blocks',
    'frequencies':{1000:1,2000:1},
    'method':'uniform',
    'obstruction_strength':'random.uniform(0.3,0.7)',
    'raytrace_filebase':'rt',
    'raytrace_folder':'scratch/maps/rt_2k_attacker',
    'attack': {
        'signal_strength': 10,
        'positions':1000
    }
}

rt_test = ExperimentRecipe(
    base_conf = CONF_DETECTION|{'nr_users':10,'max_steps':5000,'spoof_eci':True,'verbosity':{'log_user':7},'debug':True,'spoof_eci':True,'output_folder':'scratch/outputs/test_output','nr_tracking_areas':2},
    map_conf = RT_MAP_2K_SPARSE_2,
    q_hyst_range=[5],
    conf_variations={'': [0], 'paging_level': [0.2]},
    prefix='test_',
    frequencies=freqs_duo,
    add_attacker_after=1000
)
#%%
def plot_frequency(cmap,frequency,colorbar=False,color_map='hot'):
    to_plot = []
    for k,i in cmap.encoding.items():
        if k[1] == frequency:
            to_plot.append(i)
    arr = np.max(cmap.strength_map[:,:,to_plot],axis=2)
    plt.imshow(arr,cmap = color_map,origin='lower')
    for ob in cmap.obstructions:
        plt.gca().add_patch(ob.to_rectangle(facecolor='silver'))
    if colorbar:
        plt.colorbar()
# %%
RT_REJECTING = 'scratch/outputs/rt_rejecting'
RT_DETECTION = 'scratch/outputs/rt_detection'

rel_str_rt = ExperimentRecipe(
    base_conf=CONF_SHORT | {
        'output_folder': RT_REJECTING, 'rejecting': True,'nr_users':500,'nr_processes':10},
    map_conf=RT_MAP_2K_SPARSE,
    q_hyst_range=list(range(2, 10)),
    conf_variations={'attack': [
        {'frequency': f, 'signal_strength': i} for i in list(range(-10, 30, 2)) for f in [1000, 2000]]},
    prefix='att_str_',
    frequencies=freqs_D,
    nr_processes=5
)
prio_rt = ExperimentRecipe(
    base_conf=CONF_SHORT | {
        'output_folder': RT_REJECTING, 'rejecting': True,'nr_users':500,'nr_processes':10},
    map_conf=RT_MAP_2K_1500ATT,
    conf_variations={'attack': [
        {'frequency': 1500, 'signal_strength': i} for i in [-10,0,10,20,30]]},
    prefix='priority_x_',
    frequencies=freqs_B,
    nr_processes=5
)
rel_pos_rt = ExperimentRecipe(
    base_conf=CONF_SHORT | {
        'output_folder': RT_REJECTING, 'rejecting': True,'nr_users':500,'nr_processes':10},
    map_conf=RT_MAP_2K_ATTACKER,
    q_hyst_range=[5],
    conf_variations={'seed': [0], 'attack': [
        {'frequency': f, 'signal_strength': 10,'position':'all'} for f in [1000, 2000]]},
    prefix='rel_pos_',
    frequencies=freqs_D,
    nr_processes=5
)
msb_rt = ExperimentRecipe(
    base_conf=CONF_DETECTION|{'output_folder':RT_DETECTION},
    map_conf=RT_MAP_2K_SPARSE|{'nr_tracking_areas':2},
    q_hyst_range=[5],
    conf_variations={'seed': list(range(5)), 'nr_processes': [10], 'nr_users': [500], 'max_steps': [4800],
                     'tau_behavior': [
        {'attack': [(t, 'dummy'), (100-t, False)],
         'normal':[(83.7, True), (16.3, False)]}
        for t in np.linspace(13.6, 100, 50)
    ],'attack':[{'signal_strength':s, 'frequency':f} for s in [10,20] for f in [1000,2000]]},
    prefix='msb_',
    frequencies=freqs_duo,
    add_attacker_after=2400,
    nr_processes=5
)
msb_rt_test = ExperimentRecipe(
    base_conf=CONF_DETECTION|{'output_folder':RT_DETECTION},
    map_conf=RT_MAP_2K_SPARSE|{'nr_tracking_areas':2},
    q_hyst_range=[5],
    conf_variations={'seed': list(range(1)), 'nr_processes': [10], 'nr_users': [500], 'max_steps': [4800],
                     'tau_behavior': [
        {'attack': [(t, 'dummy'), (100-t, False)],
         'normal':[(83.7, True), (16.3, False)]}
        for t in np.linspace(13.6, 100, 2)
    ],'attack':[{'signal_strength':s, 'frequency':f} for s in [20] for f in [2000]]},
    prefix='msb_test_',
    frequencies=freqs_duo,
    add_attacker_after=2400,
    nr_processes=1
)
detection_test_rt = ExperimentRecipe(
    base_conf=CONF_DETECTION|{'output_folder':RT_REJECTING},
    map_conf=RT_MAP_2K_SPARSE|{'nr_tracking_areas':2},
    q_hyst_range=[5],
    conf_variations={'seed': list(range(20)), 'max_processes': [20], 'nr_users': [1000], 'loc_update_waiting_time': list(range(10, 121, 10))+[15, 25],
                     'detectors': [[{
                         'type': 'loc_update',
                         'window_size': 100,
                         'activate_after': 2000,
                         'wait_time': 100,
                         'threshold': 1,
                         'filename': 'loc_updates'
                     }]],
                     },
    nr_processes=5,
    prefix='loc_update_',
    frequencies=freqs_duo,
    add_attacker_after=2424
)
str_test_rt = ExperimentRecipe(
    base_conf=CONF_DETECTION|{'output_folder':RT_REJECTING,'spoof_eci':False},
    map_conf=RT_MAP_2K_SPARSE,
    q_hyst_range=[5],
    conf_variations={'seed': list(range(1)), 'max_processes': [20], 'nr_users': [500], 'max_steps': [7200], 'tau_behavior': [{'attack': 'dummy', 'normal': True}],
                     'attack': [{'frequency': 1000, 'signal_strength': s} for s in range(-20, 30, 2)],
                     'detectors': [[{
                         'type': 'meas_report',
                         'window_size': 100,
                         'activate_after': 2000,
                         'wait_time': 100,
                         'threshold': 1,
                         'filename': 'meas_reports'
                     }]]},
    prefix='final_str_test_',
    frequencies=freqs_duo,
    add_attacker_after=3600
)
#%%
ATTACKER_TAC = 99
DIST_FACTOR = 0.02386
def guess_user_location(folder='tmp_output', reps=1, sample_size=None):
    """Guesses the user location using gradient descent on the recorded measurement reports

    """
    u = UserHistory.from_file('userlog0',folder)
    print(u.loc_data[0])
    sig_data = u.sig_data
    def find_sig(sig):
        i, r = find_on(sig, lambda y: y[1] == ATTACKER_ECI)
        if i>=0:
            return -r[2]
    with open(os.path.join(folder, 'map_data')) as f:
        map_data = json.load(f)
    i, att_rec = util.find_on(map_data.get('stations', ()),
                            lambda rec: rec['tac'] == ATTACKER_TAC, default={})
    if i < 0:
        raise MeasurementError('No attacker found')
    frequency = att_rec.get('frequencies', [])[0]
    sigs = {rec[1]:rec[3] for rec in filter(lambda rec: rec[2] == 1000, sig_data[0][1])}
    meas = [(util.find_on(cmap.stations,lambda s: s.identifier == identifier)[1].position, sig) for identifier,sig in sigs.items()]
    meas = [((p.x,p.y),s) for p,s in meas]
    print(meas)
    bounds = map_data['size']
    def cost_function(pos, signal_strength=0, freq=frequency):
        return cost_free(pos, meas, DIST_FACTOR, signal_strength)

    def jac(pos, signal_strength=0, freq=frequency):
        x,y= grad_free(pos, meas, DIST_FACTOR, signal_strength)
        return (x,y)
        if pos[0] < 0:
            x =1
        elif pos[0] > bounds:
            x = -1
        if pos[1] < 0:
            y =1
        elif pos[1] > bounds:
            y = -1
        return (x,y)
        
    res = []
    for signal_strength in range(-20, 35, 5):
        guess = minimize(cost_function, np.array(
            [0, 0]), jac=jac, args=(signal_strength, frequency)).x
        res.append(
            ((*guess,), len(meas), Point(*guess).distance(Point(att_rec.get('x', 0), att_rec.get('y', 0)))))
    return res,cost_function,jac
#%%
def some_plotting():
    data_set = [{} for i in range(100)]
    data_set_spoofed = [{} for i in range(100)]
    for j in range(100):
        u = UserHistory.from_file(f'scratch/outputs/test_output/spoof_0/userlog{j}')

        for i in list(range(7))+[100]:
            data_set[j][i] = [rec[-1] for row in u.sig_data for rec in row[1] if rec[1] == i and rec[2] ==1000]
            data_set_spoofed[j][i] = [max([rec[-1] for rec in row[1] if rec[0] == i and rec[2] ==1000],default=np.nan) for row in u.sig_data ]
    for i in list(range(7))+[100]:
        sns.kdeplot(data_set[0][i],clip=(-140,-50),label=i)
    plt.legend()
    plt.show()
    for i in list(range(7))+[100]:
        sns.kdeplot(data_set_spoofed[0][i],clip=(-140,-50),label=i)
    plt.legend()
    #%%
    tot = np.zeros(5)
    d = []
    indexes = [0,1,3,4,5]
    for j in range(100):
        print(j)
        if j % 5 ==0:
            corr = np.corrcoef([data_set_spoofed[j][i] for i in indexes])
        else:
            corr = np.corrcoef([data_set[j][i] for i in indexes])
        plt.plot(corr[1,:],label=j)
        tot += corr[1,:]
        d.append(corr[1,:])
    #plt.legend()
    tot /= 100

    # %%