import json
import math
import multiprocessing as mp
import os
import os.path
import random
import shutil
import json
import sys
import time
from copy import deepcopy
from queue import Empty as QueueEmpty
from typing import Sequence
import traceback

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import itertools as itools
import multiprocessing as mp
from pandas.core.indexes.base import Index
from scipy.optimize import minimize
from shapely.geometry import Point
from dataclasses import dataclass, field
from typing import Any
from tqdm import tqdm

import Model.util as util
from generate_configs import FrequencyConfig, generate_sib_file, FrequencyConfigRange
from Model.baseStation import BaseStation, Attacker, ATTACKER_ECI, ATTACKER_TAC
from Model.connection_map import (ConnectionMap, NonCachedConnectionMap,
                                  create_map)
from Model.cppspeedup import cost_free, grad_free, dist
from Model.epc import AttackerTrackingArea, TrackingArea, TrackingAreaChild, ATTACKER_HANG, ATTACKER_KEEP, ATTACKER_REJECT
from Model.measurementReport import ReportLogger
from Model.user import UserHistory
from Model.usercluster import ClusterUser, UserCluster, PredictableUser
from Model.util import DefaultObject, EmptyBarrier, find_on, flatten, void, arbitrary_keywords, vary_config, ilen, tget, find_last, NestablePool, error_logging, detection_logging, info_logging
from Model.active_detection import detectors, Detector, IdDetector, plot_detection_results, ANOMALY, DetectionNotFound

"""
This file contains the experiment setup, and executes what is configured in the Experiment class.
"""
DIST_FACTOR = 0.041916900439033636


class ExperimentDone(Exception):
    pass


class MeasurementError(Exception):
    pass


MAP_CONFIG = {
    "map_size": 1000,
    "nr_stations": 3,
    "subdivision": 50,
    "mode": "empty",
    "plmn_nr": 1,
    "distance_mode": 'empty',
    "method": "random",
    "frequencies": {"1000": 1, "2000": 0.5},
    "max_plmn_distance": 0,
    "noise": 1,
    "cache": False,
    "seed": 0,
    "signal_strengths": [0]
}


def print_exp(file_handle=sys.stdout):
    def _print_exp(*args, **kwargs):
        print("[EXPERIMENT]:", *args, file=file_handle, **kwargs)
    return _print_exp


class ExperimentSetup():
    cmap: NonCachedConnectionMap
    ta: TrackingArea
    att_ta: AttackerTrackingArea
    attacker: Attacker = DefaultObject()
    user_connect_event: mp.Event
    frequencies: Sequence[FrequencyConfig]
    output: print_exp()
    report_logger = DefaultObject()
    detectors = []
    output_handle = open('/dev/null', 'w')
    stdout_handle = open('/dev/null', 'w')
    error_handle = open('/dev/null', 'w')
    is_output_stdout: bool = True
    max_users_frequencies: dict = {}

    def __init__(self, conf, map_conf=MAP_CONFIG, frequencies=[], q_hyst=0) -> None:
        self.conf = conf
        self.seed = conf['seed']
        base_map_conf = deepcopy(MAP_CONFIG)
        self.map_conf = {**base_map_conf, **map_conf}
        if frequencies:
            self.set_frequencies(frequencies, q_hyst)

    def set_frequencies(self, frequencies, q_hyst=0):
        self.conf['sib_filename'] = generate_sib_file(frequencies, q_hyst)
        self.frequencies = frequencies

    def set_seed(self, seed):
        self.seed = seed
        self.conf['seed'] = seed

    def add_attacker_after(self, nr_steps):
        if self.map_conf['cache']:
            def add_attacker(cluster_self):
                cluster_self.cmap.disabled_stations = []
                error_logging('Added attacker!')
        else:
            def add_attacker(cluster_self):
                cluster_self.cmap.stations.append(self.attacker)
                error_logging('Added attacker!')
        for cluster in self.clusters:
            cluster.do_after_step(nr_steps, add_attacker)

    def remove_attacker(self):
        i, _ = find_on(self.cmap.stations,
                       lambda s: s.eci == self.attacker.eci)
        if i >= 0:
            self.cmap.stations.pop(i)

    def simple_map(self, attack=None):
        self.map_conf['seed'] = self.conf.get('map_seed', self.conf['seed'])
        if self.frequencies:
            self.map_conf['frequencies'] = {
                f.freq: f.prevalence for f in self.frequencies}
            self.max_users_frequencies = {
                f.freq: f.max_users for f in self.frequencies
            }
        self.cmap = create_map(self.map_conf, save=False, attack=attack)

    def simple_attacker(self, signal_strength: int = 10, add=True, frequency=1000):
        if self.map_conf['cache']:
            i, attacker = find_on(self.cmap.stations, lambda x: isinstance(
                x, Attacker), default=(-1, None))
            if i < 0:
                error_logging('No attacker found in cached map!')
                return
            self.attacker = attacker
            if add:
                self.cmap.disabled_stations = []
            else:
                print("[EXPERIMENT] disabling station", i)
                self.cmap.disabled_stations = [i]
        else:
            pos = Point(*(self.map_conf['map_size']/2,)*2)
            self.attacker = Attacker(ATTACKER_ECI, tac=ATTACKER_TAC, cmap=self.cmap,
                                     position=pos, signal_strength=signal_strength, frequencies=[frequency])
            if add:
                if not any(isinstance(s, Attacker) for s in self.cmap.stations):
                    self.cmap.stations.append(self.attacker)
                else:
                    self.cmap.stations[-1] = self.attacker

    def setup_context(self):
        self.manager = mp.Manager()
        self.events = {
            'update': mp.Event(),
            'done': mp.Event(),
            'barrier': mp.Barrier(1+self.conf['nr_processes'] + self.conf['network_children']+bool(self.attacker)) if self.conf.get('synchronize', False) else EmptyBarrier(),
            'attacker_event': mp.Event(),
            'baseline_done': mp.Event() if self.conf.get('detectors') else DefaultObject(),
            'detector_bar': mp.Barrier(1+len(self.conf.get('detectors', ()))) if self.conf.get('detectors', False) else EmptyBarrier()
        }
        self.events['update'].set()
        self.events['done'].clear()
        self.report_queue = self.manager.Queue()
        self.attack_queue = self.manager.Queue()
        self.error_queue = self.manager.Queue()
        self.attacker_queue = self.manager.Queue()

        self.downlink_queues = [self.manager.Queue()
                                for _ in range(self.conf['nr_users'])]
        self.uplink_queue = self.manager.Queue()
        self.report_logger = ReportLogger(self.report_queue,
                                          self.error_queue,
                                          self.attack_queue,
                                          event_done=self.events['done'],
                                          filename_rep="reportlog",
                                          filename_err='errlog',
                                          filename_att='attlog',
                                          folder=self.conf['output_folder'])
        self.detectors = []
        if (confs := self.conf.get('detectors')):
            for conf in confs:
                self.detectors.append(create_detector(
                    conf, events=self.events, queue=self.manager.Queue(), filename=conf.get('filename')))

    def simple_ta(self, meas_setup):
        if not meas_setup:
            with open(os.path.join('meas_configs', 'basic1.json'), 'r') as f:
                meas_setup = json.load(f)

        ta_detectors = [(detector.wants, detector.uplink_queue)
                        for detector in self.detectors]

        if children := self.conf.get('network_children'):
            ta_userlist = self.manager.list(
                [None for _ in range(self.conf['nr_users'])])
            paged_users = self.manager.list([])
            freq_users = self.manager.dict(
                {f.freq: 0 for f in self.frequencies})
        else:
            ta_userlist = None
            paged_users = None
            freq_users = None
        info_logging('Creating TA with max users per frequency',
                      self.max_users_frequencies)
        self.ta = TrackingArea(tac=0,
                               config=self.conf,
                               stations=[
                                   s for s in self.cmap.stations if not isinstance(s, Attacker)],
                               user_ids=range(self.conf['nr_users']),
                               downlink_queues=self.downlink_queues,
                               attacker_queue=self.attacker_queue,
                               error_log=self.error_queue,
                               uplink_queue=self.uplink_queue,
                               events=self.events,
                               meas_setup=meas_setup,
                               userlist=ta_userlist,
                               paged_users_list=paged_users,
                               detectors=ta_detectors,
                               max_users_frequencies=self.max_users_frequencies,
                               freq_users=freq_users
                               )
        if children:
            self.child_ta_list = [TrackingAreaChild(self.ta, ta_userlist, paged_users, i, meas_setup, self.conf, detectors=ta_detectors, buffersize=self.conf.get(
                'children_buffer_size', 10), freq_users=freq_users) for i in range(children)]
        try:
            self.ta.station_ids.remove(self.attacker.identifier)
        except (ValueError, AttributeError):
            pass

    def set_output(self, outfolder="tmp"):
        if not outfolder:
            self.output = print_exp()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.is_output_stdout = True
            return
        if not os.path.isdir(outfolder):
            os.mkdir(outfolder)
        self.stdout_handle.close()
        self.output_handle.close()
        self.error_handle.close()
        self.report_logger.update_folder(outfolder)
        self.conf['output_folder'] = outfolder
        self.stdout_handle = open(os.path.join(outfolder, 'stdout'), 'w')
        self.output_handle = open(os.path.join(outfolder, 'log'), 'w')
        self.error_handle = open(os.path.join(outfolder, 'stderr'), 'w')
        sys.stderr = self.error_handle
        sys.stdout = self.stdout_handle
        self.output = print_exp(self.output_handle)
        self.is_output_stdout = False
        for detector in self.detectors:
            detector.filename = os.path.join(
                outfolder, os.path.split(detector.filename)[-1])

    def simple_att_ta(self):
        with open(os.path.join('meas_configs', 'basic1.json'), 'r') as f:
            meas_setup = json.load(f)
        self.user_connect_event = self.manager.Event()
        action = ATTACKER_REJECT if self.conf.get('rejecting') else (
            ATTACKER_KEEP if self.conf.get('attacker_keep') else ATTACKER_HANG
        )
        self.att_ta = AttackerTrackingArea(tac=1,
                                           config=self.conf,
                                           stations=[self.attacker],
                                           user_ids=range(
                                               self.conf['nr_users']),
                                           downlink_queues=self.downlink_queues,
                                           attacker_queue=None,
                                           error_log=self.error_queue,
                                           uplink_queue=self.attacker_queue,
                                           events=self.events,
                                           meas_setup=meas_setup,
                                           user_connect_event=self.user_connect_event,
                                           action=action
                                           )

    def detach_att_ta(self):
        self.att_ta.uplink_queue = self.manager.Queue()

    def cluster_and_user(self, user_path=None):
        self.clusters = [UserCluster(i, self.cmap, self.report_queue, self.attack_queue, self.events,
                                     error_queue=self.error_queue, nr_steps=self.conf.get(
                                         'max_steps', np.infty),
                                     detectors=[(detector.wants, detector.uplink_queue) for detector in self.detectors]) for i in range(self.conf['nr_processes'])]
        eci_map = {s.identifier: s.eci for s in self.cmap.stations}
        if self.conf.get('spoof_eci'):
            eci_map[self.attacker.identifier] = self.attacker.eci
        else:
            self.cmap.stations[-1].eci = self.attacker.identifier
            self.attacker.eci = self.attacker.identifier
            eci_map[self.attacker.identifier] = self.attacker.eci
        users_per_cluster = self.conf['nr_users'] // self.conf['nr_processes']
        if user_path is not None:
            self.users = [PredictableUser(
                0, self.clusters[0], 0, position=user_path[0], downlink_queue=self.downlink_queues[0], uplink_queue=self.uplink_queue, seed=self.seed, config=self.conf, logging_folder=self.conf.get('output_folder'), eci_map=eci_map)]
            self.users[0].targets = deepcopy(user_path)
        else:
            self.users = [ClusterUser(
                i, self.clusters[i // users_per_cluster], 0, downlink_queue=self.downlink_queues[i], uplink_queue=self.uplink_queue, seed=self.seed, config=self.conf, logging_folder=self.conf.get('output_folder'), eci_map=eci_map) for i in range(self.conf['nr_users'])]
        for user in self.users:
            self.clusters[user.id // users_per_cluster].users.append(user)

    def run(self):
        report_process = mp.Process(
            target=ReportLogger.run, args=(self.report_logger,), name='report_process')
        ta_process = mp.Process(target=TrackingArea.run,
                                args=(self.ta,), name='ta_process')
        user_processes = [mp.Process(target=UserCluster.run, args=(
            cluster,), name='usercluster '+str(cluster.id)) for cluster in self.clusters]
        if self.conf.get('network_children'):
            child_ta_procs = [mp.Process(target=TrackingAreaChild.run, args=(
                child_ta,), name=f'ta_child {child_ta.child_id}') for child_ta in self.child_ta_list]
        else:
            child_ta_procs = ()
        detection_logging(self.detectors)
        detector_processes = [mp.Process(target=Detector.run, args=(
            detector,), name=f'detector_process_{i}') for i, detector in enumerate(self.detectors)]
        if self.attacker:
            att_ta_process = mp.Process(target=AttackerTrackingArea.run, args=(
                self.att_ta,), name='attacker_ta_process')

            att_ta_process.start()
        report_process.start()
        ta_process.start()
        time.sleep(1)
        for user_process in user_processes:
            user_process.start()
        for p in child_ta_procs:
            p.start()
        for p in detector_processes:
            p.start()
        if self.conf.get('wait_after'):
            self.events['attacker_event'].clear()
            while ta_process.is_alive():
                if self.events['update'].is_set():
                    time.sleep(0.1)
                else:
                    self.events['attacker_event'].set()
                    self.events['update'].set()
        else:
            self.events['attacker_event'].set()
        while ta_process.is_alive():
            time.sleep(0.1)
        for user_process in user_processes:
            user_process.join()
        if self.attacker:
            att_ta_process.terminate()
        report_process.join()
        for p in detector_processes:
            p.join()
        self.events['done'].clear()

    def shutdown(self):
        self.manager.shutdown()

    @staticmethod
    def plot_user(folder, save=False):
        u = UserHistory.from_file('userlog0', folder=folder)
        with open(os.path.join(folder, 'map_data')) as f:
            map_data = json.load(f)

        def find_sig(rec, target_eci, target_frequency):
            i, tmp = find_on(rec[1], lambda x: x[0] ==
                             target_eci and x[1] == target_frequency)
            if i == -1:
                return np.nan
            else:
                return tmp[2]
        frequencies = list(set(flatten([rec.get('frequencies', ())
                               for rec in map_data.get('stations', ())])))
        # _, ax = plt.subplots(1, len(frequencies)+1,figsize=(20,4))
        plt.plot([p[1] for p in u.loc_data], [p[2] for p in u.loc_data], "b")
        plt.gca().set_aspect('equal')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(0, map_data.get('size', 0))
        plt.gca().set_ylim(0, map_data.get('size', 0))
        plt.gca().set_title("Map trajectory")
        _, ax = plt.subplots(1, 1, figsize=(20, 4))
        stations_to_check = [s.get('eci')
                             for s in map_data.get('stations', ())]

        def find_serving(loc_iter, timestamp):
            for x in loc_iter:
                if x[0] == timestamp:
                    return x[3], x[4]
        locs = iter(u.loc_data)
        serving_stats_freqs = [find_serving(locs, x[0]) for x in u.sig_data]
        colors = iter(cm.get_cmap('Set1').colors)
        for _, freq in enumerate(frequencies):
            # ax[i+1].set_title(f'{freq} MHz')
            # + [att_station]:
            for stat in filter(lambda stat: freq in stat.get('frequencies') and stat.get('eci') in stations_to_check, map_data.get('stations')):
                col = next(colors)
                eci = stat.get('eci')
                tmp = pd.DataFrame([[rec[2], find_sig(rec, eci, freq)]
                                    for rec in u.sig_data if rec[2] > 5000], columns=['t', freq])
                tmp1 = pd.DataFrame([[rec[2], find_sig(rec, eci, freq)] if (eci, freq) == serving else [float('nan'), float('nan')]
                                    for rec, serving in zip(u.sig_data, serving_stats_freqs) if rec[2] > 5000], columns=['t', freq])
                tmp.plot(x='t', y=freq, legend=True, ax=ax,
                         alpha=0.2, lw=2, color=col,)
                if not tmp1.empty:
                    tmp1.plot(x='t', y=freq, ax=ax, legend=True,
                              label=freq, lw=2, alpha=1, color=col)
                    pass
        handles, _ = ax.get_legend_handles_labels()
        ax.set_ylabel('RSRP')
        ax.legend(handles[1::2], ['low priority',
                  'medium priority', 'high priority'])
        if save:
            plt.savefig(os.path.join(
                folder, f'user_history_{os.path.basename(folder)}.pdf'), format='pdf')
            plt.gca().clear()
            plt.close()
        else:
            plt.show()
        return serving_stats_freqs

    @staticmethod
    def user_attack_ratio(folder):
        res = {
            'attacker_connected_after': [],
            'attacker_connected_ratio': [],
            'attacker_distance': [],
            'attacker_closest_station': [],
            'attacker_signal': [],
            'best_nonattacker_signal': [],
            'distance_nonattacker': []
        }
        with open(os.path.join(folder, 'map_data')) as f:
            map_data = json.load(f)
        _, att_rec = find_on(map_data.get('stations', ()),
                                lambda rec: rec['tac'] == ATTACKER_TAC, default={})
        if not att_rec:
            return {
                'attacker_connected_after': -1,
                'attacker_connected_ratio': -1,
                'attacker_distance': -1,
                'attacker_closest_station': -1
            }
        good,bad = 0,0
        for filename in filter(lambda f: f.startswith('userlog'), os.listdir(folder)):
            u = UserHistory.from_file(filename, folder)
            l = len(u.loc_data)
            connected_after, x = find_on(u.loc_data, lambda rec: rec[3] >= ATTACKER_ECI)
            if connected_after == -1:
                connected_ratio, d, att_stat_dist, attacker_signal, best_nonattacker_signal, distance_nonattacker = 0, 0, 0, 0, 0, 0
                bad +=1
                # print('bad',filename)
                continue
            good += 1
            # print('good',filename)
            loc = Point(x[2], x[1])
            d = Point(att_rec.get('x', 0),
                        att_rec.get('y', 0)).distance(loc)
            connected_ratio = sum(x[3] >= ATTACKER_ECI for x in u.loc_data)/l
            y = find_last(
                u.sig_data, lambda rec: rec[0] <= x[0], find_first_if_none=True)
            e,maybe_attacker_signal = find_on(
                y[1], lambda rec: rec[0] >= ATTACKER_ECI)
            if e < 0:
                print("No attacker signal!")
                continue
            attacker_signal = maybe_attacker_signal[3]
            best_nonattacker = max(
                filter(lambda rec: rec[0] < ATTACKER_ECI, y[1]), key=lambda rec: rec[3])
            best_nonattacker_signal = best_nonattacker[3]
            _, rec_nonattacker = find_on(
                map_data['stations'], lambda rec: rec['eci'] == best_nonattacker[0])
            distance_nonattacker = dist(
                (rec_nonattacker['x'], rec_nonattacker['y']), (x[1], x[2]))
            att_stat_dist = min(dist((att_rec['x'], att_rec['y']), (s['x'], s['y']))
                                for s in map_data['stations'] if s['eci'] < ATTACKER_ECI)

            res['attacker_connected_after'].append(connected_after)
            res['attacker_connected_ratio'].append(connected_ratio),
            res['attacker_distance'].append(d),
            res['attacker_closest_station'].append(att_stat_dist),
            res['attacker_signal'].append(attacker_signal),
            res['best_nonattacker_signal'].append(best_nonattacker_signal),
            res['distance_nonattacker'].append(distance_nonattacker)
            
        if not res['attacker_connected_after']:
            return {
                'success': 0,
                'attacker_connected_after': -1,
                'attacker_connected_ratio': -1,
                'attacker_distance': -1,
                'attacker_closest_station': -1,
                'attacker_signal': -1,
                'best_nonattacker_signal': -1,
                'distance_nonattacker': -1
            }
        return {key: np.average(item) for key, item in res.items()} | {'success': good/(good+bad)}

    @ staticmethod
    def calculate_ratios(folder):
        u = UserHistory.from_file('userlog0', folder)
        total = len(u.loc_data) if u.loc_data else float('nan')
        has_signal = ilen(filter(lambda x: x[3] >= 0, u.loc_data))
        unconnected, idle, connected = 0, 0, 0
        for x in u.loc_data:
            if x[5] == 'unconnected':
                unconnected += 1
            elif x[5] == 'rrc_idle':
                idle += 1
            elif x[5] == 'rrc_connected':
                connected += 1
        return {
            'total_steps': total,
            'has_signal': has_signal/total,
            'unconnected': unconnected/total,
            'idle': idle/total,
            'connected': connected/total
        }

    def print_frequencies(self):
        self.output('Frequencies generated for this map:')
        for s in self.cmap.stations:
            util.debug(f'{s.eci}: {s.frequencies}')

    @ staticmethod
    def guess_attacker_location(folder='tmp_output', reps=1):
        """Guesses the attacker location using gradient descent on the recorded measurement reports

        """
        mr = ReportLogger.generate_report(
            'reportlog', folder=folder).measurement_reports
        neigh_events = mr.query("event in ['a3','a4','a5']")
        sig_data = zip(((a[0], a[1])
                        for a in neigh_events['user_loc']), neigh_events['signals'])
        if not sig_data:
            return None

        def find_sig(sig):
            _, r = find_on(sig, lambda y: y[0] == ATTACKER_ECI)
            if not r is None:
                return -r[2]
        with open(os.path.join(folder, 'map_data')) as f:
            map_data = json.load(f)
        i, att_rec = find_on(map_data.get('stations', ()),
                             lambda rec: rec['eci'] == ATTACKER_ECI, default={})
        if i < 0:
            raise MeasurementError('No attacker found')
        frequency = att_rec.get('frequencies', [])[0]
        meas = list(filter(lambda x: x[1] is not None, ((
            loc, find_sig(sig)) for loc, sig in sig_data)))
        if not meas:
            raise MeasurementError("No measurements!")

        def cost_function(pos, signal_strength=0, freq=frequency):
            return cost_free(pos, meas, 1/(freq*DIST_FACTOR), 20)

        def jac(pos, signal_strength=0, freq=frequency):
            return grad_free(pos, meas, 1/(freq*DIST_FACTOR), 20)
        res = []
        for signal_strength in range(-20, 35, 5):
            guess = minimize(cost_function, np.array(
                [0, 0]), jac=jac, args=(signal_strength, frequency)).x
            res.append(
                ((*guess,), len(meas), Point(*guess).distance(Point(att_rec.get('x', 0), att_rec.get('y', 0)))))
        return res

    @ staticmethod
    def guess_attacker_location_rt(folder='tmp_output', reps=1, sample_size=None):
        """Guesses the attacker location using gradient descent on the recorded measurement reports

        """
        mr = ReportLogger.generate_report(
            'reportlog', folder=folder).measurement_reports
        if not sample_size:
            neigh_events = mr.query("event in ['a3','a4','a5']")
            if not len(neigh_events):
                raise MeasurementError("No measurements!")
            sig_data = neigh_events[['user_loc', 'signals']]
        else:
            sig_data = mr[['user_loc', 'signals']].sample(sample_size)
        def find_sig(sig):
            i, r = find_on(sig, lambda y: y[1] == ATTACKER_ECI)
            if i>=0:
                return -r[2]
        with open(os.path.join(folder, 'map_data')) as f:
            map_data = json.load(f)
        i, att_rec = find_on(map_data.get('stations', ()),
                             lambda rec: rec['tac'] == ATTACKER_TAC, default={})
        if i < 0:
            raise MeasurementError('No attacker found')
        frequency = att_rec.get('frequencies', [])[0]
        meas = list(filter(lambda x: x[1] is not None, 
            zip(sig_data['user_loc'], (find_sig(sig) for sig in sig_data['signals']))))
        if not meas:
            raise MeasurementError("No measurements!")

        def cost_function(pos, signal_strength=0, freq=frequency):
            return cost_free(pos, meas, 1/(freq*DIST_FACTOR), 20)

        def jac(pos, signal_strength=0, freq=frequency):
            return grad_free(pos, meas, 1/(freq*DIST_FACTOR), 20)
        res = []
        for signal_strength in range(-20, 35, 5):
            guess = minimize(cost_function, np.array(
                [0, 0]), jac=jac, args=(signal_strength, frequency)).x
            res.append(
                ((*guess,), len(meas), Point(*guess).distance(Point(att_rec.get('x', 0), att_rec.get('y', 0)))))
        return res

    @ staticmethod
    def calculate_signal_data(folder):
        u = UserHistory.from_file('userlog0', folder)
        locs = iter(u.loc_data)
        if not u.loc_data:
            return {}
        current_loc = next(locs)
        signal_avg = 0
        below_best_nr = 0
        below_best_avg = 0
        serving_frequencies = {}
        for i, sigs in enumerate(u.sig_data):
            while float(current_loc[0]) < float(sigs[0]):
                try:
                    current_loc = next(locs)
                except StopIteration:
                    break
            serving_cell = current_loc[3]
            serving_freq = current_loc[4]
            if not serving_freq in serving_frequencies:
                serving_frequencies[serving_freq] = 1
            else:
                serving_frequencies[serving_freq] += 1
            if serving_cell < 0:
                continue
            for rec in sigs[1]:
                if rec[0] == serving_cell and rec[1] == serving_freq:
                    signal_avg = i/(i+1)*signal_avg + rec[2]/(i+1)
                    serving_signal = rec[2]
                    break
            else:
                print('Did not find serving signal in', folder, 'for cell',
                      serving_cell, 'and frequency', serving_freq, file=sys.stderr)
                continue
            best_signal = max(sigs[1], key=tget(2))
            if best_signal[0] != serving_cell and best_signal[1] != serving_freq:
                below_best_avg = below_best_nr / (below_best_nr+1)*below_best_avg + (best_signal[2]-serving_signal)/(below_best_nr+1)
                below_best_nr += 1
        return {
            'average': signal_avg,
            'below_best_nr': below_best_nr,
            'below_best_avg': below_best_avg,
            'serving_frequency': {freq: nr/len(u.sig_data) for freq, nr in serving_frequencies.items()}
        }

    @ staticmethod
    def id_detector_output(folder):
        with open(os.path.join(folder, 'conf'), 'r') as fp:
            conf = json.load(fp)
        for detect_conf in conf.get('config', {}).get('detectors', ()):
            if detect_conf['type'] == IdDetector.code:
                break
        detect_output = plot_detection_results(
            folder, filename=detect_conf['filename'], plot=False)
        if (attacker_activation_time := conf.get('add_attacker_after', -1)) < 0:
            raise DetectionNotFound('No field for attacker addition time')
        for rec, data in filter(lambda x: x[0][1] == ANOMALY, detect_output):
            steps, _, _ = rec
            if 'freq' in data:
                ...
            if 'cell' in data:
                if data['cell'] == ATTACKER_ECI:
                    return steps - attacker_activation_time
        raise DetectionNotFound('Attacker ECI was not found in detection logs')

    def print_map_data(self):
        res = {
            'size': int(self.cmap.map_bounds),
            'distance_mode': self.cmap.distance_mode,
            'stations': [
                {'eci': int(s.eci),
                 'x': int(s.x),
                 'y': int(s.y),
                 'signal_strength': int(s.signal_strength),
                 'noise': int(s.noise),
                 'frequencies': [int(f) for f in s.frequencies],
                 'tac': int(s.tac)
                 } for s in self.cmap.stations + ([self.attacker] if self.attacker else [])
            ]}
        if self.is_output_stdout:
            self.output('map: ', json.dumps(res, indent=2))
            return
        with open(os.path.join(self.conf['output_folder'], 'map_data'), 'w') as f:
            json.dump(res, f, indent=2)


def create_detector(conf, filename, events=None, queue=None) -> Detector:
    for detector in detectors:
        if detector.code == conf.get('type'):
            return detector(events, queue, conf.get('wait_time', 100), conf.get('window_size', 100), conf.get('threshold', 1), filename=filename)
    util.error_logging('Unknown detector type', conf.get('type'))


def vary_sim_config(base_conf, conf_variations, map_conf, frequencies, seed=0, plot=False, n_iter=1):
    """Varies items in the configuration

    """
    for config in vary_config(base_conf, conf_variations):
        experiment = ExperimentSetup(
            config, frequencies=frequencies, map_conf=map_conf)
        yield experiment, config


@ dataclass
class ExperimentProducer:
    base_conf: dict = None
    conf_variations: dict = None
    map_conf: dict = None
    frequencies: FrequencyConfigRange = None
    seed: int = 0
    plot = False
    configs: Any = field(init=False)
    last_value: Any = field(init=False)

    def __post_init__(self):
        self.configs = vary_config(self.base_conf, self.conf_variations)
        config = next(self.configs)
        experiment = ExperimentSetup(
            config, frequencies=self.frequencies, map_conf=self.map_conf)
        self.last_value = experiment, config

    def produce(self):
        if self.last_value is None:
            raise StopIteration
        return self.last_value

    def skip(self):
        try:
            config = next(self.configs)
        except StopIteration:
            self.last_value = None
            return
        experiment = ExperimentSetup(
            config, frequencies=self.frequencies, map_conf=self.map_conf)
        self.last_value = experiment, config

    def __bool__(self):
        return self.last_value is not None


def vary_frequency(frequencies_ranges: FrequencyConfigRange, experiment=None, conf=None, seed=0, plot=False, n_iter=1, add_attacker_after=0, q_hyst=0, attack=True, fix_inter_to_infra=False, map_conf=None):
    """Varies the frequencies

    Uses a list FrequencyConfigRange objects to create lists of FrequencyConfig objects

    """
    for frequencies in map(list, itools.product(*(r.expand(fix_inter_to_infra) for r in frequencies_ranges))):
        experiment.set_frequencies(frequencies, q_hyst=q_hyst)
        if conf.get('attack', {}).get('position') == 'all':
            with open(os.path.join(map_conf['raytrace_folder'], map_conf['raytrace_filebase']+'_stat.json'), 'r') as f:
                stations = [BaseStation.fromdict(rec) for rec in json.load(f)]
            nr_attackers = len(
                list(filter(lambda s: isinstance(s, Attacker), stations)))
        else:
            nr_attackers = 1
        for i in range(nr_attackers):
            att_conf = conf.get('attack')
            if att_conf:
                att_conf['attack_id'] = i
            experiment.simple_map(attack=att_conf)
            if attack:
                experiment.simple_attacker(add=not bool(add_attacker_after), frequency=att_conf.get(
                    'frequency', 1000), signal_strength=att_conf.get('signal_strength', 0))
            experiment.setup_context()
            yield frequencies


def experiment_inner(experiment, meas_setup=None, add_attacker_after=0, attack=True, attacker_ratio=False, user_path=None):
    """Actually runs the experiment

    Requires to be run while iterating vary_conf_item and vary_frequency;
    those functions handle the setup necessary to make this work.
    """
    experiment.simple_ta(meas_setup)
    if attack:
        experiment.simple_att_ta()
    experiment.cluster_and_user(user_path)
    if add_attacker_after:
        experiment.add_attacker_after(add_attacker_after)
    experiment.run()
    tmp = experiment.user_attack_ratio() if attacker_ratio else None
    if add_attacker_after:
        experiment.remove_attacker()
    return tmp


def experiment_core(experiment, base_folder, config, frequency, meas_setup, add_attacker_after, q_hyst, prefix, output_nr, user_path=None):
    folder = os.path.join(base_folder, f'{prefix}{output_nr}')
    experiment.set_output(folder)
    try:
        experiment.print_map_data()
    except Exception as e:
        print("Error in json in print_map_data!")
        raise e
    try:
        experiment_inner(
            experiment, meas_setup, add_attacker_after=add_attacker_after, attack=config.get('attack', False), user_path=user_path)
    except Exception as e:
        print("Error in json in experiment_inner!")
        raise e
    conf = {
        'q_hyst': q_hyst,
        'frequencies': [fc.__dict__ for fc in frequency],
        'config': config,
        'add_attacker_after': add_attacker_after

    }
    try:
        with open(os.path.join(folder, 'conf'), 'w') as f:
            json.dump(conf, f, indent=2)
        with open(os.path.join(folder, 'meas_conf'), 'w') as f:
            json.dump(meas_setup, f, indent=2)
    except Exception as e:
        print("Error in json in experiment_core!")
        raise e
    output_nr += 1


def experiment_multiproc_part(config, map_conf=None, frequencies=None, q_hyst_range=[0], add_attacker_after=0, meas_config=None, meas_config_variations={'': [0]}, prefix='out', fix_inter_to_infra=True, output_nr=0):
    util.listen()
    experiment = ExperimentSetup(
        config, frequencies=frequencies, map_conf=map_conf)
    output_folder = config['output_folder']
    for q_hyst in q_hyst_range:
        for frequency in vary_frequency(frequencies, experiment=experiment, conf=config, q_hyst=q_hyst, fix_inter_to_infra=fix_inter_to_infra, attack=config.get('attack', False), add_attacker_after=add_attacker_after, map_conf=map_conf):
            for meas_setup in vary_config(meas_config, meas_config_variations):
                try:
                    args = (experiment, output_folder, config, frequency,
                            meas_setup, add_attacker_after, q_hyst, prefix, output_nr)
                    experiment_core(*args)
                except Exception as e:
                    print("Caught error in experiment_multiproc_part", file=sys.stderr)
                    print(traceback.format_exc(),file=sys.stderr)
                    print(e,file=sys.stderr)
                output_nr += 1
    return

@ arbitrary_keywords
def experiment_main(base_conf=None, map_conf=None, frequencies=None, conf_variations={'': [0]}, q_hyst_range=[0], n_iter=1, add_attacker_after=0, meas_config=None, meas_config_variations={'': [0]}, prefix='out', fix_inter_to_infra=True, total=0, nr_processes=0, user_path=None, redo_map=False):
    if not base_conf:
        raise Exception('No base configuration!')
    if not map_conf:
        raise Exception('No map configuration!')
    if not frequencies:
        raise Exception('No frequencies!')
    if not meas_config:
        with open(os.path.join('meas_configs', 'basic1.json'), 'r') as f:
            meas_config = json.load(f)
    output_nr = 0
    if nr_processes:
        configs = vary_config(base_conf, conf_variations)
        pool = NestablePool(nr_processes)
        nr_subsets = total/math.prod((len(i)
                                      for k, i in conf_variations.items()))
        print("Entering multiprocessing...")
        list(tqdm(pool.starmap(experiment_multiproc_part, [
            (deepcopy(config), map_conf, tuple(frequencies), tuple(q_hyst_range),  add_attacker_after,
             meas_config, meas_config_variations, prefix, fix_inter_to_infra, round(i*nr_subsets))
            for i, config in enumerate(configs)
        ]), total=total))
        # for i,config in enumerate(configs):
        #    output_nr = round(i*nr_subsets)
        #    res.append(pool.apply_async(experiment_multiproc_part, args=(deepcopy(config),map_conf, frequencies, q_hyst_range,  add_attacker_after, meas_config,meas_config_variations, prefix,fix_inter_to_infra,output_nr)))
        #    sys.stdout = sys.__stdout__
        pool.close()
        pool.join()
        output_nr = total
    else:
        if total:
            progress = tqdm(total=total)
        for experiment, config in vary_sim_config(base_conf=base_conf, conf_variations=conf_variations, map_conf=map_conf, frequencies=[]):
            for q_hyst in q_hyst_range:
                for frequency in vary_frequency(frequencies, experiment=experiment, conf=config, q_hyst=q_hyst, fix_inter_to_infra=fix_inter_to_infra, attack=config.get('attack', False), add_attacker_after=add_attacker_after):
                    for meas_setup in vary_config(meas_config, meas_config_variations):
                        args = (experiment, base_conf['output_folder'], config, frequency,
                                meas_setup, add_attacker_after, q_hyst, prefix, output_nr, user_path)
                        experiment_core(*args)
                        progress.update()
                        output_nr += 1
        # experiment.shutdown()
    return output_nr


def run_detection_afterwards(conf, folder):
    detector = create_detector(conf)
    detector_wants = detector.wants
    reportlog = ReportLogger.generate_report(
        'reportlog', folder=folder).measurement_reports
    for i, report in reportlog.iterrows():
        if report['event'] in detector_wants:
            detector.handle_data_baseline([{}])
    detector.process_window()
