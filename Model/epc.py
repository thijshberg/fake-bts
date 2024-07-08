from os import error
from .util import *
from .user import UserStateMachine, void
import multiprocessing as mp
import json
from functools import wraps
import itertools as itools
from threading import BrokenBarrierError
from copy import deepcopy
from functools import reduce, cache
import sys
from queue import Empty

import time
import traceback
import numpy.random as npr
import numpy as np
import math

"""
The network part of the simulation
"""

IDLE = 0
CONNECTED = 1

PAGING_LOW = 1/20  # 1 in 20 users is connected
PAGING_MEDIUM = 1/8  # 1 in 8 users is connected
PAGING_HIGH = 1/2  # 1 in 2 users is connected
PAGING_NONE = None  # No users are connected
MAX_PAGING_TIME = 60
PAGING_INTERVAL = 10
# How far away we consider another tower to be a neighbour. Note that setting a sensible NEIGHBOUR_MAX_NR makes this unnecessary.
NEIGHBOUR_DISTANCE = 2000
NEIGHBOUR_MAX_NR = 5


DUMMY_TAC = 0xFFFE

SIGNAL_WORSENING_STEPS = 1  # How often we lower the A1 threshold for a user
# How much the A1 threshold is lowered when a user gets a bad signal
SIGNAL_WORSENING_STEPSIZE = 10

TA_REFRESH_TIME = 0.5

_report_intervals = [0.120, 0.240, 0.480, 0.640, 1.024,
                     2.048, 5.120, 10.240, 60, 360, 720, 1800, 3600]
# The intuition behind this rounding is that the user walks one step (which is 1 m) per cycle; taking the user walks 5 km/h, which is 1.4 m/s, the above times correspond to the following in meters (and we round as the simulation is discrete)
# We cannot round but we must take the ceiling, as we cannot handle an interval of 0
REPORT_INTERVALS = [math.ceil(r*1.4) for r in _report_intervals]
REPORT_AMOUNTS = [1, 2, 4, 8, 16, 32, 64, np.inf]

# ENUMERATED {s4, s8, s16, s32, s64, s128, s256, s512}
BARRING_TIMES = [4, 8, 16, 32, 64, 128, 256, 512]


def _user_locked(method):
    """
    Decorator to acquire and release a lock at the beginning and end of the method respectively
    """
    @wraps(method)
    def _user_locked_inner(self, user_id, *args, **kwargs):
        if isinstance(self.user_lock, list):
            with self.user_lock[user_id]:
                return method(self, user_id, *args, **kwargs)
        with self.user_lock:
            return method(self, user_id, *args, **kwargs)
    return _user_locked_inner


"""
Implements one tracking area

Current model: the tracking area communicates to the user by putting messages in its queue. The user directly calls functions on the tracking area.

WARNING: The TA currently does not store the measurement configuration it has sent a user; if a (connected) user sends a measurement report it will be handled regardless of whether the user was instructed to measure this.
WARNING: make sure all towers are no further apart than NEIGHBOUR_DISTANCE

Structuring:
The _[...]_inner functions are only to be called while user_lock is acquired. They represent the inner logic, and are separate from their properly locked wrapper so they can also be called while handling a message. 

TODO: Change the name to MobileManagementEntity, as this is the official term for the kind of server that performs the described service.

"""


class TrackingArea(HasRandom):
    def __init__(self, tac, config, stations, user_ids, downlink_queues, attacker_queue, error_log, uplink_queue, events, meas_setup, userlist=None, child_procs=None, paged_users_list=None, detectors=(), max_users_frequencies={}, freq_users=None):
        self.tac = tac
        self.paging_simulation = config.get('paging_level', PAGING_NONE)
        self.print_status = config['verbosity'].get('print_status', False)
        self.stations = stations
        self.station_id_offset = min(stat.identifier for stat in stations)
        self.station_ids = [stat.identifier -
                            self.station_id_offset for stat in stations] + [-1]
        epc_logging(self, "station ids:", self.station_ids)
        self.ta_ids = {stat.tac for stat in stations}
        self.frequencies = reduce(
            set.union, [stat.frequencies for stat in stations], set())
        self.user_ids = user_ids
        # Messaging setup downlink: the core network (i.e. the tracking area) has a queue of messages for each user, which the user has access to.
        self.downlink_queues = downlink_queues
        # Messaging setup uplink: users sending messages to the TA is handled through this queue. The user thus does not need a reference to the TA, only access to the queue.
        self.uplink_queue = uplink_queue
        # In the current state there is one main tracking area handling all incoming messages; messages for unknown towers (i.e. the attacker) are put into the attacker_queue
        self.attacker_queue = attacker_queue
        self.error_log = error_log
        if userlist:
            self.registered_users = userlist
        else:
            self.registered_users = [None for _ in user_ids]
        self.total_users = config['nr_users']
        self.events = events
        self.max_steps = config.get('max_steps', np.infty)
        self.paging_time = config.get('paging_time', [5, 10])
        self.keep_neighbour_results = config.get('keep_neighbour_results', 20)
        self.reveal_network_records = (
            config['verbosity']).get('reveal_network_records')
        self.wait_after = config.get('wait_after', [])+[float('inf')]
        # Load the SIB
        if (sib_filename := config.get('sib_filename')):
            with open(sib_filename, 'r') as f:
                self.system_information_base, self.system_information_list = self.expand_system_information(
                    json.load(f))
        else:
            self.system_information_base, self.system_information_list = None, None

        # This is used to prevent race conditions where the TA keeps sending paging requests before the users have had a chance to respond.
        if not paged_users_list is None:  # We are provided a proxy
            self.paged_users = paged_users_list
        else:
            self.paged_users = []
        self.print_pagelist = (config['verbosity']).get(
            'print_users_to_page', False)
        self.log_user = config['verbosity'].get('log_user', -1)
        super().__init__(n=len(self.registered_users)
                         * 10, seed=config.get('seed', 0)+self.tac)
        self.generate_handover_graph()
        self.call_list = CallList(
            len(self.user_ids)-1, self.randint, len(self.user_ids)*10)
        self._measurement_receive, self._setup_connection = self.parse_meas_setup(
            meas_setup)

        # children stuff
        if (children := config.get('network_children')):
            self.user_lock = [mp.Lock() for _ in user_ids]
        else:
            self.user_lock = mp.Lock()
        self.has_children = children
        self.child_id = -1

        self.meas_id_aliases = {}
        self.steps = 0
        if wait_time := config.get('sync_waiting'):
            self.sync_waiting = lambda: time.sleep(wait_time)
        else:
            self.sync_waiting = lambda: ...

        # We only send messages to the detector if it wants that type of message
        @cache
        def detector_wants_dependent(key):
            def inner(msg):
                for detector_wants, detector_queue in detectors:
                    if key in detector_wants:
                        detector_queue.put(msg)
            return inner
        self.detector_register = detector_wants_dependent(
            'connection_register')
        self.detector_measurement = detector_wants_dependent(
            'measurement_report')
        self.detector_leave = detector_wants_dependent('connection_leave')
        self.detector_update = detector_wants_dependent('update')
        self.detector_database = detector_wants_dependent('database')
        self.detector_ta_update = detector_wants_dependent('ta_update')
        self.detector_loc_update = detector_wants_dependent('loc_update')
        self.detector_bts_update = detector_wants_dependent('bts_update')

        self.detector_activate_time = min([det.get('activate_after', float(
            'inf')) for det in config.get('detectors', [])], default=float('inf'))
        self.detector_wait_time = min([det.get('wait_time', 100) for det in config.get(
            'detectors', [])], default=float('inf'))

        self.ta_capacity = config.get('ta_capacity', float('inf'))
        self.ta_users = {ta_id: 0 for ta_id in self.ta_ids}
        epc_logging('Max users per cell:', json.dumps(
            {s.identifier: s.max_users for s in self.stations}, indent=2))

    def run(self):
        listen()
        try:
            detector_wait_timer = 0
            if self.reveal_network_records:
                reveal_timestamp = time.time()
            epc_logging('EPC starting!')
            while (not self.events['done'].is_set()) and self.steps <= self.max_steps:
                self.events['update'].wait()
                self.sim_step()
                if self.reveal_network_records:
                    if time.time() - reveal_timestamp > self.reveal_network_records:
                        try:
                            epc_logging(json.dumps(
                                self.registered_users, indent=4))
                        except Exception:
                            epc_logging(json.dumps(
                                self.registered_users._getvalue(), indent=4))
                        try:
                            epc_logging(f'paged users: {self.paged_users}')
                        except Exception:
                            epc_logging(
                                f'paged (proxy) users: {self.paged_users._getvalue()}')
                        reveal_timestamp = time.time()
                if self.wait_for_barrier():
                    break
                self.steps += 1
                if self.steps >= self.wait_after[0]:
                    self.wait_after.pop(0)
                    self.events['update'].clear()
                if self.steps >= self.detector_activate_time and not self.events['baseline_done'].is_set():
                    self.events['baseline_done'].set()
                if self.steps - detector_wait_timer >= self.detector_wait_time:
                    detector_wait_timer = self.steps
                    try:
                        if isinstance(self.registered_users, list):
                            self.detector_database(self.registered_users)
                        else:
                            # We don't need to make a copy as it gets passed into a queue
                            self.detector_database(
                                self.registered_users._getvalue())
                        self.events['detector_bar'].wait()
                    except BrokenBarrierError:
                        if self.steps < self.max_steps:
                            error_logging(
                                "Broken detector barrier after", self.steps, 'steps! :(')
                        break
            # i.e. we exceeded the maximum steps
            if not self.events['done'].is_set():
                self.events['done'].set()
            self.events['detector_bar'].abort()
            # time.sleep(0.2)#This prevents errors with pipes internal to the queues breaking, as it could be that the user handles the event somewhat later
        except (KeyboardInterrupt, Exception) as e:
            error_logging(f'EPC stopped after {self.steps} steps!')
            if not isinstance(e, KeyboardInterrupt):
                error_logging(traceback.format_exc())
            raise e
        return

    def wait_for_barrier(self, child=False):
        try:
            self.events['barrier'].wait()
        except BrokenBarrierError:
            if self.steps < self.max_steps:
                error_logging(
                    f"Broken barrier in {'EPC' if not child else f'child {self.child_id}'} after", self.steps, 'steps')
            return True

    def sim_step(self):
        msg_queue = flush_queue(self.uplink_queue, timeout=1)
        for msg in msg_queue:
            self._digest_message(msg)
        # We first digest messages before resending paging messages to get as accurate a state as possible
        epc_logging('looking at paging:', bool(self.paging_simulation))
        if self.paging_simulation:
            self.resend_paging_messages()
            self.new_paging_messages()
        self.sync_waiting()

    def _digest_message(self, msg):
        user_id = msg['user_id']
        bts_id = msg['serving_cell'] - self.station_id_offset
        frequency = msg['frequency']
        plmn_id = msg.get('plmn_id', -1)
        if not bts_id in self.station_ids:
            epc_logging("not in", self.station_ids, ":", bts_id)
            self.attacker_queue.put(msg)
            return
        epc_logging('(TrackingArea)', json.dumps(msg, indent=2))
        # if an unknown user contacts without indication setup, we assume it went to an attacker (this circumvents writing inter-TA handover)
        if 'connection_request' in msg:
            self.user_setup_connection(user_id, bts_id)
        if 'connection_release' in msg:
            self.user_release(user_id)
        if 'connection_leave' in msg:
            self.user_leave(user_id)
        if 'connection_register' in msg:
            self.user_register_idle(user_id, bts_id, plmn_id, frequency)
            self.detector_register(msg)
        if 'connection_hop' in msg:
            self.user_hop(user_id, bts_id, frequency)
        if 'request_ta_info' in msg:
            self.send_ta_info(user_id, bts_id)
        if 'tracking_area_update' in msg:
            self.ta_update(self.steps, user_id, bts_id,
                           msg['tracking_area_update'])
        if 'measurement_report' in msg:
            # We give report A3 and A5 priority as they can trigger handover; after handover the other messages need not be handled
            for rep in (rep for rep in msg['measurement_report'] if rep['event_id'] in ['a3', 'a5']):
                if self.measurement_receive(user_id, rep):
                    return
            for rep in (rep for rep in msg['measurement_report'] if not rep['event_id'] in ['a3', 'a5']):
                self.measurement_receive(user_id, rep)
            self.detector_measurement(msg)
        if 'loc_update' in msg:
            self.detector_loc_update((user_id, msg['loc_update']))
        if 'bts_update' in msg:
            self.detector_bts_update((user_id, bts_id, frequency))

    def new_paging_messages(self):
        nr_connected = sum(
            rec['state'] is CONNECTED for rec in self.registered_users if rec) + len(self.paged_users)
        nr_users = len(self.registered_users)
        # At this point there is no possibility any user gets paged, which should be possible
        if self.total_users <= 1/self.paging_simulation:
            estimated_calls = npr.binomial(
                1, self.paging_simulation/self.randint(*self.paging_time))
        else:
            estimated_calls = math.floor(
                nr_users*self.paging_simulation - nr_connected)
        idle_users = [rec['id'] for rec in self.registered_users if (
            bool(rec) and rec['state'] == IDLE)]
        users_to_page = [self.call_list.get() for _ in range(
            max(0, min(estimated_calls, len(idle_users)-len(self.paged_users))))]
        users_to_page = [(i, self.randint(*self.paging_time))
                         for i in users_to_page if i not in map(fst, self.paged_users)]
        users_to_page = uniques(users_to_page)
        if users_to_page:
            if self.print_pagelist:
                epc_logging("users to page:", users_to_page[:])
                epc_logging("users already paged:", self.paged_users)
            self.paged_users += [(user_id, paging_time, self.steps)
                                 for user_id, paging_time in users_to_page]
        self._send_paging_to_users(users_to_page)

    def resend_paging_messages(self):
        to_remove = []
        paging_list = []
        for i, (user_id, paging_time, timestamp) in enumerate(self.paged_users):
            if self.steps - timestamp > MAX_PAGING_TIME:
                to_remove.append(i)
            elif self.steps - timestamp % PAGING_INTERVAL == 0:
                paging_list.append((user_id, paging_time))
        for i in to_remove[::-1]:
            self.paged_users.pop(i)
        self._send_paging_to_users(paging_list)

    def spread_user_update(self, cell_id, entering=True):
        (stat := self.stations[cell_id]).connected_users += entering
        self.ta_users[stat.tac] += entering

    def barring_check(self, bts_id):
        """TS 36.331 5.3.3.11
        """
        if bts_id < 0 or bts_id > len(self.stations)-1:
            return 1, 60
        factor = clamp(
            1-4*(1-self.stations[bts_id].connected_users/self.stations[bts_id].max_users))
        barring_time = BARRING_TIMES[math.floor(len(BARRING_TIMES) * factor)]
        return factor, barring_time

    def ta_info(self, bts_id):
        tac = self.stations[bts_id].tac
        return {'tracking_area_info': {
                'tac': tac,
                'stations': [stat.identifier for stat in self.stations if stat.tac == tac],
                'attacker': isinstance(self, AttackerTrackingArea)
                }
                }

    def send_ta_info(self, user_id, bts_id):
        self._send_message_to_user(user_id, self.ta_info(bts_id))

    @_user_locked
    def user_register_idle(self, user_id, cell_id, plmn_id, freq):
        return self._user_register_idle_inner(user_id, cell_id, plmn_id, freq)

    def _user_register_idle_inner(self, user_id, cell_id, plmn_id, freq):
        self.registered_users[user_id] = {
            'id': user_id,
            'state': IDLE,
            'plmn': plmn_id,
            'last_known_cell': cell_id,
            'tracking_area': self.stations[cell_id].tac,
            'frequency': freq
        }
        msg = {'register_complete': {'tac': self.tac}}
        if self.system_information_base:
            msg['system_information'] = self.compute_system_information(
                cell_id, freq)
        self._send_message_to_user(user_id, msg)
        self.detector_update((None, cell_id, None, freq, self.steps, user_id))
        return

    @_user_locked
    def ta_update(self, timestamp, user_id, bts_id, result):
        return self._ta_update(timestamp, user_id, bts_id, result)

    def _ta_update(self, timestamp, user_id, bts_id, result):
        tac = self.stations[bts_id].tac
        self.detector_ta_update((timestamp, bts_id, tac, result))
        if self.ta_users[self.stations[bts_id].tac] >= self.ta_capacity:
            self._send_message_to_user(
                user_id, {'tracking_area_update_reject': 0, 'tac': tac})
            return
        self._send_message_to_user(
            user_id, {'tracking_area_update_accept': 0, 'tac': tac})
        self.registered_users[user_id]['last_known_cell'] = bts_id
        self.registered_users[user_id]['tracking_area'] = self.stations[bts_id].tac

    @_user_locked
    def user_hop(self, user_id, bts_id, frequency):
        """
        NOT a real thing, but necessary for modelling. Used by the user to indicate it needs an updated SIB.
        In real life this is broadcast periodically, but we don't have a broadcast (only a queue for each user).
        This function should therefore NOT trigger any state change on the network level.
        """
        return self._user_hop_inner(user_id, bts_id, frequency)

    def _user_hop_inner(self, user_id, bts_id, frequency):
        if self.registered_users[user_id] is not None:
            self.registered_users[user_id]['frequency'] = frequency
        if (sib := self.compute_system_information(bts_id, frequency)) is None:
            self._send_message_to_user(user_id, {'disconnect': 1})
        else:
            self._send_message_to_user(user_id, {'system_information': sib})
        return

    @_user_locked
    def user_leave(self, user_id):
        """
        User loses signal or for any other reason leaves the coverage
        """
        return self._user_leave_inner(user_id)

    def _user_leave_inner(self, user_id):
        if self.registered_users[user_id] is None:
            """
            This is a (probable) race condition, where the user switches too quickly between being idle and unconnected.
            It could be that the UE leaves in one step, getting the message in after the network flushes.
            The next 
            This could be solved by having a separate message for confirming the user's registration 
            """
            error_logging(
                'User', user_id, 'was not registered upon leaving!(', self.__class__.__name__, ')')
            return
        self.detector_update((self.registered_users[user_id]['last_known_cell'], None,
                             self.registered_users[user_id]['frequency'], None, self.steps, user_id))
        self.detector_leave(
            (self.steps, self.registered_users[user_id]['last_known_cell'], self.registered_users[user_id]['frequency'], user_id))
        self.registered_users[user_id] = None
        return

    def user_reject(self, user_id, result="block"):
        epc_logging('(', self.steps, ') Rejecting', user_id)
        self._send_message_to_user(user_id, {'connection_reject': result})

    @_user_locked
    def user_setup_connection(self, user_id, bts_id):
        return self._user_setup_connection_inner(user_id, bts_id)

    def _user_setup_connection_inner(self, user_id, bts_id):
        # This user was paged and needs to be popped from the paged_users list
        remove_on(self.paged_users, lambda x: fst(x) == user_id)
        # Error cases where the user is rejected
        if self.registered_users[user_id] is None:
            if self.__class__ == AttackerTrackingArea:
                self._user_register_idle_inner(
                    user_id, bts_id, 0, self.stations[0].frequencies[0])
            else:
                error_logging(
                    f'User {user_id} was unknown at connection setup(', self.__class__.__name__, ')')
                self.user_reject(user_id, result='fail')
                self.registered_users[user_id] = None
                return
        self.set_user_record_key(user_id, 'last_known_cell', bts_id)
        if not bts_id in self.station_ids:
            error_logging("Unknown station", bts_id,
                          '(', self.__class__.__name__, ')')
            self.user_reject(user_id, result='fail')
            return
        if not self.registered_users[user_id] or not self.registered_users[user_id].get('state') == IDLE:
            error_logging(
                "User", user_id, 'is not idle upon connection establishment(', self.__class__.__name__, ')')
            self.user_reject(user_id, result='fail')
            return

        if (last_known_cell := self.registered_users[user_id]['last_known_cell']) != bts_id:
            freq = self.registered_users[user_id]['frequency']
            self.detector_update(
                (last_known_cell, bts_id, freq, freq, self.steps, user_id))
        self.spread_user_update(bts_id)
        self.registered_users[user_id] |= {'state': CONNECTED}
        self.registered_users[user_id] |= {'serving_cell': bts_id}
        self._send_message_to_user(
            user_id, {'connection_setup': 1} | self._setup_connection(self, user_id, bts_id))
        return

    @_user_locked
    def measurement_receive(self, user_id, report):
        return self._measurement_receive_inner(user_id, report)

    def _measurement_receive_inner(self, user_id, report):
        reply = self._measurement_receive(self, user_id, report)
        reply = {key: item for key, item in reply.items() if item}
        self._send_message_to_user(user_id, reply)
        return 'handover' in reply or 'disconnect' in reply

    @_user_locked
    def user_release(self, user_id):
        self._user_release_inner(user_id)

    def _user_release_inner(self, user_id):
        if not self.registered_users[user_id] or not self.registered_users[user_id]['state'] == CONNECTED:
            error_logging(
                f'User {user_id} is not connected upon release(', self.__class__.__name__, ')')
        else:
            self.registered_users[user_id]['state'] = IDLE
            try:
                serving_cell = self.registered_users[user_id]['serving_cell']
            except NameError:
                serving_cell = self.registered_users[user_id]['last_known_cell']
            self.spread_user_update(serving_cell, entering=False)
            self.set_user_record_key(user_id, 'last_known_cell', serving_cell)
            self.set_user_record_key(user_id, 'serving_cell', -1)
            self._send_message_to_user(user_id,
                                       {
                                           'system_information': self.compute_system_information(None, self.registered_users[user_id]['frequency'])
                                       })

    @_user_locked
    def user_handover(self, user_id, new_bts_id):
        return self._user_handover_inner(user_id, new_bts_id)

    def _user_handover_inner(self, user_id, new_bts_id):
        if not self.registered_users[user_id] or not self.registered_users[user_id]['state'] is CONNECTED:
            error_logging(
                "User", user_id, 'is not connected on handover(', self.__class__.__name__, ')')
        else:
            self.detector_update((self.registered_users[user_id]['serving_cell'], new_bts_id,
                                 freq := self.registered_users[user_id]['frequency'], freq, self.steps, user_id))
            self.registered_users[user_id].update({'serving_cell': new_bts_id})
            self._send_message_to_user(
                user_id, self._setup_connection(self, user_id, new_bts_id))
        return

    def _send_paging_to_users(self, page_list):
        """
        Send paging requests to users in a list. Used to make users switch to the connected state.
        """
        epc_logging('Paging', page_list)
        for user_id, paging_time in page_list:
            self.downlink_queues[user_id].put(
                {'paging_request': {'time': paging_time}})

    def _send_message_to_user(self, user_id, msg):
        """
        Send a message to a user by placing it in the user's message queue
        """
        epc_logging('EPC sent to', user_id, json.dumps(msg, indent=2))
        self.downlink_queues[user_id].put(msg)

    def raise_error(self, *msg):
        self.error_log.put(''.join(msg))

    def set_user_record_key(self, user_id, key, value):
        try:
            self.registered_users[user_id][key] = value
        except Exception:
            self.raise_error(f'Error in setting record for user {user_id}')

    def generate_handover_graph(self):
        """
        Generates a distance matrix and a list of neighbours for each station. Used to give the user a set of towers to measure and possibly handover to
        """
        distance_matrix = [[(stat_a.identifier, stat_a.position.distance(
            stat_b.position)) for stat_a in self.stations] for stat_b in self.stations]
        debug("Creating handover graph")
        for l in distance_matrix:
            l.sort(key=snd)
            l = [rec for rec in l if rec[1] < NEIGHBOUR_DISTANCE]
            index, _ = find_on(
                self.stations, lambda s: s.identifier == l[0][0])
            self.stations[index].neighbourhood = [x[0]
                                                  for x in l]  # l[0] is the entry for the station itself
        return

    @staticmethod
    def expand_system_information(conf):
        system_information_base = {key: item for key, item in conf.items() if key in [
            'q_hyst', 'mobility', 'q_hyst_sf']}
        try:
            system_information_list = {rec['freq']:
                                       {
                'intra_freq': {
                    's_non_intra_search': rec['s_non_intra_search'],
                    'thresh_serving_low': rec['thresh_serving_low'],
                    'cell_reselection_priority': rec['cell_reselection_priority_intra'],

                    'q_rx_lev_min': rec['q_rx_lev_min_intra'],
                    'p_max': rec['p_max'],
                    's_intra_search': rec['s_intra_search'],
                    't_reselection': rec['t_reselection'],
                    't_reselection_sf': rec['t_reselection_sf'],
                },
                'inter_freq': {
                    'q_rx_lev_min': rec['q_rx_lev_min_inter'],
                    'p_max': rec['p_max'],
                    'thresh_high': rec['thresh_high'],
                    'thresh_low': rec['thresh_low'],
                    'cell_reselection_priority': rec['cell_reselection_priority_inter'],
                    'q_offset_freq': rec['q_offset_freq'],
                    'neigh_cell_list': rec['neigh_cell_list'],
                    'black_cell_list': rec['black_cell_list']
                }
            } for rec in conf['freq_list']}
        except KeyError as key_error:
            epc_logging(f'SIB configuration does not implement {key_error}!')
            return None, None
        return system_information_base, system_information_list

    def compute_system_information(self, bts_id, freq):
        if not freq:
            return self.system_information_base | {
                'inter_freq': {
                    other_freq: item['inter_freq'] for other_freq, item in self.system_information_list.items()
                }
            }
        if (selected_sib := self.system_information_list.get(freq, None)) is None:
            error_logging(
                f'There is no sib to compute for frequency {freq} ({self.__class__.__name__})')
            return None
        sib = self.system_information_base | {
            'intra_freq': selected_sib['intra_freq'],
            'inter_freq': {
                other_freq: item['inter_freq'] for other_freq, item in self.system_information_list.items() if other_freq != freq
            }
        }
        if (bts_id is not None) and (barring_info := self.barring_check(bts_id))[0]:
            barring_factor, barring_time = barring_info
            sib |= {'ac_barring_factor': barring_factor,
                    'ac_barring_time': barring_time}
        return sib

    @staticmethod
    def merge_setups(a, b):
        return {
            'meas_to_add_mod': a.get('meas_to_add_mod', {}) + b.get('meas_to_add_mod', {}),
            'meas_to_remove': a.get('meas_to_remove', {}) + b.get('meas_to_remove', {})
        }

    @staticmethod
    def generate_formula(fstring, _locals):
        def f(signal):
            res = eval(fstring, globals(), {**_locals, **{'signal': signal}})
            return res
        return f

    def expand_message_strings(self, msg, weakness, serving_signal, user_id, _locals):
        for i, it in enumerate(msg.get('meas_to_add_mod', [])):
            for key, item in it.get('parameters', {}).items():
                if key == 'frequency' and item == -1:
                    # Frequency = -1 is interpreted as measuring only the current frequency
                    msg['msg_to_add_mod'][i]['parameters']['frequency'] = self.registered_users[user_id]['frequency']
                if isinstance(item, str):
                    msg['meas_to_add_mod'][i]['parameters'][key] = eval(
                        item, globals(), _locals | {'weakness': weakness, 'signal': serving_signal})
        return msg

    def expand_message_targets(self, reply, user_id, bts_id):
        ret = {'meas_to_remove': reply.get('meas_to_remove', [])}
        serving_frequency = self.registered_users[user_id]['frequency']

        def expand_meas_config(conf):
            if (freq := conf.get('frequency')) == 'intra':
                return [conf | {'frequency': serving_frequency}]
            elif freq == 'inter':
                return [conf | {'frequency': frequency} for frequency in self.frequencies if frequency != serving_frequency]
            else:
                return [conf]
        ret['meas_to_add_mod'] = flatten(
            [expand_meas_config(conf) for conf in reply['meas_to_add_mod']])
        return ret

    def update_neighbour_results(self, user_id, neigh_results, frequency):
        debug(
            f'updating neighbour results {self.registered_users[user_id].get("neighbour_results")} with {neigh_results}')
        if not 'neighbour_results' in self.registered_users[user_id]:
            self.registered_users[user_id]['neighbour_results'] = [
                (*neigh_result, frequency, self.steps) for neigh_result in neigh_results]
            return
        l = self.registered_users[user_id]['neighbour_results'].copy()
        remove_on(l, lambda x: self.steps-x[-1] > self.keep_neighbour_results)
        l += [(*neigh_result, frequency, self.steps)
              for neigh_result in neigh_results]
        self.registered_users[user_id]['neighbour_results'] = l

    def parse_meas_setup(self, setup):
        self.INITIAL_A1_THRESHOLD = setup.get('initial_a1_threshold')
        self.INITIAL_A2_THRESHOLD = setup.get('initial_a2_threshold')
        self.INITIAL_A3_OFFSET = setup.get('initial_a3_offset')
        self.INITIAL_A4_THRESHOLD = setup.get('initial_a4_threshold')
        self.INITIAL_A5_THRESHOLD1 = setup.get('initial_a5_threshold1')
        self.INITIAL_A5_THRESHOLD2 = setup.get('initial_a5_threshold2')
        self.MAX_STORED_NEIGHBOURS = setup.get('max_stored_neighbours', 0)
        self.SIGNAL_WORSENING_STEPSIZE = setup.get(
            'signal_worsening_stepsize', SIGNAL_WORSENING_STEPSIZE)
        _locals = {
            'INITIAL_A1_THRESHOLD': self.INITIAL_A1_THRESHOLD,
            'INITIAL_A2_THRESHOLD': self.INITIAL_A2_THRESHOLD,
            'INITIAL_A3_OFFSET': self.INITIAL_A3_OFFSET,
            'INITIAL_A4_THRESHOLD': self.INITIAL_A4_THRESHOLD,
            'INITIAL_A5_THRESHOLD1': self.INITIAL_A5_THRESHOLD1,
            'INITIAL_A5_THRESHOLD2': self.INITIAL_A5_THRESHOLD2,
            'SIGNAL_WORSENING_STEPSIZE': self.SIGNAL_WORSENING_STEPSIZE
        }

        self.weakness_formula = self.generate_formula(
            setup.get('weakness_formula', '0'), _locals)

        def meas_initial(self, user_id, bts_id):
            reply = setup['initial']
            for i, it in enumerate(reply):
                for key, item in it.get('parameters', {}).items():
                    if isinstance(item, str):
                        reply[i]['parameters'][key] = eval(
                            item, globals(), _locals)
            return self.expand_message_targets({
                'meas_to_add_mod': reply
            }, user_id, bts_id)

        def meas_reply(self, user_id, report):
            def set_record_key(key, value):
                self.set_user_record_key(user_id, key, value)
            if serving_signal := report.get('result_serving', 0):
                set_record_key("serving_signal", serving_signal)
                weakness = self.weakness_formula(serving_signal)
                set_record_key("connection_weakness", weakness)
            reply = {
                'meas_to_add_mod': [],
                'meas_to_remove': []
            }
            if self.print_status:
                epc_logging("status: ", json.dumps(
                    self.registered_users[user_id], indent=2))
            if serving_signal:
                for conditional in setup['conditionals']:
                    if eval(conditional['condition'], globals(), {'weakness': weakness}):
                        # If we disconnect we will not set up any measurements
                        if conditional.get('disconnect'):
                            self._user_release_inner(user_id)
                            return {
                                'disconnect': 1
                            }
                        reply_append = {
                            'meas_to_add_mod': deepcopy(conditional.get('meas_to_add_mod', [])),
                            'meas_to_remove': uniques(deepcopy(conditional.copy().get('meas_to_remove', [])))
                        }
                        reply_append = self.expand_message_strings(
                            reply_append, weakness, serving_signal, user_id, _locals)
                        if conditional.get('expand_targets', 0):
                            try:
                                serving_cell = self.registered_users[user_id]['serving_cell']
                            except (KeyError, TypeError):
                                # This is a NoneType error
                                error_logging(
                                    'Could not access serving cell of user', user_id, '(', self.__class__.__name__, ')')
                                error_logging(json.dumps(
                                    self.registered_users[user_id]))
                                return {}
                            reply_append = self.expand_message_targets(
                                reply_append, user_id, serving_cell)
                        reply = self.merge_setups(reply, reply_append)
            else:
                serving_signal = self.registered_users[user_id]['serving_signal']
                weakness = self.weakness_formula(serving_signal)
            if result_neighbours := report.get('result_neighbours', False):
                self.update_neighbour_results(
                    user_id, result_neighbours, frequency=report.get('frequency', 0))
            if (eval(setup['handover']['condition'], globals(), {'weakness': weakness, 'serving_signal': serving_signal}) or self.registered_users[user_id].get('should_spread')) \
                    and (res_neighbours := self.registered_users[user_id].get('neighbour_results', [])):
                epc_logging("Attempting handover of user", user_id, "with neighbour results",
                            res_neighbours, 'and serving signal', serving_signal)
                serving_cell = self.registered_users[user_id]['serving_cell']
                debug(self.station_ids)
                for res_neighbour in sorted(res_neighbours, key=lambda x: x[1]):
                    if res_neighbour[0] == serving_cell:
                        break
                    if res_neighbour[0] in self.station_ids \
                            and (eval(setup['handover']['secondary'], globals(), {'weakness': weakness, 'serving_signal': serving_signal, 'neighbour_signal': res_neighbour[1]})):
                        self.spread_user_update(
                            self.registered_users[user_id]['serving_cell'], False)
                        self.spread_user_update(res_neighbour[0], True)
                        set_record_key('serving_cell', res_neighbour[0])
                        set_record_key('neighbour_results', [])
                        set_record_key('frequency', res_neighbour[2])
                        set_record_key('should_spread', False)
                        return {
                            'handover': {
                                'bts_id': res_neighbour[0],
                                'frequency': res_neighbour[2]
                            },
                            "meas_to_add_mod": setup['initial']
                        }
            return reply
        return meas_reply, meas_initial

#######################################################
# The child process of the core network. Reads messages from the queue as the main process would, but does not page users.
#######################################################


class TrackingAreaChild(TrackingArea):
    """Allows the work of the tracking area to be spread

    This can be detrimental for performance as we are forced to keep a shared user database when we use this.
    Shard (proxy) lists are hideously slow, so this is only a speedup if the actions taken with the user data are slow.
    """
    steps = 0

    def __init__(self, parent, registered_users, paged_users, child_id, meas_setup, config, detectors=[], buffersize=5, freq_users=[]):
        self.tac = parent.tac
        self.stations = parent.stations
        self.station_ids = parent.station_ids
        self.user_ids = parent.user_ids
        # Messaging setup downlink: the core network (i.e. the tracking area) has a queue of messages for each user, which the user has access to.
        self.downlink_queues = parent.downlink_queues
        # Messaging setup uplink: users sending messages to the TA is handled through this queue. The user thus does not need a reference to the TA, only access to the queue.
        self.uplink_queue = parent.uplink_queue
        # In the current state there is one main tracking area handling all incoming messages; messages for unknown towers (i.e. the attacker) are put into the attacker_queue
        self.attacker_queue = parent.attacker_queue
        self.error_log = parent.error_log
        self.registered_users = parent.registered_users
        self.total_users = parent.total_users
        self.events = parent.events
        self.max_steps = parent.max_steps
        self.reveal_network_records = parent.reveal_network_records
        self.max_users_frequencies = parent.max_users_frequencies
        self.paged_users = paged_users
        # This is used to prevent race conditions where the TA keeps sending paging requests before the users have had a chance to respond.
        self._measurement_receive, self._setup_connection = self.parse_meas_setup(
            meas_setup)
        self.user_lock = parent.user_lock
        self.log_user = parent.log_user
        self.child_id = child_id
        self.print_status = parent.print_status
        self.attacker_queue = DefaultObject()
        self.freq_users = freq_users
        if (sib_filename := config.get('sib_filename')):
            with open(sib_filename, 'r') as f:
                self.system_information_base, self.system_information_list = self.expand_system_information(
                    json.load(f))
        else:
            self.system_information_base, self.system_information_list = None, None
        # We only send messages to the detector if it wants that type of message

        def detector_wants_dependent(key):
            def inner(msg):
                for detector_wants, detector_queue in detectors:
                    if key in detector_wants:
                        detector_queue.put(msg)
            return inner
        self.detector_register = detector_wants_dependent(
            'connection_register')
        self.detector_measurement = detector_wants_dependent(
            'measurement_report')
        self.detector_leave = detector_wants_dependent('connection_leave')
        self.detector_update = detector_wants_dependent('update')

        self.detector_activate_time = config.get(
            'detectors', {}).get('activate_after', float('inf'))
        self.detector_wait_time = config.get(
            'detectors', {}).get('wait_time', 100)

        # We would ideally take equal shares of the queue, but there is no way to know the total size of the work once stuff is taken out
        self.buffersize = buffersize

    @_user_locked
    def user_register_idle(self, user_id, cell_id, plmn_id, freq):
        return self._user_register_idle_inner(user_id, cell_id, plmn_id, freq)

    @_user_locked
    def user_leave(self, user_id):
        return self._user_leave_inner(user_id)

    @_user_locked
    def user_setup_connection(self, user_id, bts_id):
        return self._user_setup_connection_inner(user_id, bts_id)

    @_user_locked
    def measurement_receive(self, user_id, report):
        return self._measurement_receive_inner(user_id, report)

    @_user_locked
    def user_release(self, user_id):
        self._user_release_inner(user_id)

    @_user_locked
    def user_handover(self, user_id, new_bts_id):
        return self._user_handover_inner(user_id, new_bts_id)

    def run(self):
        listen()
        while not self.events['done'].is_set():
            self.sim_step()
            if self.wait_for_barrier(child=True):
                break
        return

    def sim_step(self):
        buffer = []
        nr_in_buffer = 0
        while nr_in_buffer < self.buffersize:
            try:
                buffer.append(self.uplink_queue.get_nowait())
                nr_in_buffer += 1
            except (TimeoutError, Empty):
                break
        for msg in buffer:
            self._digest_message(msg)
        self.steps += 1


#######################################################
# The tracking area for the attacker.
# Just having a station for the attacker will not suffice, as we want it to more aggressively lure in users.
#######################################################

ATTACKER_REJECT, ATTACKER_KEEP, ATTACKER_HANG = 1, 2, 3


class AttackerTrackingArea(TrackingArea):
    """
    A tracking area model for the attacker. Does essentially the same as the 'real' tracking area, but sends different messages
    The uplink_queue for the attacker is the attacker_queue of the main TA; the attacker thus does not handle messages for the honest stations
    """
    active = False

    def __init__(self, *args, user_connect_event=None, user_connect_hook=void, action=ATTACKER_REJECT, reject_result="block", **kwargs):
        self.user_connect_event = user_connect_event
        self.user_connect_hook = user_connect_hook
        if action == ATTACKER_REJECT:
            info_logging('Attacker is rejecting')
            self._digest_message = self.rejecting_digest_message
        elif action == ATTACKER_HANG:
            info_logging('Attacker is letting users hang')
            self._digest_message = void
        else:
            info_logging('Attacker is accepting users')
        self.reject_result = reject_result
        super().__init__(*args, **kwargs)

    def run(self):
        listen()
        while not self.events['done'].is_set():
            self.wait_for_barrier()
            if self.active and not self.events['attacker_event'].is_set():
                self.teardown()
                self.active = False
            elif not self.active and self.events['attacker_event'].is_set():
                self.active = True
            elif self.active:
                self.sim_step()
            self.steps += 1
        return

    def _send_message_to_user(self, user_id, msg):
        """
        Send a message to a user by placing it in the user's message queue
        """
        epc_logging('Attacker sent to', user_id, json.dumps(msg, indent=2))
        self.downlink_queues[user_id].put(msg | {'attack': True})

    def halt_everything_hook(self, *args):
        self.events['done'].set()

    def teardown(self):
        """
        Called when the attack is stopped. All users are kicked from this TA
        """
        for rec in filter(bool, self.registered_users):
            self._send_message_to_user(rec['id'], {'disconnect': 1})
        self.registered_users = [None for _ in self.registered_users]

    def rejecting_digest_message(self, msg):
        epc_logging('(AttackerTrackingArea)', json.dumps(msg, indent=2))
        user_id = msg['user_id']
        if 'connection_request' in msg:
            # We want users to connect so they send us their tracking area update
            epc_logging('Attacker sent connection setup to user', user_id)
            self._send_message_to_user(user_id, {'connection_setup': 1})
        elif 'request_ta_info' in msg:
            self._send_message_to_user(
                user_id, {'tracking_area_info': self.ta_info(0)})
        elif 'connection_hop' in msg:
            self.user_hop(user_id, msg['serving_cell'], msg['frequency'])
        elif 'tracking_area_update' in msg:
            epc_logging(
                'Attacker sent tracking area update reject to user', user_id)
            self._send_message_to_user(
                user_id, {'tracking_area_update_reject': 1, 'tac': self.stations[0].tac})
        elif 'connection_release' in msg:
            pass
        else:
            epc_logging('Attacker sent reject to user after getting', msg)
            self.user_reject(user_id, result=self.reject_result)

    def generate_handover_graph(self):
        return list()


#######################################################
# Message setup
#######################################################
"""
downlink:


{
    disconnect
    handover:
        target
    paging_request:
        time
    measurement_config:[
        # A loose measurement_config is interpreted as a connection_reconfig, and the events should be added to the current measurement_config. 
        # A measurement_config attached to a handover overrides the old measurement_config
        event_id
        parameters:
            offset?
            threshold?
            threshold1?
            threshold2?
            target_bts?
    ]
    system_information
}

uplink:

{
    user_id
    timestamp
    serving_cell
    connection_setup
    connection_release
    connection_leave
    connection_register
    connection_hop
    measurement_report:[
        event_id
        result_serving
        ?result_neighbours (if not A1 or A2)
    ]
}

"""
