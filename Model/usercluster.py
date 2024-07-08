# coding: utf-8
from os import error
import sys
import os.path

from .user import *
from .epc import REPORT_AMOUNTS,REPORT_INTERVALS,DUMMY_TAC
from .util import snd,fst,DefaultObject,Timer,listen
from threading import BrokenBarrierError
from functools import reduce
from numbers import Number

import time
import numpy as np
import json
import traceback

from .measurementReport import *

"""
Code for the user part of the simulation.
Multiple users run in one cluster to enable multiprocessing without giving users their own process.
"""
LOGGING_WAIT_TIME = 5

LOG_USER1 = False

PENDING_UPDATE_TOLERANCE=30
REGISTER_PENDING_WAITING_TIME = 10
CONN_PENDING_WAITING_TIME = 10
P_POWER_CLASS=23#See ts 36.101 p111

TA_UPDATE_NORMAL,TA_UPDATE_REJECT = 1,2
TA_UPDATE_TIME = 75

class NoTAInfoException(Exception):
    ...

class UserCluster():
    actions_after_steps:dict={}
    def __init__(self, id_, cmap, report_queue, attack_queue, events, known_stations=None, error_queue=None, nr_steps=np.inf,detectors=[]):
        self.cmap = cmap
        self.id = id_
        self.report_queue = report_queue
        self.attack_queue = attack_queue
        self.error_queue = error_queue
        self.events = events
        self.nr_steps = nr_steps
        if not known_stations:
            self.known_stations = [stat.identifier for stat in self.cmap._callmethod('__getattribute__',('stations',))] + [-1]
        else:
            self.known_stations = known_stations + [-1]
        self.users = []
        def detector_report(keyword):
            def detector_report_inner(msg):
                for detector_wants,detector_queue in detectors:
                    if keyword in detector_wants:
                        detector_queue.put(msg)    
            return detector_report_inner
        self.detector_report_reject = detector_report('reject')
        self.detector_report_ta_reject = detector_report('ta_reject')
        self.detector_loc_update = detector_report('loc_update')
        self.detector_bts_update = detector_report('bts_update')

    def run(self):
        listen()
        self.start_time = time.time()
        steps =0
        for u in self.users:
            u.sim_init(self.start_time)
        while (not self.events['done'].is_set()) and steps <= self.nr_steps:
            self.events['update'].wait()
            crashed_users = []
            for i,u in enumerate(self.users):
                try:
                    u.update_step()
                except Exception as e:
                    u.bail(e)
                    u.sim_exit()
                    crashed_users.append(i)
            for i in crashed_users[::-1]:
                self.users.pop(i)
            try:
                self.events['barrier'].wait()
            except BrokenBarrierError:
                if steps < self.nr_steps:
                    self.raise_error(f'Broken barrier in cluster after step {steps}')
                break
            steps +=1
            if steps in self.actions_after_steps:
                (self.actions_after_steps[steps])(self)
        for u in self.users:
            u.sim_exit()
        return

    def report(self, record):
        if record.bts_id in self.known_stations:
            self.report_queue.put(record)
        else:
            self.attack_queue.put(record)

    def raise_error(self, err_object):
        if not self.error_queue:
            pass
        else:
            self.error_queue.put(err_object)

    def update_cmap(self):
        """
        Updates the connection map. 
        """
        self.cmap = self.cmap._getvalue()
            

    def do_after_step(self,step_nr,action):
        self.actions_after_steps[step_nr]=action

    

class ClusterUser(User):
    """
    An extionsion of the user class to be used in a cluster. The purpose of this is that clusters can run in a separate process, as creating a process for every user is not practical
    """

    def __init__(self, user_id, cluster, plmn_id=0,position=None, protocol_realism='none', downlink_queue=None, uplink_queue=None, return_queue=None, logging_folder="userlogs",seed=None,config={},eci_map={}):
        super().init_super(seed = seed+user_id,n=1000)
        self.tracking_area_info:dict={}
        self.barred_cells:list = []
        self.unconnected_system_information:dict={}
        self.frequency_priorities = [] #List of frequencies and their priorities for idle reseletionc. Format: (priority,frequency,is_serving)
        self.uplink_buffer = []
        self.id = user_id
        self.plmn_id = plmn_id
        self.return_queue = return_queue
        self.cluster = cluster
        # Movement initialization
        self.protocol_realism = protocol_realism
        self.downlink_queue = downlink_queue
        self.map_mode = self.cluster.cmap._callmethod('__getattribute__',("mode",))
        self.map_bounds = self.cluster.cmap._callmethod('__getattribute__',("map_bounds",))
        self.grid_size = self.cluster.cmap._callmethod('__getattribute__',("grid_size",))
        self.uplink_queue = uplink_queue
        self.logging_folder = logging_folder
        self.log_user = config.get('verbosity',{}).get('log_user',-1)
        
        #This sets up the movement state and such.
        self.setup()
        self.steps = 0
        if position:
            self.position = Point(*position)
            self.x = position[0]
            self.y = position[1]

        self.pos_init = self.position

        self.ping = config.get('ping',False)
        self.bad_connection_threshold = config.get("bad_connection_threshold",BAD_CONNECTION_THRESHOLD)
        tau_behavior = config.get('tau_behavior',{})
        if not isinstance((ta_behavior_attack:=tau_behavior.get('attack','dummy')),str):
            self.tau_behavior_attack = self.uniform_choice_distribution(ta_behavior_attack)
        else:
            self.tau_behavior_attack = ta_behavior_attack #'dummy','real',or False
        if not isinstance((ta_behavior_normal:=tau_behavior.get('normal',False)),bool):
            self.tau_behavior_normal = self.uniform_choice_distribution(ta_behavior_normal)
        else:
            self.tau_behavior_normal = ta_behavior_normal#True (send true value) or False (send dummy)

        #These are older timers not working in the Timer framework
        self.behaviour2_time = config.get("behaviour2_time",BEHAVIOUR2_TIME)
        self.auto_connect_time = config.get('auto_connect_time',6)

        #These timers work with steps not time as the steps per second can vary wildly depending on the setup. We set them up here already.
        self.unconnected_timer = Timer(config.get("unconnected_waiting_time",UNCONNECTED_WAITING_TIME))
        self.reselection_timer = Timer(config.get("cell_reselection_waiting_time",CELL_RESELECTION_WAITING_TIME))
        self.paging_check_time = config.get('paging_check_time',PAGING_CHECK_WAITING_TIME)
        self.paging_timer = Timer(self.paging_check_time)
        self.idle_check_timer = Timer(round(config.get('idle_check_time',2**30-1)*self.randrange(0.5,1.5)))
        self.register_pending_timer = Timer(config.get('register_pending_time',REGISTER_PENDING_WAITING_TIME))
        self.conn_pending_timer = Timer(config.get('conn_pending_waiting_time',CONN_PENDING_WAITING_TIME))
        self.loc_update_timer = Timer(loc_update_time:=config.get('loc_update_waiting_time',float('inf')))
        self.loc_update_timer.set_value(self.randrange(0,loc_update_time))
        self.bts_update_timer =Timer(config.get('bts_update_waiting_time',float('inf')))
        self.cell_reselection_timer = Timer(CELL_RESELECTION_WAITING_TIME)
        self.logging_timer = Timer(LOGGING_WAIT_TIME)
        self.tau_timer = Timer(TA_UPDATE_TIME)
        self.tau_timer.set_value(float('inf'))

        #Idle (re)selection parameters
        self.idle_good_signal_no_meas = config.get("idle_good_signal_no_meas",False)#see page 20 of ts 36.304; if signal is high enough the UE may choose no to perform measurements (this indicates that choice)
        self.set_default_system_information()

        self.fault=0
        self.sent_ta_info_request=False
        self.will_send_ta_update = False
        self.under_attack = False
        self.prev_tracking_area = -1 #We don't want to get a spike of dummy tacs in the first few seconds, because that is a simulation relic

        self.current_signal = -140

        self.eci_map=eci_map

        #Layer 3 filtering
        self.layer_3_last_filtered =0
        self.layer_3_coefficient = 0

        self.current_ta = None
        self.steps = 0
        
        #This makes sure we don't report the first selection. Most initial selections in the model are not on the edge of coverage, so we don't want to see them.
        self.new = True 

    def sim_init(self, start_time):
        #These timers work with time not steps, so we initialize them only at this point
        self.start_time=start_time
        self.logging_timer.set_value(start_time)
        self.unconnected_timer.immediate()
        with open(os.path.join(self.logging_folder, "userlog" + str(self.id)), 'w') as f:
            f.write(str(start_time) + '\n')
        #self.start_up_cell_selection(start_time)
        if self.id == self.log_user:
            user_logging(f'User {self.id} is logging now!')
        

    def sim_exit(self):
        self.write_logs_to_file()
        if self.return_queue:
            self.return_queue.put(
                (self.id, self.location_history, self.signal_history))

    def update_step(self):
        # pylint: disable=no-member
        self.steps +=1
        self.update_location(self.steps, ping=self.ping)
        if self.state.is_unconnected and self.unconnected_timer.check_and_set(self.steps):
            self.check_downlink(self.steps)
            self.cell_selection(self.steps)
        elif self.state.is_register_pending:
            if self.register_pending_timer.check(self.steps):
                user_logging('(',self.id,') timed out for registration at step',self.steps)
                self.register_failed(self.steps)
            self.check_downlink(self.steps)
        elif self.state.is_rrc_idle:
            paging_result = False
            if self.paging_timer.check_and_set(self.steps):
                paging_result = self.check_downlink(self.steps)
            if not paging_result:
                reselect_output = None
                if self.reselection_timer.check_and_set(self.steps):
                    debug(reselect_output:=self.cell_reselection(self.steps))
                    #It can occur that reselection puts us in the pending state for a different TA
                if reselect_output != 'leave' and self.idle_check_timer.check_and_set(self.steps) and self.state.is_rrc_idle:
                    self.connection_request(self.steps,self.auto_connect_time)
        elif self.state.is_connected_pending:
            res = self.check_downlink(self.steps)
            if not res and self.conn_pending_timer.check(self.steps):
                self.connection_failed(self.steps)
                self.cell_reselection(self.steps)
        elif self.state.is_rrc_connected:
            if self.paging_timer.check(self.steps):
                self.connection_release(self.steps)
            else:
                self.check_downlink(self.steps)
                if measurements := self.check_measurements(self.steps):
                    self.send_uplink({'measurement_report': measurements})
        if self.logging_timer.check_and_set(self.steps):
            self.write_logs_to_file()
        if self.loc_update_timer.check_and_set(self.steps):
            self.loc_update()
        if self.bts_update_timer.check_and_set(self.steps):
            self.bts_update()
        self.flush_uplink(self.steps)
        if self.tau_timer.check(self.steps):
            self.tau_timeout()
    
    def bail(self,e:Exception,received_message=None):
        self.write_logs_to_file()
        if isinstance(e,NoTAInfoException):
            error_logging(f'User {self.id} did not get TA info and bails!')
            return
        error_logging(e)
        print(traceback.format_exc())
        error_logging('user id',self.id)
        error_logging('steps',self.steps)
        error_logging('serving cell',self.serving_cell)
        error_logging('system information',json.dumps(self.system_information,indent=2))
        if self.measurement_config:
            error_logging('measurement config',json.dumps(self.measurement_config,indent=2))
        error_logging('signal strengths', self.scan_connections())
        if received_message is not None:
            error_logging(json.dumps(received_message,indent=2))
        self.cluster.events['done'].set()
        return

    def send_uplink(self,msg):
        self.uplink_buffer.append(msg)

    def flush_uplink(self, timestamp):
        if not self.uplink_buffer:
            return
        uplink_msg = {
            'user_id': self.id,
            'timestamp': timestamp,
            'steps':self.steps,
            'serving_cell': self.serving_cell,
            'frequency':self.system_information['freq']
        }
        for msg in self.uplink_buffer:
            uplink_msg.update(msg)
        if self.id == self.log_user:
            user_logging(f"Sent (step {self.steps}): "+ json.dumps(uplink_msg,indent=4))
        self.uplink_queue.put(uplink_msg)
        self.uplink_buffer = []

    def check_downlink(self, timestamp):
        """
        Checks the downlink connection and handles the messages in it. 
        Returns true if state update is triggered, to prevent other actions happening
        """
        # pylint: disable=no-member
        msgs = flush_queue(self.downlink_queue,timeout=1)
        if self.under_attack:
            msgs = filter(lambda msg: 'attack' in msg,msgs)
        #else:
        #    msgs = filter(lambda msg: not 'attack' in msg,msgs)
        for msg in msgs:
            debug('Digesting',msg)
            try:
                if self.id == self.log_user:
                    user_logging(f"Received (step {self.steps}): "+ json.dumps(msg,indent=4))
                if 'connection_reject' in msg:
                    user_logging('(',self.steps,')',self.id,'got rejected in state',self.state.current_state.identifier)
                    self.handle_connection_reject(timestamp,msg['connection_reject'])
                    return True
                if 'system_information' in msg:
                    self.handle_system_information(msg['system_information'])
                    if 'intra_freq' in self.system_information:   
                        self.change_frequency_priority()
                if 'tracking_area_info' in msg:
                    self.handle_tracking_area_info(msg['tracking_area_info'])
                if self.state.is_rrc_idle or self.state.is_register_pending:
                    if 'paging_request' in msg:
                        self.connection_request(timestamp,msg['paging_request']['time'])
                        return True
                    elif 'register_complete' in msg:
                        self.handle_register_complete(msg)
                        return True
                    elif 'disconnect' in msg:
                        self.state.idle_to_unconnected()
                        self.serving_cell = -1
                        return True
                if self.state.is_connected_pending:
                    if 'meas_to_add_mod' in msg:
                        self.add_mod_meas_configs(msg['meas_to_add_mod'])
                    if 'meas_to_remove' in msg:    
                        self.remove_meas_configs(msg['meas_to_remove'])
                    if 'connection_setup' in msg:
                        self.handle_connection_setup()
                        return True
                elif self.state.is_rrc_connected:
                    if 'disconnect' in msg:
                        self.serving_cell = -1
                        self.state.connected_to_unconnected()
                        if self.system_information['freq'] in self.system_information['inter_freq']:
                            self.send_uplink({'connection_hop':1})
                        self.measurement_config = []
                        self.idle_check_timer.set_value(self.steps)
                        self.cluster.report(self.generate_measurement_report(
                            [], timestamp, event='disconnect', meas_type='connected'))
                        return True
                    elif 'handover' in msg:
                        self.serving_cell = msg['handover']['bts_id']
                        self.system_information['freq'] = msg['handover']['frequency']
                        self.measurement_config = []
                        self.add_mod_meas_configs(msg['meas_to_add_mod'])
                        self.cluster.report(self.generate_measurement_report(
                            [self.serving_cell,msg['handover']['bts_id']], timestamp, event='handover', meas_type='connected'))
                    elif 'paging_request' in msg:
                        self.paging_timer.set_value(timestamp)
                        self.call_time = msg['paging_request']['time']
                    elif 'tracking_area_update_reject' in msg:
                        self.handle_tracking_area_update_reject(severe=msg['tracking_area_update_reject'],tac=msg['tac'])
                    elif 'tracking_area_update_accept' in msg:
                        self.handle_tracking_area_update_accept(msg['tac'])
                    if 'meas_to_add_mod' in msg:
                        self.add_mod_meas_configs(msg['meas_to_add_mod'])
                    if 'meas_to_remove' in msg:    
                        self.remove_meas_configs(msg['meas_to_remove'])
                if 'hysteresis' in msg:
                    self.hysteresis = msg['hysteresis']
            except Exception as e:
                self.bail(e,received_message=msg)

    def loc_update(self):
        # pylint: disable=no-member
        if self.under_attack or self.serving_cell == 100:
            user_logging('Attack prevented loc_update for user',self.id)
        elif not (self.under_attack or self.serving_cell == 100 or self.state.is_unconnected or self.state.is_register_pending):
            self.cluster.detector_loc_update((self.steps,self.id,self.x,self.y))
        else:
            user_logging('Otherwise prevented loc_update for user ',self.id,'in state',self.state.current_state)
    def bts_update(self):
        # pylint: disable=no-member
        if not (self.under_attack or self.serving_cell == 100 or self.state.is_unconnected or self.state.is_register_pending):
            self.cluster.detector_bts_update(self.serving_cell)

    def connection_request(self,timestamp,call_time):
        # pylint: disable=no-member
        if not self.state.is_rrc_idle:
            return False
        if factor:=self.system_information.get('ac_barring_factor'):
            if self.random() < factor:
                return False
        self.state.idle_to_conn_pending()
        self.conn_pending_timer.set_value(self.steps)
        self.paging_timer.set_value(self.steps)
        self.paging_timer.wait_time = call_time
        self.hysteresis = 0
        self.send_uplink({'connection_request':1})
        return True

    def add_mod_meas_configs(self, configs):
        for new_conf in configs:
            new_conf = self.translate_meas_config(new_conf.copy())
            found = False
            for i,conf in filter(lambda _: not found,enumerate(self.measurement_config)):
                if conf['meas_id'] == new_conf['meas_id']:
                    self.measurement_config[i] = new_conf
                    break
            else:
                self.measurement_config.append(new_conf)

    def remove_meas_configs(self,ids):
        for i in ids:
            self.measurement_config = delete_on(self.measurement_config,lambda r: r['meas_id']==i)


    @staticmethod
    def translate_meas_config(rec):
        rec['report_interval'] = REPORT_INTERVALS[rec.get('report_interval',0)]
        rec['report_amount'] = REPORT_AMOUNTS[rec.get('report_amount',0)]
        rec['periodical_timer'] = rec['report_interval']
        rec['max_report_cells'] = rec.get('max_report_cells',1)
        rec['hysteresis'] = rec.get('hysteresis',0)
        return rec
    
    def handle_tracking_area_update_reject(self,tac = None,severe=True):
        # pylint: disable=no-member
        if tac != self.current_ta:
            self.current_ta = tac
        if not self.state.is_rrc_connected:
            self.bail(Exception('TAU reject in non-connected state'))
        self.cluster.detector_report_ta_reject((self.steps,self.serving_cell,self.current_ta))
        self.connection_release(self.steps)
        if severe:
            if self.current_ta in self.tracking_area_info:
                self.barred_cells += self.tracking_area_info[self.current_ta]
            else:
                self.barred_cells.append(self.serving_cell)
        else:
            if self.current_ta in self.tracking_area_info:
                self.barred_cells += [(x,64) for x in self.tracking_area_info[self.current_ta]]
            else:
                self.barred_cells.append((self.serving_cell,64))
        user_logging('Barred cells:',self.barred_cells,self.id)
        self.will_send_ta_update = TA_UPDATE_REJECT
        self.reselection_timer.immediate()
        self.tau_timer.set_value(float('inf'))
        print(f"rejected({self.id}) on freq",self.system_information['freq'])
    
    def tau_timeout(self):
        if self.state.is_rrc_connected:
            self.state.connected_to_unconnected()
        elif self.state.is_connected_pending:
            self.state.conn_pending_failed()
            self.state.idle_to_unconnected()
        self.serving_cell = -1
        user_logging("TAU update timed out")
        self.tau_timer.set_value(float('inf'))
        self.system_information = self.unconnected_system_information
        self.system_information['freq'] = -5
        self.under_attack = False
        self.tracking_area_info = {}


    def handle_tracking_area_update_accept(self,tac = None):
        if tac != self.current_ta:
            user_logging('The ta update sent a different tac than expected:',self.current_ta,tac)
        self.tau_timer.set_value(float('inf'))

    def connection_failed(self,timestamp):
        self.state.conn_pending_failed()
        self.barred_cells.append((self.serving_cell,64))
        user_logging('Barred cells:',self.barred_cells,self.id)
        user_logging('(',self.id, ') Connection setup failed')

    def handle_connection_reject(self,timestamp,result):
        # pylint: disable=no-member
        debug('Got barred cell:',self.serving_cell)
        user_logging('Reporting reject')
        self.cluster.detector_report_reject((self.steps,self.serving_cell))
        self.cluster.report(self.generate_measurement_report([],timestamp,meas_type='idle',event='reject'))
        if result == 'block':
            self.barred_cells.append(self.serving_cell)
            self.tracking_area_info = {}
            user_logging('Barred cells:',self.barred_cells,self.id)
        self.serving_cell = -1
        if self.state.is_rrc_idle:
            self.state.idle_to_unconnected()
        elif self.state.is_register_pending:
            self.state.register_failed()
        elif self.state.is_rrc_connected:
            self.state.connected_to_unconnected()
        elif self.state.is_connected_pending:
            self.state.conn_pending_failed()
            self.state.idle_to_unconnected()
        self.system_information = self.unconnected_system_information
        self.system_information['freq'] = -3
        self.unconnected_timer.immediate()
        self.under_attack = False

    def handle_connection_setup(self):
        # pylint: disable=no-member
        if not self.state.is_connected_pending:
            self.bail(Exception("state was not connected_pending"))
        self.state.conn_pending_to_connected()
        self.paging_timer.set_value(self.steps)
        if self.will_send_ta_update == TA_UPDATE_REJECT:
            if self.tau_behavior_attack == 'dummy':
                tac_to_send=DUMMY_TAC
            elif self.tau_behavior_attack == 'real':
                tac_to_send = self.prev_tracking_area
            else:
                return
        elif self.will_send_ta_update == TA_UPDATE_NORMAL:
            tac_to_send = self.prev_tracking_area if self.tau_behavior_normal else DUMMY_TAC
        else:
            return
        if tac_to_send is None:
            tac_to_send = DUMMY_TAC
        self.send_uplink({'tracking_area_update':tac_to_send})
        self.tau_timer.set_value(self.steps)
        self.will_send_ta_update = False


    def handle_register_complete(self,msg):
        # pylint: disable=no-member
        self.current_ta = msg['register_complete']['tac']
        if self.state.is_register_pending:
            self.state.register_accepted()
        else:
            error_logging('(',self.steps,') User',self.id,'got a register_complete in',self.state.current_state.identifier)
            return
        if self.current_ta not in self.tracking_area_info:
            self.send_uplink({'request_ta_info':self.serving_cell})

    def handle_tracking_area_info(self,msg):
        is_attacker = msg['attacker']
        tac = msg['tac']
        stations = msg['stations']
        self.tracking_area_info[tac] = (stations,is_attacker)
        if self.serving_cell in stations:
            self.current_ta = tac
            self.under_attack = is_attacker

    def handle_system_information(self,rec):
        if 'inter_freq' in rec:
            rec['inter_freq'] = {
                int(key):item for key,item in rec['inter_freq'].items()
            }
        self.system_information |= rec
        if not self.unconnected_system_information:
            self.unconnected_system_information = deepcopy(self.system_information)
        self.cell_reselection_timer.wait_time = self.system_information.get('intra_freq',{}).get('t_reselection',CELL_RESELECTION_WAITING_TIME)

    def set_default_system_information(self):
        self.system_information = {
            "freq":0,
            "q_hyst":0,
            "intra_freq":{'q_rx_lev_min':-140,'p_max':P_POWER_CLASS,'t_reselection':CELL_RESELECTION_WAITING_TIME,'cell_reselection_priority':0},
            "inter_freq":{}
        }

    def change_frequency_priority(self):
        """
        Given the latest system information, create lists of frequencies of higher, equal, or lower priority.
        """
        freq = self.system_information['freq']
        intra_freq = self.system_information.get('intra_freq')
        if not intra_freq:
            self.bail("system_information does not contain intra-frequency part")
        intra_priority = intra_freq['cell_reselection_priority']
        #construct a list of (priority,frequency,is_serving_frequency)-tuples
        frequency_priorities = [(intra_priority,freq)]
        for inter_freq,rec in self.system_information['inter_freq'].items():
            frequency_priorities.append((rec['cell_reselection_priority'],inter_freq))
        self.frequency_priorities = {
            'higher':[x[1] for x in sorted(frequency_priorities,key=fst,reverse=True) if x[0] > intra_priority],
            'equal':[x[1] for x in frequency_priorities if x[0] == intra_priority],
            'lower':[x[1] for x in sorted(frequency_priorities,key=fst,reverse=True) if x[0] < intra_priority]
        }

    def connection_release(self,timestamp):
        """
        Connected -> Idle state transition. Is called when the call time ends and the user becomes inactive.
        """
        self.send_uplink({'connection_release': 1})
        self.state.connected_to_idle()
        self.measurement_config = []
        self.reconnection_count = 0
        self.paging_timer.wait_time = self.paging_check_time
        self.paging_timer.set_value(self.steps)
        self.idle_check_timer.set_value(self.steps)

    def register_failed(self,timestamp):
        self.state.register_failed()
        self.barred_cells.append(self.serving_cell)
        user_logging('Barred cells:',self.barred_cells,self.id)
        self.serving_cell = -1
        self.current_signal = -1
        self.register_pending_timer.set_value(self.steps)
        self.system_information = self.unconnected_system_information
        self.system_information['freq']=-1
        self.unconnected_timer.immediate()
        self.cluster.report(self.generate_measurement_report([],timestamp,meas_type='idle',event='failed'))

    def cell_reselection_ranking(self,rec):
        """
        Used for intra-frequency and equal-priority cell reselection.
        To be used as a sorting key.
        """
        eci,_,freq,meas=rec
        if freq == self.system_information['freq']:
            if 'intra_freq' not in self.system_information:
                self.bail("System information does not contain intra-frequency part during ranking")
            if eci == self.serving_cell:
                return meas + self.system_information.get('q_hyst',0)
            else:
                #Mistake: this should _not_ be the reselection priority here, but the q_offset. 
                #Since the priority is the same for all cells for which we do this this does nothing (so there is essentially no q_offset mechanism)
                return meas + self.system_information['intra_freq'].get('cell_reselection_priority')
        else:
            return meas + self.system_information['inter_freq'][freq].get('cell_reselection_priority')

    def cell_reselection(self, timestamp,_test_break=None) -> str:
        """
        Performs cell reselection as specified in TS 31.304.
        Does not do mobility state.
        
        Returns the outcome of the procedure for testing/debug purposes.

        If _test_break is set, we return at certain intermediate points to signify certain decisions being made.
        This is used in testing.
        """
        if self.system_information['freq'] <= 0:
            print(f"No frequency!!! ({self.id})")
        if not self.system_information['intra_freq']:
            return "no intra_freq"
        freq = self.system_information['freq']
        if freq in self.system_information['inter_freq']:
            #This happens when we have sent a hop, but are still waiting to receive a new SIB
            self.fault+=1
            if self.fault > PENDING_UPDATE_TOLERANCE/2:
                self.send_uplink({'connection_hop':1})
            if self.fault > PENDING_UPDATE_TOLERANCE:
                #This is arbitrary, and catches things going wrong in the model and not any real behaviour
                self.bail(Exception(f"Too many 'pending update's: {self.fault}!"))
            return "Pending update"
        self.fault=0
        #We always measure everything, and it will depend on the serving signal what we will actually evaluate
        signal_strengths = flatten(self.scan_connections(inter_freq=True))
        if self.ping:
            self.ping_location_history(self.steps, signals=deepcopy(signal_strengths))
        signal_strengths = [rec for rec in signal_strengths if not rec[0] in self.barred_cells]
        signal_strengths = sorted(signal_strengths,key=tget(3))
        self.current_signal,_ = self.find_signal(signal_strengths, self.serving_cell,freq)   
        srx_lev = self.current_signal - self.system_information['intra_freq']['q_rx_lev_min']-max(self.system_information.get('p_max',P_POWER_CLASS)-P_POWER_CLASS,0) 
        debug('signal strengths: ',signal_strengths)
        debug('current signal', self.current_signal)
        debug('serving srx_lev: ',srx_lev)
        #Higher priority frequencies are always evaluated.
        for high_priority_frequency in self.frequency_priorities['higher']:
            debug('high priority: ',high_priority_frequency)
            i,measurement= find_on(signal_strengths,lambda rec: rec[2] == high_priority_frequency)
            if i==-1:
                continue #This frequency was not received here.
            elif self.idle_evaluate_cell(timestamp,measurement,self.system_information['inter_freq'][high_priority_frequency],0,signal_strengths):
                return "higher"
        #Equal-ranked and intra-frequency procedure.
        equal_ranked_cells = [rec for rec in signal_strengths if rec[2] in self.frequency_priorities['equal']]
        debug('equal ranked cells',sorted(equal_ranked_cells,key=self.cell_reselection_ranking))
        intra_search = srx_lev <= self.system_information['intra_freq']['s_intra_search']
        for measurement in sorted(equal_ranked_cells,key=self.cell_reselection_ranking,reverse=True): 
            if measurement[2] == freq:
                if measurement[0] == self.serving_cell:
                    if not intra_search:
                        #we found the serving cell in the list, and there is no need to measure anything else
                        if _test_break == "intra":
                            return "unchanged_intra"
                    #we found the serving cell in the intrafrequency/equal-priority list. The rest of this priority level is by definition not good enough.
                    break
                if intra_search or not self.idle_good_signal_no_meas:
                    #we found an intrafrequency cell before the serving cell, and we need to do intrafrequency stuff
                    if self.idle_evaluate_cell(timestamp,measurement,self.system_information['intra_freq'],1,signal_strengths):
                        return "intra"
            elif self.idle_evaluate_cell(timestamp,measurement,self.system_information['inter_freq'][measurement[2]],2,signal_strengths):
                return "higher-ranked equal priority"
        if srx_lev < self.system_information['intra_freq']['thresh_serving_low']:
            #do lower priority measurements. If after the serving cell was found we calculated this was not necessary we should have returned already.
            for lower_priority_frequency in self.frequency_priorities['lower']:
                debug('lower priority: ',lower_priority_frequency)
                i,measurement= find_on(signal_strengths,lambda rec: rec[2] == lower_priority_frequency)
                if i==-1:
                    continue
                elif self.idle_evaluate_cell(timestamp,measurement,self.system_information['inter_freq'][lower_priority_frequency],2,signal_strengths):
                    return 'lower'
        else:
            if _test_break=='lower':
                return "unchanged_lower"
        if srx_lev <= 0:
            #Connection lost.
            self.send_uplink({'connection_leave': 1,'signal_strength':self.current_signal})
            self.cluster.report(self.generate_measurement_report(
                signal_strengths, timestamp, event='signal_lost', meas_type='idle'))
            self.serving_cell = -1
            self.current_signal = -140
            self.ta_stations = list()
            if self.state.is_rrc_idle:
                self.state.idle_to_unconnected()
            else:
                self.state.register_failed()
            self.unconnected_timer.set_value(self.steps)
            self.system_information = deepcopy(self.unconnected_system_information) 
            self.system_information['freq'] = -2
            debug(self.system_information)
            return 'leave'
        #Default case.
        return 'unchanged'

    def idle_evaluate_cell(self,timestamp,measurement,sib_entry,priority,signal_strengths) -> bool:
        """Run after the idle procedure. Chooses a viable candidate.

        Decides whether to actually make the reselection (and return True) when the appropriate threshold is passed.
        """
        station_id = measurement[0]
        debug('measurement',measurement)
        srx_lev = measurement[3] - sib_entry.get('q_rx_lev_min',-140)-max(sib_entry.get('p_max',P_POWER_CLASS)-P_POWER_CLASS,0) 
        if priority ==0:    
            threshold = sib_entry.get('thresh_high',-140) 
        elif priority ==1:
            threshold = 0
        elif priority ==2:
            threshold = sib_entry.get('thresh_low',-140)
        else:
            return False
        debug('evaluating: ',measurement,'srx_lev: ',srx_lev,'threshold: ',threshold)
        if srx_lev > threshold:
            debug(self.tracking_area_info)
            for _tac,stations in self.tracking_area_info.items():
                if station_id in stations:
                    tac = _tac
                    break
            else:
                tac = None
            debug("switching to",measurement)
            self.idle_switch_cell(timestamp, measurement[0],measurement[2],tac)
            self.cluster.report(self.generate_measurement_report(
                signal_strengths, timestamp, event='reselection', meas_type='idle'))
            debug('Switching!')
            return True
        return False

    def idle_switch_cell(self,timestamp,new_bts,new_freq,tac):
        """Run when idle reselection has found a viable candidate matching the requirements.

        """
        if self.system_information['freq'] != new_freq:
            self.system_information['freq'] = new_freq
            self.send_uplink({'connection_hop':1})
        if tac != self.current_ta or tac is None:
            #Tracking area switch. 
            #If the tracking area switch does not work we can switch back
            self.prev_tracking_area = self.current_ta
            if tac is not None:
                self.current_ta = tac
            if (self.id ==self.log_user):
                user_logging(f"Switching to other tracking area {tac}")
            #Since this point can be reached after having encountered a TA reject or upon normal reselection, 
            #we must remember to send the correct TA update. Since TA_UPDATE_ATTACK = 2, we can do this using max().
            self.will_send_ta_update = max(TA_UPDATE_NORMAL,self.will_send_ta_update)
            self.conn_pending_timer.set_value(self.steps)
            self.under_attack = self.tracking_area_info.get('tac',(0,False))[1]
            if not tac in self.tracking_area_info:
                self.send_uplink({'request_ta_info':new_bts})
            self.connection_request(timestamp,self.auto_connect_time)
        self.serving_cell = new_bts
        #Set the appropriate timer such that we will check the downlink on the next cycle.
        self.paging_timer.immediate()
                            
    def cell_selection(self, timestamp) -> bool:
        """Cell selection procedure

        Returns whether a suitable cell was found.

        Does not go for "acceptable" cells, as we don't model those.
        """
        signal_strengths = flatten(self.scan_connections())
        if self.ping:
            self.ping_location_history(self.steps, signals=signal_strengths.copy())
        signal_strengths = [rec for rec in signal_strengths if not rec[0] in self.barred_cells]
        debug('Selecting from signal strengths:',signal_strengths)
        #We sort by strength and then remove duplicate frequencies.
        signal_strengths.sort(key=tget(2),reverse=True)
        l = len(signal_strengths)
        for i,rec in enumerate(signal_strengths[::-1]):
            if rec[1] in [r[1] for r in signal_strengths[:l-i-1]]:
                signal_strengths.pop(l-i-1)
        debug('Narrowed down to:',signal_strengths)
        if not 'intra_freq' in self.system_information:
            ...
            #return False
        for new_stat in signal_strengths:
            if new_stat[2] - self.system_information['intra_freq'].get('q_rx_lev_min',-140)-max(self.system_information.get('p_max',P_POWER_CLASS)-P_POWER_CLASS,0) >0:
                #Suitable cell found
                debug("selecting",new_stat)
                self.serving_cell = new_stat[0]
                self.current_signal = new_stat[3]
                self.system_information['freq']=new_stat[2]
                self.send_uplink({
                    'connection_register': 1,
                    'plmn_id':self.plmn_id
                })
                if not self.new:
                    self.cluster.report(self.generate_measurement_report(
                        signal_strengths, timestamp, event='selection', meas_type='idle'))
                    self.will_send_ta_update = TA_UPDATE_NORMAL
                else:
                    self.new = False
                self.state.unconnected_to_pending()
                self.register_pending_timer.set_value(self.steps)
                return True
        return False

    def handle_A12(self,conf,serving_strength,neighbour_strengths,timestamp):
        """Handles A1 and A2 events

        """
        if (self.event_mapping[conf['event_id']])(serving_strength, threshold=conf['parameters']['threshold'],hysteresis=conf['hysteresis']):
            self.cluster.report(self.generate_measurement_report(
                neighbour_strengths, timestamp, event=conf['event_id'],	 meas_type='connected'))
            return [{
                'event_id': conf['event_id'],
                'result_serving': serving_strength
            }]
        return []
        
    def handle_A345(self,conf,serving_strength, neighbour_strengths,timestamp):
        """Handles A3, A4, and A5 events

        """
        records = [rec for rec in neighbour_strengths if rec[2] == conf['frequency']]
        if not records:
            return []
        results = []
        for record in records:
            if (self.event_mapping[conf['event_id']])(
                serving_strength, 
                record[3],
                offset=conf['parameters'].get('offset'), 
                hysteresis=conf['hysteresis'],
                threshold= conf['parameters'].get('threshold'),
                threshold1= conf['parameters'].get('threshold1'),
                threshold2= conf['parameters'].get('threshold2')):
                results.append(record)
        if results:
            self.cluster.report(self.generate_measurement_report(
                neighbour_strengths, timestamp, event=conf['event_id'], meas_type='connected'))
            return [{
                'event_id': conf['event_id'],
                'result_serving': serving_strength,
                'frequency':conf['frequency'],
                'result_neighbours': [[x[0],x[3]] for x in  results]
            }]
        return results

    def handle_periodical(self,conf,serving_strength, neighbour_strengths,timestamp):
        """Handles measurement objects of 'periodical' type
        """
        freq = self.system_information['freq']
        results = [rec for rec in neighbour_strengths if rec[2] == freq]
        self.cluster.report(self.generate_measurement_report(
            neighbour_strengths, timestamp, event=conf['event_id'], meas_type='connected'))
        return [{
            'event_id': 'periodical',
            'result_serving': serving_strength,
            'frequency':freq,
            'result_neighbours': [[x[0],x[3]] for x in  results]
        }]

    def check_measurements(self, timestamp):
        signal_strengths = flatten(self.scan_connections())
        freq = self.system_information['freq']
        serving_strength,neighbour_results = self.find_signal(
            signal_strengths, self.serving_cell,freq)
        debug('serving:',serving_strength,'neighbour:',neighbour_results)
        if self.layer_3_coefficient:
            serving_strength = self.layer_3_filtering(serving_strength)

        if self.ping:
            self.ping_location_history(self.steps, signals=signal_strengths)
        reports = list()
        for index,conf in enumerate(self.measurement_config):
            self.measurement_config[index]['periodical_timer'] -= 1
            if conf['periodical_timer'] == 0:
                if conf['event_id'] in ['a1', 'a2']:
                    reports += self.handle_A12(conf,serving_strength,neighbour_results,timestamp)
                elif conf['event_id'] in ['a3', 'a4','a5']:
                    reports += self.handle_A345(conf,serving_strength,neighbour_results,timestamp)
                elif conf['event_id'] == 'periodical':
                    reports += self.handle_periodical(conf,serving_strength,neighbour_results,timestamp)
                if conf['report_amount'] <= 1:
                    self.measurement_config.pop(index)
                else:
                    self.measurement_config[index]['report_amount'] -= 1
                    self.measurement_config[index]['periodical_timer'] = conf['report_interval']
        return reports

    def layer_3_filtering(self,signal):
        self.layer_3_last_filtered = (1-self.layer_3_coefficient)*self.layer_3_last_filtered + self.layer_3_coefficient*signal
        return self.layer_3_last_filtered

    @staticmethod
    def find_signal(signal_list, bts_id,frequency=None):
        index, record = find_on(signal_list, lambda rec: rec[0] == bts_id and rec[2] == frequency)
        if index == -1:
            return -140,signal_list
        else:
            return record[3],signal_list[:index]+signal_list[index+1:]

    def scan_connections(self, bts_id=None,inter_freq=True):
        """
        Returns signal at the current location. Currently statically determined beforehand, so this is a simple list lookup.
        """
        freq = 0 if inter_freq else self.system_information['freq']
        res =  self.cluster.cmap._callmethod('strength_at',(self.x, self.y,),{'plmn_id':self.plmn_id,'frequency':freq})
        if res is None:
            error_logging(f"Failed at: {self.x}, {self.y}")
            return []
        res = [[(self.eci_map[eci],eci,freq,value) for eci,freq,value in stat_rec] for stat_rec in res ]
        return res

    def write_logs_to_file(self):
        with open(os.path.join(self.logging_folder, 'userlog' + str(self.id)), 'a') as f:
            f.write('\n'.join((str(rec)
                               for rec in self.location_history))+'\n')
            self.location_history = list()

class PredictableUser(ClusterUser):
    targets:list
    def update_location(self,timestamp,ping=True):
        current_target = self.targets[0]
        if current_target[0] == self.x and current_target[1] == self.y:
            self.targets.pop(0)
            if not self.targets:
                self.cluster.events['done'].set()
                self.cluster.events['barrier'].abort()
                return
            current_target = self.targets[0]
        dx = current_target[0] - self.x
        dy = current_target[1] - self.y
        if abs(dx) > abs(dy):
            self.x += dx/abs(dx)
        else:
            self.y += dy/abs(dy)
        if self.ping:
            self.ping_location_history(timestamp,nan=False)
