# coding: utf-8

from re import I
from shapely.geometry import Point

import os.path

import numpy as np
import pandas as pd
import statemachine as sm
import math
import time

from .util import HasRandom,arbitrary_keywords
from .measurementReport import *
STEP_SIZE = 1

BEHAVIOUR_ENUM = [  1,#straight long lines
                    2 #random localized movement
                    ]
BEHAVIOUR2_TIME = 10

DIRECTION_CHECK_INTERVAL = 10

INITIAL_HYSTERISIS = 0
INITIAL_A1_THRESHOLD = -100
INITIAL_A2_THRESHOLD = -100
INITIAL_A3_OFFSET    = 10
INITIAL_A4_THRESHOLD = -110
INITIAL_A5_THRESHOLD1 = -120
INITIAL_A5_THRESHOLD2 = -100

GOOD_CONNECTION_THRESHOLD = -90
BAD_CONNECTION_THRESHOLD = -110#idle procedure specification, page 14
CONNECTION_RESELECTION_THRESHOLD = 20

UNCONNECTED_WAITING_TIME = 5
CELL_RESELECTION_WAITING_TIME = 5
PAGING_CHECK_WAITING_TIME = 5

"""
The old user class. This houses the statemachine and movement logic.
Any logic for running a simulation is depracated.
"""

DIRECTIONS=[(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]


class UserStateMachine(sm.StateMachine):
    rrc_idle = sm.State('rrc_idle')
    rrc_connected = sm.State('rrc_connected')
    unconnected = sm.State('unconnected',initial=True)
    register_pending = sm.State('register_pending')
    connected_pending  = sm.State('connected_pending')

    #unconnected_to_idle = unconnected.to(rrc_idle)
    unconnected_to_pending = unconnected.to(register_pending)
    register_accepted = register_pending.to(rrc_idle)
    register_failed = register_pending.to(unconnected)
    idle_to_conn_pending = rrc_idle.to(connected_pending)
    conn_pending_to_connected = connected_pending.to(rrc_connected)
    conn_pending_failed = connected_pending.to(rrc_idle)
    connected_to_idle = rrc_connected.to(rrc_idle)
    idle_to_unconnected = rrc_idle.to(unconnected)
    connected_to_unconnected = rrc_connected.to(unconnected)
    

class User(HasRandom):
    """
    Model for the user
    """
    steps:int
    def __init__(self, user_id,cmap, protocol_realism='none', message_queue=None,return_pipe=None,ping=True):
        super().__init__(n=100000)
        self.id=user_id
        self.cmap=cmap
        self.protocol_realism = protocol_realism
        self.return_pipe = return_pipe
        self.message_queue = message_queue
        self.map_bounds = self.cmap.map_bounds
        self.map_mode = self.cmap.mode
        self.grid_size = self.cmap.grid_size
        self.ping=ping
        self.cluster=None
        self.plmn_id=0
        self.system_information = {'freq':0}
        self.behaviour2_time=BEHAVIOUR2_TIME
        self.setup()
        
    def init_super(self,n=100000,seed=None):
        super().__init__(n=n,seed=seed)

    def setup(self):    
        """
        Shared setup between User and ClusterUser. Initializes the location and some timers
        """
        if self.map_mode == 'blocks':
            self.behaviour = 1
        else:
            self.behaviour = BEHAVIOUR_ENUM[self.randint(0,1)]
        self.direction = self.randint(1,4)
        self.state = UserStateMachine()
        if self.map_mode in ['blocks','empty','hata']:
            self.mobility_directions = {
                1: {'x': [0, self.map_bounds/2],                       'y': [0, self.map_bounds/2],                       'x_inc': STEP_SIZE, 'y_inc': 0},
                2: {'x': [0, self.map_bounds/2],                       'y': [self.map_bounds/2, self.map_bounds],    'x_inc': 0, 'y_inc': -STEP_SIZE},
                3: {'x': [self.map_bounds/2, self.map_bounds],    'y': [self.map_bounds/2, self.map_bounds],    'x_inc': -STEP_SIZE, 'y_inc': 0},
                4: {'x': [self.map_bounds/2, self.map_bounds],    'y': [0, self.map_bounds/2],                       'x_inc': 0, 'y_inc': STEP_SIZE}}  

            if self.behaviour == 2:
                self.x = self.randint(self.mobility_directions[self.direction]['x'][0], self.mobility_directions[self.direction]['x'][1])
                self.y = self.randint(self.mobility_directions[self.direction]['y'][0], self.mobility_directions[self.direction]['y'][1])
            else: #behaviour ==1
                p = self.get_random_location()
                self.x = p.x
                self.y=p.y
                
        elif self.map_mode == 'premade':
            if self.cluster:
                p = self.cluster.cmap.random_suitable_point(seed=self.randint(1,100))[0]
                self.map_bool = self.cluster.cmap._callmethod('get_map')
            else:
                p = self.cmap.random_suitable_point()[0]
                self.map_bool = self.cmap.get_map()
            self.direction = 2*self.random()
            self.x = p[0]
            self.y = p[1]
        self.position= Point(self.x,self.y)
        self.serving_cell = -1
        self.behaviour2_timer = 0
        self.behaviour1_timer = 0 
        self.redirect_timer = 0
        self.measurement_config = list()
        self.location_history = list()
        self.signal_history = list()
        
        self.ta_stations = list()

        self.a1_threshold = INITIAL_A1_THRESHOLD
        self.a2_threshold = INITIAL_A2_THRESHOLD
        self.a3_offset = INITIAL_A3_OFFSET
        self.a4_threshold = INITIAL_A4_THRESHOLD
        self.a5_threshold1 = INITIAL_A5_THRESHOLD1
        self.a5_threshold2 = INITIAL_A5_THRESHOLD2
        self.hysteresis = INITIAL_HYSTERISIS

        self.paging_time = 0 #The time the current connected state was initiated
        self.call_time  =0#The time the connected state will last
        self.unconnected_timer = 0 #The time we last checked for a suitable cell while unconnected
        self.reselection_timer = 0 #The time we last evaluated our cell signal
        self.paging_timer = 0 #The time we last checked the paging channel
        
    def get_random_location(self):
        if not self.cluster is None:
            p = Point(*self.cluster.cmap.random_suitable_point(seed=self.randint(1,100),method='walkable'))
        else:
            p = Point(*self.cmap.random_suitable_point(method='walkable'))
        return p
        
    
    def moveto_close_point(self):
        for i,direction in enumerate((DIRECTIONS[math.floor(self.direction*4)],
                    DIRECTIONS[math.ceil(self.direction*4) % 8], 
                    DIRECTIONS[math.floor(self.direction*4+7)%8],
                    DIRECTIONS[math.ceil(self.direction*4+1) % 8],
                    DIRECTIONS[math.floor(self.direction*4+6)%8],
                    DIRECTIONS[math.ceil(self.direction*4+2) % 8], 
                    DIRECTIONS[math.floor(self.direction*4+5)%8],
                    DIRECTIONS[math.ceil(self.direction*4+3) % 8],
            )):
            x = self.x + direction[0]
            y = self.y + direction[1]

            if self.map_bool[x][y]:
                if i >3:
                    self.direction = self.random()*2
                self.x = x
                self.y = y
                return
        raise Exception("Could not find place to go!")
        
    def run_simulation(self,max_time):
        self.start_time = time.time()
        self.unconnected_timer = self.start_time- UNCONNECTED_WAITING_TIME
        self.reselection_timer = self.start_time - CELL_RESELECTION_WAITING_TIME
        self.paging_timer = self.start_time - PAGING_CHECK_WAITING_TIME
        while True:
            t= time.time()
            if t-self.start_time >= max_time:
                self.return_pipe.send(self.location_history)  
                return         
            self.update_step(t)
        

    def update_step(self,timestamp):
        self.update_location(timestamp,ping=self.ping)
        if self.state.is_unconnected and timestamp - self.unconnected_timer > UNCONNECTED_WAITING_TIME:
            self.cell_selection()
        elif self.state.is_rrc_idle:
            paging_result = self.check_paging(timestamp) if timestamp - self.paging_timer > PAGING_CHECK_WAITING_TIME else False
            if (not paging_result) and(timestamp - self.reselection_timer > CELL_RESELECTION_WAITING_TIME):
                self.cell_reselection()
        elif self.state.is_rrc_connected:
            if timestamp - self.paging_timer > self.call_time:
                self.state.connected_to_idle()
            elif (not paging_result) and(timestamp - self.reselection_timer > CELL_RESELECTION_WAITING_TIME):
                self.cell_reselection()
   
    def update_location(self,timestamp,ping=True):
        """
        Base movement method. Updates location, updates direction if behaviour dictates it, and checks whether we are out of bounds
        """
        
        nan = False
        if self.map_mode == 'premade':
            self.moveto_close_point()
        else:    
            self.x += self.mobility_directions[self.direction]['x_inc']
            self.y += self.mobility_directions[self.direction]['y_inc']
        if self.map_mode == 'default':
            if self.behaviour == 2:
                if self.behaviour2_timer == 0:
                    self.direction = self.randint(1,4)
                    self.behaviour2_timer = self.randint(1,100)
                else:
                    self.behaviour2_timer -= 1
            if timestamp % DIRECTION_CHECK_INTERVAL ==0:
                if self.y <= 0:
                    if self.direction == 2:
                        self.direction -= 1 #if going down-right, turn left
                    elif self.direction ==3:
                        self.direction += 1 #if going down-left, turn right         
                elif self.y >= self.map_bounds:
                    if self.direction == 1:
                        self.direction += 1 #up-right
                    elif self.direction==4:
                        self.direction -= 1 #up-left
                elif self.x <= 0:
                    if self.direction == 3:
                        self.direction -= 1
                    elif self.direction==4:
                        self.direction = 1
                elif self.x >= self.map_bounds:
                    if self.direction == 1:
                        self.direction = 4
                    elif self.direction==2:
                        self.direction += 1
            
        elif self.map_mode == 'blocks':
            if self.behaviour == 1:
                if self.behaviour1_timer == 0:
                    self.direction = (self.direction + self.choice([-1,1]) -1 ) % 4 + 1
                    self.behaviour1_timer = self.randint(1,10)*math.floor(self.map_bounds/self.grid_size)
                else:
                    self.behaviour1_timer -= 1
            if self.behaviour == 2:
                if self.behaviour2_timer == 0:
                    self.direction = self.randint(1,4)
                    self.behaviour2_timer = self.behaviour1_timer = self.randint(5,15)*self.behaviour2_time
                else:
                    self.behaviour2_timer -= 1
            dirs = [self.map_bounds - self.x < 1, self.y < 1, self.x < 1, self.map_bounds -self.y < 1]
            if (s :=sum(dirs)) == 2:
                self.x = self.map_bounds //2
                self.y = self.map_bounds //2
                nan = True
            elif s == 1:
                for i,b in enumerate(dirs):
                    if b:
                        self.direction = (i + 4 - 2)%4 +1

        elif self.map_mode == 'empty':
            if self.behaviour == 1:
                if self.behaviour1_timer == 0:
                    choices = [1,2,3,4]
                    choices.remove((self.direction+1)%4+1)
                    self.direction = self.choice(choices)
                    self.behaviour1_timer = self.randint(1,10)*math.floor(self.map_bounds/self.grid_size)
                else:
                    self.behaviour1_timer -= 1
            if self.behaviour == 2:
                if self.behaviour2_timer == 0:
                    choices = [1,2,3,4]
                    choices.remove((self.direction+1)%4+1)
                    self.direction = self.choice(choices)
                    self.behaviour2_timer = self.randint(15,25)*self.behaviour2_time
                else:
                    self.behaviour2_timer -= 1
            
            dirs = [self.map_bounds - self.x < 5, self.y < 5, self.x < 5, self.map_bounds -self.y < 5]
            if (s :=sum(dirs)) == 2:
                self.x = self.map_bounds //2
                self.y = self.map_bounds //2
            elif s == 1:
                for i,b in enumerate(dirs):
                    if b:
                        self.direction = (i + 4 - 2)%4 +1

                self.behaviour1_timer = self.behaviour2_timer = round(self.random()**2 *9* self.map_bounds/10) + self.map_bounds/10
        
        if self.ping:
            self.ping_location_history(timestamp,nan=nan)
        if self.redirect_timer >0:
            self.redirect_timer -= 1


    def get_location_point(self):
        return Point(self.x, self.y)

    def check_trigger_event(self,timestamp):
        """
        Checks a number of events, and generates measurement reports for those which triggered
        """
        reports=list()
        signal_strengths = self.scan_connections()
        new_stat = max(signal_strengths, key=(lambda x: x[1]))
        if self.serving_cell == -1:
            if self.event_A1(new_stat[1]):# and 'event_a1' in self.measurement_config['reportConfigs']:
                self.serving_cell = new_stat[0]
                reports.append(self.generate_measurement_report(signal_strengths,timestamp,event='event_a1'))
        else:
            _,current_stat = find_on(signal_strengths, lambda x: x[0]==self.serving_cell)
            if self.event_A2(current_stat[1]):# and 'event_a2' in self.measurement_config['reportConfigs']:
                self.serving_cell = -1
                reports.append(self.generate_measurement_report(signal_strengths,timestamp,event='event_a2'))
            if new_stat[0] != self.serving_cell:
                if self.event_A5(current_stat[1],new_stat[1]):# and 'event_a5' in self.measurement_config['reportConfigs']:
                    self.serving_cell = new_stat[0]
                    reports.append(self.generate_measurement_report(signal_strengths,timestamp,event='event_a5'))
        return reports

    def scan_connections(self):
        """
        Returns signal at the current location. Currently statically determined beforehand, so this is a simple list lookup.
        """
        return self.cmap.strength_at(self.x,self.y)

    def generate_measurement_report(self,signal_strengths,timestamp,meas_type='none',event=None):
        return MeasurementReport(id=self.id,x=math.floor(self.x),y=math.floor(self.y),timestamp=timestamp-self.start_time,bts_id=self.serving_cell,signal_strengths=signal_strengths,event=event,meas_type=meas_type,plmn_id=self.plmn_id,frequency=self.system_information['freq'])

    def ping_location_history(self,timestamp,signals=None, nan=False):
        """
    	Used to plot the user location data (mainly for debuggin). 
        The parameter nan can be set to make matplotlib ignore this point, used to prevent plotting users teleporting between the edges.
        """
        if isinstance(timestamp, float):
            timestamp -= self.start_time
        if self.location_history:
            last_entry = self.location_history[-1]
            if last_entry[1] == self.x and last_entry[2] == self.y:
                self.location_history.pop(-1)
        if nan:
            if signals:
                self.location_history.append((timestamp,np.nan,np.nan,self.serving_cell,self.system_information.get('freq',-1),self.state.model.state,signals)) 
            else:
                self.location_history.append((timestamp,np.nan,np.nan,self.serving_cell,self.system_information.get('freq',-1),self.state.model.state))  
        else:
            if signals:
                self.location_history.append((timestamp,self.x,self.y,self.serving_cell,self.system_information.get('freq',-1),self.state.model.state,signals))
            else:
                self.location_history.append((timestamp,self.x,self.y,self.serving_cell,self.system_information.get('freq',-1),self.state.model.state))

    def set_measurement_config(self,measurement_config):
        self.measurement_config = measurement_config

########################################################################
# Triggering event implementation
########################################################################

    """ Triggering Events
        - Event A1 (Serving becomes better than threshold)
        - Event A2 (Serving becomes worse than threshold)
        - Event A3 (Neighbour becomes offset better than serving)
        - Event A4 (Neighbour becomes better than threshold)
        - Event A5 (Serving becomes worse than threshold1 and neighbour becomes better than threshold2)
        - Event B1 (Inter RAT neighbour becomes better than threshold)
        - Event B2 (Serving becomes worse than threshold1 and inter RAT neighbour becomes better than threshold2)   
    """

    @staticmethod
    @arbitrary_keywords
    def event_A1(measurement,/,threshold,hysteresis, leaving=False):
        """
        Event A1 (Serving becomes better than threshold)
        The UE shall:
        1> consider the entering condition for this event to be satisfied when condition A1-1, as specified below, is fulfilled;
        1> consider the leaving condition for this event to be satisfied when condition A1-2, as specified below, is fulfilled;
        Inequality A1-1 (Entering condition)
        Ms − Hys > Thresh
        Inequality A1-2 (Leaving condition)
        Ms + Hys < Thresh
        The variables in the formula are defined as follows:
        Ms is the measurement result of the serving cell, not taking into account any offsets.
        Hys is the hysteresis parameter for this event (i.e. hysteresis as defined within reportConfigEUTRA for this event).
        Thresh is the threshold parameter for this event (i.e. a1-Threshold as defined within reportConfigEUTRA for this
        event).
        Ms is expressed in dBm in case of RSRP, or in dB in case of RSRQ.
        Hys is expressed in dB.
        Thresh is expressed in the same unit as Ms.
        """
        if not leaving:
            return measurement -hysteresis > threshold
        else:
            return measurement +hysteresis < threshold

    @staticmethod
    @arbitrary_keywords
    def event_A2(measurement,/,threshold,hysteresis, leaving = False):
        """
        Event A2 (Serving becomes worse than threshold)
        The UE shall:
        1> consider the entering condition for this event to be satisfied when condition A2-1, as specified below, is fulfilled;
        1> consider the leaving condition for this event to be satisfied when condition A2-2, as specified below, is fulfilled;
        Inequality A2-1 (Entering condition)
        Ms+ Hys < Thresh
        Inequality A2-2 (Leaving condition)
        Ms−Hys > Thresh
        The variables in the formula are defined as follows:
        Ms is the measurement result of the serving cell, not taking into account any offsets.
        Hys is the hysteresis parameter for this event (i.e. hysteresis as defined within reportConfigEUTRA for this event).
        Thresh is the threshold parameter for this event (i.e. a2-Threshold as defined within reportConfigEUTRA for this
        event).
        Ms is expressed in dBm in case of RSRP, or in dB in case of RSRQ.
        Hys is expressed in dB.
        Thresh is expressed in the same unit as Ms.
        """
        if not leaving:
            return measurement + hysteresis < threshold
        else:
            return measurement - hysteresis > threshold

    @staticmethod
    @arbitrary_keywords
    def event_A3(serving_measurement, neigh_measurement,/, offset,hysteresis,leaving=False):
        if offset is None:
            offset = INITIAL_A3_OFFSET
        """
        Event A3 (Neighbour becomes offset better than serving)
        The UE shall:
        1> consider the entering condition for this event to be satisfied when condition A3-1, as specified below, is fulfilled;
        1> consider the leaving condition for this event to be satisfied when condition A3-2, as specified below, is fulfilled;
        Inequality A3-1 (Entering condition)
        Mn +Ofn +Ocn − Hys > Ms +Ofs +Ocs +Off
        Inequality A3-2 (Leaving condition)
        Mn +Ofn +Ocn + Hys < Ms +Ofs +Ocs +Off
        The variables in the formula are defined as follows:
        Mn is the measurement result of the neighbouring cell, not taking into account any offsets.
        Ofn is the frequency specific offset of the frequency of the neighbour cell (i.e. offsetFreq as defined within
        measObjectEUTRA corresponding to the frequency of the neighbour cell).
        Ocn is the cell specific offset of the neighbour cell (i.e. cellIndividualOffset as defined within measObjectEUTRA
        corresponding to the frequency of the neighbour cell), and set to zero if not configured for the neighbour cell.
        Ms is the measurement result of the serving cell, not taking into account any offsets.
        Ofs is the frequency specific offset of the serving frequency (i.e. offsetFreq as defined within measObjectEUTRA
        corresponding to the serving frequency).
        Ocs is the cell specific offset of the serving cell (i.e. cellIndividualOffset as defined within measObjectEUTRA
        corresponding to the serving frequency), and is set to zero if not configured for the serving cell.
        Hys is the hysteresis parameter for this event (i.e. hysteresis as defined within reportConfigEUTRA for this event).
        Off is the offset parameter for this event (i.e. a3-Offset as defined within reportConfigEUTRA for this event).
        Mn, Ms are expressed in dBm in case of RSRP, or in dB in case of RSRQ.
        Ofn, Ocn, Ofs, Ocs, Hys, Off are expressed in dB.
        """
        if not leaving:
            return neigh_measurement - hysteresis > serving_measurement + offset
        else:
            return neigh_measurement + hysteresis < serving_measurement + offset

    @staticmethod
    @arbitrary_keywords
    def event_A4(_,neigh_measurement,/,threshold,hysteresis,leaving=False):
        """
        Event A4 (Neighbour becomes better than threshold)
        The UE shall:
        1> consider the entering condition for this event to be satisfied when condition A4-1, as specified below, is fulfilled;
        1> consider the leaving condition for this event to be satisfied when condition A4-2, as specified below, is fulfilled;
        Inequality A4-1 (Entering condition)
        Mn +Ofn +Ocn − Hys > Thresh
        Inequality A4-2 (Leaving condition)
        Mn +Ofn +Ocn + Hys < Thresh
        The variables in the formula are defined as follows:
        Mn is the measurement result of the neighbouring cell, not taking into account any offsets.
        Ofn is the frequency specific offset of the frequency of the neighbour cell (i.e. offsetFreq as defined within
        measObjectEUTRA corresponding to the frequency of the neighbour cell).
        Ocn is the cell specific offset of the neighbour cell (i.e. cellIndividualOffset as defined within measObjectEUTRA
        corresponding to the frequency of the neighbour cell), and set to zero if not configured for the neighbour cell.
        Hys is the hysteresis parameter for this event (i.e. hysteresis as defined within reportConfigEUTRA for this event).
        Thresh is the threshold parameter for this event (i.e. a4-Threshold as defined within reportConfigEUTRA for this
        event).
        Mn is expressed in dBm in case of RSRP, or in dB in case of RSRQ.
        Ofn, Ocn, Hys are expressed in dB.
        Thresh is expressed in the same unit as Ms.

        Note: the first _ argument is to keep it in line with A3,A5 calling conventions
        """
        if not leaving:
            return neigh_measurement - hysteresis > threshold
        else:
            return neigh_measurement + hysteresis < threshold

    @staticmethod
    @arbitrary_keywords
    def event_A5(serving_measurement,neigh_measurement,/,threshold1,threshold2,hysteresis,leaving=False):
        """
        Event A5 (Serving becomes worse than threshold1 and neighbour becomes better than threshold2)
        The UE shall:
        1> consider the entering condition for this event to be satisfied when both conditions A5-1 and condition A5-2, as
        specified below, are fulfilled;
        1> consider the leaving condition for this event to be satisfied when condition A5-3 or condition A5-4, i.e. at least
        one of the two, as specified below, is fulfilled;
        Inequality A5-1 (Entering condition 1)
        Ms + Hys< Thresh1
        Inequality A5-2 (Entering condition 2)
        Mn +Ofn +Ocn − Hys > Thresh2
        Inequality A5-3 (Leaving condition 1)
        Ms − Hys > Thresh1
        Inequality A5-4 (Leaving condition 2)
        Mn +Ofn +Ocn + Hys < Thresh2
        The variables in the formula are defined as follows:
        Ms is the measurement result of the serving cell, not taking into account any offsets.
        Mn is the measurement result of the neighbouring cell, not taking into account any offsets.
        Ofn is the frequency specific offset of the frequency of the neighbour cell (i.e. offsetFreq as defined within
        measObjectEUTRA corresponding to the frequency of the neighbour cell).
        Ocn is the cell specific offset of the neighbour cell (i.e. cellIndividualOffset as defined within measObjectEUTRA
        corresponding to the frequency of the neighbour cell), and set to zero if not configured for the neighbour cell.
        Hys is the hysteresis parameter for this event (i.e. hysteresis as defined within reportConfigEUTRA for this event).
        Thresh1 is the threshold parameter for this event (i.e. a5-Threshold1 as defined within reportConfigEUTRA for this
        event).
        Thresh2 is the threshold parameter for this event (i.e. a5-Threshold2 as defined within reportConfigEUTRA for this
        event).
        Mn, Ms are expressed in dBm in case of RSRP, or in dB in case of RSRQ.
        Ofn, Ocn, Hys are expressed in dB.
        Thresh1 is expressed in the same unit as Ms.
        Thresh2 is expressed in the same unit as Mn.
        """
        if not leaving:
            return (serving_measurement + hysteresis < threshold1) & (neigh_measurement - hysteresis > threshold2)
        else:
            return (serving_measurement - hysteresis > threshold1) | (neigh_measurement + hysteresis) <threshold2

    def event_A6(self):
        """
        Neighbour become offset better than S Cell (This event is introduced in Release 10 for CA)
        """
        pass

    event_mapping={
        'a1': event_A1.__func__,
        'a2': event_A2.__func__,
        'a3': event_A3.__func__,
        'a4': event_A4.__func__,
        'a5': event_A5.__func__,
    }    
    
class UserHistory():
    def __init__(self,start_time,loc_data, sig_data):
        self.start_time = start_time
        self.loc_data = loc_data
        self.sig_data = sig_data

    @staticmethod
    def from_file(filename,folder=None):
        loc_data = list()
        sig_data = list()
        if folder is None:
            file_handle = open(filename,'r')
        else:
            file_handle = open(os.path.join(folder,filename),'r')
        start_time = float(file_handle.readline())
        for index,line in enumerate(file_handle.readlines()):
            l = line.replace(' ','').replace('(','').replace(')','').replace('[','').replace(']','').replace('\n','').split(',')
            if len(l) > 1: #trailing empty lines...
                timestamp = l[0]
                loc_data.append([float(timestamp),float(l[1]),float(l[2]),int(l[3]),int(l[4]),l[5].strip("'")])
            if len(l) > 7:
                signals = [(math.floor(int(sig[0])),int(sig[1]),int(sig[2]),float(sig[3])) for sig in zip(*(iter(l[6:]),)*4)]
                sig_data.append((float(timestamp),signals,index))
        file_handle.close()
        return UserHistory(start_time,loc_data,sig_data)
    
    def dataframe(self):
        sig_data = iter(self.sig_data)
        current_rec = next(sig_data)
        next_rec = next(sig_data)
        def find_signal(t,bts_id,freq):
            nonlocal current_rec
            nonlocal next_rec
            if t >= next_rec[0]:
                current_rec =next_rec
                try:
                    next_rec = next(sig_data)
                except StopIteration:
                    pass
            if bts_id == -1:
                return np.nan
            for _,_id,_freq,sig in current_rec[1]:
                if _id == bts_id and _freq == freq:
                    return sig
        data = [[t,x,y,cell_id,find_signal(t,cell_id,freq)] for t,x,y,cell_id,freq,_ in self.loc_data]
        return pd.DataFrame(data, columns=['time','x','y','cell_id','signal'])
