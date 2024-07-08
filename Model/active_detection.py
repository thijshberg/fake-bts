import time
import itertools as itools
from threading import BrokenBarrierError
import json
from copy import deepcopy
import os.path
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .util import DefaultObject, flush_queue, detection_logging, Everything, error_logging,ilen,writelines_sep,fst,snd,listen

"""
This file contains code related to detection mechanisms.
A detector will run during the simulation to aggregate and print the data given by the simulation parts.
"""
BASELINE,NORMAL,ANOMALY = 0,1,2

class DetectionNotFound(Exception):
    pass

class Detector():
    """Base class for detection mechanisms
    """
    detected_events: list = []
    state = {}
    events = {}#NOT to be confused with the detected_events, these are the synchronization primitives shared with the other processes
    wait_time = 1
    uplink_queue = DefaultObject()
    filename = '/tmp/x'
    wants: list = []
    window = []
    threshold = 1
    window_size: int
    code:str = ''
    plotstring:str='b'

    def __init__(self, events, uplink_queue, wait_time=1, window_size=100,threshold=1,filename='/tmp/x'):
        self.events = events
        self.wait_time = wait_time
        self.uplink_queue = uplink_queue
        self.threshold=threshold
        self.window_size = window_size
        self.filename=filename
        self.post_init()

    def post_init(self):
        ...

    def print_detected_events(self):
        for e in self.detected_events:
            detection_logging(e)

    def detect_event(self, timestamp, event_type,measure=0, data={}):
        self.detected_events.append(((timestamp, event_type,measure), data))

    def __iter__(self):
        return iter(self.detected_events)

    def handle_data_baseline(self, data):
        ...

    def handle_data_active(self, data):
        ...

    def wait_for_barrier(self):
        try:
            self.events['detector_bar'].wait()
        except BrokenBarrierError:
            detection_logging('Barrier is broken!')
            return True

    def run(self):
        """Used for active detection. Depends on events.
        """
        listen()
        detection_logging(self.__class__,'starting')
        while not (self.events['done'].is_set() or self.events['baseline_done'].is_set()):
            self.handle_data_baseline(flush_queue(self.uplink_queue))
            if self.wait_for_barrier():
                return
        detection_logging('Baseline done!')
        self.process_baseline()
        while not self.events['done'].is_set():
            self.handle_data_active(flush_queue(self.uplink_queue))
            self.process_window()
            if self.wait_for_barrier():
                break
        self.print_report()

    def print_report(self):
        detection_logging('Printing to',self.filename)
        with open(self.filename,'w') as fp:
            writelines_sep(fp,(self.print_report_inner(),
                '-'*10,
                self.code,
                *(f"{event_base}|{json.dumps(event_data)}" for event_base,event_data in self.detected_events)))

    def print_report_inner(self):
        return ""

    def process_window(self):
        ...

    def process_baseline(self):
        ...

class DistributionDetector(Detector):
    """Detects abnormal distribution of measurement reports (all of them together, except the periodical ones).
    """
    wants = ['measurement_report']
    baseline = {}
    code='dist'
    last_timestamp = 0

    def handle_data_baseline(self, data):
        for msg in data:
            self.last_timestamp = max(self.last_timestamp,msg['steps'])
            reports = msg['measurement_report']
            cell_id = msg['serving_cell']
            freq = msg['frequency']
            if freq not in self.baseline:
                self.baseline[freq] = {}
            if cell_id not in self.baseline[freq]:
                self.baseline[freq][cell_id]=1
            self.baseline[freq][cell_id] += ilen(filter(lambda rep: rep['event_id'] != 'periodical',reports))
        detection_logging('Processed to baseline',json.dumps(self.baseline))
        self.detect_event(self.last_timestamp,BASELINE,0)

    def handle_data_active(self, data):
        self.window += data

    def process_baseline(self):
        total = sum(sum(cell_rec.values()) for cell_rec in self.baseline.values())
        if not total:
            error_logging('Baseline was empty!')
        self.registrations = {freq: {cell_id: rec/total for cell_id, rec in freq_val.items()}
                              for freq, freq_val in self.baseline.items()}
        detection_logging(self.registrations)
        

    def process_window(self):
        if len(self.window) < self.window_size:
            return
        self.window = self.window[-self.window_size:]
        total = sum(len(msg['measurement_report']) for msg in self.window)
        window_aggregate = {}
        for msg in self.window:
            self.last_timestamp = max(self.last_timestamp,msg['steps'])
            freq = msg['frequency']
            cell_id = msg['serving_cell']
            if freq not in window_aggregate:
                window_aggregate[freq] = {}
            if cell_id not in window_aggregate[freq]:
                window_aggregate[freq][cell_id]=0
            window_aggregate[freq][cell_id] += len(msg['measurement_report'])/total
        dist =  sum(
                    sum((1 - window_aggregate.get(freq,{}).get(cell_id,0)/value)**2 
                    for cell_id,value in freq_rec.items()) 
                for freq,freq_rec in self.baseline.items())
        detection_logging('Computed distance:',dist)
        if dist > self.threshold:
            self.detect_event(self.last_timestamp,ANOMALY,dist,json.dumps(window_aggregate))
        else:
            self.detect_event(self.last_timestamp,NORMAL,dist,json.dumps(window_aggregate))



class IdDetector(Detector):
    """Detects unknown cell identities on frequencies governed by by TA as reported in measurements

    This is probably a bit silly as we have such tight control over the network.
    It could nevertheless be interesting to see how many users will fall victim before we detect it.
    """
    wants = ['measurement_report']
    known_identities = {}
    code='id'
    last_timestamp = 0
    plotstring='b+'
    def handle_data_baseline(self,data):
        for msg in data:
            freq = msg['frequency']
            self.last_timestamp = max(self.last_timestamp,msg['steps'])
            if freq not in self.known_identities:
                detection_logging('Reporting new frequency',freq)
                self.known_identities[freq] = list()
                self.detect_event(self.last_timestamp,BASELINE,0,{'freq':freq})
            for report in msg.get('measurement_report',()):
                for cell, _ in report.get('result_neighbours',()):
                    if cell not in self.known_identities[freq]:
                        detection_logging('Reporting new cell',cell,'for frequency',freq,'after',self.last_timestamp)
                        self.known_identities[freq].append(cell)
                        self.detect_event(self.last_timestamp,BASELINE,0,{'freq':freq,'cell':cell})

    def handle_data_active(self,data):
        for msg in data:
            self.last_timestamp = max(self.last_timestamp,msg['steps'])
            freq = msg['frequency']
            if freq not in self.known_identities:
                self.detect_event(msg['steps'],ANOMALY,1,{'freq':freq})
                detection_logging('Unknown frequency',freq)
                self.known_identities[freq] = list(msg.get('result_neighbours',{}).keys())
                continue
            for report in msg.get('measurement_report',()):
                for cell, _ in report.get('result_neighbours',()):
                    if not cell in self.known_identities[freq]:
                        detection_logging('Unknown cell on',freq,':',cell,'after',self.last_timestamp)
                        self.detect_event(msg['steps'],ANOMALY,1,{'cell':cell})
                        self.known_identities[freq].append(cell)

    def print_report_inner(self):
        res = []
        freq_events = sorted((ev for ev in self.events if 'freq' in ev),key=lambda ev: ev['freq'])
        cell_events = sorted((ev for ev in self.events if 'cell_id' in ev),key=lambda ev: ev['cell_id'])
        for freq,group in itools.groupby(freq_events,key=lambda ev: ev['freq']):
            res.append(f"Detected anomalous frequency {freq} at step {min(ev['timestamp'] for ev in group)}\n")
        res.append('\n')
        for cell_id,group in itools.groupby(cell_events,key=lambda ev: ev['cell_id']):
            res.append(f"Detected anomalous frequency {cell_id} at step {min(ev['timestamp'] for ev in group)}\n")
        return "".join(res)

class FullIdDistributionDetector(Detector):
    """Detects anomalies in the distribution of users over different frequencies/cells. Processes the whole user database, so this is inefficient/invasive.

    While this is the least error-prone method for realistically using user distributions in theory it is probably widely inefficient in practice.
    If a provider actively monitors when users were last online (I hope they don't) they can probably filter out the active ones to run this on.
    """
    wants = ['database']
    last_timestamp = 0
    code = 'full_id_dist'
    activated_count = 0
    distribution = {}
    baseline_points = []
            
    def handle_distribution(self,data,create_new=False):
        self.activated_count += 1
        if not isinstance(data,list) or len(data) ==0:
            return
        data = data[0]
        def update_new(user_id,cell,freq):
            if freq not in self.distribution:
                self.distribution[freq] = {cell:[user_id]}
                return False
            if cell not in self.distribution[freq]:
                self.distribution[freq][cell] = [user_id]
                return False
            rec = self.distribution[freq][cell]
            if user_id in rec:
                return False
            rec.append(user_id)
            return True
        def remove_old(user_id,new_cell=None,new_freq=None):
            for freq,freq_rec in self.distribution.items():
                for cell,rec in freq_rec.items():
                    if freq == new_freq and cell== new_cell:
                        continue
                    try:
                        rec.remove(user_id)
                        return
                    except (ValueError,KeyError):
                        pass#continue
        for user_id,rec in enumerate(data):
            if rec is None:
                remove_old(user_id)
                continue
            if update_new(user_id,rec['last_known_cell'],rec['frequency']):
                remove_old(user_id,rec['last_known_cell'],rec['frequency'])
        self.last_timestamp = self.activated_count*self.wait_time

    def handle_data_active(self,data):
        self.handle_distribution(data,create_new=False)

    def process_baseline(self):
        total = len(self.baseline_points)
        self.baseline_distribution = {freq:{cell_id:0 for cell_id in freq_val.keys()} for freq,freq_val in self.baseline_points[-1].items()}
        for rec in self.baseline_points:
            for freq,freq_val in rec.items():
                for cell_id,val in freq_val.items():
                    self.baseline_distribution[freq][cell_id] += len(val)/total
        detection_logging('Processed baseline',json.dumps(self.baseline_distribution,indent=2))
        self.detect_event(self.last_timestamp,BASELINE,0)

    def process_window(self):
        dist = {
                    freq_key:{cell_key:(1-val/max(len(self.distribution[freq_key][cell_key]),0.1))**2
                for cell_key,val in freq_rec.items()}
            for freq_key,freq_rec in self.baseline_distribution.items()}
        distsum = sum(sum(rec.values()) for rec in dist.values())
        detection_logging('Processed window',distsum)
        if distsum > self.threshold:
            detection_logging('Anomaly:',json.dumps(dist,indent=2))
            self.detect_event(self.last_timestamp,ANOMALY,distsum,json.dumps(self.distribution))
        else:
            self.detect_event(self.last_timestamp,NORMAL,distsum)

class DebugDetector(Detector):
    """Debugs how many messages it has received
    """
    wants = Everything()
    total = 0
    code = 'debug'

    def handle_data_baseline(self, data):
        self.total += len(data)

    def handle_data_active(self, data):
        self.total += len(data)

    def process_window(self):
        detection_logging(f'Received {self.total} messages so far!')

class LeaveDetector(Detector):
    """Detects users leaving
    """
    wants = ['connection_leave']
    code = 'leave'
    baseline_tot = 0
    last_step = 0
    baseline = 0
 
    def handle_data_baseline(self, data):
        for msg in data:
            self.last_step = max(self.last_step,msg[0])
            self.baseline_tot += 1

    def handle_data_active(self, data):
        total = 0
        dist = {}
        for msg in data:
            total += 1
            self.last_step = max(self.last_step,msg[0])
            if msg[1] not in dist:
                dist[msg[1]] = {msg[2]:1}
            elif msg[2] not in dist[msg[1]]:
                dist[msg[1]][msg[2]] = 1
            else:
                dist[msg[1]][msg[2]] += 1
        if anomalous:=(total > self.threshold):
            guessed_location = {cell_id:sum(freq_count * (freq/1000)**2 for freq,freq_count in cell_rec.items()) for cell_id,cell_rec in dist.items()}
            dist['guess'] = guessed_location
        self.detect_event(self.last_step,ANOMALY if anomalous else NORMAL,measure=total,data=dist)

    def process_baseline(self):
        self.baseline = self.baseline_tot/max(self.last_step,1)

    def process_window(self):
        ...

class LocUpdateDetector(Detector):
    """
    """
    wants = ['loc_update']
    code='loc_update'

    def handle_data(self, data):
        for steps,user_id,x,y in data:
            self.detect_event(steps,NORMAL,data=(user_id,x,y))
    
    def handle_data_baseline(self, data):
        self.handle_data(data)
    def handle_data_active(self, data):
        self.handle_data(data)

    def print_report(self):
        frame = pd.DataFrame([[t,user_id,x,y] for (t,_,_),(user_id,x,y) in self.detected_events],columns=['t','user_id','x','y'])
        frame.to_csv(self.filename)




class RejectDetector(Detector):
    """Collects data of users receiving a reject message
    """
    wants = ['reject','ta_reject']
    previous_rejects = 0
    baseline_rejects = 0
    last_step = 1
    code = 'reject'

    def handle_data_baseline(self, data):
        for msg in data:
            error_logging(msg)
            self.last_step = max(self.last_step,msg[0])
            self.baseline_rejects += 1

    def handle_data_active(self, data):
        for msg in data:
            error_logging('Active',msg)
            self.last_step = max(self.last_step,msg[0])
        severity = len(data)/self.window_size
        self.detect_event(self.last_step,ANOMALY if severity > self.baseline_rejects else NORMAL,measure=severity,data=data)

    def process_baseline(self):
        self.baseline_rejects /= self.last_step
    
class TAUpdateDetector(Detector):
    code='ta_update'
    wants=['ta_update']
    ta_updates = []

    #self.detector_ta_update((bts_id,self.stations[bts_id].tac,result))
    def handle_data_active(self, data):
        self.ta_updates += data
    def handle_data_baseline(self, data):
        self.ta_updates += data

    def print_report_inner(self):
        return repr(self.ta_updates)

class MeasReportDetector(Detector):
    code='meas_report'
    wants=['measurement_report']
    meas_reports = []

    def handle_data_baseline(self, data):
        self.meas_reports += data
    def handle_data_active(self, data):
        self.meas_reports += data
    def print_report_inner(self):
        return repr(self.meas_reports)


detectors = {DebugDetector,IdDetector,FullIdDistributionDetector,DistributionDetector,RejectDetector,TAUpdateDetector,LocUpdateDetector,MeasReportDetector}


############################################################
# Plotting of results
############################################################

def _parse_detection_line(line):
    a,b = line.split('|')
    return make_tuple(a),json.loads(b)


def gather_detection_results(filename):
    res = []
    with open(filename,'r') as f:
        readlines = iter(f.readlines())
        for line in readlines:
            if len(line) >= 5 and line[:5] =='-'*5:
                break
        code = next(readlines)
        for line in readlines:
            res.append(_parse_detection_line(line))
    return code,res


def plot_detection_results(folder,plot=False,filename='detect'):
    filename = os.path.join(folder,filename)
    code,events = gather_detection_results(filename)
    detection_logging(code)
    for det in detectors:
        if code == det.code:
            plotstring= det.plotstring
            break
    else:
        plotstring = 'b'
    if plot:
        plt.plot([rec[0] for rec,data in events],[rec[2] for rec,data in events],plotstring)
        for rec,_ in events:
            if rec[1] == ANOMALY:
                plt.plot([rec[0]],[rec[2]],'r+')
    return events

def loc_update_parse(filename,folder=None):
    if folder:
        filename = os.path.join(folder,filename)
    data = []
    with open(filename) as f:
        f.readline()
        f.readline()
        f.readline()
        for l in f.readlines():
            a,b = l.split('|')
            steps,_,_ = eval(a)
            user_id,x,y = eval(b)
            data.append([steps,user_id,x,y])

    return np.array(data)

            