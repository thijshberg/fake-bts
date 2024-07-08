from shapely.geometry import Point
import pandas as pd
import os.path
import sys
from .util import *

"""
Code for keeping track of user activity.
Confusingly enough this is strictly more than just Measurement Reports.
"""

class MeasurementReport():
    def __init__(self,id=0,x=0,y=0,bts_id=0,timestamp=-1,signal_strengths=[],meas_type='none',event='-',plmn_id=0,frequency=0):
        self.id = id
        self.x = x
        self.y = y
        self.typ = meas_type
        self.bts_id = bts_id
        self.signal_strengths = signal_strengths
        self.timestamp = timestamp
        self.event = event
        self.plmn_id = plmn_id
        self.frequency = frequency

    def to_csv(self):
        return ','.join([str(self.id),str(self.x),str(self.y),str(self.typ),str(self.bts_id),str(self.signal_strengths[:]),str(self.timestamp),str(self.event),str(self.plmn_id),str(self.frequency)])


    @staticmethod
    def from_csv(line: str):
        line = line.replace(" ",'').replace("(",'').replace(")","").replace("[",'').replace("]","").replace('\n','')
        line_list = line.split(',')
        ret = MeasurementReport()
        ret.id = int(line_list[0])
        ret.x = int(line_list[1])
        ret.y = int(line_list[2])
        ret.typ = line_list[3]
        ret.bts_id = int(line_list[4])
        signal_list = zip(*(iter(line_list[5:-4]),)*4)#group by four
        #print(list(signal_list)[0:2])
        signal_list = [(int(sig[0]),int(sig[1]),float(sig[2]),float(sig[3])) for sig in signal_list]#cast to floats
        ret.signal_strengths = signal_list
        ret.timestamp = float(line_list[-4])
        ret.event = line_list[-3]
        ret.plmn_id = line_list[-2]
        ret.frequency = line_list[-1]
        return ret


class MeasurementReports():
    def __init__(self):
        mr_dict = {'trigger_type': [],'event':[],'bts_id': [], 'user_id': [], 'user_loc': [], 'timestamp': [], 'bts': [], 'signals': [], 'distances': [],'plmn_id':[],'frequency':[]}
        self.measurement_reports = pd.DataFrame.from_dict(mr_dict)
        self.mr_keys = list(mr_dict.keys())

    def _append_measurement_report(self, typ,event, bts_id, user_id, user_geometry, timestamp, bts, signals, distances,plmn_id,frequency):
        self.measurement_reports = self.measurement_reports.append(pd.DataFrame([[typ,event,bts_id, user_id, user_geometry, timestamp, bts, signals, distances,plmn_id,frequency]], columns=self.mr_keys))

    def append_bulk(self,data):
        self.measurement_reports = self.measurement_reports.append(pd.DataFrame([rec for rec in data], columns=self.mr_keys)) 

    def append_measurement_report(self, *mrs):
        self.measurement_reports = self.measurement_reports.append(pd.DataFrame([[mr.typ,mr.event,mr.bts_id,mr.id,(mr.x,mr.y),mr.timestamp,0,mr.signal_strengths,0,mr.plmn_id,mr.frequency] for mr in mrs], columns=self.mr_keys))


    def get_report_for_user(self, user_id):
        report_subset = self.measurement_reports[(self.measurement_reports['user_id'] == user_id) & (self.measurement_reports['timestamp'] == max(self.measurement_reports['timestamp']))]

        serving_cell = report_subset['bts_id'][0]
        neighbors = list(report_subset['bts'][0])
        signals = list(report_subset['signals'][0])
        distances = list(report_subset['distances'][0])

        return serving_cell, neighbors, signals, distances
    
class LoggedMeasurements():
    def __init__(self):
        pass


class ReportLogger():
    def __init__(self,report_queue,error_queue,attacker_queue, event_done,filename_rep,filename_err,filename_att,folder=None):
        self.event_done = event_done
        self._filename_err = filename_err
        self._filename_rep = filename_rep
        self._filename_att = filename_att
        if not filename_rep:
            self.filing=False
            self.rep_list = list()
            self.err_list = list()
            self.att_list = list()
        else:
            self.filing=True
            self.error_queue = error_queue
            self.report_queue = report_queue
            self.attacker_queue = attacker_queue
            if folder is None:
                self.filename_err = filename_err
                self.filename_rep = filename_rep
                self.filename_att = filename_att
            else:
                self.filename_err = os.path.join(folder,filename_err)
                self.filename_rep = os.path.join(folder,filename_rep)
                self.filename_att = os.path.join(folder,filename_att)
        #open(self.filename_att,'w')
        #open(self.filename_rep,'w')
        #open(self.filename_err,'w')
            
    def update_folder(self,folder):
        self.filename_err = os.path.join(folder,self._filename_err)
        self.filename_rep = os.path.join(folder,self._filename_rep)
        self.filename_att = os.path.join(folder,self._filename_att)


    def flush(self):
        flushed_items = flush_queue(self.error_queue)
        if self.filing:
            writable_string = '\n'.join([str(rec) for rec in flushed_items]) + '\n'
            with open(self.filename_err,'a+') as f:
                f.write(writable_string)
        else:
            self.err_list += flushed_items

        flushed_items = flush_queue(self.report_queue)
        if self.filing:
            writable_string = '\n'.join([rec.to_csv() for rec in flushed_items]) + '\n'
            with open(self.filename_rep,'a+') as f:
                f.write(writable_string)
        else:
            self.rep_list += flushed_items 

        flushed_items = flush_queue(self.attacker_queue)
        if self.filing:
            writable_string = '\n'.join([rec.to_csv() for rec in flushed_items]) + '\n'
            with open(self.filename_att,'a+') as f:
                f.write(writable_string)
        else:
            self.att_list += flushed_items 

    def run(self):
        listen()
        while not self.event_done.is_set():
            time.sleep(1)
            self.flush()
        self.flush()
        if not self.filing:
            self.report_queue.put(self.rep_list)
            self.error_queue.put(self.err_list)
        return

    @staticmethod
    def generate_report(filename,folder=None):
        if not folder is None:
            filename = os.path.join(folder,filename)
        rep= MeasurementReports()
        with open(filename,'r') as f:
            rep.append_measurement_report(*[MeasurementReport.from_csv(line) for line in f.readlines() if len(line) > 1])
        return rep    