from os import stat
from shapely.geometry import Point
import numpy as np
try:
    from .cppspeedup import dist_point_to_line,dist2
except Exception:
    from .cppspeedup_alt import dist_point_to_line,dist2
import random
import math
import json
from .util import uniques_count,flatten,snd

"""
Contains code for base stations, mainly performing signal strength simulation.
"""

DIST_FACTOR = 1
QUEUE_WAITING_TIME = 1
NOISE = 2

ATTACKER_TAC=99
ATTACKER_ECI = 100

class BaseStation():
    """
    The model for a base station. Contains methods for the distance model.
    
    In the TA-version of the simulation the station is no more than a point on the map; the actual user connection control is handled through the TA.
    """
    system_information:dict = {} 
    message_queue = None
    tracking_area = None       
    height = 50
    connected_users = 0
    max_users:int
    nr_beams: int = 1
    def __init__(self, eci,tac,plmn_ids=[],position=None,cmap=None,signal_strength=20,noise=NOISE,frequencies=None,max_users=99999,identifier=None):
        self.plmn_ids = plmn_ids
        self.eci = eci
        self.identifier = identifier if identifier is not None else eci
        self.tac = tac
        if position is None:
            self.x = -1
            self.y = -1
        else:
            self.x = position.x
            self.y = position.y 
        self.position = Point(self.x, self.y)
        self.cmap=cmap
        self.signal_strength = signal_strength
        self.frequencies = frequencies if frequencies is not None else [] #used for radio model, in MHz
        self.noise = noise
        #see https://en.wikipedia.org/wiki/Free-space_path_loss
        wavelengths = [299792458/(1000000*frequency) for frequency in self.frequencies]
        self._dist_factors = [DIST_FACTOR * (wavelength/(4*np.pi)) for wavelength in wavelengths]
        self.dist_factors = [2*math.log10(factor) for factor in self._dist_factors]                           
        self.max_users = max_users

    def asdict(self):
        return {'plmn_ids':self.plmn_ids,
        'eci':self.eci,
        'identifier':self.identifier,
        'tac':self.tac,
        'x':self.x,
        'y':self.y,
        'signal_strength':self.signal_strength,
        'frequencies':self.frequencies,
        'max_users':self.max_users}

    @staticmethod
    def fromdict(d):
        if d.get('attacker'):
            return Attacker(d['eci'],d['tac'],plmn_ids = d['plmn_ids'],position = Point(d['x'],d['y']),signal_strength = d['signal_strength'],frequencies = d['frequencies'], max_users = d['max_users'],identifier = d['identifier'])

        else:
            return BaseStation(d['eci'],d['tac'],plmn_ids = d['plmn_ids'],position = Point(d['x'],d['y']),signal_strength = d['signal_strength'],frequencies = d['frequencies'], max_users = d['max_users'],identifier = d['identifier'])

    def update_frequencies(self, freq_list):
        self.frequencies = freq_list
        wavelengths = [299792458/(1000000*frequency) for frequency in self.frequencies]
        self._dist_factors = [DIST_FACTOR * (wavelength/(4*np.pi)) for wavelength in wavelengths]

    def get_signal_strength(self, location, mode='default'):
        """
        calculates the signal strength of this tower at the location. Quite expensive to calculate.
        Possible fix: the obstruction dampening is done via orthogonal distance from the obstruction to the line from the location to this tower. In the 'blocks' model this could be made more realistic.
        """
        noise = random.uniform(-self.noise,self.noise)
        strengths= []
        for freq,factor in zip(self.frequencies,self._dist_factors):    
            if mode == 'default':
                strength=self.signal_strength-self.free_space_signal_loss(self.position,location,factor = factor)+noise
                for ob in self.cmap.obstructions:
                    #(the line through the station and the user passes through the obstruction) and (the obstruction is not behind the location (note that we assume that the user is never in an obstruction))
                    if (dist_point_to_line((location.x,location.y), (self.x,self.y), (ob['location'].x,ob['location'].y)) < ob['radius'] ) and (dist2((self.x,self.y),(ob['location'].x,ob['location'].y)) < dist2((ob['location'].x,ob['location'].y),(location.x,location.y)) + dist2((location.x,location.y),(self.x,self.y))): 
                        if not ob['strength']:
                            continue
                        strength += 10* math.log10(ob['strength'])
                        if strength < -140:
                            continue
            elif mode == 'hata':
                strength =self.signal_strength - self.hata_model(self.position,location,frequency=freq,height=self.height)+noise 
            elif mode=='empty':
                strength=self.signal_strength-self.free_space_signal_loss(self.position,location, factor = factor)+noise 
            elif isinstance(mode,float):
                strength = self.signal_strength -mode* self.free_space_signal_loss(self.position,location,factor=factor)+noise
            if strength < -140:
                continue
            strengths.append((freq,strength))
        return strengths

    @staticmethod
    def free_space_signal_loss(a:Point,b:Point,factor:float = 1):
        return 20*(math.log10(max(a.distance(b),1)/factor))

    @staticmethod
    def dx_free_space_signal_loss(a: Point,b: Point,factor: float = 1):
        return 40*(a.x-b.x)*factor/(max(a.distance(b),1)*BaseStation.free_space_signal_loss(a,b,factor))

    @staticmethod
    def dy_free_space_signal_loss(a: Point,b: Point,factor: float=1):
        return 40*(a.y-b.y)*factor/(max(a.distance(b),1)*BaseStation.free_space_signal_loss(a,b,factor))

    @staticmethod
    def hata_model(a: Point,b:Point,frequency:int=1000,height:int=1):
        """
        Hata, M. (August 1980). "Empirical Formula for Propagation Loss in Land Mobile Radio Services". IEEE Transactions on Vehicular Technology. VT-29 (3): 317–25.
        Formula taken from wikipedia
        Assumptions: UE is at ground level (1 meter), small/medium city
        """
        distance = max(a.distance(b),0.01)/1000#the unit of distance is km
        antenna_height_correction = -1.3060606848816776
        return 69.55 + 16.16*math.log10(frequency) - 13.82*math.log10(height) - antenna_height_correction + (44.9 - 6.55*math.log10(height))*math.log10(distance)

    @staticmethod
    def dx_hata_mode(a: Point,b: Point,height:int = 1):
        distance = max(a.distance(b),1)
        return(44.9 - 6.55*math.log10(height))*2*(a.x-b.x)/(distance * math.log10(distance))

    @staticmethod
    def dy_hata_mode(a: Point,b: Point,height:int = 1):
        distance = max(a.distance(b),1)
        return(44.9 - 6.55*math.log10(height))*2*(a.y-b.y)/(distance * math.log10(distance))


class Attacker(BaseStation):

    @staticmethod
    def find_position(cmap, mode='minimal_signal',frequency=None):
        if mode == 'minimal_signal':
            choices = cmap.random_suitable_point(method='all_points')
            choice_strenghts = np.array(list(map(lambda p: max(map(snd,cmap.strength_at(p.x,p.y,frequency=frequency))),choices)))
            return choices[np.argmin(choice_strenghts)]
        if mode == 'maximum_distance':
            choices = cmap.random_suitable_point(method='all_points')
            def edge_distance_check(p):
                return min(p.x,cmap.map_bounds-p.x,p.y,cmap.map_bounds-p.y) > cmap.map_bounds//10
            choices = list(filter(edge_distance_check,choices))
            def min_distances(p):
                return min(p.distance(s.position) for s in cmap.stations)
            choice_distances = np.array(list(map(min_distances,choices)))
            return choices[np.argmax(choice_distances)]
        else:
            position = Point(random.randint(0,cmap.map_bounds),random.randint(0,cmap.map_bounds))
            while any([position.distance(v['location']) < v['radius'] for v in cmap.obstructions]):
                position = Point(random.randint(0,cmap.map_bounds),random.randint(0,cmap.map_bounds))
            return position

    def asdict(self):
        return super().asdict()|{'attacker':True}