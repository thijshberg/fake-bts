import time
import multiprocessing as mp
import multiprocessing.queues as mpq
import multiprocessing.pool as mpp
from shapely.geometry import Point
import functools as ftools
import math
import random
from copy import deepcopy
from typing import Callable, Iterator, Union,Any,Sequence
import matplotlib.pyplot as plt
import sys
from inspect import signature
from queue import Empty,Queue
import json
import faulthandler
import os
import threading
import numpy as np

"""
Various helpful and unhelpful functions that are used in multiple places throughout the code.
"""

#These variables control what is logged to stdout
global DEBUG, USER, EPC,ERROR,INFO
DEBUG,USER,EPC,ERROR,DETECTION,INFO = False,False,False,True,False,True

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print ('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed
        
class QueueIterator():
    def __init__(self, queue):
        self.queue = queue
    
    def __next__(self):
        try:
            return self.queue.get_nowait()
        except Exception:
            raise StopIteration
    

class FlushableQueue(mpq.Queue):
    def __init__(self):
        super().__init__(ctx = mp.get_context())

    def __iter__(self):
        return QueueIterator(self)

    def flush(self):
        return [it for it in self]

    
def queue_helper(q):
    try:
        return q.get_nowait()
    except Exception:
        pass

def flush_queue(q,timeout=100):
    begin_t = time.time()
    res_list = []
    while not (time.time() - begin_t > timeout):
        try:
            res_list.append(q.get(block=False,timeout=time.time()-begin_t))
        except Empty:
            break
    return res_list

def put_multi(q, *items):
    for it in items:
        q.put(it)

def add_point(*points):
    return Point(round(sum(map(lambda p: p.x,points))),round(sum(map(lambda p: p.y,points))))

def find_on(l:list, property,default=None):
    """
    Finds the first index and element satisfying the property. We also return the index in case the user wants to change the list.
    """
    for i,x in enumerate(l):
        if property(x):
            return i,x
    return -1,default

def find_last(l:list, property,find_first_if_none=False):
    """Finds the last element satisfying the property.
    """
    last = None
    it = iter(l)
    for x in it:
        if not property(x):
            break
        last = x
    if find_first_if_none and last is None:
        try:
            return next(it)
        except StopIteration:
            return None
    return last

def delete_on(l:list,func: Callable):
    """deletes one element from the list satisfying the predicate.
    Returns the list itself.
    """
    for i,x in enumerate(l):
        if func(x):
            l.pop(i)
            return l
    return l

def ilen(iterable: Iterator):
    """Computes the length of an iterator.
    If the iterator is infinite this will not terminate.
    """
    return ftools.reduce(lambda sum, element: sum + 1, iterable, 0)

def uniques(l:list):
    res = []
    for x in l:
        if x not in res:
            res.append(x)
    return res

def uniques_count(l:list)->dict:
    res = dict()
    for x in l:
        res[x] = res.get(x,0) + 1
    return res

def tget(i: int):
    def tget_(p: tuple):
        return p[i]
    return tget_
fst = tget(0)
snd = tget(1)

def combine_gen(gen_a: Iterator,gen_b: Iterator):
    def _gen():
        yield next(gen_a, next(gen_b))
    return _gen

def iterator_average(it: Iterator):
    s = 0
    n = 0
    for i in it:
        s += i
        n += 1
    return s/n

def ifilter(func: Callable, it: Union[Iterator,None]):
    if it is None:
        return ()
    else:
        return filter(func,it)

class CallList():
    def __init__(self,max_nr,randint,length):
        self.randint = randint
        self.max_nr = max_nr
        self.values = Queue(length+1)
        for i in range(length):
            self.values.put(self.randint(0,self.max_nr))

    def get(self):
        self.values.put(self.randint(0,self.max_nr))
        return self.values.get()

class HasRandom():
    """A class which as its own internal random method.
    It creates a (potentially) huge list of pregenerated random numbers.
    It solves the problem of fixing randomization in multiprocessed programs.
    Sometimes multiprocessed programs, even with fixing random states, give diverging random numbers.
    """
    def __init__(self,n=100000, seed=None):
        random.seed(seed)
        self.random_max = n-3
        self.randoms = [random.random() for _ in range(n)]
        self.random_index = 0
        self.random_repeats=0
        self.seed = seed

    def randrange(self,a,b):
        if a == b:
            return a
        else:
            return a + self.random()*(b-a)
    
    def random(self):
        if self.random_index > self.random_max:
            self.random_repeats +=1
            random.seed(self.seed + 10000*self.random_repeats)
            self.randoms = [random.random() for _ in range(self.random_max+3)]
            self.random_index = 0 
        self.random_index += 1
        return self.randoms[self.random_index]

    def randint(self,a,b):
        return math.floor((b-a+1)*self.random())+a

    def choice(self, l,n=1):
        nr_indices = 0
        indices = []
        ll = len(l)
        while nr_indices < n:
            index = self.randint(0,ll-1)
            if not index in indices:
                indices.append(index)
                nr_indices += 1
        if n == 1:
            return l[indices[0]]
        return [l[i] for i in indices]

    def uniform_choice_distribution(self,possiblities):
        """Given a list of (weight,outcome)-tuples, choose one of the outcomes based on weight.
        """
        total = sum(fst(p) for p in possiblities)
        choice = self.random()
        running_total = 0
        for p in possiblities:
            if choice <= (running_total := running_total + fst(p)/total):
                return snd(p)
        assert False

def min_distance_point(target: Point, points: Sequence[Point]) -> Point:
    min_dist = float('inf')
    res = points[0]
    for p in points:
        if (d:=p.distance(target)) < min_dist:
            min_dist = d
            res = p
    return res



def plot_points(points,marker='b.',**kwargs):
    if points and isinstance(next(iter(points)),Point):
        plt.plot([a.x for a in points],[a.y for a in points],marker,**kwargs)
    else:
        plt.plot([a[0] for a in points],[a[1] for a in points],marke,**kwargsr)

def clamp(a:float) -> float:
    return max(0,min(1,a))


def RDtoGPS(x,y):
    """Converts "Rijksdriehoekscoordinaten" to GPS.
    """
    dx = (x - 155000) * 10 ** -5
    dy = (y - 463000) * 10 ** -5

    sum_n = (3235.65389 * dy) + (-32.58297 * dx ** 2) + (-0.2475 * dy ** 2) + (-0.84978 * dx ** 2 * dx) + (-0.0655 * dy ** 3) + (-0.01709 * dx ** 2 * dy ** 2) + (-0.00738 * dx) + (0.0053 * dx ** 4) + (-0.00039 * dx ** 2 * dy ** 3) + (0.00033 * dx ** 4 * dy) + (-0.00012 * dx * dy)
    sum_e = (5260.52916 * dx) + (105.94684 * dx * dy) + (2.45656 * dx * dy ** 2) + (-0.81885 * dx ** 3) + (0.05594 * dx * dy ** 3) + (-0.05607 * dx ** 3 * dy) + (0.01199 * dy) + (-0.00256 * dx ** 3 * dy ** 2) + (0.00128 * dx * dy ** 4) + (0.00022 * dy ** 2) + (-0.00022 * dx ** 2) + (0.00026 * dx ** 5)

    latitude = (52.15517 + (sum_n / 3600))
    Longitude = (5.387206 + (sum_e / 3600))
    return (latitude, Longitude)


class EmptyBarrier():
    def __init__(self):
        pass

    def wait(self,*args,**kwargs):
        return 0

    def abort(self):...


def flatten(l:list):
    return ftools.reduce(list.__add__,l,[])


def print_flush(*args,**kwargs):
    print(*args,**kwargs)
    sys.stdout.flush()

def debug(*args):
    if DEBUG:
        print_flush('[DEBUG]: ',*args)

def user_logging(*args):
    if USER:
        print_flush('[USER]: ',*args)

def epc_logging(*args):
    if EPC:
        print_flush('[EPC]: ',*args)

def error_logging(*args):
    if ERROR:
        print_flush('[ERROR]: ',*args)

def detection_logging(*args):
    if DETECTION:
        print_flush('[DETECTION]: ',*args)

def info_logging(*args):
    if INFO:
        print_flush('[INFO]: ',*args)

def void(*args,**kwargs):
    pass


def arbitrary_keywords(method):
    """Makes the wrapped method ignore any unexpected keyword arguments

    pylint: disable=unexpected-keyword-arg
    """
    @ftools.wraps(method)
    def inner(*args,**kwargs):
        sig = signature(method)
        _kwargs = {
            key:value for key,value in kwargs.items() if key in sig.parameters
        }
        return method(*args,**_kwargs)
    return inner


class DefaultObject():
    """
    An object made to be given as a default. It can safely be called or its attributes checked/set.
    """
    def __getattribute__(self, name: str) -> Any:
        return DefaultObject()
    
    def __setattr__(self, name: str, value: Any) -> None:
        ...

    def __bool__(self):
        return False

    def __call__(self,*args,**kwargs):
        ...



def vary_config(base_config,key_values:dict):
    def vary_config_rec(config,key_values):
        key,values = key_values[0]
        if len(key_values) == 1:
            for value in values:
                config[key]=value
                yield config
        else:
            for value in values:
                config[key]=value
                for x in vary_config_rec(config,key_values[1:]):
                    yield x
    config= deepcopy(base_config)
    key_value_list = list(key_values.items())
    assert key_value_list
    return vary_config_rec(config,key_value_list)

#https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class Timer():
    """A class for describing a timer which does not keep track of time itself.
    It stores the time the timer was set as 'value', and checks it against the given time and the time it is supposed to wait.
    Must be explicitly checked with time as an argument.
    """
    value = 0
    def __init__(self,wait_time,value=0):
        self.wait_time = wait_time

    def check(self,timestamp):
        return timestamp - self.value > self.wait_time

    def set_value(self,value):
        self.value = value

    def check_and_set(self,timestamp):
        if self.check(timestamp):
            self.value = timestamp-1
            return True
        return False

    def immediate(self):
        """Sets the timer such that the next check returns True
        """
        self.value = -self.wait_time
    

class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(mpp.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

class Everything():
    """An object with the sole purpose of returning true to 'in' statements.
    This is expressly not an iterable.
    """

    def __contains__(self,_):
        return True

    def __repr__(self):
        return "Everything"

def writelines_sep(fp,lines=(),sep='\n'):
    """A version of writelines that supports taking in a line separator
    """
    for line in lines:
        fp.write(line)
        fp.write(sep)


def remove_on(l:list, f:Callable):
    """Removes from l all elements satisfying f
    """
    len_l = len(l)
    for i,x in enumerate(l[::-1]):
        if f(x):
            l.pop(len_l-i-1)
    return l

def json_load(filename:str):
    with open(filename, 'r') as f:
        return json.load(f)


def listen():
    faulthandler.enable()
    if not os.path.exists('stacktraces'):
        os.mkdir('stacktraces')
    faulthandler.register(10,open(os.path.join('stacktraces',str(mp.current_process().pid)),'w'))

def cartesian_product(a:np.array,b:np.array):
    return np.transpose([np.tile(a,len(a)),np.repeat(a,len(a))])