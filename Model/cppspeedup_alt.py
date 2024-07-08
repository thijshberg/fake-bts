from typing import Tuple
from shapely.geometry import Point
import math
"""
Python alternative for some methods from cppspeedup.cpp 
"""

def minus_(a: Tuple[int,...],b: Tuple[int,...]):
    return (a[0]-b[0],a[1]-b[1])

def norm(a: Tuple[int,...]):
    return math.sqrt(inner_product(a,a))

def scalar_mult(c, a:Tuple[int,...]):
    return (c*a[0],c*a[1])

def inner_product(a: Tuple[int,...], b: Tuple[int,...]):
    return a[0]*b[0] + a[1]*b[1]

def dist2(a: Tuple[int,...],b: Tuple[int,...]):
    return max(inner_product(c:=minus_(a,b),c),1)

def dist_point_to_line(a: Tuple[int,...],b: Tuple[int,...],c: Tuple[int,...]):
    n = scalar_mult(1/norm(minus_(b,a)),minus_(b,a));#unit vector for the line through a and b, given by a + t * n
    return norm(minus_(minus_(a,c),scalar_mult(inner_product(minus_(a,c),n),n)))