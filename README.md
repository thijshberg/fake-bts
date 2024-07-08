# General

This code is quite a mess, and there are many parts that were not kept up-to-date with changing code.
The code itself is structured around the Model folder, with other files aimed at running that code in a structured way.
The primary way of interacting with the code is through experiments.py, which features experiment recipes for running the simulation a number of times with different parameters.

# Creating a map

There are utility methods in Model/connection_map.py for creating a raytraced map.
This requires the raytracing implementation.
The other implementations have not been deprecated, but they have not been kept up well, and (probably) need some patching to make them work again.
