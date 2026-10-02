# iPinYou Real-Time Bidding 

This repository recreates Zhang et al.'s work on real-time bidding frameworks in Python3. It consolidates two repositories:
- https://github.com/wnzhang/make-ipinyou-data
- https://github.com/wnzhang/optimal-rtb

since these contain code that no longer works due to the [deprecation of Python 2](https://www.python.org/doc/sunset-python-2/). 

The folder `make-ipinyou-data` contains the steps needed to recreate the data needed for RTB experiments.
The folder `optimal-rtb` contains the RTB experimentation code which uses the data created by `make-ipinyou-data`. 