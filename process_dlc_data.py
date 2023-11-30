import os
import glob
import pandas as pd
import numpy as np
from numpy import matlib as mb
import re
import pickle
import pycircstat

# set working directory
ome_dir = 'D:/analysis' # WINDOWS
# home_dir = "/media/jake/LaCie/data_analysis" # Linux/Ubuntu
os.chdir(home_dir)

animal = 'Rat64'
session = '08-11-2023'
data_dir = os.path.join(home_dir, animal, session)

dlc_dir = os.path.join(data_dir, 'deeplabcut')
