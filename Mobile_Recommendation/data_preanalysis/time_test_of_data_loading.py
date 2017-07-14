# -*- coding: utf-8 -*
    
'''
@author: PY131
'''

import os
import sys
import timeit
import pandas as pd

start_time = timeit.default_timer()

# data loading using pandas
with open("../../data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv", mode = 'r') as data_file:
    df = pd.read_csv(data_file)
    
end_time = timeit.default_timer()

print(('The code for file ' + os.path.split(__file__)[1] +
       ' ran for %.2fm' % ((end_time - start_time) / 60.)), file = sys.stderr)

