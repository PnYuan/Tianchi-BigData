# -*- coding: utf-8 -*
    
'''
@author: PY131
'''

import os
import sys
import timeit
import pandas as pd

start_time = timeit.default_timer()

'''
generation of new data sets:
    df_act_34 = {<time, user_id, item_id, behavior_type = 3 or 4>}
here we write .csv file multi-step to handle the large scale data.
'''

batch = 0
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
for df in pd.read_csv(open("../../data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv", 'r'), 
                      chunksize=100000): 
    try:
        df_act_34 = df[df['behavior_type'].isin([3,4])]     
        df_act_34.to_csv('../data/act_34.csv',
                         columns=['time','user_id','item_id','behavior_type'],
                         index=False, header=False,
                         mode = 'a')
        batch += 1
        print('chunk %d done.' %batch) 
        
    except StopIteration:
        print("finish.")
        break 


'''
generation of new data sets:
    df_time_34 = {<user_id, item_id, time_3, time_4>}
'''

data_file = open('../data/act_34.csv', 'r')
try:
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H')
    df_act_34 = pd.read_csv(data_file, 
                            parse_dates = [0],
                            date_parser = dateparse,
                            index_col = False)
    df_act_34.columns = ['time','user_id','item_id','behavior_type']
    df_act_34 = df_act_34.drop_duplicates(['user_id','item_id','behavior_type'])  
finally:
    data_file.close()
    
df_time_3 = df_act_34[df_act_34['behavior_type'].isin(['3'])][['user_id','item_id','time']]
df_time_4 = df_act_34[df_act_34['behavior_type'].isin(['4'])][['user_id','item_id','time']]
df_time_3.columns = ['user_id','item_id', 'time3']
df_time_4.columns = ['user_id','item_id', 'time4']
del df_act_34  # to save memory
df_time = pd.merge(df_time_3,df_time_4,on=['user_id','item_id'],how='outer')
df_time_34 = df_time.dropna()

# df_time_3 store the sample contain only behavior_type = 3
# for predict
df_time_3 = df_time[df_time['time4'].isnull()].drop(['time4'], axis = 1)
df_time_3 = df_time_3.dropna()
df_time_3.to_csv('../data/time_3.csv',
                  columns=['user_id','item_id','time3'],
                  index=False)

# save middle data set
df_time_34.to_csv('../data/time_34.csv',
                  columns=['user_id','item_id','time3', 'time4'],
                  index=False)


'''
for decay time calculation and visualization 
'''

data_file = open('../data/time_34.csv', 'r')
try:
    df_time_34 = pd.read_csv(data_file, 
                             parse_dates = ['time3', 'time4'],
                             index_col = False)
finally:
    data_file.close()
    
delta_time = df_time_34['time4']-df_time_34['time3']
delta_hour = [] 
for i in range(len(delta_time)):
    d_hour = delta_time[i].days*24+delta_time[i]._h
    if d_hour < 0: continue     # clean invalid result
    else: delta_hour.append(d_hour)

# draw the histogram of delta_hour
import matplotlib.pyplot as plt
f1 = plt.figure(1)
plt.hist(delta_hour, 30)
plt.xlabel('hours')
plt.ylabel('count')
plt.title('time decay for shopping trolley to buy 1')
plt.grid(True)
plt.show()

    
data_file = open('../data/time_3.csv', 'r')
try:
    df_time_3 = pd.read_csv(data_file, 
                            parse_dates = ['time3'],
                            index_col = ['time3'])
finally:
    data_file.close()
    
ui_pred = df_time_3['2014-12-18']  

# generate from P
data_file = open('../../data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv', 'r')
try:
    df_item = pd.read_csv(data_file,index_col = False)
finally:
    data_file.close()

ui_pred_in_P = pd.merge(ui_pred,df_item,on = ['item_id'])  

# user_id - item_id to csv file
ui_pred_in_P.to_csv('../data/tianchi_mobile_recommendation_predict.csv',
                    columns=['user_id','item_id'],
                    index=False)

end_time = timeit.default_timer()
print(('The code for file ' + os.path.split(__file__)[1] +
       ' ran for %.2fm' % ((end_time - start_time) / 60.)), file = sys.stderr)
        
        