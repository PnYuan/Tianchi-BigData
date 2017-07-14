# -*- coding: utf-8 -*
    
'''
@author: PY131

@thoughts: construct feature based on U, I, C, UI, UC, IC features types

this file for data_set_part_2

    part 1 - train: 11.22~11.27 > 11.28;
    part 2 - train: 11.29~12.04 > 12.05;
    part 3 - test: 12.13~12.18 (> 12.19);
    
'''

##### file path
# input 
path_df_D = "../../data/raw/tianchi_fresh_comp_train_user.csv"

path_df_part_1 = "../../data/mobile/raw/df_part_1.csv"
path_df_part_2 = "../../data/mobile/raw/df_part_2.csv"
path_df_part_3 = "../../data/mobile/raw/df_part_3.csv"

path_df_part_1_tar = "../../data/mobile/raw/df_part_1_tar.csv"
path_df_part_2_tar = "../../data/mobile/raw/df_part_2_tar.csv"

path_df_part_1_uic_label = "../../data/mobile/raw/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "../../data/mobile/raw/df_part_2_uic_label.csv"
path_df_part_3_uic       = "../../data/mobile/raw/df_part_3_uic.csv"

# output
path_df_part_1_U   = "../../data/mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "../../data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "../../data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "../../data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "../../data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "../../data/mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "../../data/mobile/feature/df_part_2_U.csv"  
path_df_part_2_I   = "../../data/mobile/feature/df_part_2_I.csv"
path_df_part_2_C   = "../../data/mobile/feature/df_part_2_C.csv"
path_df_part_2_IC  = "../../data/mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "../../data/mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "../../data/mobile/feature/df_part_2_UC.csv"

path_df_part_3_U   = "../../data/mobile/feature/df_part_3_U.csv"  
path_df_part_3_I   = "../../data/mobile/feature/df_part_3_I.csv"
path_df_part_3_C   = "../../data/mobile/feature/df_part_3_C.csv"
path_df_part_3_IC  = "../../data/mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "../../data/mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "../../data/mobile/feature/df_part_3_UC.csv"


import pandas as pd
import numpy as np

##========================================================##
##======================== Part 2 ========================##
##========================================================##

###########################################
'''Step 1.1 feature data set U of df_part_2
    (1)
    u_b1_count_in_6
    u_b2_count_in_6
    u_b3_count_in_6
    u_b4_count_in_6
    u_b_count_in_6
    (2)
    u_b1_count_in_3
    u_b2_count_in_3
    u_b3_count_in_3
    u_b4_count_in_3
    u_b_count_in_3
    (2)
    u_b1_count_in_1
    u_b2_count_in_1
    u_b3_count_in_1
    u_b4_count_in_1
    u_b_count_in_1
    (3)
    u_b4_rate  (in_6)
    u_b4_diff_hours  (in_6)
'''


# loading data
path_df = open(path_df_part_2, 'r')
try:
    df_part_2 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
    df_part_2.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    path_df.close()

# u_b_count_in_6
df_part_2['cumcount'] = df_part_2.groupby(['user_id', 'behavior_type']).cumcount()
df_part_2_u_b_count_in_6 = df_part_2.drop_duplicates(['user_id','behavior_type'], 'last')[['user_id','behavior_type','cumcount']]
df_part_2_u_b_count_in_6 = pd.get_dummies(df_part_2_u_b_count_in_6['behavior_type']).join(df_part_2_u_b_count_in_6[['user_id','cumcount']])
df_part_2_u_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_u_b_count_in_6['u_b1_count_in_6'] = df_part_2_u_b_count_in_6['behavior_type_1'] * (df_part_2_u_b_count_in_6['cumcount']+1)
df_part_2_u_b_count_in_6['u_b2_count_in_6'] = df_part_2_u_b_count_in_6['behavior_type_2'] * (df_part_2_u_b_count_in_6['cumcount']+1)
df_part_2_u_b_count_in_6['u_b3_count_in_6'] = df_part_2_u_b_count_in_6['behavior_type_3'] * (df_part_2_u_b_count_in_6['cumcount']+1)
df_part_2_u_b_count_in_6['u_b4_count_in_6'] = df_part_2_u_b_count_in_6['behavior_type_4'] * (df_part_2_u_b_count_in_6['cumcount']+1)
df_part_2_u_b_count_in_6 = df_part_2_u_b_count_in_6.groupby('user_id').agg({'u_b1_count_in_6': np.sum,
                                                                            'u_b2_count_in_6': np.sum,
                                                                            'u_b3_count_in_6': np.sum,
                                                                            'u_b4_count_in_6': np.sum})
df_part_2_u_b_count_in_6.reset_index(inplace = True)
df_part_2_u_b_count_in_6['u_b_count_in_6'] = df_part_2_u_b_count_in_6[['u_b1_count_in_6',
                                                                       'u_b2_count_in_6',
                                                                       'u_b3_count_in_6',
                                                                       'u_b4_count_in_6']].apply(lambda x: x.sum(), axis = 1)

# u_b_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')]
df_part_2_in_3['cumcount'] = df_part_2_in_3.groupby(['user_id', 'behavior_type']).cumcount()
df_part_2_u_b_count_in_3 = df_part_2.drop_duplicates(['user_id','behavior_type'], 'last')[['user_id','behavior_type','cumcount']]
df_part_2_u_b_count_in_3 = pd.get_dummies(df_part_2_u_b_count_in_3['behavior_type']).join(df_part_2_u_b_count_in_3[['user_id','cumcount']])
df_part_2_u_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_u_b_count_in_3['u_b1_count_in_3'] = df_part_2_u_b_count_in_3['behavior_type_1'] * (df_part_2_u_b_count_in_3['cumcount']+1)
df_part_2_u_b_count_in_3['u_b2_count_in_3'] = df_part_2_u_b_count_in_3['behavior_type_2'] * (df_part_2_u_b_count_in_3['cumcount']+1)
df_part_2_u_b_count_in_3['u_b3_count_in_3'] = df_part_2_u_b_count_in_3['behavior_type_3'] * (df_part_2_u_b_count_in_3['cumcount']+1)
df_part_2_u_b_count_in_3['u_b4_count_in_3'] = df_part_2_u_b_count_in_3['behavior_type_4'] * (df_part_2_u_b_count_in_3['cumcount']+1)
df_part_2_u_b_count_in_3 = df_part_2_u_b_count_in_3.groupby('user_id').agg({'u_b1_count_in_3': np.sum,
                                                                            'u_b2_count_in_3': np.sum,
                                                                            'u_b3_count_in_3': np.sum,
                                                                            'u_b4_count_in_3': np.sum})
df_part_2_u_b_count_in_3.reset_index(inplace = True)
df_part_2_u_b_count_in_3['u_b_count_in_3'] = df_part_2_u_b_count_in_3[['u_b1_count_in_3',
                                                                       'u_b2_count_in_3',
                                                                       'u_b3_count_in_3',
                                                                       'u_b4_count_in_3']].apply(lambda x: x.sum(), axis = 1)

# u_b_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')]
df_part_2_in_1['cumcount'] = df_part_2_in_1.groupby(['user_id', 'behavior_type']).cumcount()
df_part_2_u_b_count_in_1 = df_part_2_in_1.drop_duplicates(['user_id','behavior_type'], 'last')[['user_id','behavior_type','cumcount']]
df_part_2_u_b_count_in_1 = pd.get_dummies(df_part_2_u_b_count_in_1['behavior_type']).join(df_part_2_u_b_count_in_1[['user_id','cumcount']])
df_part_2_u_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_u_b_count_in_1['u_b1_count_in_1'] = df_part_2_u_b_count_in_1['behavior_type_1'] * (df_part_2_u_b_count_in_1['cumcount']+1)
df_part_2_u_b_count_in_1['u_b2_count_in_1'] = df_part_2_u_b_count_in_1['behavior_type_2'] * (df_part_2_u_b_count_in_1['cumcount']+1)
df_part_2_u_b_count_in_1['u_b3_count_in_1'] = df_part_2_u_b_count_in_1['behavior_type_3'] * (df_part_2_u_b_count_in_1['cumcount']+1)
df_part_2_u_b_count_in_1['u_b4_count_in_1'] = df_part_2_u_b_count_in_1['behavior_type_4'] * (df_part_2_u_b_count_in_1['cumcount']+1)
df_part_2_u_b_count_in_1 = df_part_2_u_b_count_in_1.groupby('user_id').agg({'u_b1_count_in_1': np.sum,
                                                                            'u_b2_count_in_1': np.sum,
                                                                            'u_b3_count_in_1': np.sum,
                                                                            'u_b4_count_in_1': np.sum})
df_part_2_u_b_count_in_1.reset_index(inplace = True)
df_part_2_u_b_count_in_1['u_b_count_in_1']  = df_part_2_u_b_count_in_1[['u_b1_count_in_1',
                                                                        'u_b2_count_in_1',
                                                                        'u_b3_count_in_1',
                                                                        'u_b4_count_in_1']].apply(lambda x: x.sum(), axis = 1)

# merge the result of count_in_6, count_in_3, count_in_1

df_part_2_u_b_count = pd.merge(df_part_2_u_b_count_in_6, 
                               df_part_2_u_b_count_in_3, on = ['user_id'], how = 'left').fillna(0)
df_part_2_u_b_count = pd.merge(df_part_2_u_b_count, 
                               df_part_2_u_b_count_in_1, on = ['user_id'], how = 'left').fillna(0)
                                    
df_part_2_u_b_count[['u_b1_count_in_6',
                     'u_b2_count_in_6',
                     'u_b3_count_in_6',
                     'u_b4_count_in_6',
                      'u_b_count_in_6',
                     'u_b1_count_in_3',
                     'u_b2_count_in_3',
                     'u_b3_count_in_3',
                     'u_b4_count_in_3',
                      'u_b_count_in_3',
                     'u_b1_count_in_1',
                     'u_b2_count_in_1',
                     'u_b3_count_in_1',
                     'u_b4_count_in_1',
                      'u_b_count_in_1']] = df_part_2_u_b_count[['u_b1_count_in_6',
                                                                'u_b2_count_in_6',
                                                                'u_b3_count_in_6',
                                                                'u_b4_count_in_6',
                                                                 'u_b_count_in_6',
                                                                'u_b1_count_in_3',
                                                                'u_b2_count_in_3',
                                                                'u_b3_count_in_3',
                                                                'u_b4_count_in_3',
                                                                 'u_b_count_in_3',
                                                                'u_b1_count_in_1',
                                                                'u_b2_count_in_1',
                                                                'u_b3_count_in_1',
                                                                'u_b4_count_in_1',
                                                                 'u_b_count_in_1']].astype(int)
                                                        
# u_b4_rate
df_part_2_u_b_count['u_b4_rate'] = df_part_2_u_b_count['u_b4_count_in_6'] / df_part_2_u_b_count['u_b_count_in_6']

# u_b4_diff_time
df_part_2 = df_part_2.sort_values(by = ['user_id', 'time'])
df_part_2_u_b4_time = df_part_2[df_part_2['behavior_type'] == 4].drop_duplicates(['user_id'],'first')[['user_id','time']]
df_part_2_u_b4_time.columns = ['user_id','b4_first_time']
df_part_2_u_b_time = df_part_2.drop_duplicates(['user_id'],'first')[['user_id','time']]
df_part_2_u_b_time.columns = ['user_id','b_first_time']
df_part_2_u_b_b4_time = pd.merge(df_part_2_u_b_time, df_part_2_u_b4_time, on = ['user_id'])
df_part_2_u_b_b4_time['u_b4_diff_time'] = df_part_2_u_b_b4_time['b4_first_time'] - df_part_2_u_b_b4_time['b_first_time']
df_part_2_u_b_b4_time = df_part_2_u_b_b4_time[['user_id', 'u_b4_diff_time']]
df_part_2_u_b_b4_time['u_b4_diff_hours'] = df_part_2_u_b_b4_time['u_b4_diff_time'].apply(lambda x: x.days * 24 + x.seconds//3600)

# generating feature set U
f_U_part_2 = pd.merge(df_part_2_u_b_count, 
                      df_part_2_u_b_b4_time, 
                      on = ['user_id'], how = 'left')[['user_id',
                                                       'u_b1_count_in_6', 
                                                       'u_b2_count_in_6', 
                                                       'u_b3_count_in_6', 
                                                       'u_b4_count_in_6', 
                                                       'u_b_count_in_6',
                                                       'u_b1_count_in_3',
                                                       'u_b2_count_in_3', 
                                                       'u_b3_count_in_3',
                                                       'u_b4_count_in_3', 
                                                       'u_b_count_in_3',
                                                       'u_b1_count_in_1',
                                                       'u_b2_count_in_1', 
                                                       'u_b3_count_in_1',
                                                       'u_b4_count_in_1', 
                                                       'u_b_count_in_1', 
                                                       'u_b4_rate', 
                                                       'u_b4_diff_hours']]
                      
# write to csv file
f_U_part_2 = f_U_part_2.round({'u_b4_rate': 3})
f_U_part_2.to_csv(path_df_part_2_U, index = False)


###########################################
'''Step 1.2 feature data set I of df_part_2
    (1)
    i_u_count_in_6
    i_u_count_in_3
    i_u_count_in_1
    (2)
    i_b1_count_in_6
    i_b2_count_in_6
    i_b3_count_in_6
    i_b4_count_in_6
     i_b_count_in_6
    i_b1_count_in_3
    i_b2_count_in_3
    i_b3_count_in_3
    i_b4_count_in_3
     i_b_count_in_3
    i_b1_count_in_1
    i_b2_count_in_1
    i_b3_count_in_1
    i_b4_count_in_1
     i_b_count_in_1
    (3)
    i_b4_rate  (in_6)
    i_b4_diff_hours  (in_6)
'''


# loading data
path_df = open(path_df_part_2, 'r')
try:
    df_part_2 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
    df_part_2.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    path_df.close()

# i_u_count_in_6
df_part_2_in_6 = df_part_2.drop_duplicates(['item_id', 'user_id'])
df_part_2_in_6['i_u_count_in_6'] = df_part_2_in_6.groupby('item_id').cumcount() + 1
df_part_2_i_u_count_in_6 = df_part_2_in_6.drop_duplicates(['item_id'], 'last')[['item_id', 'i_u_count_in_6']]

# i_u_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')].drop_duplicates(['item_id', 'user_id'])
df_part_2_in_3['i_u_count_in_3'] = df_part_2_in_3.groupby('item_id').cumcount() + 1
df_part_2_i_u_count_in_3 = df_part_2_in_3.drop_duplicates(['item_id'], 'last')[['item_id', 'i_u_count_in_3']]

# i_u_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')].drop_duplicates(['item_id', 'user_id'])
df_part_2_in_1['i_u_count_in_1'] = df_part_2_in_1.groupby('item_id').cumcount() + 1
df_part_2_i_u_count_in_1 = df_part_2_in_1.drop_duplicates(['item_id'], 'last')[['item_id', 'i_u_count_in_1']]

# merge for generation of i_u_count
df_part_2_i_u_count = pd.merge(df_part_2_i_u_count_in_6, 
                               df_part_2_i_u_count_in_3,
                               on=['item_id'],how='left').fillna(0)
df_part_2_i_u_count = pd.merge(df_part_2_i_u_count, 
                               df_part_2_i_u_count_in_1,
                               on=['item_id'],how='left').fillna(0)
df_part_2_i_u_count[['i_u_count_in_6',
                     'i_u_count_in_3',
                     'i_u_count_in_1']] = df_part_2_i_u_count[['i_u_count_in_6',
                                                               'i_u_count_in_3',
                                                               'i_u_count_in_1']].astype(int)

# i_b_count_in_6
df_part_2['cumcount'] = df_part_2.groupby(['item_id', 'behavior_type']).cumcount()
df_part_2_i_b_count_in_6 = df_part_2.drop_duplicates(['item_id','behavior_type'], 'last')[['item_id','behavior_type','cumcount']]
df_part_2_i_b_count_in_6 = pd.get_dummies(df_part_2_i_b_count_in_6['behavior_type']).join(df_part_2_i_b_count_in_6[['item_id','cumcount']])
df_part_2_i_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_i_b_count_in_6['i_b1_count_in_6'] = df_part_2_i_b_count_in_6['behavior_type_1'] * (df_part_2_i_b_count_in_6['cumcount']+1)
df_part_2_i_b_count_in_6['i_b2_count_in_6'] = df_part_2_i_b_count_in_6['behavior_type_2'] * (df_part_2_i_b_count_in_6['cumcount']+1)
df_part_2_i_b_count_in_6['i_b3_count_in_6'] = df_part_2_i_b_count_in_6['behavior_type_3'] * (df_part_2_i_b_count_in_6['cumcount']+1)
df_part_2_i_b_count_in_6['i_b4_count_in_6'] = df_part_2_i_b_count_in_6['behavior_type_4'] * (df_part_2_i_b_count_in_6['cumcount']+1)
df_part_2_i_b_count_in_6 = df_part_2_i_b_count_in_6[['item_id', 
                                                     'i_b1_count_in_6', 
                                                     'i_b2_count_in_6', 
                                                     'i_b3_count_in_6',
                                                     'i_b4_count_in_6']]
df_part_2_i_b_count_in_6 = df_part_2_i_b_count_in_6.groupby('item_id').agg({'i_b1_count_in_6': np.sum,
                                                                            'i_b2_count_in_6': np.sum,
                                                                            'i_b3_count_in_6': np.sum,
                                                                            'i_b4_count_in_6': np.sum})
df_part_2_i_b_count_in_6.reset_index(inplace = True)
df_part_2_i_b_count_in_6['i_b_count_in_6'] = df_part_2_i_b_count_in_6['i_b1_count_in_6'] + \
                                             df_part_2_i_b_count_in_6['i_b2_count_in_6'] + \
                                             df_part_2_i_b_count_in_6['i_b3_count_in_6'] + \
                                             df_part_2_i_b_count_in_6['i_b4_count_in_6']

# i_b_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')]
df_part_2_in_3['cumcount'] = df_part_2_in_3.groupby(['item_id', 'behavior_type']).cumcount()
df_part_2_i_b_count_in_3 = df_part_2.drop_duplicates(['item_id','behavior_type'], 'last')[['item_id','behavior_type','cumcount']]
df_part_2_i_b_count_in_3 = pd.get_dummies(df_part_2_i_b_count_in_3['behavior_type']).join(df_part_2_i_b_count_in_3[['item_id','cumcount']])
df_part_2_i_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_i_b_count_in_3['i_b1_count_in_3'] = df_part_2_i_b_count_in_3['behavior_type_1'] * (df_part_2_i_b_count_in_3['cumcount']+1)
df_part_2_i_b_count_in_3['i_b2_count_in_3'] = df_part_2_i_b_count_in_3['behavior_type_2'] * (df_part_2_i_b_count_in_3['cumcount']+1)
df_part_2_i_b_count_in_3['i_b3_count_in_3'] = df_part_2_i_b_count_in_3['behavior_type_3'] * (df_part_2_i_b_count_in_3['cumcount']+1)
df_part_2_i_b_count_in_3['i_b4_count_in_3'] = df_part_2_i_b_count_in_3['behavior_type_4'] * (df_part_2_i_b_count_in_3['cumcount']+1)
df_part_2_i_b_count_in_3 = df_part_2_i_b_count_in_3[['item_id', 
                                                     'i_b1_count_in_3', 
                                                     'i_b2_count_in_3', 
                                                     'i_b3_count_in_3',
                                                     'i_b4_count_in_3']]
df_part_2_i_b_count_in_3 = df_part_2_i_b_count_in_3.groupby('item_id').agg({'i_b1_count_in_3': np.sum,
                                                                            'i_b2_count_in_3': np.sum,
                                                                            'i_b3_count_in_3': np.sum,
                                                                            'i_b4_count_in_3': np.sum})
df_part_2_i_b_count_in_3.reset_index(inplace = True)
df_part_2_i_b_count_in_3['i_b_count_in_3'] = df_part_2_i_b_count_in_3['i_b1_count_in_3'] + \
                                             df_part_2_i_b_count_in_3['i_b2_count_in_3'] + \
                                             df_part_2_i_b_count_in_3['i_b3_count_in_3'] + \
                                             df_part_2_i_b_count_in_3['i_b4_count_in_3']

# i_b_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')]
df_part_2_in_1['cumcount'] = df_part_2_in_1.groupby(['item_id', 'behavior_type']).cumcount()
df_part_2_i_b_count_in_1 = df_part_2_in_1.drop_duplicates(['item_id','behavior_type'], 'last')[['item_id','behavior_type','cumcount']]
df_part_2_i_b_count_in_1 = pd.get_dummies(df_part_2_i_b_count_in_1['behavior_type']).join(df_part_2_i_b_count_in_1[['item_id','cumcount']])
df_part_2_i_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)                     
df_part_2_i_b_count_in_1['i_b1_count_in_1'] = df_part_2_i_b_count_in_1['behavior_type_1'] * (df_part_2_i_b_count_in_1['cumcount']+1)
df_part_2_i_b_count_in_1['i_b2_count_in_1'] = df_part_2_i_b_count_in_1['behavior_type_2'] * (df_part_2_i_b_count_in_1['cumcount']+1)
df_part_2_i_b_count_in_1['i_b3_count_in_1'] = df_part_2_i_b_count_in_1['behavior_type_3'] * (df_part_2_i_b_count_in_1['cumcount']+1)
df_part_2_i_b_count_in_1['i_b4_count_in_1'] = df_part_2_i_b_count_in_1['behavior_type_4'] * (df_part_2_i_b_count_in_1['cumcount']+1)
df_part_2_i_b_count_in_1 = df_part_2_i_b_count_in_1[['item_id', 
                                                     'i_b1_count_in_1', 
                                                     'i_b2_count_in_1', 
                                                     'i_b3_count_in_1',
                                                     'i_b4_count_in_1']]
df_part_2_i_b_count_in_1 = df_part_2_i_b_count_in_1.groupby('item_id').agg({'i_b1_count_in_1': np.sum,
                                                                            'i_b2_count_in_1': np.sum,
                                                                            'i_b3_count_in_1': np.sum,
                                                                            'i_b4_count_in_1': np.sum})
df_part_2_i_b_count_in_1.reset_index(inplace = True)
df_part_2_i_b_count_in_1['i_b_count_in_1'] = df_part_2_i_b_count_in_1['i_b1_count_in_1'] + \
                                             df_part_2_i_b_count_in_1['i_b2_count_in_1'] + \
                                             df_part_2_i_b_count_in_1['i_b3_count_in_1'] + \
                                             df_part_2_i_b_count_in_1['i_b4_count_in_1']

# merge for generation of i_b_count
df_part_2_i_b_count = pd.merge(df_part_2_i_b_count_in_6, 
                               df_part_2_i_b_count_in_3, 
                               on = ['item_id'], how = 'left').fillna(0)
df_part_2_i_b_count = pd.merge(df_part_2_i_b_count, 
                               df_part_2_i_b_count_in_1, 
                               on = ['item_id'], how = 'left').fillna(0)
df_part_2_i_b_count[['i_b1_count_in_6',
                     'i_b2_count_in_6',
                     'i_b3_count_in_6',
                     'i_b4_count_in_6',
                      'i_b_count_in_6',
                     'i_b1_count_in_3',
                     'i_b2_count_in_3',
                     'i_b3_count_in_3',
                     'i_b4_count_in_3',
                      'i_b_count_in_3',
                     'i_b1_count_in_1',
                     'i_b2_count_in_1',
                     'i_b3_count_in_1',
                     'i_b4_count_in_1',
                      'i_b_count_in_1']] = df_part_2_i_b_count[['i_b1_count_in_6',
                                                                'i_b2_count_in_6',
                                                                'i_b3_count_in_6',
                                                                'i_b4_count_in_6',
                                                                 'i_b_count_in_6',
                                                                'i_b1_count_in_3',
                                                                'i_b2_count_in_3',
                                                                'i_b3_count_in_3',
                                                                'i_b4_count_in_3',
                                                                 'i_b_count_in_3',
                                                                'i_b1_count_in_1',
                                                                'i_b2_count_in_1',
                                                                'i_b3_count_in_1',
                                                                'i_b4_count_in_1',
                                                                 'i_b_count_in_1']].astype(int)

# i_b4_rate
df_part_2_i_b_count['i_b4_rate'] = df_part_2_i_b_count['i_b4_count_in_6'] / df_part_2_i_b_count['i_b_count_in_6']

# i_b4_diff_time
df_part_2 = df_part_2.sort_values(by=['item_id', 'time'])
df_part_2_i_b4_time = df_part_2[df_part_2['behavior_type'] == 4].drop_duplicates(['item_id'], 'first')[['item_id','time']]
df_part_2_i_b4_time.columns = ['item_id','b4_first_time']
df_part_2_i_b_time = df_part_2.drop_duplicates(['item_id'], 'first')[['item_id','time']]
df_part_2_i_b_time.columns = ['item_id','b_first_time']
df_part_2_i_b_b4_time = pd.merge(df_part_2_i_b_time, df_part_2_i_b4_time, on = ['item_id'])
df_part_2_i_b_b4_time['i_b4_diff_time']  = df_part_2_i_b_b4_time['b4_first_time'] - df_part_2_i_b_b4_time['b_first_time']
df_part_2_i_b_b4_time['i_b4_diff_hours'] = df_part_2_i_b_b4_time['i_b4_diff_time'].apply(lambda x: x.days * 24 + x.seconds//3600)
df_part_2_i_b_b4_time = df_part_2_i_b_b4_time[['item_id', 'i_b4_diff_hours']]

# generating feature set I
f_I_part_2 = pd.merge(df_part_2_i_b_count, 
                      df_part_2_i_b_b4_time, 
                      on = ['item_id'], how = 'left')
f_I_part_2 = pd.merge(f_I_part_2, 
                      df_part_2_i_u_count, 
                      on = ['item_id'], how = 'left')[['item_id', 
                                                       'i_u_count_in_6', 
                                                       'i_u_count_in_3', 
                                                       'i_u_count_in_1',
                                                       'i_b1_count_in_6', 
                                                       'i_b2_count_in_6', 
                                                       'i_b3_count_in_6', 
                                                       'i_b4_count_in_6', 
                                                       'i_b_count_in_6', 
                                                       'i_b1_count_in_3',
                                                       'i_b2_count_in_3',
                                                       'i_b3_count_in_3',
                                                       'i_b4_count_in_3',
                                                       'i_b_count_in_3',
                                                       'i_b1_count_in_1', 
                                                       'i_b2_count_in_1', 
                                                       'i_b3_count_in_1', 
                                                       'i_b4_count_in_1', 
                                                       'i_b_count_in_1',
                                                       'i_b4_rate', 
                                                       'i_b4_diff_hours']]
                      
# write to csv file
f_I_part_2 = f_I_part_2.round({'i_b4_rate': 3})
f_I_part_2.to_csv(path_df_part_2_I, index = False)


###########################################
'''Step 1.3 feature data set C of df_part_2
    (1)
    c_u_count_in_6
    c_u_count_in_3
    c_u_count_in_1
    (2)
    c_b1_count_in_6
    c_b2_count_in_6
    c_b3_count_in_6
    c_b4_count_in_6
     c_b_count_in_6
    c_b1_count_in_3
    c_b2_count_in_3
    c_b3_count_in_3
    c_b4_count_in_3
     c_b_count_in_3
    c_b1_count_in_1
    c_b2_count_in_1
    c_b3_count_in_1
    c_b4_count_in_1
     c_b_count_in_1
    (3)
    c_b4_rate  (in_6)
    c_b4_diff_hours  (in_6)
'''


# loading data
path_df = open(path_df_part_2, 'r')
try:
    df_part_2 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
    df_part_2.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    path_df.close()
    
# c_u_count_in_6
df_part_2_in_6 = df_part_2.drop_duplicates(['item_category', 'user_id'])
df_part_2_in_6['c_u_count_in_6'] = df_part_2_in_6.groupby('item_category').cumcount() + 1
df_part_2_c_u_count_in_6 = df_part_2_in_6.drop_duplicates(['item_category'], 'last')[['item_category', 'c_u_count_in_6']]

# c_u_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')].drop_duplicates(['item_category', 'user_id'])
df_part_2_in_3['c_u_count_in_3'] = df_part_2_in_3.groupby('item_category').cumcount() + 1
df_part_2_c_u_count_in_3 = df_part_2_in_3.drop_duplicates(['item_category'], 'last')[['item_category', 'c_u_count_in_3']]

# c_u_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')].drop_duplicates(['item_category', 'user_id'])
df_part_2_in_1['c_u_count_in_1'] = df_part_2_in_1.groupby('item_category').cumcount() + 1
df_part_2_c_u_count_in_1 = df_part_2_in_1.drop_duplicates(['item_category'], 'last')[['item_category', 'c_u_count_in_1']]

df_part_2_c_u_count = pd.merge(df_part_2_c_u_count_in_6, df_part_2_c_u_count_in_3,on=['item_category'],how='left').fillna(0)
df_part_2_c_u_count = pd.merge(df_part_2_c_u_count, df_part_2_c_u_count_in_1,on=['item_category'],how='left').fillna(0)
df_part_2_c_u_count[['c_u_count_in_6',
                     'c_u_count_in_3',
                     'c_u_count_in_1']] = df_part_2_c_u_count[['c_u_count_in_6',
                                                               'c_u_count_in_3',
                                                               'c_u_count_in_1']].astype(int)

# c_b_count_in_6
df_part_2['cumcount'] = df_part_2.groupby(['item_category', 'behavior_type']).cumcount()
df_part_2_c_b_count_in_6 = df_part_2.drop_duplicates(['item_category','behavior_type'], 'last')[['item_category','behavior_type','cumcount']]
df_part_2_c_b_count_in_6 = pd.get_dummies(df_part_2_c_b_count_in_6['behavior_type']).join(df_part_2_c_b_count_in_6[['item_category','cumcount']])
df_part_2_c_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_c_b_count_in_6['c_b1_count_in_6'] = df_part_2_c_b_count_in_6['behavior_type_1'] * (df_part_2_c_b_count_in_6['cumcount']+1)
df_part_2_c_b_count_in_6['c_b2_count_in_6'] = df_part_2_c_b_count_in_6['behavior_type_2'] * (df_part_2_c_b_count_in_6['cumcount']+1)
df_part_2_c_b_count_in_6['c_b3_count_in_6'] = df_part_2_c_b_count_in_6['behavior_type_3'] * (df_part_2_c_b_count_in_6['cumcount']+1)
df_part_2_c_b_count_in_6['c_b4_count_in_6'] = df_part_2_c_b_count_in_6['behavior_type_4'] * (df_part_2_c_b_count_in_6['cumcount']+1)
df_part_2_c_b_count_in_6 = df_part_2_c_b_count_in_6[['item_category', 
                                                     'c_b1_count_in_6', 
                                                     'c_b2_count_in_6', 
                                                     'c_b3_count_in_6',
                                                     'c_b4_count_in_6']]
df_part_2_c_b_count_in_6 = df_part_2_c_b_count_in_6.groupby('item_category').agg({'c_b1_count_in_6': np.sum,
                                                                                  'c_b2_count_in_6': np.sum,
                                                                                  'c_b3_count_in_6': np.sum,
                                                                                  'c_b4_count_in_6': np.sum})
df_part_2_c_b_count_in_6.reset_index(inplace = True)
df_part_2_c_b_count_in_6['c_b_count_in_6'] = df_part_2_c_b_count_in_6['c_b1_count_in_6'] + \
                                             df_part_2_c_b_count_in_6['c_b2_count_in_6'] + \
                                             df_part_2_c_b_count_in_6['c_b3_count_in_6'] + \
                                             df_part_2_c_b_count_in_6['c_b4_count_in_6']

# c_b_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')]
df_part_2_in_3['cumcount'] = df_part_2_in_3.groupby(['item_category', 'behavior_type']).cumcount()
df_part_2_c_b_count_in_3 = df_part_2_in_3.drop_duplicates(['item_category','behavior_type'], 'last')[['item_category','behavior_type','cumcount']]
df_part_2_c_b_count_in_3 = pd.get_dummies(df_part_2_c_b_count_in_3['behavior_type']).join(df_part_2_c_b_count_in_3[['item_category','cumcount']])
df_part_2_c_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_c_b_count_in_3['c_b1_count_in_3'] = df_part_2_c_b_count_in_3['behavior_type_1'] * (df_part_2_c_b_count_in_3['cumcount']+1)
df_part_2_c_b_count_in_3['c_b2_count_in_3'] = df_part_2_c_b_count_in_3['behavior_type_2'] * (df_part_2_c_b_count_in_3['cumcount']+1)
df_part_2_c_b_count_in_3['c_b3_count_in_3'] = df_part_2_c_b_count_in_3['behavior_type_3'] * (df_part_2_c_b_count_in_3['cumcount']+1)
df_part_2_c_b_count_in_3['c_b4_count_in_3'] = df_part_2_c_b_count_in_3['behavior_type_4'] * (df_part_2_c_b_count_in_3['cumcount']+1)
df_part_2_c_b_count_in_3 = df_part_2_c_b_count_in_3[['item_category', 
                                                     'c_b1_count_in_3', 
                                                     'c_b2_count_in_3', 
                                                     'c_b3_count_in_3',
                                                     'c_b4_count_in_3']]
df_part_2_c_b_count_in_3 = df_part_2_c_b_count_in_3.groupby('item_category').agg({'c_b1_count_in_3': np.sum,
                                                                                  'c_b2_count_in_3': np.sum,
                                                                                  'c_b3_count_in_3': np.sum,
                                                                                  'c_b4_count_in_3': np.sum})
df_part_2_c_b_count_in_3.reset_index(inplace = True)
df_part_2_c_b_count_in_3['c_b_count_in_3'] = df_part_2_c_b_count_in_3['c_b1_count_in_3'] + \
                                             df_part_2_c_b_count_in_3['c_b2_count_in_3'] + \
                                             df_part_2_c_b_count_in_3['c_b3_count_in_3'] + \
                                             df_part_2_c_b_count_in_3['c_b4_count_in_3']

# c_b_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')]
df_part_2_in_1['cumcount'] = df_part_2_in_1.groupby(['item_category', 'behavior_type']).cumcount()
df_part_2_c_b_count_in_1 = df_part_2_in_1.drop_duplicates(['item_category','behavior_type'], 'last')[['item_category','behavior_type','cumcount']]
df_part_2_c_b_count_in_1 = pd.get_dummies(df_part_2_c_b_count_in_1['behavior_type']).join(df_part_2_c_b_count_in_1[['item_category','cumcount']])
df_part_2_c_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                           2:'behavior_type_2',
                                           3:'behavior_type_3',
                                           4:'behavior_type_4'}, inplace=True)
df_part_2_c_b_count_in_1['c_b1_count_in_1'] = df_part_2_c_b_count_in_1['behavior_type_1'] * (df_part_2_c_b_count_in_1['cumcount']+1)
df_part_2_c_b_count_in_1['c_b2_count_in_1'] = df_part_2_c_b_count_in_1['behavior_type_2'] * (df_part_2_c_b_count_in_1['cumcount']+1)
df_part_2_c_b_count_in_1['c_b3_count_in_1'] = df_part_2_c_b_count_in_1['behavior_type_3'] * (df_part_2_c_b_count_in_1['cumcount']+1)
df_part_2_c_b_count_in_1['c_b4_count_in_1'] = df_part_2_c_b_count_in_1['behavior_type_4'] * (df_part_2_c_b_count_in_1['cumcount']+1)
df_part_2_c_b_count_in_1 = df_part_2_c_b_count_in_1[['item_category', 
                                                     'c_b1_count_in_1', 
                                                     'c_b2_count_in_1', 
                                                     'c_b3_count_in_1',
                                                     'c_b4_count_in_1']]
df_part_2_c_b_count_in_1 = df_part_2_c_b_count_in_1.groupby('item_category').agg({'c_b1_count_in_1': np.sum,
                                                                                  'c_b2_count_in_1': np.sum,
                                                                                  'c_b3_count_in_1': np.sum,
                                                                                  'c_b4_count_in_1': np.sum})
df_part_2_c_b_count_in_1.reset_index(inplace = True)
df_part_2_c_b_count_in_1['c_b_count_in_1'] = df_part_2_c_b_count_in_1['c_b1_count_in_1'] + \
                                             df_part_2_c_b_count_in_1['c_b2_count_in_1'] + \
                                             df_part_2_c_b_count_in_1['c_b3_count_in_1'] + \
                                             df_part_2_c_b_count_in_1['c_b4_count_in_1']    
                                             
df_part_2_c_b_count = pd.merge(df_part_2_c_b_count_in_6, df_part_2_c_b_count_in_3, on = ['item_category'], how = 'left').fillna(0)                                      
df_part_2_c_b_count = pd.merge(df_part_2_c_b_count, df_part_2_c_b_count_in_1, on = ['item_category'], how = 'left').fillna(0)
df_part_2_c_b_count[['c_b1_count_in_6',
                     'c_b2_count_in_6',
                     'c_b3_count_in_6',
                     'c_b4_count_in_6',
                      'c_b_count_in_6',
                     'c_b1_count_in_3',
                     'c_b2_count_in_3',
                     'c_b3_count_in_3',
                     'c_b4_count_in_3',
                      'c_b_count_in_3',
                     'c_b1_count_in_1',
                     'c_b2_count_in_1',
                     'c_b3_count_in_1',
                     'c_b4_count_in_1',
                      'c_b_count_in_1']] = df_part_2_c_b_count[['c_b1_count_in_6',
                                                                'c_b2_count_in_6',
                                                                'c_b3_count_in_6',
                                                                'c_b4_count_in_6',
                                                                 'c_b_count_in_6',
                                                                'c_b1_count_in_3',
                                                                'c_b2_count_in_3',
                                                                'c_b3_count_in_3',
                                                                'c_b4_count_in_3',
                                                                 'c_b_count_in_3',
                                                                'c_b1_count_in_1',
                                                                'c_b2_count_in_1',
                                                                'c_b3_count_in_1',
                                                                'c_b4_count_in_1',
                                                                 'c_b_count_in_1']].astype(int)

# c_b4_rate
df_part_2_c_b_count['c_b4_rate'] = df_part_2_c_b_count['c_b4_count_in_6'] / df_part_2_c_b_count['c_b_count_in_6']

# c_b4_diff_time
df_part_2 = df_part_2.sort_values(by=['item_category', 'time'])
df_part_2_c_b4_time = df_part_2[df_part_2['behavior_type'] == 4].drop_duplicates(['item_category'], 'first')[['item_category','time']]
df_part_2_c_b4_time.columns = ['item_category','b4_first_time']
df_part_2_c_b_time = df_part_2.drop_duplicates(['item_category'], 'first')[['item_category','time']]
df_part_2_c_b_time.columns = ['item_category','b_first_time']
df_part_2_c_b_b4_time = pd.merge(df_part_2_c_b_time, df_part_2_c_b4_time, on = ['item_category'])
df_part_2_c_b_b4_time['c_b4_diff_time']  = df_part_2_c_b_b4_time['b4_first_time'] - df_part_2_c_b_b4_time['b_first_time']
df_part_2_c_b_b4_time['c_b4_diff_hours'] = df_part_2_c_b_b4_time['c_b4_diff_time'].apply(lambda x: x.days * 24 + x.seconds//3600)
df_part_2_c_b_b4_time = df_part_2_c_b_b4_time[['item_category',
                                               'c_b4_diff_hours']]

# generating feature set C
f_C_part_2 = pd.merge(df_part_2_c_u_count, df_part_2_c_b_count, on = ['item_category'], how = 'left')
f_C_part_2 = pd.merge(f_C_part_2, df_part_2_c_b_b4_time, on = ['item_category'], how = 'left')
f_C_part_2 = f_C_part_2.round({'c_b4_rate': 3})

# write to csv file
f_C_part_2.to_csv(path_df_part_2_C, index = False)


############################################
'''Step 1.4 feature data set IC of df_part_2
    ic_u_rank_in_c  (in_6)
    ic_b_rank_in_c  (in_6)
    ic_b4_rank_in_c  (in_6)
'''


# get df_part_2_i_ub_count
path_df = open(path_df_part_2_I, 'r')
try:
    df_part_2_I = pd.read_csv(path_df, index_col = False)
finally:
    path_df.close()
df_part_2_i_ub_count = df_part_2_I[['item_id','i_u_count_in_6','i_b_count_in_6','i_b4_count_in_6']]
del(df_part_2_I)

# get df_part_2_uic for merge i & c
path_df = open(path_df_part_2_uic_label, 'r')
try:
    df_part_2_uic = pd.read_csv(path_df, index_col = False)
finally:
    path_df.close()
df_part_2_ic_u_b_count = pd.merge(df_part_2_uic, df_part_2_i_ub_count, on=['item_id'], how='left').fillna(0)
df_part_2_ic_u_b_count = df_part_2_ic_u_b_count.drop_duplicates(['item_id','item_category'])

# ic_u_rank_in_c
df_part_2_ic_u_b_count['ic_u_rank_in_c'] = df_part_2_ic_u_b_count.groupby('item_category')['i_u_count_in_6'].rank(method='min',ascending=False).astype('int')
# ic_b_rank_in_c
df_part_2_ic_u_b_count['ic_b_rank_in_c'] = df_part_2_ic_u_b_count.groupby('item_category')['i_b_count_in_6'].rank(method='min',ascending=False).astype('int')
# ic_b4_rank_in_c
df_part_2_ic_u_b_count['ic_b4_rank_in_c'] = df_part_2_ic_u_b_count.groupby('item_category')['i_b4_count_in_6'].rank(method='min',ascending=False).astype('int')

f_IC_part_2 = df_part_2_ic_u_b_count[['item_id', 
                                      'item_category', 
                                      'ic_u_rank_in_c', 
                                      'ic_b_rank_in_c', 
                                      'ic_b4_rank_in_c']]
# write to csv file
f_IC_part_2.to_csv(path_df_part_2_IC, index = False)


############################################
'''Step 1.5 feature data set UI of df_part_2
    (1)
    ui_b1_count_in_6
    ui_b2_count_in_6
    ui_b3_count_in_6
    ui_b4_count_in_6
     ui_b_count_in_6
    ui_b1_count_in_3
    ui_b2_count_in_3
    ui_b3_count_in_3
    ui_b4_count_in_3
     ui_b_count_in_3
    ui_b1_count_in_1
    ui_b2_count_in_1
    ui_b3_count_in_1
    ui_b4_count_in_1
     ui_b_count_in_1
    (2)
    ui_b_count_rank_in_u  (in_6)
    ui_b_count_rank_in_uc (in_6)
    (3)
    ui_b1_last_hours  (in_6)
    ui_b2_last_hours  (in_6)
    ui_b3_last_hours  (in_6)
    ui_b4_last_hours  (in_6)
'''


path_df = open(path_df_part_2, 'r')
try:
    df_part_2 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
    df_part_2.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    path_df.close()

# ui_b_count_in_6
df_part_2['cumcount'] = df_part_2.groupby(['user_id', 'item_id', 'behavior_type']).cumcount()
df_part_2_ui_b_count_in_6 = df_part_2.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','cumcount']]
df_part_2_ui_b_count_in_6 = pd.get_dummies(df_part_2_ui_b_count_in_6['behavior_type']).join(df_part_2_ui_b_count_in_6[['user_id','item_id','cumcount']])
df_part_2_ui_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_2_ui_b_count_in_6['ui_b1_count_in_6'] = df_part_2_ui_b_count_in_6['behavior_type_1'] * (df_part_2_ui_b_count_in_6['cumcount']+1)
df_part_2_ui_b_count_in_6['ui_b2_count_in_6'] = df_part_2_ui_b_count_in_6['behavior_type_2'] * (df_part_2_ui_b_count_in_6['cumcount']+1)
df_part_2_ui_b_count_in_6['ui_b3_count_in_6'] = df_part_2_ui_b_count_in_6['behavior_type_3'] * (df_part_2_ui_b_count_in_6['cumcount']+1)
df_part_2_ui_b_count_in_6['ui_b4_count_in_6'] = df_part_2_ui_b_count_in_6['behavior_type_4'] * (df_part_2_ui_b_count_in_6['cumcount']+1)
df_part_2_ui_b_count_in_6 = df_part_2_ui_b_count_in_6[['user_id', 
                                                       'item_id', 
                                                       'ui_b1_count_in_6', 
                                                       'ui_b2_count_in_6', 
                                                       'ui_b3_count_in_6',
                                                       'ui_b4_count_in_6']]
df_part_2_ui_b_count_in_6 = df_part_2_ui_b_count_in_6.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in_6': np.sum,
                                                                                           'ui_b2_count_in_6': np.sum,
                                                                                           'ui_b3_count_in_6': np.sum,
                                                                                           'ui_b4_count_in_6': np.sum})
df_part_2_ui_b_count_in_6.reset_index(inplace = True)
df_part_2_ui_b_count_in_6['ui_b_count_in_6'] = df_part_2_ui_b_count_in_6['ui_b1_count_in_6'] + \
                                               df_part_2_ui_b_count_in_6['ui_b2_count_in_6'] + \
                                               df_part_2_ui_b_count_in_6['ui_b3_count_in_6'] + \
                                               df_part_2_ui_b_count_in_6['ui_b4_count_in_6']

# ui_b_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')]
df_part_2_in_3['cumcount'] = df_part_2_in_3.groupby(['user_id', 'item_id', 'behavior_type']).cumcount()
df_part_2_ui_b_count_in_3 = df_part_2.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','cumcount']]
df_part_2_ui_b_count_in_3 = pd.get_dummies(df_part_2_ui_b_count_in_3['behavior_type']).join(df_part_2_ui_b_count_in_3[['user_id','item_id','cumcount']])
df_part_2_ui_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_2_ui_b_count_in_3['ui_b1_count_in_3'] = df_part_2_ui_b_count_in_3['behavior_type_1'] * (df_part_2_ui_b_count_in_3['cumcount']+1)
df_part_2_ui_b_count_in_3['ui_b2_count_in_3'] = df_part_2_ui_b_count_in_3['behavior_type_2'] * (df_part_2_ui_b_count_in_3['cumcount']+1)
df_part_2_ui_b_count_in_3['ui_b3_count_in_3'] = df_part_2_ui_b_count_in_3['behavior_type_3'] * (df_part_2_ui_b_count_in_3['cumcount']+1)
df_part_2_ui_b_count_in_3['ui_b4_count_in_3'] = df_part_2_ui_b_count_in_3['behavior_type_4'] * (df_part_2_ui_b_count_in_3['cumcount']+1)
df_part_2_ui_b_count_in_3 = df_part_2_ui_b_count_in_3[['user_id', 
                                                       'item_id', 
                                                       'ui_b1_count_in_3', 
                                                       'ui_b2_count_in_3', 
                                                       'ui_b3_count_in_3',
                                                       'ui_b4_count_in_3']]
df_part_2_ui_b_count_in_3 = df_part_2_ui_b_count_in_3.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in_3': np.sum,
                                                                                           'ui_b2_count_in_3': np.sum,
                                                                                           'ui_b3_count_in_3': np.sum,
                                                                                           'ui_b4_count_in_3': np.sum})
df_part_2_ui_b_count_in_3.reset_index(inplace = True)
df_part_2_ui_b_count_in_3['ui_b_count_in_3'] = df_part_2_ui_b_count_in_3['ui_b1_count_in_3'] + \
                                               df_part_2_ui_b_count_in_3['ui_b2_count_in_3'] + \
                                               df_part_2_ui_b_count_in_3['ui_b3_count_in_3'] + \
                                               df_part_2_ui_b_count_in_3['ui_b4_count_in_3']

# ui_b_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')]
df_part_2_in_1['cumcount'] = df_part_2_in_1.groupby(['user_id', 'item_id', 'behavior_type']).cumcount()
df_part_2_ui_b_count_in_1 = df_part_2_in_1.drop_duplicates(['user_id','item_id','behavior_type'], 'last')[['user_id','item_id','behavior_type','cumcount']]
df_part_2_ui_b_count_in_1 = pd.get_dummies(df_part_2_ui_b_count_in_1['behavior_type']).join(df_part_2_ui_b_count_in_1[['user_id','item_id','cumcount']])
df_part_2_ui_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)
df_part_2_ui_b_count_in_1['ui_b1_count_in_1'] = df_part_2_ui_b_count_in_1['behavior_type_1'] * (df_part_2_ui_b_count_in_1['cumcount']+1)
df_part_2_ui_b_count_in_1['ui_b2_count_in_1'] = df_part_2_ui_b_count_in_1['behavior_type_2'] * (df_part_2_ui_b_count_in_1['cumcount']+1)
df_part_2_ui_b_count_in_1['ui_b3_count_in_1'] = df_part_2_ui_b_count_in_1['behavior_type_3'] * (df_part_2_ui_b_count_in_1['cumcount']+1)
df_part_2_ui_b_count_in_1['ui_b4_count_in_1'] = df_part_2_ui_b_count_in_1['behavior_type_4'] * (df_part_2_ui_b_count_in_1['cumcount']+1)
df_part_2_ui_b_count_in_1 = df_part_2_ui_b_count_in_1[['user_id',
                                                       'item_id', 
                                                       'ui_b1_count_in_1', 
                                                       'ui_b2_count_in_1', 
                                                       'ui_b3_count_in_1',
                                                       'ui_b4_count_in_1']]
df_part_2_ui_b_count_in_1 = df_part_2_ui_b_count_in_1.groupby(['user_id', 'item_id']).agg({'ui_b1_count_in_1': np.sum,
                                                                                           'ui_b2_count_in_1': np.sum,
                                                                                           'ui_b3_count_in_1': np.sum,
                                                                                           'ui_b4_count_in_1': np.sum})
df_part_2_ui_b_count_in_1.reset_index(inplace = True)
df_part_2_ui_b_count_in_1['ui_b_count_in_1'] = df_part_2_ui_b_count_in_1['ui_b1_count_in_1'] + \
                                               df_part_2_ui_b_count_in_1['ui_b2_count_in_1'] + \
                                               df_part_2_ui_b_count_in_1['ui_b3_count_in_1'] + \
                                               df_part_2_ui_b_count_in_1['ui_b4_count_in_1']
                                             
df_part_2_ui_b_count = pd.merge(df_part_2_ui_b_count_in_6, df_part_2_ui_b_count_in_3, on = ['user_id','item_id'], how = 'left').fillna(0)
df_part_2_ui_b_count = pd.merge(df_part_2_ui_b_count, df_part_2_ui_b_count_in_1, on = ['user_id','item_id'], how = 'left').fillna(0)
df_part_2_ui_b_count[['ui_b1_count_in_6',
                      'ui_b2_count_in_6',
                      'ui_b3_count_in_6',
                      'ui_b4_count_in_6',
                       'ui_b_count_in_6',
                      'ui_b1_count_in_3',
                      'ui_b2_count_in_3',
                      'ui_b3_count_in_3',
                      'ui_b4_count_in_3',
                       'ui_b_count_in_3',
                      'ui_b1_count_in_1',
                      'ui_b2_count_in_1',
                      'ui_b3_count_in_1',
                      'ui_b4_count_in_1',
                       'ui_b_count_in_1']] = df_part_2_ui_b_count[['ui_b1_count_in_6',
                                                                   'ui_b2_count_in_6',
                                                                   'ui_b3_count_in_6',
                                                                   'ui_b4_count_in_6',
                                                                    'ui_b_count_in_6',
                                                                   'ui_b1_count_in_3',
                                                                   'ui_b2_count_in_3',
                                                                   'ui_b3_count_in_3',
                                                                   'ui_b4_count_in_3',
                                                                    'ui_b_count_in_3',
                                                                   'ui_b1_count_in_1',
                                                                   'ui_b2_count_in_1',
                                                                   'ui_b3_count_in_1',
                                                                   'ui_b4_count_in_1',
                                                                    'ui_b_count_in_1']].astype(int)

# ui_b_count_rank_in_u
df_part_2_ui_b_count['ui_b_count_rank_in_u'] = df_part_2_ui_b_count.groupby(['user_id'])['ui_b_count_in_6'].rank(method='min',ascending=False).astype('int')

# ui_b_count_rank_in_uc
path_df = open(path_df_part_2_uic_label, 'r')
try:
    df_part_2_uic = pd.read_csv(path_df, index_col = False)
finally:
    path_df.close()
df_part_2_ui_b_count = pd.merge(df_part_2_uic, df_part_2_ui_b_count, on = ['user_id','item_id'], how = 'left')
df_part_2_ui_b_count['ui_b_count_rank_in_uc'] = df_part_2_ui_b_count.groupby(['user_id','item_category'])['ui_b_count_rank_in_u'].rank(method='min',ascending=True).astype('int')


# ui_b_last_time
df_part_2.sort_values(by=['user_id','item_id','behavior_type','time'], inplace=True)
df_part_2_ui_b_last_time = df_part_2.drop_duplicates(['user_id','item_id','behavior_type'],'last')[['user_id','item_id','behavior_type','time']]

df_part_2_ui_b_last_time['ui_b1_last_time'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['behavior_type'] == 1]['time']
df_part_2_ui_b_last_time['ui_b2_last_time'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['behavior_type'] == 2]['time']
df_part_2_ui_b_last_time['ui_b3_last_time'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['behavior_type'] == 3]['time']
df_part_2_ui_b_last_time['ui_b4_last_time'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['behavior_type'] == 4]['time']

df_part_2_ui_b_last_time.loc[df_part_2_ui_b_last_time['ui_b1_last_time'].notnull(), 'ui_b1_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_ui_b_last_time['ui_b1_last_time'])             
df_part_2_ui_b_last_time['ui_b1_last_hours'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['ui_b1_last_hours'].notnull()]['ui_b1_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_ui_b_last_time.loc[df_part_2_ui_b_last_time['ui_b2_last_time'].notnull(), 'ui_b2_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_ui_b_last_time['ui_b2_last_time'])             
df_part_2_ui_b_last_time['ui_b2_last_hours'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['ui_b2_last_hours'].notnull()]['ui_b2_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_ui_b_last_time.loc[df_part_2_ui_b_last_time['ui_b3_last_time'].notnull(), 'ui_b3_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_ui_b_last_time['ui_b3_last_time'])             
df_part_2_ui_b_last_time['ui_b3_last_hours'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['ui_b3_last_hours'].notnull()]['ui_b3_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_ui_b_last_time.loc[df_part_2_ui_b_last_time['ui_b4_last_time'].notnull(), 'ui_b4_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_ui_b_last_time['ui_b4_last_time'])             
df_part_2_ui_b_last_time['ui_b4_last_hours'] = df_part_2_ui_b_last_time[df_part_2_ui_b_last_time['ui_b4_last_hours'].notnull()]['ui_b4_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_ui_b_last_time = df_part_2_ui_b_last_time[['user_id',
                                                     'item_id',
                                                     'ui_b1_last_hours',
                                                     'ui_b2_last_hours',
                                                     'ui_b3_last_hours',
                                                     'ui_b4_last_hours']] 

df_part_2_ui_b_last_time = df_part_2_ui_b_last_time.groupby(['user_id', 'item_id']).agg({'ui_b1_last_hours': np.sum,
                                                                                         'ui_b2_last_hours': np.sum,
                                                                                         'ui_b3_last_hours': np.sum,
                                                                                         'ui_b4_last_hours': np.sum})
df_part_2_ui_b_last_time.reset_index(inplace = True)

# merge for generation of f_UI_part_2
f_UI_part_2 = pd.merge(df_part_2_ui_b_count, df_part_2_ui_b_last_time, how='left', on=['user_id', 'item_id'])

# write to csv file
f_UI_part_2.to_csv(path_df_part_2_UI, index = False)


############################################
'''Step 1.6 feature data set UC of df_part_2
    (1)
    uc_b1_count_in_6
    uc_b2_count_in_6
    uc_b3_count_in_6
    uc_b4_count_in_6
     uc_b_count_in_6
    uc_b1_count_in_3
    uc_b2_count_in_3
    uc_b3_count_in_3
    uc_b4_count_in_3
     uc_b_count_in_3
    uc_b1_count_in_1
    uc_b2_count_in_1
    uc_b3_count_in_1
    uc_b4_count_in_1
     uc_b_count_in_1
    (2)
    uc_b_count_rank_in_u  (in_6)
    (3)
    uc_b1_last_hours  (in_6)
    uc_b2_last_hours  (in_6)
    uc_b3_last_hours  (in_6)
    uc_b4_last_hours  (in_6)
'''

path_df = open(path_df_part_2, 'r')
try:
    df_part_2 = pd.read_csv(path_df, index_col = False, parse_dates = [0])
    df_part_2.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    path_df.close()

# uc_b_count_in_6
df_part_2['cumcount'] = df_part_2.groupby(['user_id', 'item_category', 'behavior_type']).cumcount()
df_part_2_uc_b_count_in_6 = df_part_2.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','cumcount']]
df_part_2_uc_b_count_in_6 = pd.get_dummies(df_part_2_uc_b_count_in_6['behavior_type']).join(df_part_2_uc_b_count_in_6[['user_id','item_category','cumcount']])
df_part_2_uc_b_count_in_6.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_2_uc_b_count_in_6['uc_b1_count_in_6'] = df_part_2_uc_b_count_in_6['behavior_type_1'] * (df_part_2_uc_b_count_in_6['cumcount']+1)
df_part_2_uc_b_count_in_6['uc_b2_count_in_6'] = df_part_2_uc_b_count_in_6['behavior_type_2'] * (df_part_2_uc_b_count_in_6['cumcount']+1)
df_part_2_uc_b_count_in_6['uc_b3_count_in_6'] = df_part_2_uc_b_count_in_6['behavior_type_3'] * (df_part_2_uc_b_count_in_6['cumcount']+1)
df_part_2_uc_b_count_in_6['uc_b4_count_in_6'] = df_part_2_uc_b_count_in_6['behavior_type_4'] * (df_part_2_uc_b_count_in_6['cumcount']+1)
df_part_2_uc_b_count_in_6 = df_part_2_uc_b_count_in_6[['user_id', 
                                                       'item_category', 
                                                       'uc_b1_count_in_6', 
                                                       'uc_b2_count_in_6', 
                                                       'uc_b3_count_in_6',
                                                       'uc_b4_count_in_6']]
df_part_2_uc_b_count_in_6 = df_part_2_uc_b_count_in_6.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in_6': np.sum,
                                                                                                 'uc_b2_count_in_6': np.sum,
                                                                                                 'uc_b3_count_in_6': np.sum,
                                                                                                 'uc_b4_count_in_6': np.sum})
df_part_2_uc_b_count_in_6.reset_index(inplace = True)
df_part_2_uc_b_count_in_6['uc_b_count_in_6'] = df_part_2_uc_b_count_in_6['uc_b1_count_in_6'] + \
                                               df_part_2_uc_b_count_in_6['uc_b2_count_in_6'] + \
                                               df_part_2_uc_b_count_in_6['uc_b3_count_in_6'] + \
                                               df_part_2_uc_b_count_in_6['uc_b4_count_in_6']

# uc_b_count_in_3
df_part_2_in_3 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-02')]
df_part_2_in_3['cumcount'] = df_part_2_in_3.groupby(['user_id', 'item_category', 'behavior_type']).cumcount()
df_part_2_uc_b_count_in_3 = df_part_2.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','cumcount']]
df_part_2_uc_b_count_in_3 = pd.get_dummies(df_part_2_uc_b_count_in_3['behavior_type']).join(df_part_2_uc_b_count_in_3[['user_id','item_category','cumcount']])
df_part_2_uc_b_count_in_3.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)  
df_part_2_uc_b_count_in_3['uc_b1_count_in_3'] = df_part_2_uc_b_count_in_3['behavior_type_1'] * (df_part_2_uc_b_count_in_3['cumcount']+1)
df_part_2_uc_b_count_in_3['uc_b2_count_in_3'] = df_part_2_uc_b_count_in_3['behavior_type_2'] * (df_part_2_uc_b_count_in_3['cumcount']+1)
df_part_2_uc_b_count_in_3['uc_b3_count_in_3'] = df_part_2_uc_b_count_in_3['behavior_type_3'] * (df_part_2_uc_b_count_in_3['cumcount']+1)
df_part_2_uc_b_count_in_3['uc_b4_count_in_3'] = df_part_2_uc_b_count_in_3['behavior_type_4'] * (df_part_2_uc_b_count_in_3['cumcount']+1)
df_part_2_uc_b_count_in_3 = df_part_2_uc_b_count_in_3[['user_id', 
                                                       'item_category', 
                                                       'uc_b1_count_in_3', 
                                                       'uc_b2_count_in_3', 
                                                       'uc_b3_count_in_3',
                                                       'uc_b4_count_in_3']]
df_part_2_uc_b_count_in_3 = df_part_2_uc_b_count_in_3.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in_3': np.sum,
                                                                                                 'uc_b2_count_in_3': np.sum,
                                                                                                 'uc_b3_count_in_3': np.sum,
                                                                                                 'uc_b4_count_in_3': np.sum})
df_part_2_uc_b_count_in_3.reset_index(inplace = True)
df_part_2_uc_b_count_in_3['uc_b_count_in_3'] = df_part_2_uc_b_count_in_3['uc_b1_count_in_3'] + \
                                               df_part_2_uc_b_count_in_3['uc_b2_count_in_3'] + \
                                               df_part_2_uc_b_count_in_3['uc_b3_count_in_3'] + \
                                               df_part_2_uc_b_count_in_3['uc_b4_count_in_3']

# uc_b_count_in_1
df_part_2_in_1 = df_part_2[df_part_2['time'] >= np.datetime64('2014-12-04')]
df_part_2_in_1['cumcount'] = df_part_2_in_1.groupby(['user_id', 'item_category', 'behavior_type']).cumcount()
df_part_2_uc_b_count_in_1 = df_part_2_in_1.drop_duplicates(['user_id','item_category','behavior_type'], 'last')[['user_id','item_category','behavior_type','cumcount']]
df_part_2_uc_b_count_in_1 = pd.get_dummies(df_part_2_uc_b_count_in_1['behavior_type']).join(df_part_2_uc_b_count_in_1[['user_id','item_category','cumcount']])
df_part_2_uc_b_count_in_1.rename(columns = {1:'behavior_type_1',
                                            2:'behavior_type_2',
                                            3:'behavior_type_3',
                                            4:'behavior_type_4'}, inplace=True)
df_part_2_uc_b_count_in_1['uc_b1_count_in_1'] = df_part_2_uc_b_count_in_1['behavior_type_1'] * (df_part_2_uc_b_count_in_1['cumcount']+1)
df_part_2_uc_b_count_in_1['uc_b2_count_in_1'] = df_part_2_uc_b_count_in_1['behavior_type_2'] * (df_part_2_uc_b_count_in_1['cumcount']+1)
df_part_2_uc_b_count_in_1['uc_b3_count_in_1'] = df_part_2_uc_b_count_in_1['behavior_type_3'] * (df_part_2_uc_b_count_in_1['cumcount']+1)
df_part_2_uc_b_count_in_1['uc_b4_count_in_1'] = df_part_2_uc_b_count_in_1['behavior_type_4'] * (df_part_2_uc_b_count_in_1['cumcount']+1)
df_part_2_uc_b_count_in_1 = df_part_2_uc_b_count_in_1[['user_id',
                                                       'item_category', 
                                                       'uc_b1_count_in_1', 
                                                       'uc_b2_count_in_1', 
                                                       'uc_b3_count_in_1',
                                                       'uc_b4_count_in_1']]
df_part_2_uc_b_count_in_1 = df_part_2_uc_b_count_in_1.groupby(['user_id', 'item_category']).agg({'uc_b1_count_in_1': np.sum,
                                                                                                 'uc_b2_count_in_1': np.sum,
                                                                                                 'uc_b3_count_in_1': np.sum,
                                                                                                 'uc_b4_count_in_1': np.sum})
df_part_2_uc_b_count_in_1.reset_index(inplace = True)
df_part_2_uc_b_count_in_1['uc_b_count_in_1'] = df_part_2_uc_b_count_in_1['uc_b1_count_in_1'] + \
                                               df_part_2_uc_b_count_in_1['uc_b2_count_in_1'] + \
                                               df_part_2_uc_b_count_in_1['uc_b3_count_in_1'] + \
                                               df_part_2_uc_b_count_in_1['uc_b4_count_in_1']
                                             
df_part_2_uc_b_count = pd.merge(df_part_2_uc_b_count_in_6, df_part_2_uc_b_count_in_3, on = ['user_id','item_category'], how = 'left').fillna(0)
df_part_2_uc_b_count = pd.merge(df_part_2_uc_b_count, df_part_2_uc_b_count_in_1, on = ['user_id','item_category'], how = 'left').fillna(0)
df_part_2_uc_b_count[['uc_b1_count_in_6',
                      'uc_b2_count_in_6',
                      'uc_b3_count_in_6',
                      'uc_b4_count_in_6',
                       'uc_b_count_in_6',
                      'uc_b1_count_in_3',
                      'uc_b2_count_in_3',
                      'uc_b3_count_in_3',
                      'uc_b4_count_in_3',
                       'uc_b_count_in_3',
                      'uc_b1_count_in_1',
                      'uc_b2_count_in_1',
                      'uc_b3_count_in_1',
                      'uc_b4_count_in_1',
                       'uc_b_count_in_1']] = df_part_2_uc_b_count[['uc_b1_count_in_6',
                                                                   'uc_b2_count_in_6',
                                                                   'uc_b3_count_in_6',
                                                                   'uc_b4_count_in_6',
                                                                    'uc_b_count_in_6',
                                                                   'uc_b1_count_in_3',
                                                                   'uc_b2_count_in_3',
                                                                   'uc_b3_count_in_3',
                                                                   'uc_b4_count_in_3',
                                                                    'uc_b_count_in_3',
                                                                   'uc_b1_count_in_1',
                                                                   'uc_b2_count_in_1',
                                                                   'uc_b3_count_in_1',
                                                                   'uc_b4_count_in_1',
                                                                    'uc_b_count_in_1']].astype(int)

# uc_b_count_rank_in_u
df_part_2_uc_b_count['uc_b_count_rank_in_u'] = df_part_2_uc_b_count.groupby(['user_id'])['uc_b_count_in_6'].rank(method='min',ascending=False).astype('int')

# uc_b_last_time
df_part_2.sort_values(by=['user_id','item_category','behavior_type','time'], inplace=True)
df_part_2_uc_b_last_time = df_part_2.drop_duplicates(['user_id','item_category','behavior_type'],'last')[['user_id','item_category','behavior_type','time']]

df_part_2_uc_b_last_time['uc_b1_last_time'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['behavior_type'] == 1]['time']
df_part_2_uc_b_last_time['uc_b2_last_time'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['behavior_type'] == 2]['time']
df_part_2_uc_b_last_time['uc_b3_last_time'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['behavior_type'] == 3]['time']
df_part_2_uc_b_last_time['uc_b4_last_time'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['behavior_type'] == 4]['time']

df_part_2_uc_b_last_time.loc[df_part_2_uc_b_last_time['uc_b1_last_time'].notnull(), 'uc_b1_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_uc_b_last_time['uc_b1_last_time'])             
df_part_2_uc_b_last_time['uc_b1_last_hours'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['uc_b1_last_hours'].notnull()]['uc_b1_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_uc_b_last_time.loc[df_part_2_uc_b_last_time['uc_b2_last_time'].notnull(), 'uc_b2_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_uc_b_last_time['uc_b2_last_time'])             
df_part_2_uc_b_last_time['uc_b2_last_hours'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['uc_b2_last_hours'].notnull()]['uc_b2_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_uc_b_last_time.loc[df_part_2_uc_b_last_time['uc_b3_last_time'].notnull(), 'uc_b3_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_uc_b_last_time['uc_b3_last_time'])             
df_part_2_uc_b_last_time['uc_b3_last_hours'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['uc_b3_last_hours'].notnull()]['uc_b3_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_uc_b_last_time.loc[df_part_2_uc_b_last_time['uc_b4_last_time'].notnull(), 'uc_b4_last_hours'] = (pd.to_datetime('2014-12-05') - df_part_2_uc_b_last_time['uc_b4_last_time'])             
df_part_2_uc_b_last_time['uc_b4_last_hours'] = df_part_2_uc_b_last_time[df_part_2_uc_b_last_time['uc_b4_last_hours'].notnull()]['uc_b4_last_hours'].apply(lambda x: x.days*24 + x.seconds//3600)

df_part_2_uc_b_last_time = df_part_2_uc_b_last_time[['user_id',
                                                     'item_category',
                                                     'uc_b1_last_hours',
                                                     'uc_b2_last_hours',
                                                     'uc_b3_last_hours',
                                                     'uc_b4_last_hours']] 

df_part_2_uc_b_last_time = df_part_2_uc_b_last_time.groupby(['user_id', 'item_category']).agg({'uc_b1_last_hours': np.sum,
                                                                                               'uc_b2_last_hours': np.sum,
                                                                                               'uc_b3_last_hours': np.sum,
                                                                                               'uc_b4_last_hours': np.sum})
df_part_2_uc_b_last_time.reset_index(inplace = True)

# merge for generation of f_UC_part_2
f_UC_part_2 = pd.merge(df_part_2_uc_b_count, df_part_2_uc_b_last_time, how='left', on=['user_id', 'item_category'])

# write to csv file
f_UC_part_2.to_csv(path_df_part_2_UC, index = False)


print(' - PY131 - ')