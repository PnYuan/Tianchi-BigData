# -*- coding: utf-8 -*
    
'''
@author: PY131

@thoughts:  as the samples are extremely imbalance (N/P ratio ~ 1.2k),
            here we use sub-sample on negative samples.
            1-st: using k_means to make clustering on negative samples (clusters_number ~ 1k)
            2-nd: subsample on each clusters based on the same ratio,
                  the ratio was selected to be the best by testing in random sub_sample + GBDT
            3-rd: selecting the best parameter for GBDT classifier
            4-th: using GBDT model for training and predicting on sub_sample set.
            
            here is 2-nd to 4-th step
'''

########## file path ##########
##### input file
# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "../../data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "../../data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic       = "../../data/mobile/raw/df_part_3_uic.csv"

# data_set features
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

# item_sub_set P
path_df_P = "../../data/raw/tianchi_fresh_comp_train_item.csv"

##### output file
path_df_result = "../../data/mobile/gbdt/res_gbdt_k_means_subsample.csv"
path_df_result_tmp = "../../data/mobile/gbdt/df_result_tmp.csv"

# depending package
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import time

# some functions
def df_read(path, mode = 'r'):
    '''the definition of dataframe loading function 
    '''
    data_file = open(path, mode)
    try:     df = pd.read_csv(data_file, index_col = False)
    finally: data_file.close()
    return   df

def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size
    
    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df) : return df
    else : return df.sample(n = sub_size)

##### loading data of part 1 & 2
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)

df_part_1_U  = df_read(path_df_part_1_U )   
df_part_1_I  = df_read(path_df_part_1_I )
df_part_1_C  = df_read(path_df_part_1_C )
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U  = df_read(path_df_part_2_U )   
df_part_2_I  = df_read(path_df_part_2_I )
df_part_2_C  = df_read(path_df_part_2_C )
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

##### generation of training set & valid set
def train_set_construct(np_ratio = 1, sub_ratio = 1):
    '''
    # generation of train set
    @param np_ratio: int, the sub-sample rate of training set for N/P balanced.
    @param sub_ratio: float ~ (0~1], the further sub-sample rate of training set after N/P balanced.
    '''
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    
    frac_ratio = sub_ratio * np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    print("training subset uic_label keys is selected.")
    
    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)
    
    # using all the features for training gbdt model
    train_X = train_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                  'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                  'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                  'u_b4_rate','u_b4_diff_hours',
                                  'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                  'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                  'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                  'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                  'i_b4_rate','i_b4_diff_hours',
                                  'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                                  'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                  'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                  'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                  'c_b4_rate','c_b4_diff_hours',
                                  'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                  'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                  'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                  'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                  'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                  'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                                  'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                  'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                  'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                  'uc_b_count_rank_in_u',
                                  'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours'])
    train_y = train_df['label'].values
    print("train subset is generated.")
    return train_X, train_y
    
def valid_set_construct(sub_ratio = 0.1):
    '''
    # generation of valid set
    @param sub_ratio: float ~ (0~1], the sub-sample rate of original valid set
    '''
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)

    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
    
        valid_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
    # constructing valid set
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)
    
    # using all the features for valid gbdt model
    valid_X = valid_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                  'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                  'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                  'u_b4_rate','u_b4_diff_hours',
                                  'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                  'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                  'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                  'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                  'i_b4_rate','i_b4_diff_hours',
                                  'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                                  'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                  'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                  'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                  'c_b4_rate','c_b4_diff_hours',
                                  'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                  'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                  'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                  'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                  'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                  'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                                  'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                  'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                  'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                  'uc_b_count_rank_in_u',
                                  'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours'])
    valid_y = valid_df['label'].values
    print("valid subset is generated.")
 
    return valid_X, valid_y


##### generation and splitting to training set & valid set
def valid_train_set_construct(valid_ratio = 0.5, valid_sub_ratio = 0.5, train_np_ratio = 1, train_sub_ratio = 0.5):
    '''
    # generation of train set
    @param valid_ratio: float ~ [0~1], the valid set ratio in total set and the rest is train set
    @param valid_sub_ratio: float ~ (0~1), random sample ratio of valid set
    @param train_np_ratio:(1~1200), the sub-sample ratio of training set for N/P balanced.
    @param train_sub_ratio: float ~ (0~1), random sample ratio of train set after N/P subsample
    
    @return valid_X, valid_y, train_X, train_y
    '''
    msk_1 = np.random.rand(len(df_part_1_uic_label_cluster)) < valid_ratio
    msk_2 = np.random.rand(len(df_part_2_uic_label_cluster)) < valid_ratio
        
    valid_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[msk_1]
    valid_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[msk_2]
    
    valid_part_1_uic_label = valid_df_part_1_uic_label_cluster[ valid_df_part_1_uic_label_cluster['class'] == 0 ].sample(frac = valid_sub_ratio)
    valid_part_2_uic_label = valid_df_part_2_uic_label_cluster[ valid_df_part_2_uic_label_cluster['class'] == 0 ].sample(frac = valid_sub_ratio)
    
    ### constructing valid set
    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == i]
        if len(valid_part_1_uic_label_0_i) != 0:
            valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = valid_sub_ratio)
            valid_part_1_uic_label     = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
        
        valid_part_2_uic_label_0_i = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == i]
        if len(valid_part_2_uic_label_0_i) != 0:
            valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = valid_sub_ratio)
            valid_part_2_uic_label     = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)
    
    # using all the features for valid rf model
    valid_X = valid_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                  'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                  'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                  'u_b4_rate','u_b4_diff_hours',
                                  'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                  'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                  'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                  'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                  'i_b4_rate','i_b4_diff_hours',
                                  'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                                  'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                  'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                  'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                  'c_b4_rate','c_b4_diff_hours',
                                  'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                  'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                  'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                  'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                  'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                  'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                                  'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                  'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                  'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                  'uc_b_count_rank_in_u',
                                  'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours'])
    valid_y = valid_df['label'].values
    print("valid subset is generated.")

    ### constructing training set
    train_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[~msk_1]
    train_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[~msk_2] 
    
    train_part_1_uic_label = train_df_part_1_uic_label_cluster[ train_df_part_1_uic_label_cluster['class'] == 0 ].sample(frac = train_sub_ratio)
    train_part_2_uic_label = train_df_part_2_uic_label_cluster[ train_df_part_2_uic_label_cluster['class'] == 0 ].sample(frac = train_sub_ratio)
    
    frac_ratio = train_sub_ratio * train_np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == i]
        if len(train_part_1_uic_label_0_i) != 0:
            train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
            train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == i]
        if len(train_part_2_uic_label_0_i) != 0:
            train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
            train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    
    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])

    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)
    
    # using all the features for training rf model
    train_X = train_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                  'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                  'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                  'u_b4_rate','u_b4_diff_hours',
                                  'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                  'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                  'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                  'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                  'i_b4_rate','i_b4_diff_hours',
                                  'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                                  'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                  'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                  'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                  'c_b4_rate','c_b4_diff_hours',
                                  'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                  'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                  'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                  'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                  'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                  'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                                  'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                  'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                  'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                  'uc_b_count_rank_in_u',
                                  'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours'])
    train_y = train_df['label'].values
    print("train subset is generated.")
    
    return valid_X, valid_y, train_X, train_y


#######################################################################
'''Step 1: training for analysis of the best GBDT model (parameters tuning)
        (1). selection a suitable N/P ratio of sample space
        (2). selection for best n_estimators & learning_rate for GBDT
        (3). selection for best max_depth & min_samples_split & min_samples_leaf for base tree
        (4). selection for best prediction cutoff for GBDT
'''

'''
########## (1) selection for best N/P ratio of subsample
# generation of gbdt model
gbdt_clf = GradientBoostingClassifier(learning_rate=0.05, 
                                      n_estimators=200, 
                                      max_depth=7, 
                                      subsample=0.8,
                                      max_features="sqrt", 
                                      verbose=True)
    
f1_scores = []
np_ratios = []
for np_ratio in [1,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,120,150,200,300]:
    t1 = time.time()
    # generation of training and valid set
    valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2,
                                                                   valid_sub_ratio = 1, 
                                                                   train_np_ratio = np_ratio,
                                                                   train_sub_ratio = 1)
    
    gbdt_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = gbdt_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    np_ratios.append(np_ratio)
    
    print('gbdt_clf [NP ratio = %d] is fitted' % np_ratio)
    t2 = time.time() 
    print('time used %d s' % (t2-t1))
    
# plot the result
f1 = plt.figure(1)
plt.plot(np_ratios, f1_scores, label="lr=0.05,nt=200,md=7,sub=0.8,sqrt")
plt.xlabel('NP ratio')
plt.ylabel('f1_score')
plt.title('f1_score as function of NP ratio - GBDT')
plt.legend(loc=4)
plt.grid(True, linewidth=0.5)
plt.show()
'''

'''
########## (2) selection for best n_estimators and learning_rate of GBDT
# training and validating data set generation
valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2, 
                                                               valid_sub_ratio = 1, 
                                                               train_np_ratio = 60,
                                                               train_sub_ratio = 1)

learning_rates = []
f1_matrix = []
for lr in [0.05, 0.1, 0.15, 0.2]:
    n_trees = []
    f1_scores = []
    for nt in [10,20,30,40,50,60,70,80,90,100,120,150,180,240,300,400]:
        t1 = time.time()
        # generation of training and valid set
        
        # generation of GBDT model and fit
        GBDT_clf = GradientBoostingClassifier(learning_rate=lr,
                                              n_estimators=nt,
                                              max_depth=7,  
                                              subsample=0.8,
                                              max_features="sqrt",
                                              verbose=True)
        GBDT_clf.fit(train_X, train_y)
        
        # validation and evaluation
        valid_y_pred = GBDT_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        n_trees.append(nt)
        
        print('GBDT_clf [lr = %.2f, nt = %d] is fitted' % (lr, nt))
        t2 = time.time() 
        print('time used %d s' % (t2-t1))
        
    f1_matrix.append(f1_scores)
    learning_rates.append(lr)
    
# plot the result
f1 = plt.figure(1)
i = 0
for f1_scores in f1_matrix:
    plt.plot(n_trees, f1_scores, label="lr=%.2f" % learning_rates[i])
    i += 1
    
plt.xlabel('n_trees')
plt.ylabel('f1_score')
plt.title('f1_score as function of GBDT lr & nt (np=60,md=7)')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()
'''

'''
########## (2) selection for best n_estimators as learning_rate=0.05 of GBDT
# training and validating data set generation
valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2, 
                                                               valid_sub_ratio = 1, 
                                                               train_np_ratio = 50,
                                                               train_sub_ratio = 1)

n_trees = []
f1_scores = []
for nt in [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,400,500]:
    t1 = time.time()
    
    # generation of GBDT model and fit
    GBDT_clf = GradientBoostingClassifier(n_estimators=nt,
                                          learning_rate=0.05,
                                          max_depth=5,
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    GBDT_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = GBDT_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    n_trees.append(nt)
    
    print('GBDT_clf nt = %d is fitted' % nt)
    t2 = time.time() 
    print('time used %d s' % (t2-t1))
        
# plot the result
f1 = plt.figure(1)
plt.plot(n_trees, f1_scores, label="np=50,lr=0.05,md=5,sub=0.8,sqrt")
plt.xlabel('n_trees')
plt.ylabel('f1_score')
plt.title('f1_score as function of GBDT nt')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()
'''

'''
########## (3.1) selection for best max_depth of GBDT
# training and validating
valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2, 
                                                               valid_sub_ratio = 1, 
                                                               train_np_ratio = 60,
                                                               train_sub_ratio = 1)

max_depths = []
f1_scores = []
for md in [2,3,4,5,6,7,8,9,10,12,15,20]:
    t1 = time.time()
    # generation of training and valid set
    
    # generation of GBDT model and fit
    GBDT_clf = GradientBoostingClassifier(max_depth=md,
                                          learning_rate=0.05,
                                          n_estimators=150, 
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    GBDT_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = GBDT_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    max_depths.append(md)
    print('GBDT_clf [max_depth = %d] is fitted' % md)
    
    t2 = time.time() 
    print('time used %d s' % (t2-t1))
   
# plot the result
f1 = plt.figure(1)
plt.plot(max_depths, f1_scores, label="np=60,lr=0.05,nt=150,sub0.8,sqrt")

plt.xlabel('max_depths')
plt.ylabel('f1_score')
plt.title('f1_score as function of GBDT max_depths')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()
'''

'''
########## (3.2) selection for best min_samples_split of GBDT
# training and validating
valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2, 
                                                               valid_sub_ratio = 1, 
                                                               train_np_ratio = 60,
                                                               train_sub_ratio = 1)

min_samples_splits = []
f1_scores = []
for mss in [2,5,10,20,50,100,500,1000,5000]:
    t1 = time.time()
    # generation of training and valid set
    
    # generation of GBDT model and fit
    GBDT_clf = GradientBoostingClassifier(min_samples_split=mss,
                                          max_depth=5, 
                                          learning_rate=0.05,
                                          n_estimators=150, 
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    GBDT_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = GBDT_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    min_samples_splits.append(mss)
    print('GBDT_clf [min_samples_splits = %d] is fitted' % mss)
    
    t2 = time.time() 
    print('time used %d s' % (t2-t1))
   
# plot the result
f1 = plt.figure(1)
plt.plot(min_samples_splits, f1_scores, label="np=60,lr=0.05,nt=150,md=5,sub0.8,")

plt.xlabel('min_samples_split')
plt.ylabel('f1_score')
plt.title('f1_score as function of GBDT min_samples_split')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()
'''

'''
########## (3.2) selection for best min_samples_leaf of GBDT
# training and validating
valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2, 
                                                               valid_sub_ratio = 1, 
                                                               train_np_ratio = 60,
                                                               train_sub_ratio = 1)

min_samples_leafs = []
f1_scores = []
for msl in range(2,30,2):
    t1 = time.time()
    # generation of training and valid set
    
    # generation of GBDT model and fit
    GBDT_clf = GradientBoostingClassifier(min_samples_leaf=msl,
                                          learning_rate=0.05,
                                          n_estimators=150, 
                                          max_depth=4, 
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    GBDT_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = GBDT_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    min_samples_leafs.append(msl)
    print('GBDT_clf [min_samples_leaf = %d] is fitted' % msl)
    
    t2 = time.time() 
    print('time used %d s' % (t2-t1))
   
# plot the result
f1 = plt.figure(1)
plt.plot(min_samples_leafs, f1_scores, label="np=60,lr=0.05,nt=150,md=5,sub0.8,sqrt")

plt.xlabel('min_samples_leaf')
plt.ylabel('f1_score')
plt.title('f1_score as function of GBDT min_samples_leaf')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()
'''

'''
########## (4) selection for best prediction cutoff for GBDT
# training and validating
valid_X, valid_y, train_X, train_y = valid_train_set_construct(valid_ratio = 0.2, 
                                                               valid_sub_ratio = 1, 
                                                               train_np_ratio = 60,
                                                               train_sub_ratio = 1)
# generation of GBDT model and fit
GBDT_clf = GradientBoostingClassifier(learning_rate=0.025,
                                      n_estimators=300, 
                                      max_depth=4,
                                      min_samples_leaf=10,
                                      subsample=0.8,
                                      max_features="sqrt",
                                      verbose=True)
GBDT_clf.fit(train_X, train_y)
cut_offs = []
f1_scores = []
for co in np.arange(0.05,1,0.05):
    t1 = time.time()
    # validation and evaluation
    valid_y_pred = (GBDT_clf.predict_proba(valid_X)[:,1] > co).astype(int)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    cut_offs.append(co)
    print('GBDT_clf [cutoff = %.2f] is fitted' % co)
    
    t2 = time.time() 
    print('time used %d s' % (t2-t1))
   
# plot the result
f1 = plt.figure(1)
plt.plot(cut_offs, f1_scores, label="np=60,lr=0.025,nt=300,md=4,msl=10,sub=0.8,sqrt")

plt.xlabel('cut_offs')
plt.ylabel('f1_score')
plt.title('f1_score as function of GBDT predict cutoff')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()
'''

#######################################################################
'''Step 2: training the optimal GBDT model and predicting on part_3 
'''
train_X, train_y = train_set_construct(np_ratio=60, sub_ratio=1)

# build model and fitting
GBDT_clf = GradientBoostingClassifier(max_depth=4,
                                      min_samples_leaf=10,
                                      learning_rate=0.025,
                                      n_estimators=300,
                                      subsample=0.8,
                                      max_features="sqrt",
                                      verbose=True)
GBDT_clf.fit(train_X, train_y)

##### predicting
# loading feature data
df_part_3_U  = df_read(path_df_part_3_U )   
df_part_3_I  = df_read(path_df_part_3_I )
df_part_3_C  = df_read(path_df_part_3_C )
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)

# process by chunk as ui-pairs size is too big
batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize = 100000): 
    try:     
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
        pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
        pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])

        # fill the missing value as -1 (missing value are time features)
        pred_df.fillna(-1, inplace=True)
        
        # using all the features for training RF model
        pred_X = pred_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                    'u_b4_rate','u_b4_diff_hours',
                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                    'i_b4_rate','i_b4_diff_hours',
                                    'c_u_count_in_6','c_u_count_in_3','c_u_count_in_1',
                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                    'c_b4_rate','c_b4_diff_hours',
                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                    'ui_b1_last_hours','ui_b2_last_hours','ui_b3_last_hours','ui_b4_last_hours',
                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                    'uc_b_count_rank_in_u',
                                    'uc_b1_last_hours','uc_b2_last_hours','uc_b3_last_hours','uc_b4_last_hours'])

        # predicting
        pred_y = (GBDT_clf.predict_proba(pred_X)[:,1] > 0.45).astype(int)

        # generation of U-I pairs those predicted to buy
        pred_df['pred_label'] = pred_y
        # add to result csv
        pred_df[pred_df['pred_label'] == 1].to_csv(path_df_result_tmp, 
                                                   columns=['user_id','item_id'],
                                                   index=False, header=False, mode='a')
        
        batch += 1
        print("prediction chunk %d done." % batch) 
        
    except StopIteration:
        print("prediction finished.")
        break         
   
    
#######################################################################
'''Step 3: generation result on items' sub set P 
'''
# loading data
df_P = df_read(path_df_P)
df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
df_pred = pd.read_csv(open(path_df_result_tmp,'r'), index_col=False, header=None)
df_pred.columns = ['user_id', 'item_id']

# output result
df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
df_pred_P.to_csv(path_df_result, index=False)


print(' - PY131 - ')