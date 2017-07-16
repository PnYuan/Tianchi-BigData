# -*- coding: utf-8 -*
    
'''
@author: PY131

@thoughts:  as the samples are extremely imbalance (N/P ratio ~ 1.2k),
            here we use sub-sample on negative samples.
            1-st: using k_means to make clustering on negative samples (clusters_number ~ 1k)
            2-nd: subsample on each clusters based on the same ratio,
                  the ratio was selected to be the best by testing in random sub_sample + GBDT
            3-rd: using GBDT model for training and predicting on sub_sample set.
            
            here is 1-st step

@Description:
    here we use clustering to data reduction: 
        we want to get N/P ration ~ 35 for each training set
        the method refer to:
            https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set/answers/1144228?spm=5176.100239.blogcont93547.8.UsslUB&srid=h3G6o

'''

##### file path
### input
# data_set keys and lebels
path_df_part_1_uic_label = "../../data/mobile/raw/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "../../data/mobile/raw/df_part_2_uic_label.csv"
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

### out file

### intermediate file
# data partition with diffferent label
path_df_part_1_uic_label_0 = "../../data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_0.csv"
path_df_part_1_uic_label_1 = "../../data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_1.csv"
path_df_part_2_uic_label_0 = "../../data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_0.csv"
path_df_part_2_uic_label_1 = "../../data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_1.csv"

# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "../../data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "../../data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_cluster.csv"

# scalers for data standardization store as python pickle
# for each part's features
path_df_part_1_scaler = "../../data/mobile/gbdt/k_means_subsample/df_part_1_scaler"
path_df_part_2_scaler = "../../data/mobile/gbdt/k_means_subsample/df_part_2_scaler"


import pandas as pd
import numpy as np

def df_read(path, mode = 'r'):
    '''the definition of dataframe loading function 
    '''
    path_df = open(path, mode)
    try:     df = pd.read_csv(path_df, index_col = False)
    finally: path_df.close()
    return   df

def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size
    
    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df) : return df
    else : return df.sample(n = sub_size)
    
########################################################################
'''Step 1: dividing of positive and negative sub-set by u-i-c-label keys
    
    p.s. we first generate u-i-C key, then merging for data set and operation by chunk 
    such strange operation designed for saving my poor PC-MEM.
'''
''' 
df_part_1_uic_label = df_read(path_df_part_1_uic_label)  # loading total keys
df_part_2_uic_label = df_read(path_df_part_2_uic_label)

df_part_1_uic_label_0 = df_part_1_uic_label[df_part_1_uic_label['label'] == 0]
df_part_1_uic_label_1 = df_part_1_uic_label[df_part_1_uic_label['label'] == 1]
df_part_2_uic_label_0 = df_part_2_uic_label[df_part_2_uic_label['label'] == 0]
df_part_2_uic_label_1 = df_part_2_uic_label[df_part_2_uic_label['label'] == 1]

df_part_1_uic_label_0.to_csv(path_df_part_1_uic_label_0, index=False)
df_part_1_uic_label_1.to_csv(path_df_part_1_uic_label_1, index=False)
df_part_2_uic_label_0.to_csv(path_df_part_2_uic_label_0, index=False)
df_part_2_uic_label_1.to_csv(path_df_part_2_uic_label_1, index=False)
'''

#######################################################################
'''Step 2: clustering on negative sub-set
    clusters number ~ 35, using mini-batch-k-means
'''

# clustering based on sklearn
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import pickle

##### part_1 #####
# loading features
df_part_1_U  = df_read(path_df_part_1_U )   
df_part_1_I  = df_read(path_df_part_1_I )
df_part_1_C  = df_read(path_df_part_1_C )
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

# process by chunk as ui-pairs size is too big

# for get scale transform mechanism to large scale of data
scaler_1 = preprocessing.StandardScaler() 
batch = 0
for df_part_1_uic_label_0 in pd.read_csv(open(path_df_part_1_uic_label_0, 'r'), chunksize=150000): 
    try:
        # construct of part_1's sub-training set
        train_data_df_part_1 = pd.merge(df_part_1_uic_label_0, df_part_1_U, how='left', on=['user_id'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_I,  how='left', on=['item_id'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_C,  how='left', on=['item_category'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_IC, how='left', on=['item_id','item_category'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UC, how='left', on=['user_id','item_category'])
        
        # getting all the complete features for clustering
        train_X_1 = train_data_df_part_1.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                                    'u_b4_rate',
                                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                                    'i_b4_rate',
                                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                                    'c_b4_rate',
                                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                                    'uc_b_count_rank_in_u'])
        # feature standardization
        scaler_1.partial_fit(train_X_1)        
        
        batch += 1
        print('chunk %d done.' %batch) 
        
    except StopIteration:
        print("finish.")
        break

# initial clusters
mbk_1 = MiniBatchKMeans(init='k-means++', n_clusters=1000, batch_size=500, reassignment_ratio=10**-4) 
classes_1 = []
batch = 0
for df_part_1_uic_label_0 in pd.read_csv(open(path_df_part_1_uic_label_0, 'r'), chunksize=15000): 
    try:
        # construct of part_1's sub-training set
        train_data_df_part_1 = pd.merge(df_part_1_uic_label_0, df_part_1_U, how='left', on=['user_id'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_I,  how='left', on=['item_id'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_C,  how='left', on=['item_category'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_IC, how='left', on=['item_id','item_category'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
        train_data_df_part_1 = pd.merge(train_data_df_part_1, df_part_1_UC, how='left', on=['user_id','item_category'])
        
        train_X_1 = train_data_df_part_1.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                                    'u_b4_rate',
                                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                                    'i_b4_rate',
                                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                                    'c_b4_rate',
                                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                                    'uc_b_count_rank_in_u'])
        # feature standardization
        standardized_train_X_1 = scaler_1.transform(train_X_1)
         
        # fit clustering model
        mbk_1.partial_fit(standardized_train_X_1)
        classes_1 = np.append(classes_1, mbk_1.labels_)
        
        batch += 1
        print('chunk %d done.' %batch) 
        
    except StopIteration:
        print(" ------------ k-means finished on part 1 ------------.")
        break 

del(df_part_1_U )
del(df_part_1_I )
del(df_part_1_C )
del(df_part_1_IC)
del(df_part_1_UI)
del(df_part_1_UC)

##### part_2 #####
# loading features
df_part_2_U  = df_read(path_df_part_2_U )   
df_part_2_I  = df_read(path_df_part_2_I )
df_part_2_C  = df_read(path_df_part_2_C )
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

# process by chunk as ui-pairs size is too big

# for get scale transform mechanism to large scale of data
scaler_2 = preprocessing.StandardScaler()
batch = 0
for df_part_2_uic_label_0 in pd.read_csv(open(path_df_part_2_uic_label_0, 'r'), chunksize=150000): 
    try:
        # construct of part_1's sub-training set
        train_data_df_part_2 = pd.merge(df_part_2_uic_label_0, df_part_2_U, how='left', on=['user_id'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_I,  how='left', on=['item_id'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_C,  how='left', on=['item_category'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_IC, how='left', on=['item_id','item_category'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_UC, how='left', on=['user_id','item_category'])

        train_X_2 = train_data_df_part_2.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                                    'u_b4_rate',
                                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                                    'i_b4_rate',
                                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                                    'c_b4_rate',
                                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                                    'uc_b_count_rank_in_u'])
        # fit the scaler
        scaler_2.partial_fit(train_X_2)
        
        batch += 1
        print('chunk %d done.' %batch) 
        
    except StopIteration:
        print("finish.")
        break 

# initial clusters
mbk_2 = MiniBatchKMeans(init='k-means++', n_clusters=1000, batch_size=500, reassignment_ratio=10**-4)  

# process by chunk as ui-pairs size is too big
batch = 0
classes_2 = []
for df_part_2_uic_label_0 in pd.read_csv(open(path_df_part_2_uic_label_0, 'r'), chunksize=15000): 
    try:
        # construct of part_1's sub-training set
        train_data_df_part_2 = pd.merge(df_part_2_uic_label_0, df_part_2_U, how='left', on=['user_id'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_I,  how='left', on=['item_id'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_C,  how='left', on=['item_category'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_IC, how='left', on=['item_id','item_category'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
        train_data_df_part_2 = pd.merge(train_data_df_part_2, df_part_2_UC, how='left', on=['user_id','item_category'])
        
        train_X_2 = train_data_df_part_2.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                                    'u_b4_rate',
                                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                                    'i_b4_rate',
                                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                                    'c_b4_rate',
                                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                                    'uc_b_count_rank_in_u'])
        # feature standardization
        standardized_train_X_2 = scaler_2.transform(train_X_2)
        
        # fit clustering model
        mbk_2.partial_fit(standardized_train_X_2)
        classes_2 = np.append(classes_2, mbk_2.labels_)
        
        batch += 1
        print('chunk %d done.' %batch) 
        
    except StopIteration:
        print(" ------------ k-means finished on part 2 ------------.")
        break 

del(df_part_2_U )
del(df_part_2_I )
del(df_part_2_C )
del(df_part_2_IC)
del(df_part_2_UI)
del(df_part_2_UC)

pickle.dump(scaler_1, open(path_df_part_1_scaler,'wb')) 
pickle.dump(scaler_2, open(path_df_part_2_scaler,'wb'))


#######################################################################
'''Step 3: generation of new training set

    each training sub-set contains a clusters' negative samples' and all positive samples
    
    here we just generation of u-i-c-label-class keys of training data 
        ps. label -> whether to buy
            class -> clusters labels 
                for positive : 0
                for negitive : 1 to clusters_numbers
'''


# add a new attr for keys
df_part_1_uic_label_0 = df_read(path_df_part_1_uic_label_0)
df_part_1_uic_label_1 = df_read(path_df_part_1_uic_label_1)
df_part_2_uic_label_0 = df_read(path_df_part_2_uic_label_0)
df_part_2_uic_label_1 = df_read(path_df_part_2_uic_label_1)
    
df_part_1_uic_label_0['class'] = classes_1.astype('int') + 1
df_part_1_uic_label_1['class'] = 0
df_part_2_uic_label_0['class'] = classes_2.astype('int') + 1
df_part_2_uic_label_1['class'] = 0

df_part_1_uic_label_class = pd.concat([df_part_1_uic_label_0, df_part_1_uic_label_1])
df_part_2_uic_label_class = pd.concat([df_part_2_uic_label_0, df_part_2_uic_label_1])
   
df_part_1_uic_label_class.to_csv(path_df_part_1_uic_label_cluster, index=False)
df_part_2_uic_label_class.to_csv(path_df_part_2_uic_label_cluster, index=False)






print(' - PY131 - ')