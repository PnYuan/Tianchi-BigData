# -*- coding: utf-8 -*

'''
@author: PY131

@thoughts:  as the samples are extremely imbalance (N/P ratio ~ 1.2k),

            1-st: using k_means to make clustering on negative samples (clusters_number ~ 1k)
            2-nd: selecting the best n/p ratio and parameters for XGB classifier
            3-rd: using XGB model for training and predicting on prediction set.
            
            here is 2-nd and 3-th step
'''

########## file path ##########
##### input file
# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "../../data/mobile/xgb/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "../../data/mobile/xgb/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic = "../../data/mobile/raw/df_part_3_uic.csv"

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
path_df_result     = "../../data/mobile/xgb/res_xgb.csv"
path_df_result_tmp = "../../data/mobile/xgb/res_xgb_tmp.csv"

from data_load import *

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import gc

#######################################################################
'''Step 1: training for analysis of the best XGB model (parameters tuning)
        (1). tuning n_estimators with a relative high learning_rate
        (2). tuning max_depth & min_child_weight
        (3). tuning gamma
        (4). tuning subsample and colsample_bytree
        (5). tuning lambda & alpha
        (6). reducing Learning rate and recycle for a more stable parameter combination
'''

### definition of custom f1_score for training parameter feval
def f1_score(preds, dtrain):
    y_labels = dtrain.get_label()
    y_preds  = (preds > 0.5).astype(int) 
    return 'f1-score', metrics.f1_score(y_labels, y_preds)

### cross-validation under subsample training set and original validation set
def CV_with_subsample(xg_param, xg_bst=None, cv_fold=5, train_np_ratio=1, valid_sub=1, train_sub=1, metric='f1-score', seed=None):
    '''
    @describe: with nagetive sample -> subsample, construct a cross-validation
    
    @param xg_bst: former booster file or None, ndarray like [cv_folds, ]
    @param xg_param: maps, booster's params
    @param cv_fold: default 3 folds
    @param train_np_ratio: subsample on training set for NP balance
    @param metric: metric method of return
    @param seed: random seed
    
    @bst booster after CV, ndarray like [cv_folds, ]
    @return metric_arr: ndarray like [cv_folds, num_boost_round], under current bst_params
    '''
    metric_arr = []
    bst = []
        
    ### generating data set
    for k in range(cv_fold):
        valid_df, train_df = valid_train_set_split(folds = cv_fold, fold = k, 
                                                   train_np_ratio = train_np_ratio,
                                                   valid_sub_ratio = valid_sub, 
                                                   train_sub_ratio = train_sub, 
                                                   seed = seed)

        feature_cols = [i for i in train_df.columns if i not in ['user_id','item_id','item_category','label','class']]
        dtrain = xgb.DMatrix(data=train_df[feature_cols].values, label=train_df['label'].values, feature_names=feature_cols)
        dvalid = xgb.DMatrix(data=valid_df[feature_cols].values, label=valid_df['label'].values, feature_names=feature_cols)   
        
        if metric == 'f1-score':
            if xg_bst != None: 
                bst_tmp = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'],
                                     verbose_eval=True, xgb_model=xg_bst[k]) 
            else: 
                bst_tmp = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'],
                                     verbose_eval=True)
                
            y_preds = (bst_tmp.predict(dvalid) > 0.5).astype(int)
            
            metric_arr.append(metrics.f1_score(dvalid.get_label(), y_preds))
            bst.append(bst_tmp)
        
        del(valid_df)
        del(train_df)
        del(dtrain)
        del(dvalid)
        del(bst_tmp)
        gc.collect()  # for saving my poor memory
        
    return bst, metric_arr

# initial parameters
xg_param = {
        'nthread':4,
#         'seed':13,
        
        'eta':0.02,
#         'n_estimators':240,
#         'max_depth':6,
#         'min_child_weight':10,
#         'gamma':0.3,
#         'subsample':1,
#         'colsample_bytree':0.4,
#         'lambda':1,
#         'alpha':1,
#         'scale_pos_weight':1,
#         
        'booster':'gbtree',
        'objective':'binary:logistic'
    }

'''1.1 tuning n_estimators(num_boost_round) with a relative high learning_rate(eta)
'''

xg_param['n_estimators'] = 10000
        
###### 1.1.2 training and evaluating with fi-score curve (continues test)
train_df, _ = data_set_construct_by_part(np_ratio = 70, sub_ratio = 1)
feature_cols = [i for i in train_df.columns if i not in ['user_id','item_id','item_category','label','class']]
dtrain = xgb.DMatrix(data=train_df[feature_cols].values, label=train_df['label'].values, feature_names=feature_cols)

_, valid_df = data_set_construct_by_part(np_ratio = 1200, sub_ratio = 0.4)
dvalid = xgb.DMatrix(data=valid_df[feature_cols].values, label=valid_df['label'].values, feature_names=feature_cols)   

watchlist = [(dtrain,'train'), (dvalid,'valid')]  # set valid set f1-score as the optimize objective

del(valid_df)
del(train_df)
gc.collect()

evals_res = {}
watchlist = [(dtrain,'train'), (dvalid,'valid')]  # set valid set f1-score as the optimize objective

bst = None
bst = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'], early_stopping_rounds=400,
                evals=watchlist, feval=f1_score, maximize=True, evals_result=evals_res,
                xgb_model=bst, verbose_eval=True) 

# pks = pickle.dumps(bst)  # store the bst
# bst = pickle.loads(pks)

# info visualization for judgment
plt.figure(1)
plt.plot(evals_res['train']['error'], label='train-loss')
plt.plot(evals_res['valid']['error'], label='valid-loss')
plt.plot(evals_res['train']['f1-score'], label='train-fi1')
plt.plot(evals_res['valid']['f1-score'], label='valid-f1')
plt.xlabel('n_estimators')
plt.ylabel('error_rate/f1-score')
plt.title('error_rate/f1-score of training - XGB \n (eta=0.1 + default)')
plt.legend()
plt.grid(True, linewidth=0.5)
plt.show()


'''1.6 tuning for the best cutoff
'''

'''
###### initial data
valid_df, train_df = valid_train_set_construct(valid_ratio=0.2, train_np_ratio=70)
feature_cols = [i for i in train_df.columns if i not in ['user_id','item_id','item_category','label','class']]
dtrain = xgb.DMatrix(data=train_df[feature_cols].values, label=train_df['label'].values, feature_names=feature_cols)
dvalid = xgb.DMatrix(data=valid_df[feature_cols].values, label=valid_df['label'].values, feature_names=feature_cols)   
del(train_df)
del(valid_df)
gc.collect()

# training
bst = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'], verbose_eval=True)

###### coarse-grain training
f1_scores = []
cut_offs = []
for co in np.arange(0.05,1,0.05):
   
    y_preds = (bst.predict(dvalid) > co).astype(int)
    f1_tmp = metrics.f1_score(dvalid.get_label(), y_preds)
    
    f1_scores.append(f1_tmp)  
    cut_offs.append(co)

f1 = plt.figure(3)
plt.plot(cut_offs, f1_scores)  
plt.xlabel('cut_offs')
plt.ylabel('f1_score')
plt.title('f1_score as function of XGB cut_offs \n (np=70,eta=0.05,nt=240,md=6,mcw=10,g=0.3,ss=1,cb=0.4,r_l=1,r_a=1 + default)')
plt.grid(True, linewidth=1)
plt.show()   
'''

#######################################################################
'''Step 2: prediction
'''

# generation of train set and training
train_df = train_set_construct(np_ratio=70, sub_ratio=1)
feature_cols = [i for i in train_df.columns if i not in ['user_id','item_id','item_category','label','class']]
dtrain = xgb.DMatrix(data=train_df[feature_cols].values, label=train_df['label'].values, feature_names=feature_cols)

bst = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'], verbose_eval=True)

### predicting
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
        
        dpred = xgb.DMatrix(data=pred_df[feature_cols].values, feature_names=feature_cols)

        # predicting
        y_preds = (bst.predict(dpred) > 0.3).astype(int)

        # generation of U-I pairs those predicted to buy
        pred_df['pred_label'] = y_preds
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