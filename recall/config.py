import pandas as pd
import os

data_path = '../raw_data'

'''
raw data file
'''
music_meta = os.path.join(data_path, 'music_meta')
user_profile = os.path.join(data_path, 'user_profile.data')
user_watch_pref = os.path.join(data_path, 'user_watch_pref.sml')

'''
middle data save path 比如：相似度矩阵
'''
cf_train_data_path = '../data/cf_train.data'

# 相似度矩阵存储path
sim_mid_data_path = '../data/sim_m_data'
user_user_sim_file = os.path.join(sim_mid_data_path, 'user_sim.data')
item_item_sim_file = os.path.join(sim_mid_data_path, 'item_sim.data')

# recommend list out file
cf_rec_lst_outfile = '../data/cf_reclst.data'

user_feat_map_file = '../data/map/user_feat_map'
cross_file = '../data/map/cross_file'

model_file = '../data/map/model_file'

# prefix for user_id in recommend list
UCF_PREFIX = 'UCF_'
ICF_PREFIX = 'ICF_'

# user base在召回中的权重
a = 0.6

'''
Generate raw data format
'''


# user action data
def gen_user_watch(nrows=None):
    return pd.read_csv(user_watch_pref,
                       sep='\001',
                       nrows=nrows,
                       names=['user_id', 'item_id', 'stay_seconds', 'hour'])


# user profile data
def gen_user_profile(nrows=None):
    return pd.read_csv(user_profile,
                       sep=',',
                       nrows=nrows,
                       names=['user_id', 'gender', 'age', 'salary', 'province'])


# item description data
def gen_music_meta(nrows=None):
    df_music_meta = pd.read_csv(music_meta,
                                sep='\001',
                                nrows=nrows,
                                names=['item_id', 'item_name', 'desc', 'total_timelen', 'location', 'tags'])
    del df_music_meta['desc']
    return df_music_meta.fillna('-')  # 把空换成-
