import recall.gen_cf_data as gcd
import recall.config as conf
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

cross_file = conf.cross_file
user_feat_map_file = conf.user_feat_map_file
model_file = conf.model_file

data = gcd.user_item_score(10000000)
# 定义label 0/1规则：听完算喜欢
data['label'] = data['score'].apply(lambda x: 1 if x>=1.0 else 0)

'''
user_id,item_id,label
加入用户和item的信息
'''

# user信息
user_profile =conf.gen_user_profile()
# item信息
music_meta = conf.gen_music_meta()
# 关联用户和item的信息到data中
data = data.merge(user_profile,
                  how='inner',
                  on='user_id').merge(music_meta,
                                      how='inner',
                                      on='item_id')
# print(data.head())
'''
特征种类
'''
user_feat = ['gender','age','salary','province']
item_feat = ['total_timelen','location']
item_text_feat = ['item_name','tags']
watch_feat = ['hours','stay_seconds','score']

category_feat = user_feat+['location']
continuous_feat = ['score']

labels = data['label']
del data['label']

# 特征处理
# 1. 离散特征one-hot处理 （word2vec-> embedding[continuous]）
df = pd.get_dummies(data[category_feat])
one_hot_columns = df.columns
# print(data[category_feat].head())
# print(df.head())
# 2.连续特征不处理直接带入  【一般做离散化GBDT（xgboost）叶子节点做离散化 GBDT+LR】
df[continuous_feat] = data[continuous_feat].astype(float)
# cross feat save
data['ui-key'] = data['user_id'].astype(str)+'_'+data['item_id'].astype(str)
cross_feat_map = dict()
for _,row in data[['ui-key','score']].iterrows():
    cross_feat_map[row['ui-key']] = row['score']
with open(cross_file,'w') as cf:
    cf.write(str(cross_feat_map))

# 随机划分训练集train test split[0.7,0.3]
X_train,X_test,y_train,y_test = train_test_split(df.values,labels,test_size=0.2,random_state=2019)
lr = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=0.1,
                        fit_intercept=True, intercept_scaling=1, class_weight=None,
                        random_state=None, solver='liblinear', max_iter=100,
                        multi_class='ovr', verbose=1, warm_start=False, n_jobs=-1)
model = lr.fit_transform(X_train,y_train)
print("w:%s, b:%s"%(lr.coef_,lr.intercept_))
print("Residual sum of squatres: %.2f"% np.mean((lr.predict(X_test)-y_test)**2))
print("score: %.2f"%lr.score(X_test,y_test))

'''
one-hot [0,1] [1,0] 性别——男：1，性别-女：0   map{字段-字段值：index}
'''
# 存储离散特征map
feat_map={}
for i in range(len(one_hot_columns)):
    key = one_hot_columns[i]
    feat_map[key] = i

with open(user_feat_map_file,'w',encoding='utf-8') as f:
    f.write(str(feat_map))

# 存储模型
model_dict = {'W':lr.coef_.tolist()[0],
              'b':lr.intercept_.tolist()[0]}

with open(model_file,'w',encoding='utf-8') as f:
    f.write(str(model_dict))

