import pandas as pd
import operator
from recall import user_base
from recall import item_base

# 处理训练数据-> d
df = pd.read_csv('../data/u.data',
                 sep='\t',
                 # nrows=100,
                 names=['user_id', 'item_id', 'rating', 'timestamp'])
print(max(df['rating']))
d = dict()
for _, row in df.iterrows():
    user_id = str(row['user_id'])
    item_id = str(row['item_id'])
    rating = row['rating']
    if user_id not in d.keys():
        d[user_id] = {item_id: rating}
    else:
        d[user_id][item_id] = rating

# user base
C = user_base.user_sim(d)
user = '196'
rank_u = user_base.recommend('196', d, C, 10)
print(len(rank_u))
# print(rank_u)
print('196用户基于用户相似度推荐list：')
print(sorted(rank_u.items(), key=operator.itemgetter(1),
             reverse=True)[0:10])

# item base
print('196用户基于物品相似度推荐list：')
C = item_base.item_sim(d)
rank_i = item_base.recommendation(d, user, C, 10)
print(sorted(rank_i.items(), key=lambda x: x[1], reverse=True)[0:10])
