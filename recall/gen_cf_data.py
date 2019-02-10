import recall.config as conf

train_file = conf.cf_train_data_path


def user_item_score(action_num=100):
    '''
    将原始数据处理成cf的输入数据，类似udata中的数据 user_id,item_id,rating
    :return: data(DataFrame)[user_id,item_id,score]
    '''
    user_watch = conf.gen_user_watch(action_num)
    # 取music_meta中音乐的总时长
    music_meta = conf.gen_music_meta()
    # pandas里面的merge和sql的join一样，数据拼起来了
    data = user_watch.merge(music_meta, how='inner', on='item_id')
    # 清除读进来不再用的数据
    del user_watch
    del music_meta
    # apply相当于spark rdd map操作  听这首音乐的比例：score = 200s/ 304s
    data['score'] = data.apply(lambda x: float(x['stay_seconds']) / float(x['total_timelen']), axis=1)
    # 对应列降序排列
    # data.sort_values('score',ascending=False)
    data = data[['user_id', 'item_id', 'score']]
    data = data.groupby(['user_id', 'item_id']).score.sum().reset_index()
    # user_avg_df = data.groupby('user_id').score.avg().reset_index()
    return data


def train_from_df(df, col_name=['user_id', 'item_id', 'score']):
    '''
    将DataFram数据处理成cf输入的数据形式（dict）
    :param df: DataFrame数据
    :param col_name:对应所需要取到的列名数组
    :return: 最终dict数据
    '''
    d = dict()
    for _, row in df.iterrows():
        user_id = str(row[col_name[0]])
        item_id = str(row[col_name[1]])
        rating = row[col_name[2]]
        if user_id not in d.keys():
            d[user_id] = {item_id: rating}
        else:
            d[user_id][item_id] = rating
    return d


# main
if __name__ == '__main__':
    data = user_item_score(50000)
    train = train_from_df(data, col_name=['user_id', 'item_id', 'score'])
    # 将训练数据存储起来，下次用直接读取不需要再处理
    # 可以尝试 json.dumps()
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(str(train))
