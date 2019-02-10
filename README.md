# 音乐推荐系统
代码模块主要有三部分：

1. **分析模块：**
    - `notebook`：主要是原始数据进行分析，对数据做一个理解

2. **数据模块：**
    - `raw_data`:音乐数据原始数据存放目录
    - `data`:中间结果存放路径（模拟nosql），存放在这里面的数据，其实线上是一般需要存储到redis中的。

3. **代码模块：**
    - `recall`：召回模块主要用的协同过滤做召回
    - `rank`: rank模块主要是recall传过来的数据做（LR、GBDT等）打分排序，包括这些模型的训练
    
## recall部分：召回/match
`item_base`和`user_base`是我们在协同过滤课程中已经实现了的代码，在这里需要进行方法调用。

- `cf_rec_list`: 实现利用`item_base`和`user_base`生成的推荐列表进行存储（现在存磁盘，如果在线上一般存；redis离线存数据）
- `config`: 所有数据存储输入输出的路径，以及原始数据raw data的读取方式
- `gen_cf_data`: 生成协同过滤需要用到的训练数据的格式
- `user_base`: 之前实现的基于用户的协同过滤
- `item_base`: 之前实现的基于物品的协同过滤


## rank部分： LR模型训练、工程