import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

# 导入数据
tr_data = pd.read_csv('xfdata/房屋租赁价格预测挑战赛公开数据/train.csv')    
te_data = pd.read_csv('xfdata/房屋租赁价格预测挑战赛公开数据/test.csv')

data = pd.concat([tr_data, te_data], ignore_index=True)

print("#"*10 + "进行特征工程..." + "#"*10)
# 时间处理
data["建成年份"] = data["建成年份"].apply(lambda x: int(x) if 1811 <= x <= 2020 else np.nan)
data["最后翻新年份"] = data["最后翻新年份"].apply(lambda x : int(x) if 1990<= x <= 2020 else np.nan)
data["上传年份"] = data["上传日期"].replace({"Feb20": 2020, "May19": 2019,"Oct19": 2019,"Sep18": 2018})   # 2019年有两个月份
data["上传月份"] = data["上传日期"].replace({"Feb20": 3, "May19": 6, "Oct19": 10,"Sep18": 9})

data["上传-建成年差"] = data["上传年份"] - data["建成年份"]
data["最后翻新-建成年差"] = data["最后翻新年份"] - data["建成年份"]
data["上传-最后翻新年差"] = data["上传年份"] - data["最后翻新年份"]

# 处理负值的上传年差：
data["上传-建成年差"] = data["上传-建成年差"].apply(lambda x: x if x >=0 else np.nan)
data["最后翻新-建成年差"] = data["最后翻新-建成年差"].apply(lambda x: x if x >=0 else np.nan)
data["上传-最后翻新年差"] = data["上传-最后翻新年差"].apply(lambda x: x if x >=0 else np.nan)


# 费用、房间面积
data["房间数量"] = data["房间数量"].apply(lambda x: int(x) if 1 <= x <= 8 else 8)

data["供暖费用"] = data["供暖费用"].apply(lambda x : x if x <= 198 else np.nan)  
data["服务费"] = data["服务费"].apply(lambda x : x if x <= 1000 else np.nan)

data["log_供暖费用"] = np.log(data["供暖费用"] + 1)         # # 供暖费相关性要比log_供暖费强
data["log_服务费"] = np.log(data["服务费"]+1)         # 服务费相关性要比log_服务费强

# 居住面积有0，这是问题的，但是test中也有，总不能删掉test吧？
data["居住面积"] = data["居住面积"].apply(lambda x: x if 10<=x<=500 else np.nan)  # 再议
data["log_居住面积"] = np.log(data["居住面积"])         # 居住面积的异常值感觉要处理，7/15 去除+1，已经没有0了

# 处理房屋租金
data.drop(data[(data["房屋租金"]>=10000)|(data["房屋租金"]<=100)].index, inplace=True)
data["log_房屋租金"] = np.log(data["房屋租金"]+1)   # 还有必要+1吗？

data["单位log面积服务费"] = data["服务费"] / data["log_居住面积"]
data["单位面积服务费"] = data["服务费"] / data["居住面积"]
data["单位房间服务费"] = data["服务费"] / data["房间数量"]

data["单位log面积上传图片数"] = data["上传图片数"] / data["log_居住面积"]
data["单位面积上传图片数"] = data["上传图片数"] / data["居住面积"]
data["单位房间上传图片数"] = data["上传图片数"] / data["房间数量"]


# 楼层处理
def judge_use_GB50096(x):         
    # 使用住宅设计规范判定楼层：低层、多层、中高层、高层，外加地下室
    if x == 10086:   # 缺失值
        return np.nan
    elif x <= 0:
        return 0
    elif 1<=x and x<=3:
        return 1
    elif 4<=x and x<=6:
        return 2
    elif 7<=x and x<=9:      # & 是位运算符，and才是对于的逻辑运算符，python中建议常用逻辑运算符
        return 3
    else:
        return 4
    
data["所处楼层"] = data["所处楼层"].apply(lambda x: int(x) if x <= 31 else np.nan) # 31  # 由异常值函数得到下限10
data["建筑楼层"] = data["建筑楼层"].apply(lambda x : int(x) if x <= 54 else np.nan) # 54 # 由函数得到下限11
data["楼层占比"] = data["所处楼层"] / data["建筑楼层"]   # -1 地下一层
data["楼层占比"] = data["楼层占比"].apply(lambda x: x if np.abs(x) <= 1 else np.nan)
# 划分等级
data["所处楼层划分"] = data["所处楼层"].fillna(10086).apply(lambda x: judge_use_GB50096(x))
data["建筑楼层划分"] = data["建筑楼层"].fillna(10086).apply(lambda x: judge_use_GB50096(x))


# 电梯效应，所在楼层大于7楼没电梯，影响比1楼没电梯大
# <=11, 12<=18 19<=33(31)
# data["电梯效应"] = data.apply(lambda x: x["所处楼层"] if np.exp(x["有电梯"]+1) else -np.exp(x["所处楼层"]+1), axis=1)   # axis=1将行传入apply

# 其他
data["可带宠物_2"] = data["可带宠物"].replace({"negotiable": 2, "no": 1, "yes": 3})


# 类型特征编码
# 有缺失值的特征不适于one-hot，会重叠其中一类和缺失值
data = pd.get_dummies(data, columns=['上传月份'], drop_first=True)


# 特征统计量 ["区域1",'区域2', '区域3', '街道']
# 服务费         6385
# 所处楼层       47743
# 建筑楼层       90806
# 价格趋势        1699
# 房间数量           0
# 居住面积     本无缺失
temp = data.groupby("区域1")['服务费'].agg([('区域1_服务费_mean','mean'), ('区域1_服务费_std','std')])   # 这么高级的吗？
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域1', how='left')

temp = data.groupby("区域2")['服务费'].agg([('区域2_服务费_mean','mean'), ('区域2_服务费_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域2', how='left')

temp = data.groupby("区域3")['服务费'].agg([('区域3_服务费_mean','mean'), ('区域3_服务费_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域3', how='left')

temp = data.groupby("街道")['服务费'].agg([('街道_服务费_mean','mean'), ('街道_服务费_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='街道', how='left')

# 增加居住面积
temp = data.groupby("区域1")['居住面积'].agg([('区域1_居住面积_mean','mean'), ('区域1_居住面积_std','std')])
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域1', how='left')

temp = data.groupby("区域2")['居住面积'].agg([('区域2_居住面积_mean','mean'), ('区域2_居住面积_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域2', how='left')

temp = data.groupby("区域3")['居住面积'].agg([('区域3_居住面积_mean','mean'), ('区域3_居住面积_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域3', how='left')

temp = data.groupby("街道")['居住面积'].agg([('街道_居住面积_mean','mean'), ('街道_居住面积_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='街道', how='left')

# 增加更多特征统计量
temp = data.groupby("区域1")['所处楼层'].agg([('区域1_所处楼层_mean','mean'), ('区域1_所处楼层_std','std')])  
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域1', how='left')

temp = data.groupby("区域1")['价格趋势'].agg([('区域1_价格趋势_mean','mean'), ('区域1_价格趋势_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域1', how='left')

temp = data.groupby("区域1")['房间数量'].agg([('区域1_房间数量_mean','mean'), ('区域1_房间数量_std','std')])  
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域1', how='left')
# 1
temp = data.groupby("区域2")['所处楼层'].agg([('区域2_所处楼层_mean','mean'), ('区域2_所处楼层_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域2', how='left')

temp = data.groupby("区域2")['价格趋势'].agg([('区域2_价格趋势_mean','mean'), ('区域2_价格趋势_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域2', how='left')

temp = data.groupby("区域2")['房间数量'].agg([('区域2_房间数量_mean','mean'), ('区域2_房间数量_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域2', how='left')
# 2
temp = data.groupby("区域3")['所处楼层'].agg([('区域3_所处楼层_mean','mean'), ('区域3_所处楼层_std','std')])  
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域3', how='left')

temp = data.groupby("区域3")['价格趋势'].agg([('区域3_价格趋势_mean','mean'), ('区域3_价格趋势_std','std')])  
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域3', how='left')

temp = data.groupby("区域3")['房间数量'].agg([('区域3_房间数量_mean','mean'), ('区域3_房间数量_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='区域3', how='left')
# 3
temp = data.groupby("街道")['所处楼层'].agg([('街道_所处楼层_mean','mean'), ('街道_所处楼层_std','std')])  
temp.fillna(0, inplace=True)
data = data.merge(temp, on='街道', how='left')

temp = data.groupby("街道")['价格趋势'].agg([('街道_价格趋势_mean','mean'), ('街道_价格趋势_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='街道', how='left')

temp = data.groupby("街道")['房间数量'].agg([('街道_房间数量_mean','mean'), ('街道_房间数量_std','std')])   
temp.fillna(0, inplace=True)
data = data.merge(temp, on='街道', how='left')

l = data.columns.to_list()
ft = ["ID","房屋租金","log_供暖费用",
      "没有停车位","log_房屋租金",
      "上传日期","可带宠物","log_服务费",'log_居住面积','单位面积服务费','单位面积上传图片数']

features = [i for i in l if i not in ft]
category_fea = []
for i in ['房屋状况','内饰质量','加热类型','房屋类型','所处楼层划分', '建筑楼层划分','可带宠物_2']:
    category_fea.append(features.index(i))
       
train_data = data[data["房屋租金"].notnull()].reset_index(drop=True)
test_data = data[data["房屋租金"].isnull()].reset_index(drop=True)
x_train_temp = train_data[features]
y_train = train_data["log_房屋租金"]
x_test_temp = test_data[features]

# 处理种类基数较多的分类变量
from category_encoders import TargetEncoder
import pandas as pd
enc = TargetEncoder(cols=["区域1",'区域2', '区域3', '街道',"邮政编码"])
# transform the datasets
x_train = enc.fit_transform(x_train_temp, y_train)
x_test = enc.transform(x_test_temp)

# 保存x_test，用于预测
x_test.to_csv("user_data/tmp_data/x_test.csv", index=False)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
print(l)
print("\n")
print(features)
print("all_len: {}, fea_len: {}".format(len(l),len(features)))
print("train_data_num: {}, test_data_num: {}".format(len(train_data),len(test_data)))
print("category_fea: ", category_fea)
print("#"*10 + "特征工程已完成！" + "#"*10)
print("\n\n")
print("#"*10 + "进行训练模型..." + "#"*10)
############################################################
# 训练模型
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 10
    seed = 2022
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
#     test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'l1',          # mae mean_absolute_error
                
                'seed': 2022,
                'n_jobs':-1,
                
                'bagging_fraction': 0.9426917523072508,
                'bagging_freq': 3,
                'feature_fraction': 0.6495602688979487, 
                'lambda_l1': 0.02052223766043643,
                'lambda_l2': 3.7224143113186443,
                'learning_rate': 0.03319994978090327,
                'max_bin': 158,
                'min_data_in_leaf': 26, 
                'min_sum_hessian_in_leaf': 8.865519169470296,
                'num_leaves': 89,
            }
            
            # category_fea
            model = clf.train(params, train_matrix, 90000, valid_sets=[train_matrix, valid_matrix], 
                              categorical_feature=category_fea, verbose_eval=3000, early_stopping_rounds=1000)   
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
#             test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True)))
        # 保存模型
        model.save_model("user_data/model_data/lgb_model_{}.txt".format(str(i+1)))    
        train[valid_index] = val_pred
#         test += test_pred / kf.n_splits     # 这里必有加号
        cv_scores.append(mean_absolute_error(val_y, val_pred))
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    return train
    
lgb_train = cv_model(lgb, x_train, y_train, x_test, "lgb")
# lgb_train.to_csv("user_data/tmp_data/lgb_train.csv", index=False)   # 可用于stacking，但本方案并未采取模型融合
print("#"*10 + "模型训练已完成！" + "#"*10)