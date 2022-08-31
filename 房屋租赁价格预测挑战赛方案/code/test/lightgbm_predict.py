import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import warnings
warnings.filterwarnings('ignore')

print("#"*10 + "模型预测" + "#"*10)
def model_predict(x_test):
    
    test = np.zeros(x_test.shape[0])
    for i in range(10):
        model_file = "user_data/model_data/lgb_model_{}.txt".format(str(i+1))
        model = lgb.Booster(model_file=model_file)
        test_pred = model.predict(x_test, num_iteration=model.best_iteration)
        test += test_pred / 10
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("#"*10 + "model{} end".format(str(i+1)) +"#"*9)
    return test

x_test = pd.read_csv("user_data/tmp_data/x_test.csv")
test_data = pd.read_csv("xfdata/房屋租赁价格预测挑战赛公开数据/test.csv")
test_data['log_房屋租金'] = model_predict(x_test)
test_data['房屋租金'] = np.exp(test_data['log_房屋租金']) - 1
test_data[['ID','房屋租金']].to_csv("prediction_result/result.csv", index=False)
