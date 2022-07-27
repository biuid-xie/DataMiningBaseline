# 房屋租赁价格预测挑战赛方案

本方案使用lightGBM算法，对经特征工程后的房屋信息进行回归，因变量为房屋租金。

## 官网地址

[2022 iFLYTEK A.I.开发者大赛-讯飞开放平台 (xfyun.cn)](https://challenge.xfyun.cn/topic/info?type=realtor&option=ssgy)

## 项目目录结构

```
project
 |-- README.md 
 |-- requirements.txt 
 |-- xfdata					# ⽐赛数据集
 |-- user_data				# 选⼿数据⽂件夹
 	|-- model_data 
 		|-- lgb_model_1.txt	# 用于预测，可重新训练生成
 		|-- lgb_model_2.txt
 		|-- lgb_model_3.txt
 		|-- lgb_model_4.txt
 		|-- lgb_model_5.txt
 		|-- lgb_model_6.txt
 		|-- lgb_model_7.txt
 		|-- lgb_model_8.txt
 		|-- lgb_model_9.txt
 		|-- lgb_model_10.txt
 	|-- tmp_data		    # 临时存储⽂件夹
 		|-- x_test.csv		# 用于得到预测结果，可重新训练生成
 |-- prediction_result 		# 预测结果
 |-- code 					# 选⼿代码⽂件夹
 	|-- train             	# 训练代码⽂件夹
 		|-- lightgbm_tr.py  # 模型训练文件
 	|-- test             	# 预测代码⽂件夹
 		|-- lightgbm_predict.py # 模型预测文件
 	|-- test.sh         		# 预测执⾏脚本
 	|-- train.sh       		# 训练示例脚本
```

## 运行环境

Python版本为3.7，各个Python包版本见requirements.txt，使用如下命令即可安装：

```shell
pip install -r code/requirements.txt
```

# 运行代码

## 1. 直接预测

在`user_data/model_data`以及`user_data/tmp_data`下已经预存了模型文件和测试文件，直接预测可以执行：

```shell
bash test.sh
```

## 2. 训练模型再预测

如果需要自行训练模型，再进行预测，可以依次执行：

```shell
bash train.sh
```

```shell
bash test.sh
```



