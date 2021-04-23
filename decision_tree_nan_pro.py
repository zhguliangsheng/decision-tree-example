# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

"""
函数功能：计算香农熵
参数说明：
    dataSet：原始数据集
返回：
	ent:香农熵的值
"""
def calEnt(dataSet):
    n = dataSet.shape[0]                             #数据集总行数
    '''datasSet.iloc用于查看指定行，列数据；value_counts()用于统计各个值出现的次数'''
    iset = dataSet.iloc[:,-1].value_counts()         #标签的所有类别
    p = iset/n                                       #每一类标签所占比
    ent = (-p*np.log2(p)).sum()                      #计算信息熵
    return ent


"""
函数功能：根据信息增益选择出最佳数据集切分的列
参数说明：
	dataSet：原始数据集
返回：
	axis:数据集最佳切分列的索引
"""

#选择最优的列进行切分
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)                                #计算原始熵
    bestGain = 0                                             #初始化信息增益
    axis = -1                                                #初始化最佳切分列，标签列
    for i in range(dataSet.shape[1]-1):                      #对特征的每一列进行循环
        levels= dataSet.iloc[:,i].value_counts().index       #提取出当前列的所有取值
        ents = 0      										 #初始化子节点的信息熵       
        for j in levels:									 #对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:,i]==j]         #某一个子节点的dataframe
            ent = calEnt(childSet)							 #计算某一个子节点的信息熵
            ents += (childSet.shape[0]/dataSet.shape[0])*ent #计算当前列的信息熵
        #print(f'第{i}列的信息熵为{ents}')
        infoGain = baseEnt-ents								 #计算当前列的信息增益
        #print(f'第{i}列的信息增益为{infoGain}')
        if (infoGain > bestGain):
            bestGain = infoGain                              #选择最大信息增益
            axis = i                                         #最大信息增益所在列的索引
    return axis


"""
函数功能：按照给定的列划分数据集
参数说明：
	dataSet：原始数据集
	axis：指定的列索引
	value：指定的属性值
返回：
	redataSet：按照指定列索引和属性值切分后的数据集
"""

def mySplit(dataSet,axis,value):
    col = dataSet.columns[axis]                             #返回指定axis的属性名称
    redataSet = dataSet.loc[dataSet[col]==value,:].drop(col,axis=1)   #drop--删除指定维度数据
    return redataSet                                        #loc--提取指定数据


"""
函数功能：基于最大信息增益切分数据集，递归构建决策树
参数说明：
	dataSet：原始数据集（最后一列是标签）
返回：
	myTree：字典形式的树
"""
def createTree(dataSet):
    featlist = list(dataSet.columns)[0:len(dataSet.columns)-1]                 #提取出数据集所有的列标，即属性
    classlist = dataSet.iloc[:,-1].value_counts()             #获取最后一列类标签，即类别;按照类别出现次数进行升序排列
    #print(classlist)
    #判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if classlist.iloc[0]==dataSet.shape[0] or dataSet.shape[1] == 1:  
        #try:
            #print(classlist)
            #acc=float(classlist.index[0])
        #except:
        return classlist.index[0]                             #如果是，返回类标签
    #dataSet=dataSet[list(dataSet.columns)[0:len(dataSet.columns)-1]]
    axis = bestSplit(dataSet)
    if(axis == -1):                                           #标签列不参与类别划分
        del featlist[axis]                                        #删除当前特征
        #print("axis:")                                 #确定出当前最佳切分列的索引
        #print(axis)
        #print(classlist.index[0])
        return classlist.index[0]  
        #print(myTree)
    else:
        bestfeat = featlist[axis]                                 #获取该索引对应的特征
        myTree = {bestfeat:{}}                                    #采用字典嵌套的方式存储树信息
        del featlist[axis]                                        #删除当前特征
        valuelist = set(dataSet.iloc[:,axis])                     #提取最佳切分列所有属性值
        #print(valuelist)
        #print("\n")
        for value in valuelist:	
            #try:
                #print("value:")
                #print(float(value))                                  #对每一个属性值递归建树
            myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
            #except:
                #break
        return myTree

'''
myTree = createTree(dataSet)


#树的存储
np.save('myTree.npy',myTree)

#树的读取
read_myTree = np.load('myTree.npy').item()   
#read_myTree
'''

"""
函数功能：对一个测试实例进行分类
参数说明：
	inputTree：已经生成的决策树
	labels：存储选择的最优特征标签
	testVec：测试数据列表，顺序对应原数据集
返回：
	classLabel：分类结果
"""
def classify(inputTree,labels, testVec):
    firstStr = next(iter(inputTree))                   #获取决策树第一个节点
    secondDict = inputTree[firstStr]                   #下一个字典
# =============================================================================
#     featIndex = labels.index(firstStr)     			   #第一个节点所在列的索引
#     for key in secondDict.keys():
#         if testVec[featIndex] == key:
#             if type(secondDict[key]) == dict :
#                 classLabel = classify(secondDict[key], labels, testVec)
#             else: 
#                 classLabel = secondDict[key]
#     return classLabel
# =============================================================================
    while(type(firstStr)==str):                       
         #字典递进，部分内容报错,进行异常处理
         try:
             secondDict = secondDict[testVec[firstStr]]  #数据量过少部分数据没有对应分叉
             #print(secondDict)                  
         # 异常处理   
         except:
             #break
             #print(secondDict)
             #print(testVec)
             #print("\n")
             #选择当前分叉下的分类结果
             if(type(list(secondDict.keys())[0])==str):
                 secondDict=list(secondDict.keys())[0]
                 #while(type(secondDict)=dict):
                     #secondDict=
             else:
                 while(type(secondDict)==dict):
                     secondDict=secondDict[list(secondDict.keys())[0]]
                 #print(secondDict)
                 
         #secondDict可能没有下一个iter,增加异常处理
         try:
             #print(secondDict)
             firstStr = next(iter(secondDict))
             secondDict = secondDict[next(iter(secondDict))]                   #下一个字典
             #print(firstStr)
         except:
             #print(secondDict)
             #print(testVec)
             #print("\n")
             break
         #print(firstStr)   			   
    return secondDict

    



"""
函数功能：对测试集进行预测，并返回预测后的结果
参数说明：
	train：训练集
	test：测试集
返回：
	test：预测好分类的测试集
"""
def acc_classify(train,test):
    inputTree = createTree(train)						#根据测试集生成一棵树
    print(inputTree)
    labels = list(train.columns)						#数据集所有的列名称，即所有属性
    result = []
    for i in range(test.shape[0]):						#对测试集中每一条数据进行循环
        testVec = test.iloc[i,:-1]						#测试集中的一个实例
        classLabel = classify(inputTree,labels,testVec)	#预测该实例的分类
        result.append(classLabel)						#将分类结果追加到result列表中
    test['predict']=result								#将预测结果追加到测试集最后一列
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()		#计算准确率
    print(f'模型预测准确率为{acc}')
    return acc,test 


#新增函数
def discrete_sort(data_conse):

     for i in data_conse.index[0:7]:
         if(data_conse[i] >= 0 and data_conse[i] < 0.25):
             data_conse[i] = 0
         elif(data_conse[i] >= 0.25 and data_conse[i] < 0.5):
             data_conse[i] = 1
         elif(data_conse[i] >= 0.5 and data_conse[i] < 0.75):
             data_conse[i] = 2
         elif(data_conse[i] >= 0.75 and data_conse[i] <= 1):
             data_conse[i] = 3

     return data_conse

#k-means聚类
def kmeansCluster(X):
  X=np.array(X).reshape(-1, 1)
  list_score=[]
  for n_clu in range(2,10):
      y_pred = KMeans(n_clusters=n_clu).fit_predict(X)
      list_score.append(metrics.calinski_harabaz_score(X, y_pred))  
  n_clu=list_score.index(max(list_score))+2
  #n_clu=5
  y_pred = KMeans(n_clusters=n_clu).fit_predict(X)
  return y_pred  
list_acc_nan=[]
list_num_nan=[]


list_acc=[]
list_num=[]
for i  in range(0,20):
    #pandas 处理数据
    df_all = pd.read_excel("Exasens.xlsx",sheet_name='Exasens') #数据导入
    #删除缺失值
    df_nan_all=df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_nan = df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    df_all.dropna(axis=0,how='any',inplace=True)
    #缺失值补全 
    #df_all.fillna(df_all.mean(),inplace = True)  #平均值填充
    #df_all.fillna(df_all.median(),inplace = True) #中值填充
    #df_all.fillna(method='ffill',inplace = True)  #最靠近的前一个数据填充
    df_nan=df_all
    
    #列名列表
    list_columns = [column_name for column_name in df_nan.columns]
    #选择特征列
    df_nan_data = df_nan.loc[:,list(list_columns[3:10])]
    #字符转浮点，强制类型转换
    df_nan_data = df_nan_data.astype('float')    
    #0-1标准化 将数据规范到0-1之间
    for c_name in df_nan_data.columns:
        df_nan_data[c_name] = MinMaxScaler().fit_transform(df_nan_data[c_name].values.reshape(-1,1))
    #df_no_nan=df_nan
    #df_nan_data=df_nan_data.apply(lambda x: kmeansCluster(x), axis=0)
    df_nan_data["Diagnosis"] = df_nan["Diagnosis"]
   
    df_no_nan=df_nan_data
    #随机划分数据
    df_train = df_no_nan[df_no_nan["Diagnosis"]=='COPD'].sample(frac=0.7, replace=False, random_state=None) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='HC'].sample(frac=0.7, replace=False, random_state=None)) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Asthma'].sample(frac=0.7, replace=False, random_state=None)) 
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Infected'].sample(frac=0.7, replace=False, random_state=None)) 
    df_test = df_no_nan.append(df_train).drop_duplicates(subset=df_no_nan.columns,keep=False) #差集即为剩余30%
    
   

    #train=df_train
    #test=df_test
    dataset=df_train
    testdataset=df_test
    #利用 discrete_sort离散化数据
    train = dataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    test = testdataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    print("第%d个数据划分\n" %(i+1))
    acc,test=acc_classify(train,test)
    list_acc.append(acc)
    i=i+1
    list_num.append(i)


acc_avg=np.mean(list_acc)

print("模型预测平均准确率：%f" %acc_avg)
list_acc_nan.append(acc_avg)

#############################均值填充#######################################
list_acc=[]
list_num=[]
for i  in range(0,20):
    #pandas 处理数据
    df_all = pd.read_excel("Exasens.xlsx",sheet_name='Exasens') #数据导入
    #删除缺失值
    df_nan_all=df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_nan = df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_all.dropna(axis=0,how='any',inplace=True)
    #缺失值补全 
    df_all.fillna(df_all.mean(),inplace = True)  #平均值填充
    #df_all.fillna(df_all.median(),inplace = True) #中值填充
    #df_all.fillna(method='ffill',inplace = True)  #最靠近的前一个数据填充
    df_nan=df_all
    
    #列名列表
    list_columns = [column_name for column_name in df_nan.columns]
    #选择特征列
    df_nan_data = df_nan.loc[:,list(list_columns[3:10])]
    #字符转浮点，强制类型转换
    df_nan_data = df_nan_data.astype('float')    
    #0-1标准化 将数据规范到0-1之间
    for c_name in df_nan_data.columns:
        df_nan_data[c_name] = MinMaxScaler().fit_transform(df_nan_data[c_name].values.reshape(-1,1))
    #df_no_nan=df_nan
    #df_nan_data=df_nan_data.apply(lambda x: kmeansCluster(x), axis=0)
    df_nan_data["Diagnosis"] = df_nan["Diagnosis"]
   
    df_no_nan=df_nan_data
    #随机划分数据
    df_train = df_no_nan[df_no_nan["Diagnosis"]=='COPD'].sample(frac=0.7, replace=False, random_state=None) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='HC'].sample(frac=0.7, replace=False, random_state=None)) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Asthma'].sample(frac=0.7, replace=False, random_state=None)) 
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Infected'].sample(frac=0.7, replace=False, random_state=None)) 
    df_test = df_no_nan.append(df_train).drop_duplicates(subset=df_no_nan.columns,keep=False) #差集即为剩余30%
    
   

    #train=df_train
    #test=df_test
    dataset=df_train
    testdataset=df_test
    #利用 discrete_sort离散化数据
    train = dataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    test = testdataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    print("第%d个数据划分\n" %(i+1))
    acc,test=acc_classify(train,test)
    list_acc.append(acc)
    i=i+1
    list_num.append(i)


acc_avg=np.mean(list_acc)

print("模型预测平均准确率：%f" %acc_avg)
list_acc_nan.append(acc_avg)



#############################中值填充#######################################
list_acc=[]
list_num=[]
for i  in range(0,20):
    #pandas 处理数据
    df_all = pd.read_excel("Exasens.xlsx",sheet_name='Exasens') #数据导入
    #删除缺失值
    df_nan_all=df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_nan = df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_all.dropna(axis=0,how='any',inplace=True)
    #缺失值补全 
    #df_all.fillna(df_all.mean(),inplace = True)  #平均值填充
    df_all.fillna(df_all.median(),inplace = True) #中值填充
    #df_all.fillna(method='ffill',inplace = True)  #最靠近的前一个数据填充
    df_nan=df_all
    
    #列名列表
    list_columns = [column_name for column_name in df_nan.columns]
    #选择特征列
    df_nan_data = df_nan.loc[:,list(list_columns[3:10])]
    #字符转浮点，强制类型转换
    df_nan_data = df_nan_data.astype('float')    
    #0-1标准化 将数据规范到0-1之间
    for c_name in df_nan_data.columns:
        df_nan_data[c_name] = MinMaxScaler().fit_transform(df_nan_data[c_name].values.reshape(-1,1))
    #df_no_nan=df_nan
    #df_nan_data=df_nan_data.apply(lambda x: kmeansCluster(x), axis=0)
    df_nan_data["Diagnosis"] = df_nan["Diagnosis"]
   
    df_no_nan=df_nan_data
    #随机划分数据
    df_train = df_no_nan[df_no_nan["Diagnosis"]=='COPD'].sample(frac=0.7, replace=False, random_state=None) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='HC'].sample(frac=0.7, replace=False, random_state=None)) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Asthma'].sample(frac=0.7, replace=False, random_state=None)) 
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Infected'].sample(frac=0.7, replace=False, random_state=None)) 
    df_test = df_no_nan.append(df_train).drop_duplicates(subset=df_no_nan.columns,keep=False) #差集即为剩余30%
    
   

    #train=df_train
    #test=df_test
    dataset=df_train
    testdataset=df_test
    #利用 discrete_sort离散化数据
    train = dataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    test = testdataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    print("第%d个数据划分\n" %(i+1))
    acc,test=acc_classify(train,test)
    list_acc.append(acc)
    i=i+1
    list_num.append(i)


acc_avg=np.mean(list_acc)

print("模型预测平均准确率：%f" %acc_avg)
list_acc_nan.append(acc_avg)

#############################临近值填充#######################################
list_acc=[]
list_num=[]
for i  in range(0,20):
    #pandas 处理数据
    df_all = pd.read_excel("Exasens.xlsx",sheet_name='Exasens') #数据导入
    #删除缺失值
    df_nan_all=df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_nan = df_all.dropna(axis=0,how='any') #删除表中含有任何NaN的行
    #df_all.dropna(axis=0,how='any',inplace=True)
    #缺失值补全 
    #df_all.fillna(df_all.mean(),inplace = True)  #平均值填充
    #df_all.fillna(df_all.median(),inplace = True) #中值填充
    df_all.fillna(method='ffill',inplace = True)  #最靠近的前一个数据填充
    df_nan=df_all
    
    #列名列表
    list_columns = [column_name for column_name in df_nan.columns]
    #选择特征列
    df_nan_data = df_nan.loc[:,list(list_columns[3:10])]
    #字符转浮点，强制类型转换
    df_nan_data = df_nan_data.astype('float')    
    #0-1标准化 将数据规范到0-1之间
    for c_name in df_nan_data.columns:
        df_nan_data[c_name] = MinMaxScaler().fit_transform(df_nan_data[c_name].values.reshape(-1,1))
    #df_no_nan=df_nan
    #df_nan_data=df_nan_data.apply(lambda x: kmeansCluster(x), axis=0)
    df_nan_data["Diagnosis"] = df_nan["Diagnosis"]
   
    df_no_nan=df_nan_data
    #随机划分数据
    df_train = df_no_nan[df_no_nan["Diagnosis"]=='COPD'].sample(frac=0.7, replace=False, random_state=None) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='HC'].sample(frac=0.7, replace=False, random_state=None)) #随机抽样70%不重复数据
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Asthma'].sample(frac=0.7, replace=False, random_state=None)) 
    df_train = df_train.append(df_no_nan[df_no_nan["Diagnosis"]=='Infected'].sample(frac=0.7, replace=False, random_state=None)) 
    df_test = df_no_nan.append(df_train).drop_duplicates(subset=df_no_nan.columns,keep=False) #差集即为剩余30%
    
   

    #train=df_train
    #test=df_test
    dataset=df_train
    testdataset=df_test
    #利用 discrete_sort离散化数据
    train = dataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    test = testdataset.apply(lambda x: discrete_sort(x), axis=1).reset_index(drop=True)
    print("第%d个数据划分\n" %(i+1))
    acc,test=acc_classify(train,test)
    list_acc.append(acc)
    i=i+1
    list_num.append(i)


acc_avg=np.mean(list_acc)

print("模型预测平均准确率：%f" %acc_avg)
list_acc_nan.append(acc_avg)



#处理数据缺失方法比较
list_num_text=["del","mean","median","near"]

list_num_nan=[1,2,3,4]
#绘制条形图
plt.bar(list_num_nan, list_acc_nan, facecolor='#9999ff', edgecolor='white')

#添加文字
for i in range(0,len(list_num_nan)) :
    plt.text(list_num_nan[i], list_acc_nan[i], str(list_acc_nan[i])[0:6], ha='center', va='bottom')
    plt.text(list_num_nan[i], list_acc_nan[i]/2, list_num_text[i], ha='center', va='bottom')
plt.xticks([]) # 去除横坐标
plt.ylabel('acc')
plt.show()


