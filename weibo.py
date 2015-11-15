#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: zhang
create time: 2015-09

实现了一个基本数据挖掘的流程，包括
1 加载训练数据
2 对数据进行各种预处理（抽样，去噪等）
3 提取特征
4 训练模型
5 利用模型对预测数据进行预测
6 预测结果评价（评分）

提供 刚刚入门 数据挖掘 概念者

一个较为基础的python+pandas+sklearn实现

注：从 __main__ 看起
"""

from pandas import Series, DataFrame
import pandas as pd
from sklearn import datasets, linear_model
import numpy as np
import datetime

import time
import math

def loadData(filename): #加载数据
    dicData = {} 
    uid = []            #用户id
    mid = []            #微博id
    datatime = []       #时间
    forward = []        #转发量
    comment = []        #评论量
    like = []           #赞 量
    content = []        #内容
    with open(filename) as f:
        for line in f:
            #print line
            lineSplit = line.strip().split('\t')
            #print lineSplit
			#根据需求加载相应的数据
            '''
            if len(lineSplit) == 7:
                uid.append(lineSplit[0])
                #mid.append(lineSplit[1])
                #datatime.append(lineSplit[2])
                forward.append(int(lineSplit[3]))
                comment.append(int(lineSplit[4]))
                like.append(int(lineSplit[5]))
                #content.append(lineSplit[6].strip())
            '''
            if len(lineSplit) == 6:
                uid.append(lineSplit[0])
                #mid.append(lineSplit[1])
                #datatime.append(lineSplit[2])
                forward.append(int(lineSplit[3]))
                comment.append(int(lineSplit[4]))
                like.append(int(lineSplit[5]))
                #content.append(lineSplit[6].strip())

    dicData['uid'] = np.array(uid)   #转换为array
    #dicData['mid'] = np.array(mid)
    #dicData['datatime'] = np.array(datatime)
    dicData['forward'] = np.array(forward)
    dicData['comment'] = np.array(comment)
    dicData['like'] = np.array(like)
    #dicData['content'] = np.array(content)

    
    #print dicData
    return DataFrame(dicData)        #使用DataFrame 进行数据格式化
    #print uid

def loadPredictData(filename):   #加载预测数据
    #预测的数据特征一部分来自训练数据(用户特征)，一部分来自本身（时间和文本）
    #保证 预测的特征 和 训练时的特征一一对应


    #获取用户特征 （保证预测集 全部在  训练集中出现）
    df = pd.read_csv('trainFeatures.csv')
    #print df
    FeaturesLen = len(df.values[0,:]) - 1
    #print len(df[0])
    userFeature = {}
    for data in df.values:
        userFeature[data[0]] = data[1:]
    #print userFeature

    predictData = []

    uidAndMid = []
    

    keys = userFeature.keys()
    dickeys = dict(zip(keys, keys))
    

    with open(filename) as f:  #打开预测数据文件
        for line in f:
            #print line
            lineSplit = line.strip().split('\t')
            #print lineSplit
            if len(lineSplit) == 3:
                if lineSplit[0] in dickeys:
                    data = []
                    uid = []
                    data.append(lineSplit[0])
                    data.append(lineSplit[1])
                
                    uid.append(lineSplit[0])
                    uid.append(lineSplit[1])
                    uidAndMid.append(uid)
        
                    data.append(lineSplit[2])
                
                    data = data + list(userFeature[lineSplit[0]])
                    predictData.append(data)
    
    return predictData, FeaturesLen, uidAndMid

                
def getFeatures(df):   #获取训练数据特征
    #add one feature by column
    
    functions = ['count', 'mean', 'max', 'std', 'min', 'median']  #函数功能分别为计数，求平均，求最大，求标准差，最小，中值
    
    #functions = [ 'mean']
    gourped = df.groupby('uid')
    
    re = gourped['like', 'forward', 'comment'].agg(functions)
    #print re
    #print re.unstack('uid')
    df = pd.merge(df, re, left_on='uid', right_index=True) #按uid合并
    #print df
    saveDf = df.copy()                                     #拷贝
	
    del saveDf['comment']
    del saveDf['forward']
    del saveDf['like']
	
    saveDf = saveDf.drop_duplicates()

    #存储的用户特征
    saveDf.to_csv('trainFeatures.csv', index=False)
    return df



#type 1 is forward

#type 2 is comment

#type 3 is like
def trainModel(df, Type):   #训练模型
    print len(df)
    df = df.dropna()        #过滤nan值
    print len(df)
    if Type == 1:
        train_x = np.array(df.values[:, 4:]) #特征列
        #print train_x
    #print train_x
        train_y = np.array(df.values[:, 1])  #目标列

    if Type == 2:
        train_x = df.values[:, 4:]
    #print train_x
        train_y = df.values[:, 0]

    if Type == 3:
        train_x = df.values[:, 4:]
    #print train_x
        train_y = df.values[:, 2]

    #print train_y
    # Create linear regression object
    regr = linear_model.LinearRegression()
    #regr = linear_model.Ridge (alpha = .5)
    #regr = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    #regr = linear_model.Lasso(alpha = 0.1)
    #regr = linear_model.LassoLars(alpha=.1)

    #regr = linear_model.BayesianRidge()
    # Train the model using the training sets
    regr.fit(train_x, train_y)

    # The coefficients
    #print('Coefficients: \n', regr.coef_)

    
    return regr

def predictM(regr, predict_x, FeaturesLength, df=None): #利用模型来预测结果
    
    lensum = len(predict_x[0])
    length = lensum - FeaturesLength
    #print predict_x
    predict_X = []
    for data in predict_x:
        predict_X.append(map(float,data[length:]))
    
    #print predict_X
    from sklearn.preprocessing import Imputer

    predict_X = Imputer().fit_transform(predict_X)
    #predict_X  = np.array(predict_X)
        
    preRe = regr.predict(predict_X)
    return preRe

def putPrecitResult(uidAndMid, ref, rec, rel,filename):   #将预测结果存入文件中
    
    result = ([a, b, c , d] for a, b, c, d in zip(uidAndMid, ref, rec, rel))
    with open(filename,'w') as f:
        for re in result:
            re = str(re)
            re = re.replace('[',' ')
            re = re.replace(']',' ')
            re = re.replace(',',' ')
            re = re.replace('\'',' ')
            #print re
            f.write(re.strip())
            f.write('\n')
  

def getResult(filename):                          #计算预测值 评分
    #评分标准，不同问题会有不同的评分方式
	#总体是按照一定的规则，将预测出的结果与真实的结果进行比较
    #source file
    
    #result file
    #set the value 0
    
    #and change if the value is given
    RealRe = {}
    
    with open('testTrain.txt') as f:            #打开测试集的数据
        for line in f:
            #print line
            lineSplit = line.strip().split()
            #print lineSplit
            if len(lineSplit) == 5:
                data = []
                data.append(int(lineSplit[2]))
                data.append(int(lineSplit[3]))
                data.append(int(lineSplit[4]))
                data.append(0)
                data.append(0)
                data.append(0)
                

                #print data
                RealRe[lineSplit[0]+lineSplit[1]] = data
    
    #print RealRe
    print 'realRe finish'
    keys = RealRe.keys()
    
    dickeys = dict(zip(keys, keys))
    with open(filename) as f2:     
        for line in f2:
            #print line
            lineSplit = line.strip().split()
            #print lineSplit
            if len(lineSplit) == 5:
                if lineSplit[0]+lineSplit[1] in dickeys:
                    RealRe[lineSplit[0]+lineSplit[1]][3] = int(lineSplit[2])
                    RealRe[lineSplit[0]+lineSplit[1]][4] = int(lineSplit[3])
                    RealRe[lineSplit[0]+lineSplit[1]][5] = int(lineSplit[4])
    #print RealRe
    print 'open result finish'
    countSum = 0.0
    countPre = 0.0
    for k, v in RealRe.iteritems():          #评分标准，不同问题会有不同的评分方式
        counti = v[0] + v[1] +v[2]
        if counti > 100:
            counti = 100
        df = math.fabs(v[0]-v[3])/(v[0]+5)
        dc = math.fabs(v[1]-v[4])/(v[1]+3)
        dl = math.fabs(v[2]-v[5])/(v[2]+3)
        #print df,dc,dl
        precision = 1-0.5*df-0.25*dc-0.25*dl
        #print precision
        if precision > 0.8:
            countPre += counti+1
        countSum += counti+1

    preSum = countPre/countSum

    #print preSum
    return preSum
        
def modelSave(model, filename):
    from sklearn.externals import joblib
    joblib.dump(model, filename)

def modelLoad(filename):
    from sklearn.externals import joblib
    model = joblib.load(filename)
    return model

if __name__ == "__main__":
    
    starttime = datetime.datetime.now()   #用来计算时间间隔
    #do something
    df = loadData('test.txt') #加载训练数据
    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    #print df.isnull()
    #df = df.dropna()
    
    print 'loadData time (seconds):'
    print interval
    

    starttime = datetime.datetime.now()
    #do something
    newdf = getFeatures(df)  #获取训练数据特征
    #print newdf
    #newdf = newdf.dropna()
    #print newdf.isnull()
    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    
    print 'getFeatures time (seconds):'
    print interval
  
    starttime = datetime.datetime.now()
    #do something
    regrf = trainModel(newdf, Type=1)   #训练模型
    regrc = trainModel(newdf, Type=2)
    regrl = trainModel(newdf, Type=3)
    
    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    
    print 'trainModel time (seconds):'
    print interval

    
    #modelSave(regrf, 'modelf2.pkl')
    #modelSave(regrc, 'modelc2.pkl')
    #modelSave(regrl, 'modell2.pkl')

    #model = modelLoad('model.pkl')
    #print model
    #print model.coef_

    starttime = datetime.datetime.now()
    #do something
    predictData, FeaturesLen, uidAndMid = loadPredictData('predictData.txt') #加载预测数据
    #from numpy import nan
    #print predictData[:2]
    print len(predictData)
   
    resultf = predictM(regrf, predict_x=predictData, FeaturesLength=FeaturesLen)  #预测
    resultc = predictM(regrc, predict_x=predictData, FeaturesLength=FeaturesLen)
    resultl = predictM(regrl, predict_x=predictData, FeaturesLength=FeaturesLen)
    #print (resultf)
    #print (resultf)
    #print (resultf)
    #ref = map(int,map(math.fabs,map(math.floor,resultf)))
    #rec = map(int,map(math.fabs,map(math.floor,resultc)))
    #rel = map(int,map(math.fabs,map(math.floor,resultl)))
    
    ref = map(int,resultf)
    rec = map(int,resultc)
    rel = map(int,resultl)
    
    #print ref, rec, rel
    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    print 'predictM time (seconds):'
    print interval
    
   
    #print uidAndMid
    putPrecitResult(uidAndMid, ref, rec, rel, '2015.txt')   #合并 产生结果

    
   
    starttime = datetime.datetime.now()
    
    #do something
    precision = getResult('2015.txt')   #对预测值进行评分
    print precision
    

    
    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    
    print 'getResult time (seconds):'
    print interval
    
   
