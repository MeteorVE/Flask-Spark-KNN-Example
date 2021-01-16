import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

#os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 8g"

import pyspark
from pyspark.mllib.classification import LogisticRegressionWithSGD as LR
from pyspark import SparkConf, SparkContext


def turn_iris_to_sc(sc):
    from sklearn import datasets
    iris = datasets.load_iris()
    iris_target = iris['target']
    table = list(map(lambda a, b: a+[b], iris['data'].tolist(), iris['target']))
    return sc.parallelize(table)

def turn_wine_to_sc(sc):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    wine = datasets.load_wine()
    scaler.fit(wine['data'])
    wine_std = scaler.transform(wine['data'])
    #wine_target = wine['target']
    table = list(map(lambda a, b: a+[b], wine_std.tolist(), wine['target']))
    return sc.parallelize(table)

def turn_digits_to_sc(sc):
    from sklearn import datasets
    digits = datasets.load_digits()
    digits_target = digits['target']
    table = list(map(lambda a, b: a+[b], digits['data'].tolist(), digits['target']))
    return sc.parallelize(table)

def accuracy_score(test , li, k):
    ret = 0
    #print("debug:", test, li)
    dic = { test[-1]:0 }
    for i in range(k):
        if li[i][1] in dic.keys() :
            dic[ li[i][1] ] += 1
        else:
            dic[ li[i][1] ] = 1
    
    for i in dic.keys():
        if dic[test[-1]] < dic[i] :
                return 0
    return 1

class Distance_function(object):
    def distanceAbs(training, test, numfields):
        # training :list   e.g. [0.1, 0.3, 0.4, 0.5]
        ret = 0
        #training = training.collect() #test = training.collect()
        print
        for i in range(numfields-1):
            ret += abs(float(training[i])-float(test[i]))
        return ret

    def distanceEuc(training, test, numfields):
        import math
        ret = 0
        for i in range(numfields-1):
            ret += (float(training[i])-float(test[i]))**2
        return math.sqrt(ret)

    def distanceChe(training, test, numfields):
        ret = 0
        for i in range(numfields-1):
            tmp = abs(float(training[i])-float(test[i]))
            if tmp > ret:
                ret = tmp
        return ret

    def distanceCos(training, test, numfields):
        import math
        dot=sum(a*b for a, b in zip(training, test) if (index(a)!=numfields-1 & index(b)!=numfields-1) )
        norm_training = math.sqrt(sum(a*a for a in training if index(a)!=numfields-1))
        norm_test = math.sqrt(sum(b*b for b in test if index(b)!=numfields-1))
        cos_sim = dot / (norm_training*norm_test)
        ret = 1 - cos_sim
        return ret


def KNN(sc, dataset='./dis.txt', num_nearest_neigbours=5, distance_func='distanceAbs', seed=10):

    # prepare data
    if dataset == 'buildin_iris':
        total_data = turn_iris_to_sc(sc)
    elif dataset == 'buildin_wine':
        total_data = turn_wine_to_sc(sc)
    elif dataset == 'buildin_digits':
        total_data = turn_digits_to_sc(sc)
    else:
        # url= './dis.txt'
        text_file = sc.textFile('upload/'+dataset)
        total_data = text_file.map(lambda line: line.split(" "))

    testset,trainingset = total_data.randomSplit([3,7], seed) # random seed

    numfields = len(testset.collect()[0]) # Feature columns
    #numNearestNeigbours = _numNearestNeigbours # K

    print("[debug]: test set:", testset.collect(),"\n================\n")

    counts = testset.cartesian(trainingset) \
    .map(lambda tt : (tt[0], getattr(Distance_function, distance_func)(tt[0], tt[1], numfields), tt[1][-1])) \
    .map(lambda p: (tuple(p[0]), (p[1], p[2])) ) \
    .groupByKey().map(lambda p: (p[0], sorted(p[1]) ) ) \
    .map(lambda t: accuracy_score(t[0], t[1], num_nearest_neigbours) )

    print('[debug]: in final.py :', counts.collect() )

    ret = counts.collect()

    score = 0
    for i in ret:

        score += i
    score = float(score)/len(ret)
    sc.stop()
    
    return score # ['Some distance', 0.78, 5, '.dis.txt', 4]

#KNN('buildin_wine')




########## 2021/1/14 updated

def get_userdefined_dataset(sc, filename):
    from pyspark.mllib.regression import LabeledPoint
    import numpy as np
    import pandas as pd
    userdefined_dataset = pd.read_csv(
        "upload/" + filename, header=None, sep=" ").to_numpy(dtype=np.float64)
    return sc.parallelize(zip(userdefined_dataset[:, -1], userdefined_dataset[:, :-1])).map(lambda x: LabeledPoint(x[0], x[1]))

class train_model(object):

    def LR_train_model(sc, data, modelname, iterations=10, seed=10):
        train_data, test_data = data.randomSplit([0.7, 0.3], seed)
        train_data_collected = train_data.collect()
        _Classes = set()
        for i in range(len(train_data_collected)):
            _Classes.add(train_data_collected[i].label)
        _numClasses = len(_Classes)
        from pyspark.mllib.classification import LogisticRegressionWithLBFGS as LR
        lr_model = LR.train(train_data, iterations, numClasses = _numClasses)
        #lr_model.save(sc, "./model/" + modelname)
        lr_result = (test_data.map(lambda row:row.label) \
                    .zip(lr_model.predict(test_data.map(lambda row:row.features)))) \
                    .map(lambda row:(row[0], row[1] * 1.0))
        import pyspark.mllib.evaluation as ev
        lr_score = ev.MulticlassMetrics(lr_result)
        print ("Accuracy:{}".format(lr_score.accuracy))
        return lr_score.accuracy

    def DT_train_model(sc, data, modelname, categoricalFeaturesInfo={}, seed=10):        
        train_data, test_data = data.randomSplit([0.7, 0.3], seed)
        train_data_collected = train_data.collect()
        _Classes = set()
        for i in range(len(train_data_collected)):
            _Classes.add(train_data_collected[i].label)
        _numClasses = len(_Classes)
        from pyspark.mllib.tree import DecisionTree as DT
        dt_model = DT.trainClassifier(train_data, _numClasses, categoricalFeaturesInfo)
        #dt_model.save(sc, "./model/" + modelname)
        print(dt_model.toDebugString())
        dt_result = (test_data.map(lambda row:row.label) \
                    .zip(dt_model.predict(test_data.map(lambda row:row.features)))) \
                    .map(lambda row:(row[0], row[1] * 1.0))
        import pyspark.mllib.evaluation as ev
        dt_score = ev.MulticlassMetrics(dt_result)
        print ("Accuracy:{}".format(dt_score.accuracy))
        return dt_score.accuracy

    def RF_train_model(sc, data, modelname, categoricalFeaturesInfo={}, numTrees=5, seed=10):
        train_data, test_data = data.randomSplit([0.7, 0.3], seed)
        train_data_collected = train_data.collect()
        _Classes = set()
        for i in range(len(train_data_collected)):
            _Classes.add(train_data_collected[i].label)
        _numClasses = len(_Classes)
        from pyspark.mllib.tree import RandomForest as RF
        rf_model = RF.trainClassifier(train_data, _numClasses, categoricalFeaturesInfo, numTrees)
        #rf_model.save(sc, "./model/" + modelname)
        print(rf_model.toDebugString())
        rf_result = (test_data.map(lambda row:row.label) \
                    .zip(rf_model.predict(test_data.map(lambda row:row.features)))) \
                    .map(lambda row:(row[0], row[1] * 1.0))
        import pyspark.mllib.evaluation as ev
        rf_score = ev.MulticlassMetrics(rf_result)
        print ("Accuracy:{}".format(rf_score.accuracy))
        return rf_score.accuracy


    def NB_train_model(sc, data, modelname, seed=10):
        train_data, test_data = data.randomSplit([0.7, 0.3], seed)
        from pyspark.mllib.classification import NaiveBayes as NB
        nb_model = NB.train(train_data)
        #nb_model.save(sc, "./model/" + modelname)
        nb_result = (test_data.map(lambda row:row.label) \
                    .zip(nb_model.predict(test_data.map(lambda row:row.features)))) \
                    .map(lambda row:(row[0], row[1] * 1.0))
        accuracy = 1.0 * nb_result.filter(lambda row:(row[0] == row[1])).count() / test_data.count()
        print("Accuracy:{}".format(accuracy))
        return accuracy


class test_model(object):

    def LR_test_model(sc, data, modelname):
        from pyspark.mllib.classification import LogisticRegressionModel as LR
        lr_model = LR.load(sc, "./model/default_model/" + modelname)
        lr_result = lr_model.predict(data)
        print(lr_result * 1.0)
        return lr_result * 1.0

    def DT_test_model(sc, data, modelname):
        from pyspark.mllib.tree import DecisionTreeModel as DT
        dt_model = DT.load(sc, "./model/default_model/" + modelname)
        print(dt_model.toDebugString())
        dt_result = dt_model.predict(data)
        print(dt_result)
        return dt_result


    def RF_test_model(sc, data, modelname):
        from pyspark.mllib.tree import RandomForestModel as RF
        rf_model = RF.load(sc, "./model/default_model/" + modelname)
        print(rf_model.toDebugString())
        rf_result = rf_model.predict(data)
        print(rf_result)
        return rf_result


    def NB_test_model(sc, data, modelname):
        from pyspark.mllib.classification import NaiveBayesModel as NB
        nb_model = NB.load(sc, "./model/default_model/" + modelname)
        nb_result = nb_model.predict(data)
        print(nb_result)
        return nb_result


class get_dataset(object):

    def get_iris_dataset(sc):
        from pyspark.mllib.regression import LabeledPoint
        from sklearn.datasets import load_iris
        iris_dataset = load_iris()
        return sc.parallelize(zip(iris_dataset.target, iris_dataset.data)).map(lambda x : LabeledPoint(x[0], x[1]))

    def get_wine_dataset(sc):
        from pyspark.mllib.regression import LabeledPoint
        from sklearn.datasets import load_wine
        wine_dataset = load_wine()
        return sc.parallelize(zip(wine_dataset.target, wine_dataset.data)).map(lambda x : LabeledPoint(x[0], x[1]))


##########2020/1/14 updated

def call_train_function(algorithm, mode, algorithm_parameter):
    #: algorithm : String
    #: mode : String ('train' or 'test')
    #: algorithm_parameter : Dictionary
    #:return type: test_mode= int/str, train_mode= int

    pyspark.SparkContext.setSystemProperty('spark.driver.memory', '4g')
    pyspark.SparkContext.setSystemProperty('spark.executor.memory', '4g')

    return_value = None
    sc = pyspark.SparkContext()

    if algorithm == 'KNN':
        return_value = KNN(sc=sc,**algorithm_parameter) # get KNN Score
    else:
        data_name = algorithm_parameter.pop('dataset', 'not indicate')
        data= None
        # Load dataset
        if data_name == 'buildin_wine':
            data = getattr(get_dataset, "get_wine_dataset")(sc)
        elif data_name == 'buildin_iris':
            data = getattr(get_dataset, "get_iris_dataset")(sc)
        else:
            print('read from file or throw exception ...')
            data = get_userdefined_dataset(sc, data_name)
            # read file to get dataset
            # data = read( ??? )

        # Run train/test process
        if mode == 'train':
            function_name = algorithm + '_train_model'
            local_model_name = 'wine_NB_model' if algorithm == 'NB' else 'iris_'+ algorithm + '_model'
            return_value = getattr(train_model, function_name)(sc=sc, data=data, **algorithm_parameter, modelname=local_model_name)
        elif mode == 'test':
            function_name = algorithm + '_test_model'
            local_model_name = 'wine_NB_model' if algorithm == 'NB' else 'iris_'+ algorithm + '_model'
            return_value = getattr(test_model, function_name)(
                sc, algorithm_parameter['feature'], local_model_name)
        else:
            return_value = -1
            print('[Error]: Mode need to be "train" or "test"')
        algorithm_parameter['dataset'] = data_name
    sc.stop()
    return return_value 

    #getattr(train_model, "LR_train_model")(sc, data, 10, 10, "iris_LR_model")
    #getattr(test_model, "LR_test_model")(sc, [5.1, 3.5, 1.4, 0.2], "iris_LR_model")

    #getattr(train_model, "DT_train_model")(sc, data, {}, 10, "iris_DT_model")
    #getattr(test_model, "DT_test_model")(sc, [5.1, 3.5, 1.4, 0.2], "iris_DT_model")

    #getattr(train_model, "RF_train_model")(sc, data, {}, 5,  10, "iris_RF_model")
    #getattr(test_model, "RF_test_model")(sc, [5.1, 3.5, 1.4, 0.2], "iris_RF_model")

    
def create_default_model():
    pyspark.SparkContext.setSystemProperty('spark.driver.memory', '4g')
    pyspark.SparkContext.setSystemProperty('spark.executor.memory', '4g')
    sc = pyspark.SparkContext()
    getattr(train_model, "NB_train_model")(sc, getattr(get_dataset, "get_wine_dataset")(sc), "default_model/wine_NB_model", 10)
    getattr(train_model, "LR_train_model")(sc, getattr(get_dataset, "get_iris_dataset")(sc), 10, "default_model/iris_LR_model", 10)
    getattr(train_model, "DT_train_model")(sc, getattr(get_dataset, "get_iris_dataset")(sc), {}, "default_model/iris_DT_model", 10)
    getattr(train_model, "RF_train_model")(sc, getattr(get_dataset, "get_iris_dataset")(sc), {}, 5, "default_model/iris_RF_model", 10)
    sc.stop()
