import sys
import os
SPARK_HOME = "/opt/bitnami/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

import pyspark
from pyspark.mllib.classification import LogisticRegressionWithSGD
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


def KNN(input_data='buildin_iris', _numNearestNeigbours=5, distance_func='distanceAbs'):

    # prepare sc
    sc = pyspark.SparkContext()
    opt_file = 'output'

    # prepare data
    if input_data == 'buildin_iris':
        total_data = turn_iris_to_sc(sc)
    elif input_data == 'buildin_wine':
        total_data = turn_wine_to_sc(sc)
    elif input_data == 'buildin_digits':
        total_data = turn_digits_to_sc(sc)
    else:
        # url= './dis.txt'
        text_file = sc.textFile(input_data)
        total_data = text_file.map(lambda line: line.split(" "))

    testset,trainingset = total_data.randomSplit([3,7], 10) # random seed

    numfields = len(testset.collect()[0]) # Feature columns
    print(numfields)
    numNearestNeigbours = _numNearestNeigbours # K

    print("[debug]: test set:", testset.collect(),"\n================\n")

    counts = testset.cartesian(trainingset) \
    .map(lambda tt : (tt[0], getattr(Distance_function, distance_func)(tt[0], tt[1], numfields), tt[1][-1])) \
    .map(lambda p: (tuple(p[0]), (p[1], p[2])) ) \
    .groupByKey().map(lambda p: (p[0], sorted(p[1]) ) ) \
    .map(lambda t: accuracy_score(t[0], t[1], numNearestNeigbours) )

    print('[debug]: in final.py :', counts.collect() )

    ret = counts.collect()

    score = 0
    for i in ret:
        score += i
    score = float(score)/len(ret)
    sc.stop()
    
    return [distance_func, score, _numNearestNeigbours, input_data, numfields] # ['Some distance', 0.78, 5, '.dis.txt', 4]

KNN('buildin_wine')