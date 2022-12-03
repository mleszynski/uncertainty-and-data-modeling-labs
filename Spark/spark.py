# solutions.py

import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE



# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """ 
    # open session and read in data ############################################
    spark = SparkSession.builder.appName('app_name').getOrCreate()
    data = spark.sparkContext.textFile(filename)

    # count and sort words #####################################################
    data = data.flatMap(lambda row: row.split(' '))
    data = data.map(lambda row: (row, 1))
    data = data.reduceByKey(lambda x, y: x + y)
    data = data.sortBy(lambda row: row[1]).collect()
    data = data[::-1]
    spark.stop()

    return data[:20]
    
    
### Problem 2
def monte_carlo(n=10**5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """
    # open session #############################################################
    spark = SparkSession.builder.appName('app_name').getOrCreate()

    # run monte carlo simulation ###############################################
    pi = spark.sparkContext.parallelize(range(n*parts), parts)
    pi = pi.map(lambda x: (np.random.uniform(low=-1, high=1),\
             np.random.uniform(low=-1, high=1)))
    pi = pi.map(lambda x: np.linalg.norm(x))
    pi = pi.filter(lambda x: x<=1)
    pi_est = pi.count()
    spark.stop()

    return 4*(pi_est/(n*parts))


# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.
    
    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women, 
             and the survival rate of men in that order.
    """
    # open session and read in data ############################################
    spark = SparkSession.builder.appName('app_name').getOrCreate()
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT')
    data = spark.read.csv(filename, schema=schema)

    # manipulate data by creating new, simpler dataframe #######################
    data = data.groupBy('sex', 'survived').count().sort('sex', 'survived')
    num_w = data.collect()[0][2] + data.collect()[1][2]
    num_m = data.collect()[2][2] + data.collect()[3][2]
    sr_w = data.collect()[1][2] / num_w
    sr_m = data.collect()[3][2] / num_m
    spark.stop()

    return num_w, num_m, sr_w, sr_m


### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Murder'):
    """
    Explores crime by borough and income for the specified min_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): crime minor category to analyze
    returns:
        numpy array: borough names sorted by percent months with crime, descending
    """
    # open session and read in data ############################################
    spark = SparkSession.builder.appName('app_name').getOrCreate()
    df_crime = spark.read.csv(crimefile, header=True, inferSchema=True)
    df_income = spark.read.csv(incomefile, header=True, inferSchema=True)
    data = (df_crime.filter(df_crime['major_category'] == major_cat)\
        .groupBy('borough').sum('value').join(df_income, on='borough')\
        .select('borough', 'sum(value)', 'median-08-16').sort('sum(value)', ascending=False))
    
    output = np.array(data.collect())

    # plot results #############################################################
    plt.scatter(output[:, 1].astype('float64'), output[:, 2].astype('float64'))
    plt.title('Prob 4')
    plt.xlabel('Num of ' + major_cat)
    plt.ylabel('Median Income')
    plt.show()

    spark.stop()

    return output


### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (list): a list of metrics gauging the performance of the model
            ('accuracy', 'weightedPrecision', 'weightedRecall')
    """
    # open session and read in data ############################################
    spark = SparkSession.builder.appName('app_name').getOrCreate()
    schema = ('survived INT, pclass INT, name STRING, sex STRING, age FLOAT, sibsp INT, parch INT, fare FLOAT')
    data = spark.read.csv(filename, schema=schema)
    
    # create model #############################################################
    sex = StringIndexer(inputCol='sex', outputCol='sex_binary')
    one_hot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])
    feats = ['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare']
    feats_col = VectorAssembler(inputCols=feats, outputCol='features')
    pipeline = Pipeline(stages=[sex, one_hot, feats_col])
    data = pipeline.fit(data).transform(data)
    data = data.drop('pclass', 'name', 'sex')

    train, test = data.randomSplit([.75, .25], seed=11)
    rf = RandomForestClassifier(labelCol='survived', featuresCol='features')
    params = ParamGridBuilder().addGrid(rf.maxDepth, list(range(3, 9))).build()
    tvs = TrainValidationSplit(estimator=rf, estimatorParamMaps=params,\
            evaluator=MCE(labelCol='survived'), trainRatio=.75, seed=11)
    clf = tvs.fit(train)
    results = clf.bestModel.evaluate(test)
    accuracy = results.accuracy
    recall = results.weightedRecall
    precision = results.weightedPrecision
    spark.stop()

    return accuracy, recall, precision
