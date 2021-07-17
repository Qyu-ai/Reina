"""
Returns ML models from the MLLib library.
"""

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import LinearSVC

def baseModel(model="LinearRegression", labelCol="label", model_options={}):
    
    if model == "LinearRegression":
        return LinearRegression(featuresCol="features", labelCol=labelCol, **model_options) 
    elif model == "DecisionTreeRegressor":
        return DecisionTreeRegressor(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "RandomForestRegressor":
        return RandomForstRegressor(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "GradientBoostedTreeRegressor":
        return GBTRegressor(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "LogisticRegression":
        return LogisticRegression(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "DecisionTreeClassifier":
        return DecisionTreeClassifier(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "RandomForestClassifier":
        return RandomForestClassifier(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "GradientBoostedTreeClassifier":
        return GBTClassifier(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "MultilayerPerceptronClassifier":
        return MultilayerPerceptronClassifier(featuresCol="features", labelCol=labelCol, **model_options)
    elif model == "LinearSVM":
        return LinearSVC(featuresCol="features", labelCol=labelCol, **model_options)

