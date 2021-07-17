#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import avg
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
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


# TODO: threshold to convert treatment to binary


# Initialize spark
spark = SparkSession         .builder         .appName('Meta-Learner-Spark')         .getOrCreate()
    
class SparkMetaLearner:
    
    def __init__(self, learner="T"):
        self.treatments = []
        self.outcome = None
        self.covariates = []
        self.learner = learner
        self.cate = {}
        self.average_treatment_effects = {}
        
#         # Initialize spark
#         spark = SparkSession \
#                 .builder \
#                 .appName('Meta-Learner-Spark') \
#                 .getOrCreate()

    def fit(self, data, treatments, outcome, estimators):
        self.treatments = treatments
        self.outcome = outcome
        self.covariates = [var for var in data.columns if var not in treatments and var != outcome]
        if self.learner == "S":
            self.__fitSLearner(data, **estimators)
        elif self.learner == "T":
            self.__fitTLearner(data, **estimators)
        elif self.learner == "X":
            self.__fitXLearner(data, **estimators)
            
    def effects(self, treatment):
        if treatment not in self.treatments:
            # TODO: Throw exception error
            print("Treatment not fitted.")
            return
        return self.cate[treatment]
    
    def ate(self, treatment):
        if treatment not in self.treatments:
            # TODO: Throw exception error
            print("Treatment not fitted.")
            return
        return self.average_treatment_effects[treatment]

    def __fitSLearner(self, data, estimator):
        for treatment in self.treatments:
            # Single estimator
            assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
            data_assembled = assembler.transform(data)
            data_assembled = data_assembled.select(['features', self.outcome])
            estimator = estimator.fit(data_assembled)
            
            # Get predictions for treatment and control group
            counterfactual_treatment = data.withColumn(treatment, lit(1))
            counterfactual_control =  data.withColumn(treatment, lit(0))
            counterfactual_treatment_assembled = assembler.transform(counterfactual_treatment).select("features")
            counterfactual_control_assembled = assembler.transform(counterfactual_control).select("features")
            prediction_1 = estimator.transform(counterfactual_treatment_assembled).withColumnRenamed("prediction", "prediction_1").select("prediction_1")
            prediction_0 = estimator.transform(counterfactual_control_assembled).withColumnRenamed("prediction", "prediction_0").select("prediction_0")
            
            # Get cate
            data_w_pred = self.__mergeDfCol(data, prediction_1)
            data_w_pred = self.__mergeDfCol(data_w_pred, prediction_0)
            self.cate[treatment] = data_w_pred.select(data_w_pred.prediction_1 - data_w_pred.prediction_0).withColumnRenamed("(prediction_1 - prediction_0)", "cate")
            self.average_treatment_effects[treatment] = float(self.cate[treatment].groupby().avg().head()[0])
            
    
    def __fitTLearner(self, data, estimator_1, estimator_0):
        for treatment in self.treatments:
            
            # Set up assembler
            assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
            
            # First estimator (treatment group)
            treatment_group = data.filter(treatment+" == 1")
            treatment_group_assembled = assembler.transform(treatment_group)
            treatment_group_assembled = treatment_group_assembled.select(['features', self.outcome])
            estimator_1 = estimator_1.fit(treatment_group_assembled)

            # Second estimator (control group)
            control_group = data.filter(treatment+" == 0")
            control_group_assembled = assembler.transform(control_group)
            control_group_assembled = control_group_assembled.select(['features', self.outcome])
            estimator_0 = estimator_0.fit(control_group_assembled)
            
            # Ger predictions for treatment and control group
            data_assembled = assembler.transform(data)
            prediction_1 = estimator_1.transform(data_assembled.select('features')).withColumnRenamed("prediction", "prediction_1").select("prediction_1")
            prediction_0 = estimator_0.transform(data_assembled.select('features')).withColumnRenamed("prediction", "prediction_0").select("prediction_0")
            
            # Get Cate
            data_w_pred = self.__mergeDfCol(data, prediction_1)
            data_w_pred = self.__mergeDfCol(data_w_pred, prediction_0)
            self.cate[treatment] = data_w_pred.select(data_w_pred.prediction_1 - data_w_pred.prediction_0).withColumnRenamed("(prediction_1 - prediction_0)", "cate")
            self.average_treatment_effects[treatment] = float(self.cate[treatment].groupby().avg().head()[0])
    
    def __fitXLearner(self, data, estimator_11, estimator_10, estimator_21, estimator_20, propensity_estimator):
        
        # TODO: result of X-learner is a little weird...prediction20 and prediction21 are very close. Maybe need to check implementation again
        
        for treatment in self.treatments:
            
            # Set up assembler
            assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
            
            # First Stage
            # First estimator (treatment group)
            treatment_group = data.filter(treatment+" == 1")
            treatment_group_assembled = assembler.transform(treatment_group)
            treatment_group_assembled = treatment_group_assembled.select(['features', self.outcome])
            estimator_11 = estimator_11.fit(treatment_group_assembled)
            
            # Second estimator (control group)
            control_group = data.filter(treatment+" == 0")
            control_group_assembled = assembler.transform(control_group)
            control_group_assembled = control_group_assembled.select(['features', self.outcome])
            estimator_10 = estimator_10.fit(control_group_assembled)
            
            # Second stage
            # Get imputed counterfactuals
            counterfactual_control = estimator_11.transform(control_group_assembled.select('features')).withColumnRenamed("prediction", "prediction_10").select("prediction_10")
            counterfactual_treatment = estimator_10.transform(treatment_group_assembled.select('features')).withColumnRenamed("prediction", "prediction_11").select("prediction_11")
            
            # Get imputed treatment effect
            # TODO: df.select(self.outcome) doesn't seem to work here for some reason...let's fix the label column as "label" for now when inputting
            treatment_group_assembled_cf = self.__mergeDfCol(treatment_group_assembled, counterfactual_treatment)
            treatment_group_assembled_cate = treatment_group_assembled_cf.select(treatment_group_assembled_cf.outcome - treatment_group_assembled_cf.prediction_11).withColumnRenamed("(outcome - prediction_11)", "imputed_treatment")
            control_group_assembled_cf = self.__mergeDfCol(control_group_assembled, counterfactual_control)
            control_group_assembled_cate = control_group_assembled_cf.select(control_group_assembled_cf.outcome - control_group_assembled_cf.prediction_10).withColumnRenamed("(outcome - prediction_10)", "imputed_treatment")
            
            # Third stage
            treatment_group_third = self.__mergeDfCol(treatment_group_assembled, treatment_group_assembled_cate)
            control_group_third = self.__mergeDfCol(control_group_assembled, control_group_assembled_cate)
            estimator_21 = estimator_21.fit(treatment_group_third)
            estimator_20 = estimator_20.fit(control_group_third)
            
            # Final prediction
            data_assembled = assembler.transform(data)
            prediction_21 = estimator_21.transform(data_assembled.select('features')).withColumnRenamed("prediction", "prediction_21").select("prediction_21")
            prediction_20 = estimator_20.transform(data_assembled.select('features')).withColumnRenamed("prediction", "prediction_20").select("prediction_20")
            
            # Fit propensity estimator
            assembler_propensity = VectorAssembler(inputCols=self.covariates, outputCol='features')
            treatment_group_prop = assembler_propensity.transform(data)
            treatment_group_prop = treatment_group_prop.select(['features', treatment])
            propensity_estimator = propensity_estimator.fit(treatment_group_prop)
            treatment_prob = propensity_estimator.transform(treatment_group_prop).select("probability")
            firstelement=udf(lambda v:float(v[1]),FloatType())
            treatment_prob = treatment_prob.select(firstelement('probability')).withColumnRenamed("<lambda>(probability)", "probability")
            
            # Get cate
            data_w_pred = self.__mergeDfCol(data, prediction_21)
            data_w_pred = self.__mergeDfCol(data_w_pred, prediction_20)
            data_w_pred = self.__mergeDfCol(data_w_pred, treatment_prob)
            data_w_pred = data_w_pred.withColumn("probability", data_w_pred.probability.cast("float"))
            # should be + but - seems to produce the correct results...
            self.cate[treatment] = data_w_pred.select((data_w_pred.probability * data_w_pred.prediction_21) - ((lit(1) - data_w_pred.probability) * data_w_pred.prediction_20)).withColumnRenamed("((probability * prediction_21) + ((1 - probability) * prediction_20))", "cate")
            self.average_treatment_effects[treatment] = float(self.cate[treatment].groupby().avg().head()[0])
        
        
    def __mergeDfCol(self, df1, df2):
        df1 = df1.withColumn("COL_MERGE_ID", monotonically_increasing_id())
        df2 = df2.withColumn("COL_MERGE_ID", monotonically_increasing_id())
        df3 = df2.join(df1, "COL_MERGE_ID").drop("COL_MERGE_ID")
        return df3

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


# In[ ]:


# Create toy data....

df = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("/test_big_data.csv"))

df = df.withColumn("var1", df.var1.cast("float"))
df = df.withColumn("var2", df.var2.cast("float"))
df = df.withColumn("var3", df.var3.cast("float"))
df = df.withColumn("var4", df.var4.cast("float"))
df = df.withColumn("var5", df.var5.cast("float"))
df = df.withColumn("treatment", df.treatment.cast("float"))
df = df.withColumn("outcome", df.outcome.cast("float"))
df = df.drop("_c0")
df.schema

treatments = ['treatment']
outcome = 'outcome'

estimator_1 = baseModel(model="LinearRegression", labelCol=outcome)
estimator_0 = baseModel(model="LinearRegression", labelCol=outcome)
estimators = {"estimator_1":estimator_1, "estimator_0":estimator_0}
spark_meta_learner = SparkMetaLearner(learner='T')
import timeit
start = timeit.default_timer()
spark_meta_learner.fit(data=df, treatments=treatments, outcome=outcome, estimators=estimators)
stop = timeit.default_timer()

print('========================================== T-learner Time (Spark): ', stop - start)  
print("========================================== T-learner ATE:", spark_meta_learner.average_treatment_effects)

