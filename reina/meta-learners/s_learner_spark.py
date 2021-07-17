#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
Provides a spark-based S-learner heterogeneous treatment effect estimator.
"""

from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import avg
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import when
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
    
class SparkSLearner:
    """
    Spark-based S-learner heterogeneous treatment effect estimator.
    
    This class currently only supports numeric outcomes.
    
    Assumptions
    ---------------
    This class assumes that the data is already stored in a distributed storage system (e.g., HDFS).
    This class also assumes that the treatment variable only contains 1s and 0s.
    """
    
    def __init__(self):
        self.treatments = []  # Multiple treatment effects can be estimated
        self.covariates = []
        self.outcome = ""
        self.estimator = None
        
    def fit(self, data, treatments, outcome, estimator):
        """
        Wrapper function to fit an ML-based counterfacual model.
        When multiple treatments are inputted, each treatment effect is estiamted individually.
        
        Parameters
        ----------
        data (2-D Spark dataframe): Base dataset containing features, treatment, iv, and outcome variables
        treatments (List): Names of the treatment variables             
        outcome (Str): Name of the outcome variable
        estimator (mllib model obj): Arbitrary ML estimator of choice
              
        Returns
        ------
        self
        """
        
        self.treatments = treatments
        self.outcome = outcome
        self.covariates = [var for var in data.columns if var not in treatments and var != outcome]
        self.estimator = estimator
        self.__fit(data)

    def effects(self, X, treatment):
        """
        Function to get the estimated heterogeneous treatment effect from the fitted counterfactual model.
        
        The treatment effect is calculated by taking the difference between the predicted counterfactual outcomes.
        
        Parameters
        ----------
        X (2-D Spark dataframe): Feature data to estimate treatment effect of
        treatment (Str): Name of the treatment variable   
        
        returns
        -------
        cate: conditional average treatment effect
        ate: average treatment effect
        """
        
        # Input treatment has to be fitted
        assert treatment in self.treatments
        
        # Get predictions for treatment and control group
        counterfactual_treatment = X.withColumn(treatment, lit(1))
        counterfactual_control =  X.withColumn(treatment, lit(0))
        assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
        counterfactual_treatment_assembled = assembler.transform(counterfactual_treatment).select("features")
        counterfactual_control_assembled = assembler.transform(counterfactual_control).select("features")
        prediction_1 = self.estimator.transform(counterfactual_treatment_assembled).withColumnRenamed("prediction", "prediction_1").select("prediction_1")
        prediction_0 = self.estimator.transform(counterfactual_control_assembled).withColumnRenamed("prediction", "prediction_0").select("prediction_0")

        # Get cate
        X_w_pred = self.__mergeDfCol(X, prediction_1)
        X_w_pred = self.__mergeDfCol(X_w_pred, prediction_0)
        self.cate[treatment] = X_w_pred.select(X_w_pred.prediction_1 - X_w_pred.prediction_0).withColumnRenamed("(prediction_1 - prediction_0)", "cate")
        self.average_treatment_effects[treatment] = float(self.cate[treatment].groupby().avg().head()[0])
            
        return cate, ate

    def __fit(self, data):
        for treatment in self.treatments:
            
            # Single estimator
            assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
            data_assembled = assembler.transform(data)
            data_assembled = data_assembled.select(['features', self.outcome])
            self.estimator = self.estimator.fit(data_assembled)
        
    def __mergeDfCol(self, df_1, df_2):
        """
        Function to merge two spark dataframes.
        
        Parameters
        ----------
        df_1 (2-D Spark dataframe): Spark dataframe to merge 
        df_2 (2-D Spark dataframe): Spark dataframe to merge
        
        Returns
        ------
        df_3 (2-D Spark dataframe): Spark dataframe merged by df1 and df2
        """
        
        df_1 = df_1.withColumn("COL_MERGE_ID", monotonically_increasing_id())
        df_2 = df_2.withColumn("COL_MERGE_ID", monotonically_increasing_id())
        df_3 = df_2.join(df1, "COL_MERGE_ID").drop("COL_MERGE_ID")
        return df_3
