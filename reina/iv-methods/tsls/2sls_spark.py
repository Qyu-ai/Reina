#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Provides a spark-based two-stage least squares treatment effect estimator.
"""

from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import avg
from pyspark.sql.functions import lit
from pyspark.sql.functions import when
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

class SparkSieveTSLS:
    """
    Spark-based two-stage least squared heterogeneous treatment effect estimator.
    
    Assumptions
    ---------------
    This class assumes that the data is already stored in a distributed storage system (e.g., HDFS).
    This class also assumes that the treatment variable only contains 1s and 0s.
    """

    def __init__(self):
        self.treatments = []  # Multiple treatment effects can be estimated
        self.covariates = []
        self.outcome = ""
        self.iv = ""
        self.ols_1 = LinearRegression(featuresCol = 'features', labelCol=treatment)
        self.ols_2 = LinearRegression(featuresCol = 'features', labelCol=self.outcome)

    def fit(self, data, treatments, outcome, iv):
        """
        Wrapper function to fit first and second stage linear model to get adjusted treatment and a counterfacual model.
        When multiple treatments are inputted, each treatment effect is estiamted individually.
        
        Parameters
        ----------
        data (2-D Spark dataframe): Base dataset containing features, treatment, iv, and outcome variables
        treatments (List): Names of the treatment variables             
        outcome (Str): Name of the outcome variable
        iv (Str): Name of the instrument variable
              
        Returns
        ------
        self
        """
        
        self.treatments = treatments
        self.outcome = outcome
        self.iv = iv
        self.covariates = [var for var in data.columns if var not in treatments and var != iv and var != outcome]
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
        
        # Predict adjusted treatment
        temp_1 = self.__mergeDfCol(X.select(treatment), X.select(self.iv))
        assembler = VectorAssembler(inputCols=[self.iv], outputCol='features')
        temp_1_assembled = assembler.transform(temp_1)
        temp_1_assembled = temp_1_assembled.select(['features', treatment])
        adjustedTreatment = self.ols_1 .transform(temp_1_assembled.select('features')).select("prediction")
        
        # Get predicted counterfactual for Y1 and Y0
        X_adjusted = self.__mergeDfCol(X, adjustedTreatment)
        counterfactual_treatment = X_adjusted.withColumn("prediction", lit(1))
        counterfactual_control =  X_adjusted.withColumn("prediction", lit(0))
        counterfactual_treatment_assembled = assembler.transform(counterfactual_treatment).select("features")
        counterfactual_control_assembled = assembler.transform(counterfactual_control).select("features")
        prediction_1 = self.ols_2.transform(counterfactual_treatment_assembled).withColumnRenamed("prediction", "prediction_1").select("prediction_1")
        prediction_0 = self.ols_2.transform(counterfactual_control_assembled).withColumnRenamed("prediction", "prediction_0").select("prediction_0")

        # Get cate and ate
        data_w_pred = self.__mergeDfCol(data, prediction_1)
        data_w_pred = self.__mergeDfCol(data_w_pred, prediction_0)
        cate = data_w_pred.select(data_w_pred.prediction_1 - data_w_pred.prediction_0).withColumnRenamed("(prediction_1 - prediction_0)", "cate")
        ate = float(cate.groupby().avg().head()[0])  # simply the average of cate
        return cate, ate
            
    def __fit(self, data):
        for treatment in self.treatments:
            
            # Fit first stage OLS
            trainSet_1 = self.__mergeDfCol(data.select(treatment), data.select(self.iv))
            assembler = VectorAssembler(inputCols=[self.iv], outputCol='features')
            trainSet_1_assembled = assembler.transform(trainSet_1)
            trainSet_1_assembled = trainSet_1_assembled.select(['features', treatment])
            self.ols_1 = self.ols_1.fit(trainSet_1_assembled)
            adjustedTreatment = self.ols_1 .transform(trainSet_1_assembled.select('features')).select("prediction")
            
            # Fit second stage OLS
            trainSet_2 = self.__mergeDfCol(data, adjustedTreatment)
            assembler = VectorAssembler(inputCols=self.covariates+["prediction"], outputCol='features')
            trainSet_2_assembled = assembler.transform(trainSet_2)
            trainSet_2_assembled = trainSet_2_assembled.select(['features', self.outcome])
            self.ols_2 = self.ols_2.fit(trainSet_2_assembled)


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

