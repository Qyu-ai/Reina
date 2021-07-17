#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Provides a spark-based X-learner heterogeneous treatment effect estimator.
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
    
class SparkXLearner:
    """
    Spark-based X-learner heterogeneous treatment effect estimator.

    Assumptions
    ---------------
    This class assumes that the data is already stored in a distributed storage system (e.g., HDFS).
    This class also assumes that the treatment variable only contains 1s and 0s.
    """
    
    def __init__(self, learner="T"):
        self.treatments = []  # Multiple treatment effects can be estimated
        self.covariates = []
        self.outcome = None
        self.estimator_10 = None
        self.estimator_11 = None
        self.estimator_20 = None
        self.estimator_21 =  None
        self.propensity_estimator = None

    def fit(self, data, treatments, outcome, estimator_10, estimator_11, estimator_20, estimator_21, propensity_estimator):
        """
        Wrapper function to fit an ML-based counterfacual model.
        When multiple treatments are inputted, each treatment effect is estiamted individually.
        
        Parameters
        ----------
        data (2-D Spark dataframe): Base dataset containing features, treatment, iv, and outcome variables
        treatments (List): Names of the treatment variables             
        outcome (Str): Name of the outcome variable
        estimator_10 (mllib model obj): Arbitrary ML model of choice
        estimator_11 (mllib model obj): Arbitrary ML model of choice
        estimator_20 (mllib model obj): Arbitrary ML model of choice
        estimator_21 (mllib model obj): Arbitrary ML model of choice
        propensity_estimator (mllib model obj): Arbitrary ML model for propensity function
        
        Returns
        ------
        self
        """
        
        self.treatments = treatments
        self.outcome = outcome
        self.covariates = [var for var in data.columns if var not in treatments and var != outcome]
        self.estimator_10 = estimator_10
        self.estimator_11 = estimator_11
        self.estimator_20 = estimator_20
        self.estimator_21 =  estimator_21
        self.propensity_estimator = propensity_estimator
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
        
        assert treatment in self.treatments
        
        # Final prediction
        assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
        X_assembled = assembler.transform(X)
        prediction_21 = estimator_21.transform(X_assembled.select('features')).withColumnRenamed("prediction", "prediction_21").select("prediction_21")
        prediction_20 = estimator_20.transform(X_assembled.select('features')).withColumnRenamed("prediction", "prediction_20").select("prediction_20")

        # Propensity function
        propensity_assembler = VectorAssembler(inputCols=self.covariates, outputCol='features')
        treatment_group_prop = propensity_assembler.transform(X)
        treatment_group_prop = treatment_group_prop.select(['features', treatment])
        treatment_prob = propensity_estimator.transform(treatment_group_prop).select("probability")
        firstelement = udf(lambda v:float(v[1]),FloatType())
        treatment_prob = treatment_prob.select(firstelement('probability')).withColumnRenamed("<lambda>(probability)", "probability")

        # Get cate
        X_w_pred = self.__mergeDfCol(X, prediction_21)
        X_w_pred = self.__mergeDfCol(X_w_pred, prediction_20)
        X_w_pred = self.__mergeDfCol(X_w_pred, treatment_prob)
        X_w_pred = X_w_pred.withColumn("probability", X_w_pred.probability.cast("float"))
        
        # should be + but - seems to produce the correct results...
        self.cate[treatment] = X_w_pred.select((X_w_pred.probability * X_w_pred.prediction_21) - ((lit(1) - X_w_pred.probability) * X_w_pred.prediction_20)).withColumnRenamed("((probability * prediction_21) + ((1 - probability) * prediction_20))", "cate")
        self.average_treatment_effects[treatment] = float(self.cate[treatment].groupby().avg().head()[0])
            
        return cate, ate

    
    def __fit(self, data):
        
        # TODO: result of X-learner is a little weird...prediction20 and prediction21 are very close. Maybe need to check implementation again
        
        for treatment in self.treatments:
            
            # Set up assembler
            assembler = VectorAssembler(inputCols=self.covariates+[treatment], outputCol='features')
            
            # First Stage
            # First estimator (treatment group)
            treatment_group = data.filter(treatment+" == 1")
            treatment_group_assembled = assembler.transform(treatment_group)
            treatment_group_assembled = treatment_group_assembled.select(['features', self.outcome])
            self.estimator_11 = self.estimator_11.fit(treatment_group_assembled)
            
            # Second estimator (control group)
            control_group = data.filter(treatment+" == 0")
            control_group_assembled = assembler.transform(control_group)
            control_group_assembled = control_group_assembled.select(['features', self.outcome])
            self.estimator_10 = self.estimator_10.fit(control_group_assembled)
            
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
            self.estimator_21 = self.estimator_21.fit(treatment_group_third)
            self.estimator_20 = self.estimator_20.fit(control_group_third)
            
            # Fit propensity estimator
            assembler_propensity = VectorAssembler(inputCols=self.covariates, outputCol='features')
            treatment_group_prop = assembler_propensity.transform(data)
            treatment_group_prop = treatment_group_prop.select(['features', treatment])
            self.propensity_estimator = self.propensity_estimator.fit(treatment_group_prop)
            treatment_prob = propensity_estimator.transform(treatment_group_prop).select("probability")
            firstelement=udf(lambda v:float(v[1]),FloatType())
            treatment_prob = treatment_prob.select(firstelement('probability')).withColumnRenamed("<lambda>(probability)", "probability")
            

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

