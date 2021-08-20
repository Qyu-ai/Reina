# Reina

## About Reina
ReInA (Reasoning In AI) is a causal inference platform aimed at estimating heterogeneous treatment effects in observational data. There are various open-source projects that provide convenient causal inference methods, but the current out-of-box packages are limited to local memory for computation. Hence, this project integrates Apache Spark with various machine learning (ML) powered causal inference frameworks, enabling causal analysis on big-data.

## Installation
    $ pip install reina
    
## Quick Start
    import reina
    from pyspark.sql import SparkSession

    # Initialize spark session
    spark = SparkSession \
                .builder \
                .appName('Meta-Learner-Spark') \
                .getOrCreate()
    
    # Read data locally (without cluster) or from a distributed storage (e.g., Hadoop HDFS, AWS S3) 
    data = spark.read \
          .format("csv") \
          .load("your_data.csv") \
    
    # Set up necessary parameters (parameters will vary depending on the method used)
    treatment = ['name_of_treatment']
    outcome = 'name_of_outcome'
    
    # Setup and fit model
    causal_model = reina.iv.TwoStageLeastSquares(data=data, treatment=treatment, outcome=outcome)
    causal_model.fit(data=data, treatments=treatment, outcome=outcome,...)
    
    # Get heterogeneous treatment effect
    cate, ate = causal_model.effect()
    print(cate)
    print(ate)
    
Please refer to example notebooks and [full documentation](https://qyu-ai.github.io/Reina/) for more detailed toy demonstrations.

## Contribution Guidelines
If you wish to contribute, please refer to our [contribution guidelines](./CONTRIBUTING.md).

Any contributions are greatly welcomed and appreciated.

## References
