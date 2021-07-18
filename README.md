# Reina

## About Reina
ReInA (Reasoning In AI) is a causal inference platform aimed at estimating heterogeneous treatment effects in observational data. There are various open-source projects that provide convenient causal inference methods, but the current out-of-box packages are limited to local memory for computation. Hence, this project integrates Apache Spark with various machine learning (ML) powered causal inference frameworks, enabling causal analysis on big-data.

## Installation
    $ pip install reina
    
## Quick Start
    import reina
    
    # Read data from a distributed storage (e.g., Hadoop HDFS, AWS S3)
    data = spark.read
          .format("csv")
          .load("your_data.csv")
    
    # Set up necessary parameters (parameters will vary depending on the method used)
    treatment = ['name_of_treatment']
    outcome = 'name_of_outcome'
    
    # Setup and fit model
    causal_model = reina.iv.tsls(data=data, treatment=treatment, outcome=outcome)
    causal_model.fit(data=data, treatment)
    
    # Get heterogeneous treatment effect
    cate, ate = causal_model.effect()
    print(cate)
    print(ate)
    
Please refer to example notebooks for more detailed toy demonstrations.

## Contribution Guidelines

## References
