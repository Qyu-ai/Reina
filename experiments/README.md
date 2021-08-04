# Experiments
We conducted some run-time experiments on a toy dataset to demonstrate big-data processing ability of Econml (Microsoft) versus Reina (Qyu.ai).

# Methods \& Setup
The dataset was generated randomly, purely for the purpose of big-data processing demonstration. Experiments on real datasets will come soon. Currently, we have tried 4 causal inference methods: S-learner, T-learner, X-learner, and two-stage least squares.

The machine used for running the methods was an AWS cloud instance size of m5.xlarge (please refer to *https://aws.amazon.com/ec2/instance-types/* for detailed description of the instance). We used the AWS EMR (Elastic Map Reduce) service to setup a Spark cluster of 3 instances for Reina.

# Preliminary Results

**Tables showing EconML (local machine memory) and Reina (Spark cluster) on dataset of size ~6GB, on an AWS 

| Method   | Run-time (seconds) |
| ----------- | ----------- |
| SieveTSLS (Econml)       | N/A |
| SLearner (Econml)        | Memory Error |
| TLearner (Econml)        | Memory Error |
| XLearner (Econml)        | Memory Error |

| Method   | Run-time (seconds) |
| ----------- | ----------- |
| TSLS (Reina)       |  N/A |
| SLearner (Reina)        | 519.46 |
| TLearner (Reina)        |  658.91 |
| XLearner (Reina)        | 1714.05 |

From the table above, it is clear that Reina has an advantage on making causal inference on big-data possible.