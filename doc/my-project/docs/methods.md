# Methods

## Metalearners

### S-Learner
	class reina.metalearners.SLearner

#### Methods

| Function Name        | Description |
| ----------- | ----------- |
| [__init__](dummy.com)     | Initialize object.         |
| [effects](dummy.com)  | Calculates and returns the treatment effects from this class.       |
| [fit](dummy.com)  | Train the causal model.        |

`effects(self, X, treatment)`

Calculates and returns the treatment effects from this class.

**Parameters**

>
- X (*2D Spark Dataframe*): Feature data to estimate treatment effect of
- treatment (*Str*): Name of the treatment variable 

**Returns**

>
- cate (*2D Spark DataFrame*): conditional average treatment effect
- ate (*float*): average treatment effect


`fit(self, data, treatments, outcome, estimator)`

Trains the ML-based causal model for this class.

**Parameters**

>
- data (*2-D Spark dataframe*): Base dataset containing features, treatment, iv, and outcome variables
- treatments (*List*): Names of the treatment variables
- outcome (*Str*): Name of the outcome variable
- estimator (*sklearn model obj*): Arbitrary ML estimator of choice

**Returns**

>
- None (*self*)

**Example**

Example S-learner usage can be found [here](https://github.com/Qyu-ai/Reina/blob/main/examples/notebooks/metalearner/metalearner_toy.ipynb).


### T-Learner
	class reina.metalearners.TLearner

**Methods**

| Function Name        | Description |
| ----------- | ----------- |
| [__init__](dummy.com)    | Initialize object.       |
| [effects](dummy.com)  | Calculates and returns the treatment effects from this class.        |
| [fit](dummy.com)  | Trains the causal model        |

`effects(self, X, treatment)`

Calculates and returns the treatment effects from this class.

**Parameters**

>
- X (*2D Spark Dataframe*): Feature data to estimate treatment effect of
- treatment (*Str*): Name of the treatment variable 

**Returns**

>
- cate (*2D Spark DataFrame*): conditional average treatment effect
- ate (*float*): average treatment effect


`fit(self,  data, treatments, outcome, estimator_0, estimator_1)`

Trains the ML-based causal model for this class.

**Parameters**

>
- data (*2-D Spark dataframe*): Base dataset containing features, treatment, iv, and outcome variables
- treatments (*List*): Names of the treatment variables
- outcome (*Str*): Name of the outcome variable
- estimator_0 (*mllib model obj*): Arbitrary ML model of choice
- estimator_1 (*mllib model obj*): Arbitrary ML model of choice

**Returns**

>
- None (*self*)

**Example**

Example T-learner usage can be found [here](https://github.com/Qyu-ai/Reina/blob/main/examples/notebooks/metalearner/metalearner_toy.ipynb).

### X-Learner
	class reina.metalearners.XLearner

**Methods**

| Function Name        | Description |
| ----------- | ----------- |
| [__init__](dummy.com)     | Initialize object.         |
| [effects](dummy.com) | Calculates and returns the treatment effects from this class.        |
| [fit](dummy.com)  | Trains the causal model.        |

`effects(self, X, treatment)`

Calculates and returns the treatment effects from this class.

**Parameters**

>
- X (*2D Spark Dataframe*): Feature data to estimate treatment effect of
- treatment (*Str*): Name of the treatment variable 

**Returns**

>
- cate (*2D Spark DataFrame*): conditional average treatment effect
- ate (*float*): average treatment effect


`fit(self,  data, treatments, outcome, estimator_0, estimator_1)`

Trains the ML-based causal model for this class.

**Parameters**

>
- data (*2-D Spark dataframe*): Base dataset containing features, treatment, iv, and outcome variables
- treatments (*List*): Names of the treatment variables
- outcome (*Str*): Name of the outcome variable
- estimator_10 (*mllib model obj*): Arbitrary ML model of choice
- estimator_11 (*mllib model obj*): Arbitrary ML model of choice
- estimator_20 (*mllib model obj*): Arbitrary ML model of choice
- estimator_21 (*mllib model obj*): Arbitrary ML model of choice
- propensity_estimator (*mllib model obj*): Arbitrary ML model for propensity function

**Returns**

>
- None (*self*)

**Example**

Example X-learner usage can be found [here](https://github.com/Qyu-ai/Reina/blob/main/examples/notebooks/metalearner/metalearner_toy.ipynb).

## IV-based Methods

### Two-Stage Least Squares
	class reina.iv.TwoStageLeastSquares

**Methods**

| Function Name        | Description |
| ----------- | ----------- |
| [__init__](dummy.com)     | Initialize object.         |
| [effects](dummy.com)  | Calculates and returns the treatment effects from this class.        |
| [fit](dummy.com)   | Trains the causal model.       |

`effects(self, X, treatment)`

Calculates and returns the treatment effects from this class.

**Parameters**

>
- X (*2D Spark Dataframe*): Feature data to estimate treatment effect of
- treatment (*Str*): Name of the treatment variable 

**Returns**

>
- cate (*2D Spark DataFrame*): conditional average treatment effect
- ate (*float*): average treatment effect


`fit(self, data, data, treatments, outcome, iv)`

Trains the ML-based causal model for this class.

**Parameters**

>
- data (*2-D Spark dataframe*): Base dataset containing features, treatment, iv, and outcome variables
- treatments (*List*): Names of the treatment variables
- outcome (*Str*): Name of the outcome variable
- iv (*Str*): Name of the instrument variable

**Returns**

>
- None (*self*)

**Example**

Example TwoStageLeastSquares usage can be found [here](https://github.com/Qyu-ai/Reina/blob/main/examples/notebooks/tsls/tsls_toy.ipynb).

