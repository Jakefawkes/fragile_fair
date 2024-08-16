# Fragile_Fair

Code for the paper "The Fragility of Fairness: Causal Sensitivity Analysis
for Fair Machine Learning". 

This repo allows you to bound common fairness metrics under a variety  of meeasurement biases discussed in the paper. These are specified through json config files, and editing these allows for the specification of custom biases, causal structures, and constraints.


## Guide to Repo

### src

This folder contains the codebase for our metholodgy. Of main interest are the following:

- `src/analyze_metric_sensitivity.py`. This file contains the function `analyze_metric_sensitivity` which is used throughout to bound fairness metrics under a given level of measurement bias. This function takes as input an `observed_joint_table`, a fairness metric, and numerous custom arguments which are specified through a JSON file.
- `src/construct_fairness_metrics.py`. This file contains the code required to convert all metrics used in the paper to a form that can be read by autobounds. This can also serve as a guide for how to construct custom metrics. 
- `src/data_utils.py`. This provides the function `joint_distribution` which turns a df of results into an `observed_joint_table` of the type which needs to be used in `analyze_metric_sensitivity`.
- `src/bias_configs`. Contains examples of JSON config files to be used for the standard measurement biases used in the paper. More detials below.

### JSON config files
These are required to set numerous parameters for the sensativity analysis. Walking through the arguments for the  `src/bias_configs/selection.json`:


- `dag_str`: A string containing the DAG to be used in the analysis. For example in the selection config this is:  "A->Y, A->P, A->S, U->P, U->Y, U->S, Y->S"
- `unob`: The unobserved variables in the DAG. For the above DAG this is:  ["U"],
- `cond_nodes` : If we observe data conditional on some value they should be included here. For example in selection we observe all data conditional on S=1 and so cond_nodes = ["S"]
- `constraints`: Constraints for the problem including the sensativity parameter. Provided as a list where the sensativity parameter is defined by an inequality involving the scalar D. Example: ["P(Z = 0 & Y = 0) + P(Z = 1 & Y = 1) >= 1 - D"],
- `attribute_node`: The node for the true attribute in the causal graph. Set to: "A" by default.
- `outcome_node`: The node for the true outcome in the causal graph. Set to: "Y" by default. 
- `prediction_node`: The node for the true prediction in the causal graph. Set to: "P" by default. 

### experiments
Contains notebooks to reproduce each figure in the paper alongside the `experiments/playground.ipynb` notebook which contains an introduction to the code. 

### experiments
Contains example csv's that can be inputted as `observed_joint_table` into `analyze_metric_sensitivity`. This folder also contains the csvs to reproduce the experiments in the paper. 

## Enviroment Setup 

We recommend packages with conda to prevent having to manually install ipopt. In order to run all experiments run:

"""
conda env create -f full_environment.yml
"""

If the preference is to only run the playground.ipynb file and to begin using the bounding tools without recreating the figures from the paper, please run:

"""
conda env create -f base_environment.yml
"""


