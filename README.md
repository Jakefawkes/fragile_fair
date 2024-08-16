# Fragile Fair

This is a codebase for analyzing the fragility of fairness metrics under various measurement bias. The codebase leverages Directed Acyclic Graphs (DAGs) to model the causal relationships between the variables in the data and probabilistic constraints on those variables to allowing users to flexibly specify the causal relationships between the variables in their data. Biases are reduced to a particular DAG and a set of constraints, which we specify through a JSON config. A number of standard biases and metrics are readily implemnted in the codebase, and users can easily add their own biases and metrics.

This codebase is the official implementation of the paper "The Fragility of Fairness: Causal Sensitivity Analysis for Fair Machine Learning". In the experiments dir we provide notebooks to reproduce all the figures in the paper, demonstrating how to use the codebase to do sophisticated sensitivity analysis on fairness metrics.

## Installation

To install the codebase, clone the repository and install the requirements:

```bash 
git clone
cd fragile_fair
conda env create -f environment.yml
```

If you want to run the experiments in the paper, you will need to use the experimental environment:

```bash
conda env create -f exps_environment.yml
```

## Usage

Using the codebase is straightforward. The main function to use is `analyze_metric_bias_sensitivity` which takes as input an `observed_joint_table`, a fairness metric, and one of the pre-specified bias configs in `src/bias_configs`, which specifies the DAG, constraints, and other parameters for the analysis.

```python
from src.analyze_metric_sensitivity import analyze_metric_sensitivity
from src.data_utils import joint_distribution
import pandas as pd

# Load the data
data = pd.read_csv('data/selection.csv')

# Convert the data to an observed joint table
observed_joint_table = joint_distribution(data, ['A', 'Y', 'P'])

# Run the analysis of the selection bias on the False Positive Rate disparity metric
lb, ub, lb_converged, ub_converged = analyze_metric_bias_sensitivity(
    probability_df, metric='FPR', bias='selection', 
    sensitivity_parameter_values=0.05
)
```

The `analyze_metric_bias_sensitivity` function is really a wrapper around the `analyze_metric_sensitivity` function which is the core of the codebase. Unpacking this we can instead write:

```python
from src.analyze_metric_sensitivity import analyze_metric_sensitivity
from src.data_utils import joint_distribution
import pandas as pd
imort json

# Load the data
data = pd.read_csv('data/selection.csv')

# Convert the data to an observed joint table
observed_joint_table = joint_distribution(data, ['A', 'Y', 'P'])

# Load the bias config
with open('src/bias_configs/selection.json') as f:
    bias_config = json.load(f)

# Run the analysis of the selection bias on the False Positive Rate disparity metric
lb, ub, lb_converged, ub_converged = analyze_metric_sensitivity(
    observed_joint_table, metric='FPR', 
    sensitivity_parameter_values=0.05, 
    **bias_config
)
```

The bias config file is a JSON file that specifies the DAG, constraints, and other parameters for the analysis. For example, the `src/bias_configs/selection.json` file specifies the following:

```json
{
    "dag_str": "A->Y, A->P, A->S, U->P, U->Y, U->S, Y->S",
    "unob": ["U"],
    "cond_nodes": ["S"],
    "attribute_node": "A",
    "outcome_node": "Y",
    "prediction_node": "P",
    "constraints": ["P(S = 1) >= 1 - D"]
}
```

The `dag_str` field is really just an edgelist for the DAG. The `unob` field specifies the unobserved variables in the DAG, which are used to compute the conditional independencies between the observable variables. The `cond_nodes` field specifies the variables that are observed conditional on some value, e.g. in the selection case we only observe individuals conditional on `S=1`.

The `attribute_node`, `outcome_node`, and `prediction_node` fields specify the nodes for the observed attribute, outcome, and prediction which will be used in the parity metric. Usually these will be set to "A", "Y", and "P" respectively, but some biases may require different nodes. For example in Proxy Y Bias the observed attribute `Y` is not the true outcome but a proxy for the true outcome, so the `outcome_node` field would be set to the true outcome node. In the `src/bias_configs/proxy_y.json` we call this true outcome `Z`.

The `constraints` field specifies a set of probabilistic constraints on the variables in the DAG. The variable `D` is a protected variable name which is used to specify the the sensitivity parameter, which is a bias-specific level of measurement bias. In the selection bias we have `P(S = 1) >= 1 - D`, so `D` is the probability of observing an individual. In Proxy Y Bias we have `P(Z = 0 & Y = 0) + P(Z = 1 & Y = 1) >= 1 - D`, so `D` lower bounds the probability that the observed (proxy) outcome equals the true outcome.

The `src/construct_fairness_metrics.py` file contains the code required to convert all metrics used in the paper to a form that can be read by autobounds. This can also serve as a guide for how to construct custom metrics. The `src/data_utils.py` file contains the `joint_distribution` function which turns a df of results into an `observed_joint_table` of the type which needs to be used in `analyze_metric_sensitivity`.

## Creating New Bias Configs

It is easy to create new bias configs by following the template of the existing bias configs. The `src/bias_configs` directory contains a number of bias configs for the biases discussed in the paper. The selection and proxy_y configs are good starting points for understanding the format. The `src/bias_configs/proxy_y_and_selection.json` file is an example of how these two configs can be combined, which is useful for understanding how to analyze multiple biases at once. More advanced users may also be interested in creating biases that rely on intervening on the DAG, we illustrate how to do this in the `src/bias_configs/ecp.json` config.

## Understanding the Codebase

The codebase is structured as follows:

```bash
fragile_fair/
├── src/
│   ├── analyze_metric_sensitivity.py
│   ├── construct_fairness_metrics.py
│   ├── data_utils.py
│   └── bias_configs/
│       ├── selection.json
│       ├── proxy_y.json
│       ├── proxy_y_and_selection.json
│       ├── ecp.json
├── experiments/
│   ├── fig{i}.ipynp
│   ├── utils.py
├── data/
│   ├── ...

```

The `src` directory contains the core codebase. The `analyze_metric_sensitivity.py` file contains the main functions for analyzing the sensitivity of fairness metrics to biases. The `construct_fairness_metrics.py` file contains the code for constructing the fairness metrics used in the paper. The `data_utils.py` file contains the code for converting a dataframe of results into an `observed_joint_table` which is the input to the `analyze_metric_sensitivity` function. The `bias_configs` directory contains the JSON files which specify the DAGs, constraints, and other parameters for the biases. The `experiments` directory contains the notebooks which reproduce the figures in the paper. The `data` directory contains the data used in the experiments.


