import os 
from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
from autobound.Query import Query
import json
from .construct_fairness_metrics import get_metric_expressions
                     
def analyze_metric_sensitivity(
        observed_joint_table, metric, 
        dag_str, unob, constraints, cond_nodes=None, cond_node_values=1,
        attribute_node='A',outcome_node='Y',prediction_node='P',
        sensitivity_parameter_values=0.05, verbose=0, 
        get_metric_fns=None
): 
    """
    This function analyzes the sensitivity of a fairness metric to a given bias.

    A bias is defined by:
        1. a DAG, specified by the edge list `dag_str`, the unobserved nodes `unob`, and the conditional nodes `cond_nodes`
        2. a set of probabilistic constraints, specified by the list `constraints`

    Rather than specifying each of these components we advise writing a bias config and passing that in using **config. 
    A set of examples of pre-defined bias configs is provided in the `bias_configs` directory. An example config looks like:

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

    NOTE: All variables in the DAG should be a signle character. The behavior of the function is not guaranteed if this is not the case.

    Args:
    - observed_joint_table: a pandas DataFrame containing a row for every possible combination for every observed, 
        non-conditioned variables in the DAG and the associated probability of that combination.
        The function `joint_distribution` in `src/data_utils.py` can be used to generate this table
        from a DataFrame of binary columns of observed data.
    - metric: The metric to analyze. We suuport the following metrics by default:
        Standard metrics: FPR, FNR, PPP, NPP, DP
        Causal metrics: TE, CF, SE.
    - dag_str: The edge list of the DAG.
    - unob: The list of unobserved nodes in the DAG.
    - constraints: A list of constraints to be applied when bounding the metric, these should be written in terms of the
        observed or conditioned variables in the DAG. The constraints should also include `D` as a variable which is replaced
        by the input value of the sensitivity parameter.
    - cond_nodes: A list of conditional nodes in the DAG.
    - cond_node_values: A list of binary values for the conditional nodes in the DAG.
    - attribute_node: The attribute node in the DAG, typically denoted by `A`.
    - outcome_node: The outcome node in the DAG, typically denoted by `Y`.
    - prediction_node: The prediction node in the DAG, typically denoted by `P`.
    - sensitivity_parameter_values: A float or list of floats specifing the value of the sensitivity parameter to use when 
        bounding the metric. If a list is provided, the length of the list should be the same as the length of the constraints list,
        and the sensitivity parameter values will be applied to the constraints in order allowing for different sensitivity values
        for different constraints. Note that each constraints should include `D` as the placeholder which will be replaced by the input value.
    - verbose: The verbosity level of the function.     
        0 is silent, 
        1 prints the stages of the bounding process, 
        2 prints details of internal optimization calls
    - get_metric_fns: A function that takes a `metric` string and returns the numerator and denominator queries for that metric.
        The only reason to provide this function is to allow for custom metrics to be used. By default, the function uses the
        `get_metric_expressions` function from `src/construct_fairness_metrics.py`. A custom function should have the same signature
        as `get_metric_expressions`. See `experiments/fogliato_reproduction.py` for an example of how to use a custom metric.

    Returns:
    - lower bound: The lower bound of the metric.
    - upper bound: The upper bound of the metric.
    - lower bound convergence: The convergence status of the lower bound.
    - upper bound convergence: The convergence status of the upper bound.
    """
    dag = DAG()
    dag.from_structure(dag_str, unob = unob)
    problem = causalProblem(dag)

    if get_metric_fns is None:
        get_metric_fns = get_metric_expressions

    # This handles conditional node logic:
    # If cond_nodes is not None, then we need to add the conditional nodes to the observed_joint_table
    # we do this by iterating over the cond_nodes and cond_node_values and adding them to the observed_joint_table
    if cond_nodes is None:
        cond_nodes = []
    else:
        if not hasattr(cond_node_values, '__iter__'):
            cond_node_values = [cond_node_values] * len(cond_nodes)
        for node, value in zip(cond_nodes, cond_node_values):
            observed_joint_table[node] = value

    if not hasattr(sensitivity_parameter_values, '__iter__'):
        sensitivity_parameter_values = [sensitivity_parameter_values] * len(constraints)

    if verbose >= 1:
        print("Loading_data")
    problem.load_data(observed_joint_table, cond=cond_nodes)

    # Add constraints to the problem to ensure variables are treated as probabilities
    problem.add_prob_constraints()
    
    # Add constraints to the problem
    # The constraints should be written in terms of the observed or conditioned variables in the DAG
    # The constraints should also include `D` as a variable which is replaced by the input value of the sensitivity parameter
    for constraint, sensitivity_parameter_value in zip(constraints, sensitivity_parameter_values):
        if verbose == 2:
            print(constraint, sensitivity_parameter_value)
        problem.add_natural_constraint(constraint, sensitivity_parameter_value)

    if verbose >= 1:
        print("Collecting term")
    numerator, denominator = get_metric_fns(
        problem,
        metric=metric, 
        attribute_variable=attribute_node,
        outcome_variable=outcome_node, 
        prediction_variable=prediction_node
    )

    if verbose >= 1:
        print("Setting Estimand")
    problem.set_estimand(numerator, div=denominator)
    
    program = problem.write_program()
    
    if verbose >= 1:
        print("Running ipopt")
    result = program.run_pyomo('ipopt', verbose=verbose==2)

    return result


def analyze_metric_bias_sensitivity(
        observed_joint_table, metric, bias, 
        sensitivity_parameter_values=0.05, 
        verbose=0, get_metric_fns=None
):  
    """
    This function analyzes the sensitivity of a fairness metric to a given bias saved in a json file in the `bias_configs` directory.

    This is a wrapper function around `analyze_metric_sensitivity` that loads the bias config from a json file, and the details of 
    the arguments can be found in the docstring of `analyze_metric_sensitivity`.

    Args:
    - observed_joint_table: a pandas DataFrame containing a row for every possible combination for every observed, 
        non-conditioned variables in the DAG and the associated probability of that combination.
        The function `joint_distribution` in `src/data_utils.py` can be used to generate this table
        from a DataFrame of binary columns of observed data.
    - metric: The metric to analyze. We suuport the following metrics by default:
        Standard metrics: FPR, FNR, PPP, NPP, DP
        Causal metrics: TE, CF, SE.
    - bias: The name of the bias config file to load from the `bias_configs` directory.
        Detault biases: selection, proxy_a, proxy_y, ecp, proxy_y_and_selection
    - sensitivity_parameter_values: The value of the sensitivity parameter to use when bounding the metric.
    - verbose: The verbosity level of the function.     
        0 is silent, 
        1 prints the stages of the bounding process, 
        2 prints details of internal optimization calls
    - get_metric_fns: A function that takes a `metric` string and returns the numerator and denominator queries for that metric.
        The only reason to provide this function is to allow for custom metrics to be used. By default, the function uses the
        `get_metric_expressions` function from `src/construct_fairness_metrics.py`. A custom function should have the same signature
        as `get_metric_expressions`.

    Returns:
    - lower bound: The lower bound of the metric.
    - upper bound: The upper bound of the metric.
    - lower bound convergence: The convergence status of the lower bound.
    - upper bound convergence: The convergence status of the upper bound.
    """
    current_dir = os.path.dirname(__file__)  # Get the directory of the current module
    with open(os.path.join(current_dir, "bias_configs", f"{bias}.json")) as f:
        bias_config = json.load(f)

    return analyze_metric_sensitivity(
        observed_joint_table, 
        metric=metric,
        sensitivity_parameter_values=sensitivity_parameter_values,
        verbose=verbose, get_metric_fns=get_metric_fns,
        **bias_config
    )

