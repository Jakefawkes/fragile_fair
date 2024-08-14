from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import json
from .construct_fairness_metrics import get_metric_expressions
                     
def analyze_metric_sensitivity(
        observed_joint_table, metric, 
        dag_str, unob, constraints, cond_nodes=None, cond_node_values=1,
        attribute_node='A',outcome_node='Y',prediction_node='P',
        sensitivity_parameter_value=0.05, verbose=0, 
): 
    """
    This function runs the fair bounding analysis for a given metric on a given dataset.
    
    Args:
    - observed_joint_table: The observed joint table.
    - metric: The metric to analyze.
    - dag_str: The DAG string.
    - unob: The unobserved variable.
    - constraints: The constraints to add.
    - cond_nodes: The conditioning vertex.
    - cond_node_values: The conditioning vertex values.
    - sensitivity_parameter_value: The sensitivity parameter value.
    - verbose: The verbosity level.
    """
    dag = DAG()
    dag.from_structure(dag_str, unob = " ".join(unob))
    problem = causalProblem(dag)

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

    if verbose == 1:
        print("Loading_data")
    problem.load_data(observed_joint_table, cond=cond_nodes)
    problem.add_prob_constraints()

    if verbose == 1:
        print("Collecting term")
    numerator, denominator = get_metric_expressions(
        problem,
        metric=metric, 
        attribute_variable=attribute_node,
        outcome_variable=outcome_node, 
        prediction_variable=prediction_node
    )

    if verbose == 1:
        print("Setting Estimand")
    problem.set_estimand(numerator, div=denominator)
    
    for constraint in constraints:
        problem.add_natural_constraint(constraint, sensitivity_parameter_value)
    
    program = problem.write_program()
    
    if verbose == 1:
        print("Running Autobounds")
    result = program.run_pyomo('ipopt', verbose=False)

    return result

def analyze_metric_bias_sensitivity(
        probability_df, metric, bias, 
        sensitivity_parameter_value=0.05, 
        verbose=0
): 
    with open(f"bias_configs/{bias}.json") as f:
        bias_config = json.load(f)

    return analyze_metric_sensitivity(
        probability_df, 
        metric=metric,
        sensitivity_parameter_value=sensitivity_parameter_value,
        verbose=verbose,
        **bias_config
    )

