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
    This function runs the fair bounding analysis for a given metric on a given dataset. Examples of the arguments are:
    
    Args: 
    - observed_joint_table : A pandas Dataframe containing the empirical distribution of the predictor.
    - metric: todo
    - dag_str: A string containing the DAG to be used in the analysis. For example in the selection config this is: 
        "A->Y, A->P, A->S, U->P, U->Y, U->S, Y->S"
    - unob: A list containing the unobserved variables in the DAG. For the above DAG this is:  ["U"],
    - cond_nodes : A list contianing all the conditioned nodes. For example in selection we observe all 
        data conditional on S=1 and so cond_nodes = ["S"]
    - constraints: A list of constraints for the problem including the sensativity parameter. 
        The sensativity parameter is defined by an inequality involving the scalar D. 
        Example: ["P(Z = 0 & Y = 0) + P(Z = 1 & Y = 1) >= 1 - D"],
    - attribute_node: A string with the node for the true attribute in the causal graph. Set to: "A" by default.
    - outcome_node: A string with the node for the true outcome in the causal graph. Set to: "Y" by default. 
    - prediction_node: A string with the true prediction in the causal graph. Set to: "P" by default. 
    - sensitivity_parameter_values: The value of the sensativity parameter, given by an int or list of ints.
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

    if verbose == 1:
        print("Loading_data")
    problem.load_data(observed_joint_table, cond=cond_nodes)
    problem.add_prob_constraints()
    
    for constraint, sensitivity_parameter_value in zip(constraints, sensitivity_parameter_values):
        if verbose == 2:
            print(constraint, sensitivity_parameter_value)
        problem.add_natural_constraint(constraint, sensitivity_parameter_value)

    if verbose == 1:
        print("Collecting term")
    numerator, denominator = get_metric_fns(
        problem,
        metric=metric, 
        attribute_variable=attribute_node,
        outcome_variable=outcome_node, 
        prediction_variable=prediction_node
    )

    if verbose == 1:
        print("Setting Estimand")
    problem.set_estimand(numerator, div=denominator)
    
    program = problem.write_program()
    
    if verbose == 1:
        print("Running Autobounds")
    result = program.run_pyomo('ipopt', verbose=verbose==2)

    return result


def analyze_metric_bias_sensitivity(
        observed_joint_table, metric, bias, 
        sensitivity_parameter_values=0.05, 
        verbose=0, get_metric_fns=None
):  
    """
    This function runs the fair bounding analysis for a standard bias config used in the paper.
    Args:
    - observed_joint_table : A pandas Dataframe containing the empirical distribution of the predictor.
    - metric: todo 
    - bias: A string refering to one of the biases used in the paper with a name given by one of the jsons in src/bias_configs.
        Example "proxy_y".
    - sensitivity_parameter_values: The value of the sensativity parameter, given by an int or list of ints.
    - get_metric_expressions: todo
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

