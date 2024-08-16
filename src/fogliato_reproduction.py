import itertools 
import numpy as np
from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
import json
from autobound.Query import Query
from io import StringIO

from .query_algebra_utils import get_autobounds_query

def get_p_vector_torch(df,
        prediction_node="P", attribute_node="A", outcome_node="Y"):
    p = np.zeros((2,2,2))
    for i,j,k in itertools.product(range(2),range(2),range(2)):
        p[i,j,k] =   df["prob"][(df[attribute_node] == i)  & (df[outcome_node] == j) & (df[prediction_node] == k)].item()
    return p

def get_fogliato_metrics_autobounds(
        problem, metric="FPR", group=0,
        prediction_variable="P", attribute_variable="A", outcome_variable="Y"
):
    variable_list = [outcome_variable, prediction_variable, attribute_variable]
    if metric == "FPR":
        numerator = get_autobounds_query(problem, variable_list, [0, 1, group])
        denominator = get_autobounds_query(problem, variable_list, [0, 1, group]) + get_autobounds_query(problem, variable_list, [0, 0, group])
    elif metric == "FNR":
        numerator = get_autobounds_query(problem, variable_list, [1, 0, group])
        denominator = get_autobounds_query(problem, variable_list, [1, 0, group]) + get_autobounds_query(problem, variable_list, [1, 1, group])
    elif metric == "PPV":
        numerator = get_autobounds_query(problem, variable_list, [1, 1, group])
        denominator = get_autobounds_query(problem, variable_list, [0, 1, group]) + get_autobounds_query(problem, variable_list, [1, 1, group])
    
    return numerator, denominator

def get_fogliato_true_bounds(prob_vec,sensitivity_parameter_value = 0.05,group=0,metric = "FPR",constrained = False,standard = True):

    p = prob_vec[group,:,:]

    if metric == "FPR":

        if standard:
            if not constrained:
                metric_val = ((p[0,1]-sensitivity_parameter_value)/(p[0,1]+p[0,0]-sensitivity_parameter_value),(p[0,1])/(p[0,1]+p[0,0]-sensitivity_parameter_value))
            else:
                metric_val = ( ( p[0,1]* (p[1,0] + p[1,1] ) - sensitivity_parameter_value * p[1,0] ) / ((p[0,1]+p[0,0]-sensitivity_parameter_value)*(p[1,0]+p[1,1])),(p[0,1])/(p[0,1]+p[0,0]))
        else: 
            metric_val = get_fogliato_true_bounds(prob_vec,sensitivity_parameter_value = sensitivity_parameter_value,group=group,metric = "FNR",constrained=constrained,standard = True)

    if metric == "FNR":
        if standard:
            if not constrained:
                if sensitivity_parameter_value<p[0,1]:
                    metric_val = ((p[1,0])/(p[1,0]+p[1,1]+sensitivity_parameter_value),(p[1,0]+sensitivity_parameter_value)/(p[1,0]+p[1,1]+sensitivity_parameter_value))
                else:
                    metric_val = ((p[1,0])/(p[1,0]+p[1,1]+p[0,1]),(p[1,0]+sensitivity_parameter_value)/(p[1,0]+p[1,1]+sensitivity_parameter_value))
            else:
                metric_val = metric_val = ((p[1,0])/(p[1,0]+p[1,1]),(p[1,1])/(p[1,0]+p[1,1]))
        else: 
            metric_val = get_fogliato_true_bounds(prob_vec,sensitivity_parameter_value = sensitivity_parameter_value,group=group,metric = "FPR",constrained=constrained,standard = True)
    
    if metric == "PPV":
        if standard:
            if not constrained:
                metric_val = ((p[1,1])/(p[0,1]+p[1,1]),(p[1,1]+sensitivity_parameter_value)/(p[0,1]+p[1,1]))
            else: 
                metric_val = ((p[1,1])/(p[0,1]+p[1,1]),(p[1,1]+sensitivity_parameter_value*((p[1,0])/(p[1,0]+p[1,1]) ))/(p[0,1]+p[1,1]))
        else:
            if not constrained:
                metric_val = ((p[0,0])/(p[1,0]+p[0,0]),(p[0,0]-sensitivity_parameter_value)/(p[1,0]+p[0,0]))
            else: 
                metric_val = ((p[1,1])/(p[0,1]+p[1,1]),(p[1,1]+sensitivity_parameter_value*((p[1,0])/(p[1,0]+p[1,1]) ))/(p[0,1]+p[1,1]))
    
    return (max(0,metric_val[0]),min(1,metric_val[1]))

def Fogliato_true_bounds(
        observed_joint_table, metric, 
        dag_str, constraints,
        attribute_node='A',outcome_node='Z',prediction_node='P',
        sensitivity_parameter_value=0.05,group = 0
): 
    """
    This function computes the true bounds for the Fogliato Case.
    
    Args:
    - observed_joint_table: The observed joint table.
    - metric: The metric to analyze.
    - dag_str: The DAG string.
    - constraints: The constraints to add, can only handle those from the experiment form.
    - sensitivity_parameter_value: The sensitivity parameter value.
    - verbose: The verbosity level.
    """
    constrained = not("U->Z" in dag_str.replace(" ",""))
    
    standard = False
    for c in constraints:
        if "Y=0&Z=1)==0" in c.replace(" ",""):
            standard = True
            break
        if "Z=1&Y=0)==0" in c.replace(" ",""):
            val = True
            break
    
    if not standard:
        reversed_df = observed_joint_table.copy()
        reversed_df["Z"] = 1-reversed_df["Z"]
        reversed_df["P"] = 1-reversed_df["P"]
        prob_vec = get_p_vector_torch(
            reversed_df,
            prediction_node=prediction_node, 
            attribute_node=attribute_node, 
            outcome_node=outcome_node
        )
    else:
        prob_vec = get_p_vector_torch(
            observed_joint_table,
            prediction_node=prediction_node, 
            attribute_node=attribute_node, 
            outcome_node=outcome_node
        )

    return get_fogliato_true_bounds(
        prob_vec, 
        sensitivity_parameter_value=sensitivity_parameter_value,
        group=group, metric = metric, constrained=constrained, standard=standard
    )

# def analyze_metric_sensitivity_Fogliato(
#         observed_joint_table, metric, 
#         dag_str, unob, constraints, cond_nodes=None, cond_node_values=1,
#         attribute_node='A',outcome_node='Y',prediction_node='P',
#         sensitivity_parameter_value=0.05, verbose=0, custom_parser= True,
# ): 
#     """
#     This function runs the fair bounding analysis for a given metric on a given dataset.
    
#     Args:
#     - observed_joint_table: The observed joint table.
#     - metric: The metric to analyze.
#     - dag_str: The DAG string.
#     - unob: The unobserved variable.
#     - constraints: The constraints to add.
#     - cond_nodes: The conditioning vertex.
#     - cond_node_values: The conditioning vertex values.
#     - sensitivity_parameter_value: The sensitivity parameter value.
#     - verbose: The verbosity level.
#     """
#     dag = DAG()
#     dag.from_structure(dag_str, unob = unob)
#     problem = causalProblem(dag)

#     # This handles conditional node logic:
#     # If cond_nodes is not None, then we need to add the conditional nodes to the observed_joint_table
#     # we do this by iterating over the cond_nodes and cond_node_values and adding them to the observed_joint_table
#     if cond_nodes is None:
#         cond_nodes = []
#     else:
#         if not hasattr(cond_node_values, '__iter__'):
#             cond_node_values = [cond_node_values] * len(cond_nodes)
#         for node, value in zip(cond_nodes, cond_node_values):
#             observed_joint_table[node] = value

#     if verbose == 1:
#         print("Loading_data")
#     problem.load_data(observed_joint_table, cond=cond_nodes)
#     # problem.add_prob_constraints()

#     if verbose == 1:
#         print("Collecting term")
#     numerator, denominator = get_fogliato_metrics_autobounds(
#         problem,
#         metric=metric, 
#         attribute_node=attribute_node,
#         outcome_node=outcome_node, 
#         prediction_node=prediction_node
#     )

#     if verbose == 1:
#         print("Setting Estimand")
#     problem.set_estimand(numerator, div=denominator)
    
#     if custom_parser:
#         for constraint in constraints:
#             problem.add_natural_constraint(constraint, sensitivity_parameter_value)
    
#     else:
#     #     # problem.add_constraint(problem.query("Y=1&Z=0"),symbol="==")
#     #     # problem.add_constraint(problem.query(f"Y=0&Z=1&A=0") - Query(sensitivity_parameter_value),symbol="==")
            
#     #     problem.add_constraint(problem.query("Y=1&Z=0"),symbol="==")

#     # # problem.add_constraint(problem.query(f"Y=1&Z=0&A={op_group}"),symbol="==")
#     #     problem.add_constraint(problem.query(f"Y=0&Z=1&A=0") - Query(sensitivity_parameter_value),symbol="<=")
#         problem.add_constraint(problem.query("Y=0&Z=1"),symbol="==")
#         problem.add_constraint(problem.query(f"Y=1&Z=0&A=0") - Query(sensitivity_parameter_value),symbol="<=")
    
#     program = problem.write_program()
    
#     if verbose == 1:
#         print("Running Autobounds")
#     result = program.run_pyomo('ipopt', verbose=True)

#     return result