import itertools 
import numpy as np

from ..src.query_algebra_utils import get_autobounds_query

def get_p(
        df, prediction_node="P", attribute_node="A", outcome_node="Y"
):
    p = np.zeros((2, 2, 2))
    for i, j, k in itertools.product(range(2), range(2),range(2)):
        p[i, j, k] = df["prob"][(df[attribute_node] == i) & (df[outcome_node] == j) & (df[prediction_node] == k)].item()
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
    
    return (max(0,metric_val[0]), min(1,metric_val[1]))

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
    
    if not standard:
        reversed_df = observed_joint_table.copy()
        reversed_df["Z"] = 1-reversed_df["Z"]
        reversed_df["P"] = 1-reversed_df["P"]
        prob_vec = get_p(
            reversed_df,
            prediction_node=prediction_node, 
            attribute_node=attribute_node, 
            outcome_node=outcome_node
        )
    else:
        prob_vec = get_p(
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
