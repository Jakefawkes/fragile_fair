from autobound.Query import Query
from .query_algebra_utils import get_autobounds_query

def get_numerator_for_difference_metrics(
        problem, target_variable, conditioning_variable, 
        target_value = 1, conditioning_value = 0
    ):
    """
    This function returns the numerator for difference metrics.
    Numerators and denominators need to be computed separately for computational reasons.

    Args:
    - problem: The problem object.
    - target_variable: The target variable.
    - conditioning_variable: The conditioning variable.
    - target_value: The target value.
    - conditioning_value: The conditioning value.
    """
    if conditioning_variable != "A":
        variable_list = [target_variable, conditioning_variable, "A"]
        variable_values = [target_value, conditioning_value]
        term = get_autobounds_query(problem, variable_list, variable_values+[1]) * get_autobounds_query(problem, variable_list[1:], variable_values[1:]+[0])
        term = term + get_autobounds_query(problem, variable_list, variable_values+[0], sign=-1) * get_autobounds_query(problem, variable_list[1:], variable_values[1:]+[1])
    else:
        variable_list = [target_variable, conditioning_variable]
        variable_values = [target_value]
        term = get_autobounds_query(problem, variable_list, variable_values+[1]) * get_autobounds_query(problem, ["A"], [0])  
        term = term + get_autobounds_query(problem, variable_list, variable_values+[0], sign=-1)* get_autobounds_query(problem, ["A"], [1])  
    return term

def get_denominator_for_difference_metrics(
        problem, conditioning_variable, conditioning_value = 0
    ):
    """
    This function returns the denominator for difference metrics.
    Denominators only involve the conditioning variable.

    Args:
    - problem: The problem object.
    - conditioning_variable: The conditioning variable.
    - conditioning_value: The conditioning value.
    """
    if conditioning_variable != "A":
        variable_list = [conditioning_variable, "A"]
        variable_values = [conditioning_value]
        term = get_autobounds_query(problem, variable_list, variable_values + [0]) * get_autobounds_query(problem, variable_list, variable_values+[1])
    else:
        term = get_autobounds_query(problem, ["A"], [0]) * get_autobounds_query(problem, ["A"], [1])
    return term

def get_metric_expressions(
        problem, metric="FPR", 
        prediction_variable="P", attribute_variable="A", outcome_variable="Y"
):
    """
    This function returns the numerator and denominator for a given metric.

    Args:
    - problem: The problem object.
    - metric: The metric to analyze:
          Standard metrics: FPR, FNR, PPP, NPP, DP
          Causal metrics: TE, CF, SE.
    - prediction_variable: The prediction variable.
    - attribute_variable: The attribute variable.
    - outcome_variable: The outcome variable.

    Returns:
    - numerator: The numerator query for the metric.
    - denominator: The denominator query for the metric.
    """
    if metric in ["FPR", "FNR", "PPP", "NPP", "DP"]:
        if metric == "FPR":
            target_variable, cond_variable = prediction_variable, outcome_variable
            target_value, conditioning_value = 1, 0
        elif metric == "FNR":
            target_variable, cond_variable = prediction_variable, outcome_variable
            target_value, conditioning_value = 0, 1
        elif metric == "PPP":
            target_variable, cond_variable = outcome_variable, prediction_variable
            target_value, conditioning_value = 1, 1
        elif metric == "NPP":
            target_variable, cond_variable = outcome_variable, prediction_variable
            target_value, conditioning_value = 1, 0
        elif metric == "DP":
            target_variable, cond_variable = prediction_variable, attribute_variable
            target_value, conditioning_value = 1, 0
        numerator = get_numerator_for_difference_metrics(
            problem, target_variable, cond_variable, target_value, conditioning_value
        )
        denominator = get_denominator_for_difference_metrics(
            problem, cond_variable, conditioning_value
        )
    elif metric in ["TE","CF","SE"]:

        if "(" in prediction_variable:
            raise ValueError("Prediction variable cannot be interventional for causal metrics")

        elif "(" in attribute_variable:
            raise ValueError("Attribute variable cannot be interventional for causal metrics")

        elif "(" in outcome_variable:
            raise ValueError("Outcome variable cannot be interventional for causal metrics")

        elif metric == "TE":
            numerator = problem.query(f'{prediction_variable}({attribute_variable}=1)=1') + problem.query(f'{prediction_variable}({attribute_variable}=0)=1', -1)
            denominator = Query(1)
        elif metric  == "CF":
            # numerator = problem.query('P(A=1)=1&P(A=0)=1') + problem.query('P(A=1)=0&P(A=0)=0')
            numerator = problem.query(f'{prediction_variable}({attribute_variable}=1)=1 & {prediction_variable}({attribute_variable}=0)=1') + problem.query(f'{prediction_variable}({attribute_variable}=1)=0&{prediction_variable}({attribute_variable}=0)=0')
            denominator = Query(1)
        elif metric == "SE":
            # numerator = problem.query(f'{prediction_variable}({attribute_variable}=1)=1') * problem.query(f'{attribute_variable}=1') + problem.query(f'{prediction_variable}=1&{attribute_variable}=1', -1)
            # denominator = problem.query(f'{attribute_variable}=1')
            numerator = problem.query(f'{prediction_variable}({attribute_variable}=1)=1')*problem.query('{attribute_variable}=1') + problem.query(f'{prediction_variable}=1&{attribute_variable}=0',-1)
            denominator = Query(1)

    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return numerator, denominator