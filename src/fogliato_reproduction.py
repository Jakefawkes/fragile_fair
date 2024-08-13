from query_algebra_utils import get_autobounds_query

def get_fogliato_metrics(
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