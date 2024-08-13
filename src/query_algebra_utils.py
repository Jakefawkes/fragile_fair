def subset_list(lst, subsetting_lst):
    """
    This function returns a subset of a list.
    
    Args:
    - lst: The list to subset.
    - subsetting_lst: The list of indices to subset.
    """
    return [lst[i] for i in subsetting_lst]

def sum_of_queries(query_list):
    """
    This function returns the sum of a list of queries.
    
    Args:
    - query_list: A list of queries.
    """
    query_sum = query_list[0]
    for term in query_list[1:]:
        query_sum = query_sum + term
    return query_sum

def prod_of_queries(query_list):
    """
    This function returns the product of a list of queries.

    Args:
    - query_list: A list of queries.
    """
    query_sum = query_list[0]
    for term in query_list[1:]:
        query_sum = query_sum * term
    return query_sum

def get_autobounds_query(problem, variable_list, variable_values, sign = 1):
    """
    This function returns the query for the specified variables and values.

    Args:
    - problem: The problem object.
    - variable_list: A list of variables.
    - variable_values: A list of values for the variables.
    - sign: The sign of the query (1 for >= 0, -1 for <= 0).
    """
    term_string = "={}&".join(variable_list) + "={}"
    term_string = term_string.format(*variable_values)
    if sign == 1:
        return problem.query(term_string)
    elif sign == -1:
        return problem.query(term_string, -1)
    else:
        raise(TypeError("sign must be 1 or -1."))
    
def get_autobounds_query_func(problem,variable_list,variable_values):
    """
    This function returns a function that computes the sum of queries for a subset of variables.

    Args:
    - problem: The problem object.
    - variable_list: A list of variables.
    - variable_values: A list of values for the variables.
    """
    def term_sum_func(subset, sign=1):
        """
        This function computes the sum of queries for a subset of variables.
        
        Args:
        - subset: The subset of variables.
        - sign: The sign of the query (1 for >= 0, -1 for <= 0).
        """
        return get_autobounds_query(
            problem,
            subset_list(variable_list, subset),
            subset_list(variable_values, subset),
            sign=sign
        )
    
    return term_sum_func