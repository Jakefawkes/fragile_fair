from autobound.causalProblem import causalProblem
from autobound.DAG import DAG
from autobound.Query import Query
import pandas as pd
from io import StringIO

# Below is the code used to convert different disparity metrics into terms that can be read by autobounds.
def subset_list(lst,subsetting_lst):
    return [lst[i] for i in subsetting_lst]

def sum_of_queries(query_list):
    query_sum = query_list[0]
    for term in query_list[1:]:
        query_sum = query_sum + term
    return query_sum

def prod_of_queries(query_list):
    query_sum = query_list[0]
    for term in query_list[1:]:
        query_sum = query_sum * term
    return query_sum

def get_autobounds_query(problem,variable_list,variable_values,sign = 1):
    term_string = "={}&".join(variable_list)+"={}"
    term_string = term_string.format(*variable_values)
    if sign == 1:
        return problem.query(term_string)
    elif sign == -1:
        return problem.query(term_string,-1)
    else:
        raise(TypeError("sign must be 1 or -1."))
    
def get_autobounds_query_func(problem,variable_list,variable_values):

    def term_sum_func(subset,sign = 1):
        return get_autobounds_query(problem,subset_list(variable_list,subset),subset_list(variable_values,subset),sign=sign)
    
    return term_sum_func

def get_numerator_for_difference_metrics(problem,target_variable,conditioning_variable,target_value = 1,conditioning_value = 0):

    if conditioning_variable != "A":
        variable_list = [target_variable,conditioning_variable,"A"]
        variable_values = [target_value,conditioning_value]
        term = get_autobounds_query(problem,variable_list,variable_values+[1]) * get_autobounds_query(problem,variable_list[1:],variable_values[1:]+[0])
        term = term + get_autobounds_query(problem,variable_list,variable_values+[0],sign=-1) * get_autobounds_query(problem,variable_list[1:],variable_values[1:]+[1])
    else:
        variable_list = [target_variable,conditioning_variable]
        variable_values = [target_value]
        term = get_autobounds_query(problem,variable_list,variable_values+[1]) * get_autobounds_query(problem,["A"],[0])  
        term = term + get_autobounds_query(problem,variable_list,variable_values+[0],sign=-1)* get_autobounds_query(problem,["A"],[1])  
    return term

def get_denominator_for_difference_metrics(problem,conditioning_variable,conditioning_value = 0):
    
    if conditioning_variable != "A":
        variable_list = [conditioning_variable,"A"]
        variable_values = [conditioning_value]
        term = get_autobounds_query(problem,variable_list,variable_values+[0]) * get_autobounds_query(problem,variable_list,variable_values+[1])
    else:
        term = get_autobounds_query(problem,["A"],[0]) * get_autobounds_query(problem,["A"],[1])
    return term

def get_metric_expressions(problem,metric="FPR",ECP=False,**kwargs):
    if metric == "FPR":
        target_variable = "P"
        target_value = 1
        conditioning_variable = "Y"
        if ECP: 
             conditioning_variable = "Y(D=1)"
        conditioning_value = 0

        numerator = get_numerator_for_difference_metrics(problem,target_variable,conditioning_variable,target_value,conditioning_value)
        denominator = get_denominator_for_difference_metrics(problem,conditioning_variable,conditioning_value)
    
    elif metric == "FNR":
        target_variable = "P"
        target_value = 0
        conditioning_variable = "Y"
        if ECP: 
             conditioning_variable = "Y(D=1)"
        conditioning_value = 1

        numerator = get_numerator_for_difference_metrics(problem,target_variable,conditioning_variable,target_value,conditioning_value)
        denominator = get_denominator_for_difference_metrics(problem,conditioning_variable,conditioning_value)

    elif metric == "PPP":
        target_variable = "Y"
        target_value = 1
        conditioning_variable = "P"
        conditioning_value = 1

        if ECP: 
            target_variable = "Y(D=1)"
        
        numerator = get_numerator_for_difference_metrics(problem,target_variable,conditioning_variable,target_value,conditioning_value)
        denominator = get_denominator_for_difference_metrics(problem,conditioning_variable,conditioning_value)
    
    elif metric == "NPP":
        target_variable = "Y"
        target_value = 1
        conditioning_variable = "P"
        conditioning_value = 0

        if ECP: 
            target_variable = "Y(D=1)"

        numerator = get_numerator_for_difference_metrics(problem,target_variable,conditioning_variable,target_value,conditioning_value)
        denominator = get_denominator_for_difference_metrics(problem,conditioning_variable,conditioning_value)

    elif metric == "DP":
        target_variable = "P"
        target_value = 1
        conditioning_variable = "A"
        conditioning_value = 0


        numerator = get_numerator_for_difference_metrics(problem,target_variable,conditioning_variable,target_value,conditioning_value)
        denominator = get_denominator_for_difference_metrics(problem,conditioning_variable,conditioning_value)
    
    elif metric == "FPR_Fogliato":

        if "group" in kwargs.keys():
            group = kwargs["group"]
        else:
            group = 0

        variable_list = ["Y","P","A"]
        numerator = get_autobounds_query(problem,variable_list,[0,1,group])
        denominator = get_autobounds_query(problem,variable_list,[0,1,group])+get_autobounds_query(problem,variable_list,[0,0,group])
    
    elif metric == "FNR_Fogliato":
        
        if "group" in kwargs.keys():
            group = kwargs["group"]
        else:
            group = 0
        
        variable_list = ["Y","P","A"]
        numerator = get_autobounds_query(problem,variable_list,[1,0,group])
        denominator = get_autobounds_query(problem,variable_list,[1,0,group])+get_autobounds_query(problem,variable_list,[1,1,group])
    
    elif metric == "PPV_Fogliato":

        if "group" in kwargs.keys():
            group = kwargs["group"]
        else:
            group = 0
    
        variable_list = ["Y","P","A"]
        numerator = get_autobounds_query(problem,variable_list,[1,1,group])
        denominator = get_autobounds_query(problem,variable_list,[0,1,group])+get_autobounds_query(problem,variable_list,[1,1,group])
    
    elif metric == "TE":
        return (problem.query('P(A=1)=1') + problem.query('P(A=0)=1', -1),Query(1))
    
    elif metric  == "CF":
        return (problem.query('P(A=1)=1&P(A=0)=1') + problem.query('P(A=1)=0&P(A=0)=0'),Query(1))
    
    elif metric == "SE":
        return (problem.query('P(A=1)=1')*problem.query('A=1') + problem.query('P=1&A=0',-1),problem.query)
    
    return numerator,denominator


DAG_string_dict = {
    "Proxy_Y": {"DAG":"A -> Y, A->P, A -> Z, U->P, U -> Y, U -> Z, Y->Z","Unobserved":"U"},
    "Selection": {"DAG":"A -> Y, A->P, A -> S, U->P, U -> Y, U -> S, Y->S","Unobserved":"U"},
    "ECP" :  {"DAG":"A -> Y, A->P, A -> D, U->P, U -> Y, U -> D, D->Y","Unobserved":"U"}
}

def get_sensitivity_parameter(problem,bias):

    if bias == "Proxy_Y":
        return problem.query("Y=1&Z=0")+problem.query("Y=0&Z=1")

    if bias == "ECP":
        return problem.query("Y(T=1)=1&Y(T=0)=0")+problem.query("Y(T=1)=0&Y(T=0)=1")
    
    if bias == "Selection":
        return problem.query("S=0")
     
import re
def split_string_with_delimiters(input_string, delimiters):
    # Create a regex pattern with the delimiters
    regex_pattern = '|'.join(map(re.escape, delimiters))
    
    # Split the string using the regex pattern
    split_string_and_delims = re.split(f'({regex_pattern})', input_string)

    split_string, used_delimiters = [], []
    for s in split_string_and_delims:
        if s in delimiters:
            used_delimiters.append(s)
        else:
            split_string.append(s)
    
    return split_string, used_delimiters

def parse_constraint(problem, constraint, D):
    """
    Parse and add a constraint to the problem.
    
    Args:
    - problem: The problem object where constraints are to be added.
    - constraint: A string representing the constraint, e.g., "P(A=1 & C=1) + P(A=0 & C=0) >= 1 - D".
    - D: The variable D used in the constraint.
    """
    # Define the valid operators
    valid_operators = ['+', '-', '*', '/']
    inequality_symbols = ['>=', '<=', '==']
    
    # Identify the inequality symbol in the constraint
    inequality_symbol = None
    for symbol in inequality_symbols:
        if symbol in constraint:
            inequality_symbol = symbol
            break
    
    if inequality_symbol is None:
        raise ValueError("Invalid constraint: must contain >=, <=, or ==")
    
    def token_to_query(token):
        if token.startswith('P('):
            return problem.query(token[2:-1])  # Strip P( and )
        elif token == 'D':
            print("D",D)
            return Query(D)       
        elif re.match(r'^([0-9]+(\.[0-9]+)?)$', token):
            return Query(float(token))
        else:
            raise ValueError(f"Invalid token: {token}")
        
    def combine_queries(lhs, rhs, operator):
        if operator == '+':
            return lhs + rhs
        elif operator == '-':
            return lhs - rhs
        elif operator == '*':
            return lhs * rhs
        elif operator == '/':
            return lhs / rhs
        else:
            raise ValueError(f"Invalid operator: {operator}")
    
    lhs, rhs = constraint.split(inequality_symbol)
    lhs = "".join(lhs.split())
    rhs = "".join(rhs.split())
    
    lhs_tokens, lhs_operators = split_string_with_delimiters(lhs, valid_operators)
    rhs_tokens, rhs_operators = split_string_with_delimiters(rhs, valid_operators)

    print("lhs_tokens",lhs_tokens)
    print("lhs_operators",lhs_operators)
    print("rhs_tokens",rhs_tokens)
    print("rhs_operators",rhs_operators)
    constraint = token_to_query(lhs_tokens[0])
    for operator, token in zip(lhs_operators, lhs_tokens[1:]):
        query = token_to_query(token)
        constraint = combine_queries(constraint, query, operator)

    rhs_operators = ['-'] + ['-' if op == '+' else '+' if op == '-' else op for op in rhs_operators]
    for operator, token in zip(rhs_operators, rhs_tokens):
        query = token_to_query(token)
        constraint = combine_queries(constraint, query, operator)

    problem.add_constraint(constraint, inequality_symbol)

                     
def run_fair_bounding(
        probability_df, metric, dag_str, unob, constraints, 
        sensitivity_parameter_value=0.05, verbose=0, 
        **kwargs
): 
    dag = DAG()
    dag.from_structure(dag_str, unob = unob)
    problem = causalProblem(dag)

    if verbose == 1:
        print("Loading_data")
    problem.load_data(probability_df)
    problem.add_prob_constraints()

    if verbose == 1:
        print("Collecting term")
    numerator, denominator = get_metric_expressions(
        problem, metric=metric, ECP=False, kwargs=kwargs
    )

    if verbose == 1:
        print("Setting Estimand")
    problem.set_estimand(numerator, div=denominator)
    
    for constraint in constraints:
        parse_constraint(problem, constraint, sensitivity_parameter_value)
    
    program = problem.write_program()
    
    if verbose == 1:
        print("Running Autobounds")
    result = program.run_pyomo('ipopt',verbose=False)

    return result



