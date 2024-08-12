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
    
def run_fair_bounding(probability_df,metric,bias,DAG_dict,sensitivity_parameter_value=0.05,**kwargs):
    
    dag = DAG()
    dag.from_structure(DAG_dict["DAG"], unob = DAG_dict["Unobserved"])
    problem = causalProblem(dag)

    print("Loading_data")

    if bias == "Proxy_Y":
        probability_df["Z"] = probability_df["Y"]
        probability_df.drop("Y",axis=1)

    if bias == "Selection":
        probability_df["S"] = 1
        problem.load_data(StringIO(probability_df.to_csv(index=False)), cond = ["S"])
    
    else:
        problem.load_data(StringIO(probability_df.to_csv(index=False)))
    problem.add_prob_constraints()

    print("Collecting term")
    numerator,denominator = get_metric_expressions(problem,metric=metric,ECP=(bias=="ECP"),kwargs=kwargs)

    print("Setting Estimand")
    problem.set_estimand(numerator,div=denominator)
    
    problem.add_constraint(get_sensitivity_parameter(problem,bias) - Query(sensitivity_parameter_value),symbol="<=")
    program = problem.write_program()
    
    print("Running Autobounds")
    result = program.run_pyomo('ipopt',verbose=False)


    return result



