from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import GridSearch,DemographicParity,FalsePositiveRateParity,ErrorRate,TruePositiveRateParity,EqualizedOdds
from fairlearn.metrics import false_negative_rate,false_positive_rate,demographic_parity_difference
import pandas as pd
import itertools
import numpy as np
import glob 
import yaml

bias_list = ["proxy_Y","proxy_A","selection","ECD"]
disparity_metrics_list = ["FPR","FNR","PPP","NPP","DP"]

disparity_metric_fns = {
    "FPR": lambda Y, Y_hat, A: false_positive_rate(Y[A==1], Y_hat[A==1]) - false_positive_rate(Y[A==0], Y_hat[A==0]),
    "FNR": lambda Y, Y_hat, A: false_negative_rate(Y[A==1], Y_hat[A==1]) - false_negative_rate(Y[A==0], Y_hat[A==0]),
    "PPP": lambda Y, Y_hat, A: false_negative_rate(Y_hat[A==0], Y[A==0]) - false_negative_rate(Y_hat[A==1], Y[A==1]),
    "NPP": lambda Y, Y_hat, A: false_positive_rate(Y_hat[A==1], Y[A==1]) - false_positive_rate(Y_hat[A==0], Y[A==0]),
    "DP": lambda Y, Y_hat, A: sum(Y_hat[A==1]) / len(Y_hat[A==1]) - sum(Y_hat[A==0]) / len(Y_hat[A==0]),
    "EO_max": lambda Y, Y_hat, A: max(
        abs(disparity_metric_fns["FPR"](Y, Y_hat, A)), abs(disparity_metric_fns["FNR"](Y, Y_hat, A))
    )
}

def calc_disparity_metric(Y, Y_hat, A, disparity_metric="FPR"):
    return disparity_metric_fns[disparity_metric](Y, Y_hat, A)

def train_fairness_classifiers(X_train,y_train,A_train,disparity_metric="FPR",grid_size=80,classifier=LogisticRegression):

    if disparity_metric=="FPR":
        fairlearn_metric = FalsePositiveRateParity
        fairlearn_constraint = True

    if disparity_metric=="FNR":
        fairlearn_metric = TruePositiveRateParity
        fairlearn_constraint = True

    if disparity_metric=="DP":
        fairlearn_metric = DemographicParity
        fairlearn_constraint = True
    
    if disparity_metric == "PPP":
        fairlearn_constraint = False
    
    if disparity_metric == "NPP":
        fairlearn_constraint = False
    
    if disparity_metric[:2] == "EO":
        fairlearn_metric = EqualizedOdds
        fairlearn_constraint = True

    if fairlearn_constraint:
        sweep = GridSearch(classifier(),
                        constraints=fairlearn_metric(),
                        grid_size=grid_size)
    else:
        sweep = GridSearch(classifier(),
                        constraints=DemographicParity(),constraint_weight=0,
                        grid_size=grid_size)

    sweep.fit(X_train, y_train,
            sensitive_features=A_train)

    predictors = sweep.predictors_
    errors, disparities = [], []

    for m in predictors:
        def classifier(X): return m.predict(X)

        error = ErrorRate()
        error.load_data(X_train, pd.Series(y_train), sensitive_features=A_train)
        disparity = calc_disparity_metric(y_train,classifier(X_train),A_train,disparity_metric=disparity_metric)
        disparities.append(disparity)
        errors.append(error.gamma(classifier)[0])
        

    all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

    non_dominated = []
    for row in all_results.itertuples():
        errors_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
        if row.error <= errors_for_lower_or_eq_disparity.min():
            non_dominated.append(row.predictor)

    disparity_level = [abs(calc_disparity_metric(y_train,classifer.predict(X=X_train),A_train,disparity_metric=disparity_metric)) for classifer in non_dominated]
    non_dominated = [x for _,_, x in sorted(zip(disparity_level,list(range(len(disparity_level))) ,non_dominated))]

    # predictor_list = [classifier for classifier in predictor_list if sum(classifier.predict(X=X_train)==0) != len(y_train)]
    # predictor_list = [classifier for classifier in predictor_list if sum(classifier.predict(X=X_train)==1) != len(y_train)]

    # metric_val_list = [abs(calc_disparity_metric(y_train,classifer.predict(X=X_train),A_train,disparity_metric=disparity_metric)) for classifer in predictor_list]
    # final_predictor_dict = {}

    return non_dominated



def return_prob_df(predictor,X_test,y_test,A_test):
    df = {"A":A_test,"Z":y_test,"P":predictor.predict(X_test)}
    df = pd.DataFrame(df)
    data_dict = {"A":[],"Z":[],"P":[],"prob":[]}
    for i,j,k in itertools.product(range(2),range(2),range(2)):
        p = np.zeros([2,2,2])
        p[i,j,k] =   ((df["A"] == i)  & (df["Z"] == j) & (df["P"] == k)).mean()
        data_dict["A"].append(i)
        data_dict["Z"].append(j)
        data_dict["P"].append(k)
        data_dict["prob"].append((p[i,j,k].item()))
    prob_df = pd.DataFrame(data_dict)
    return prob_df

def return_prob_df_unselected(predictor,X_test,y_test,A_test,select_vec):
    df = {"A":A_test,"Z":y_test,"P":predictor.predict(X_test),"S":select_vec.astype(int)}
    df = pd.DataFrame(df)
    data_dict = {"A":[],"P":[],"S":[],"prob":[]}
    for i,j,k in itertools.product(range(2),range(2),range(2)):
        p = np.zeros([2,2,2])
        p[i,j,k] =   ((df["A"] == i) & (df["P"] == j)& (df["S"] == k)).mean()
        data_dict["A"].append(i)
        data_dict["P"].append(j)
        data_dict["S"].append(k)
        data_dict["prob"].append((p[i,j,k].item()))
    prob_df = pd.DataFrame(data_dict)
    return prob_df

def predictor_dict_from_list(name_string, predictor_list, X_test, y_test, A_test, num_predictors, disparity_metric="FNR"):
    predictor_list = [classifier for classifier in predictor_list if sum(classifier.predict(X=X_test)==0) != len(y_test)]
    predictor_list = [classifier for classifier in predictor_list if sum(classifier.predict(X=X_test)==1) != len(y_test)]
    if len(predictor_list) < num_predictors:
        num_predictors = len(predictor_list)

    metric_val_list = [abs(calc_disparity_metric(y_test,classifer.predict(X=X_test),A_test,disparity_metric=disparity_metric)) for classifer in predictor_list]
    final_predictor_dict = {}

    for i in range(num_predictors):
        pred_index = metric_val_list.index(min(metric_val_list))
        classifier = predictor_list.pop(pred_index)
        metric_val_list.pop(pred_index)
        df = {"A":A_test,"Y":y_test,"P":classifier.predict(X_test)}
        df = pd.DataFrame(df)
        data_dict = {"A":[],"Y":[],"P":[],"prob":[]}
        for i,j,k in itertools.product(range(2),range(2),range(2)):
            p = np.zeros([2,2,2])
            p[i,j,k] =   ((df["A"] == i)  & (df["Y"] == j) & (df["P"] == k)).mean()
            data_dict["A"].append(i)
            data_dict["Y"].append(j)
            data_dict["P"].append(k)
            data_dict["prob"].append((p[i,j,k].item()))
            prob_df = pd.DataFrame(data_dict)
        results_dict = {}
        acc_vec = (y_test == classifier.predict(X_test))
        results_dict["Accuracy"] = (sum(acc_vec)/len(acc_vec)).item()
        for disp_metric in disparity_metrics_list:
            results_dict[disp_metric] = calc_disparity_metric(y_test,classifier.predict(X=X_test),A_test,disparity_metric=disp_metric).item()
        final_predictor_dict[name_string+"_"+str(i)] = prob_df,results_dict
    return final_predictor_dict

def load_main_results(results_path = "results"):
    
    data_dict = {"Dataset":[],"Model":[],"Disparity Metric":[],"Bias":[],"y_lower":[],"y_upper":[],"Bound Width":[],"Parameter Value":[]}
    for dataset_path in glob.glob(results_path+"/*"):
        dataset_name = dataset_path[dataset_path.find(results_path)+len(results_path)+1:]
        for model_path in glob.glob(dataset_path+"/*"):
            model_name = model_path[model_path.find(dataset_path)+len(dataset_path)+1:]
            for disparity_path in glob.glob(model_path+"/*"):
                disparity_name = disparity_path[disparity_path.find(model_path)+len(model_path)+1:]
                for specific_model_path in glob.glob(disparity_path+"/*"):
                    for bias in bias_list:
                        try:
                            bias_file = (bias+"_"+disparity_name+".metrics")
                            bias_path = os.path.join(specific_model_path,bias_file)
                            with open(bias_path, "r") as f:
                                bias_dict = yaml.safe_load(f)
                            for i,param_val in enumerate(bias_dict["param_range"]):
                                data_dict["Dataset"].append(dataset_name.capitalize)
                                data_dict["Model"].append(model_name)
                                data_dict["Disparity Metric"].append(disparity_name)
                                data_dict["Bias"].append(bias)
                                data_dict["y_lower"].append(bias_dict["y_lower"][i])
                                data_dict["y_upper"].append(bias_dict["y_upper"][i])
                                data_dict["Bound Width"].append(abs(bias_dict["y_lower"][i]-bias_dict["y_upper"][i]))
                                data_dict["Parameter Value"].append(param_val)
                            with open(os.path.join(specific_model_path,"stats.metrics"), "r") as f:
                                stats_dict = yaml.safe_load(f)
                            data_dict["Dataset"].append(dataset_name.capitalize)
                            data_dict["Model"].append(model_name)
                            data_dict["Disparity Metric"].append(disparity_name)
                            data_dict["Bias"].append(bias)
                            data_dict["y_lower"].append(stats_dict[disparity_name])
                            data_dict["y_upper"].append(stats_dict[disparity_name])
                            data_dict["Bound Width"].append(0)
                            data_dict["Parameter Value"].append(0)
                        except:
                            pass