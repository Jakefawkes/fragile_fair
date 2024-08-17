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

    return non_dominated
