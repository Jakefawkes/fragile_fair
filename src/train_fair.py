from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import GridSearch,DemographicParity,FalsePositiveRateParity,ErrorRate,TruePositiveRateParity,EqualizedOdds
from fairlearn.metrics import false_negative_rate,false_positive_rate,demographic_parity_difference

import pandas as pd


def calc_disparity_metric(Y,Y_hat,A,fairness_constraint="FPR"):
    
    if fairness_constraint=="FPR":
        val = false_positive_rate(Y[A==1],Y_hat[A==1]) - false_positive_rate(Y[A==0],Y_hat[A==0])
    
    if fairness_constraint=="FNR":
        val = false_negative_rate(Y[A==1],Y_hat[A==1]) - false_negative_rate(Y[A==0],Y_hat[A==0])
    
    if fairness_constraint=="PPP":
        val = -(false_negative_rate(Y_hat[A==1],Y[A==1]) - false_negative_rate(Y_hat[A==0],Y[A==0]))
    
    if fairness_constraint=="NPP":
        val = false_positive_rate(Y_hat[A==1],Y[A==1]) -false_positive_rate(Y_hat[A==0],Y[A==0])
    
    if fairness_constraint=="DP":
        val = sum(Y_hat[A==1])/len(Y_hat[A==1])-sum(Y_hat[A==0])/len(Y_hat[A==0])
    
    if fairness_constraint=="EO_max":
        val = max(abs(false_positive_rate(Y[A==1],Y_hat[A==1]) - false_positive_rate(Y[A==0],Y_hat[A==0]))
                  ,(abs(false_negative_rate(Y[A==1],Y_hat[A==1]) - false_negative_rate(Y[A==0],Y_hat[A==0]))))
    
    return val


class FairLogisticRegression():
    """
    This is the class for training a logistic regression to satisfy various ML fairness constraints. 
    We make use of fairlearn for training such predictors. 

    """
    def __init__(self,fairness_constraint="FPR",grid_size=256):

        self.fairness_constraint = fairness_constraint
        self.grid_size = grid_size

        if fairness_constraint=="FPR":
            self.fairlearn_metric = FalsePositiveRateParity
            self.fairlearn_constraint = True

        if fairness_constraint=="FNR":
            self.fairlearn_metric = TruePositiveRateParity
            self.fairlearn_constraint = True

        if fairness_constraint=="DP":
            self.fairlearn_metric = DemographicParity
            self.fairlearn_constraint = True
        
        if fairness_constraint == "PPP":
            self.fairlearn_constraint = False
        
        if fairness_constraint == "NPP":
            self.fairlearn_constraint = False
        
        if fairness_constraint[:2] == "EO":
            self.fairlearn_metric = EqualizedOdds
            self.fairlearn_constraint = True
        
        self.set_sweep()


    def set_sweep(self):
            
            if self.fairlearn_constraint:
                self.sweep = GridSearch(LogisticRegression(),
                                constraints = self.fairlearn_metric(),
                                grid_size = self.grid_size)
            else:
                self.sweep = GridSearch(LogisticRegression(),
                                constraints = DemographicParity(),constraint_weight=0,
                                grid_size = self.grid_size)    
    
    def fit(self,X_train, y_train,A_train):
            
            self.sweep.fit(X_train, y_train,sensitive_features=A_train)
            
            predictors = self.sweep.predictors_
            errors, disparities = [], []

            for m in predictors:
                def classifier(X): return m.predict(X)

                error = ErrorRate()
                error.load_data(X_train, pd.Series(y_train), sensitive_features=A_train)
                disparity = calc_disparity_metric(y_train,classifier(X_train),y_train,fairness_constraint=self.fairness_constraint)
                disparities.append(disparity)
                errors.append(error.gamma(classifier).iloc[0])
                

            all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

            non_dominated = []
            for row in all_results.itertuples():
                errors_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
                if row.error <= errors_for_lower_or_eq_disparity.min():
                    non_dominated.append(row.predictor)

            disparity_level = [abs(calc_disparity_metric(y_train,classifer.predict(X=X_train),y_train,fairness_constraint=self.fairness_constraint)) for classifer in non_dominated]
            non_dominated = [x for _,_, x in sorted(zip(disparity_level,list(range(len(disparity_level))) ,non_dominated))]

            self.model = non_dominated[0]

            return None

    def predict(self,X):

        if not hasattr(self,"model"):
            raise AttributeError("Model must be fit first.")
        
        return self.model.predict(X)