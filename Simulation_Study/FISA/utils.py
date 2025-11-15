import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycox.evaluation import EvalSurv
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.metrics.pairwise import cosine_distances
from model import *
from fairness_measure import *
from utils import *
from data_preprocess import *


# ==============================================================================
#                             Fairness penalty Constraint
# ==============================================================================    

class criterionHinge(nn.Module):
    def __init__(self):
        super(criterionHinge, self).__init__()

    def forward(self, target_fairness,prediction,X_distance, ntimes, scale):
        """
        Compute the fairness penalty Constraint measure 
        Arguments:
        target_fairness -- Maximum fairness (i.e., 0)
        prediction -- Predicted Survival Probability
        X_distance--Covariates
        ntimes--Number of pre-specified time points
        scale--scale parameter
        Returns:
        Fairness penalty Constraint measure 
        """ 
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        zeroTerm = torch.tensor(0.0).to(device)
        N = len(prediction)
        R_beta = torch.tensor(0.0).to(device)  #initialization of individual fairness  
        zeroTerm = torch.tensor(0.0).to(device)
        
        # CRITICAL FIX: Convert X_distance to numpy ONCE outside the loops
        # This avoids repeated CPU-GPU transfers in the triple loop (major performance bottleneck)
        if isinstance(X_distance, torch.Tensor):
            X_distance_np = X_distance.cpu().detach().numpy()
        else:
            X_distance_np = X_distance
            
        for i in range(ntimes):
            for j in range(prediction.shape[0]):
                for k in range(prediction.shape[0]):
                    if k<=j:
                        continue
                    else:                
                        distance=cosine_distances(X_distance_np[j].reshape(1,-1),X_distance_np[k].reshape(1,-1))[0][0]
                        R_beta = R_beta + torch.max(zeroTerm,(torch.abs(prediction[j,i]-prediction[k,i])-(scale*distance))) 
        model_fairness = (torch.tensor(2.0).to(device))*R_beta/(N*(N-1)*ntimes)
        
        return torch.max(zeroTerm, (model_fairness-target_fairness))    
    

# ==================================================================================
#                             Compute accuracy and fairnes measure using FIDP model 
# ==================================================================================  
    
    
def FIDP_Evaluation(model, data_X_test, X_test_uncen,  X_test_cen, data_time_train, train_time_uncen, train_time_cen, data_time_test, test_time_cen, test_time_uncen, data_event_train, data_event_test, test_event_uncen, test_event_cen, S_protected_attribute_1, S_protected_attribute_2, eval_time, scale_fairness, dataset_name):
    
    """
    Compute accuracy and fairnes measures using FIDP model 
    Arguments:
    model -- FIDP model
    data_X_test--Covariates in test data
    X_test_uncen--Covariates in uncensored test data
    X_test_cen--Covariates in censored test data
    data_time_train--Observed time in train data
    train_time_uncen--Observed time in uncensored train data
    train_time_cen--Observed time in censored train data
    data_time_test---Observed time in test data
    test_time_cen--Observed time in censored test data
    test_time_uncen--Observed time in uncensored test data
    data_event_train--Event status in train data
    data_event_test--Event status in test data
    test_event_uncen--Event status in uncensored test data
    test_event_cen--Event status in censored test data
    S_protected_attribute_1--Protected attribute 1 (e.g. age)
    S_protected_attribute_2--Protected attribute 2 (e.g. gender/race)
    eval_time--Pre-specified evaluation time points
    scale_fairness--scale parameter
    dataset_name--Name of the dataset (e.g., SEER/SUPPORT/FLChain)
    Returns:
    Accuracy and fairnes measures
    """     
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use forward_prob to get probabilities in [0,1] range for evaluation
    if hasattr(model, 'forward_prob'):
        sp_test = model.forward_prob(torch.tensor(np.array(data_X_test, dtype='float32')).to(device)).cpu().detach().numpy()
    else:
        # Fallback: apply sigmoid if method doesn't exist
        sp_test = torch.sigmoid(model(torch.tensor(np.array(data_X_test, dtype='float32')).to(device))).cpu().detach().numpy()
    cif_test = 1-sp_test   
            
    if hasattr(model, 'forward_prob'):
        sp_test_uncen = model.forward_prob(torch.tensor(np.array(X_test_uncen, dtype='float32')).to(device)).cpu().detach().numpy()
    else:
        sp_test_uncen = torch.sigmoid(model(torch.tensor(np.array(X_test_uncen, dtype='float32')).to(device))).cpu().detach().numpy()
    cif_test_uncen = 1-sp_test_uncen  
       
    if hasattr(model, 'forward_prob'):
        sp_test_cen = model.forward_prob(torch.tensor(np.array(X_test_cen, dtype='float32')).to(device)).cpu().detach().numpy()
    else:
        sp_test_cen = torch.sigmoid(model(torch.tensor(np.array(X_test_cen, dtype='float32')).to(device))).cpu().detach().numpy()
    cif_test_cen = 1-sp_test_cen 
    
    
    data_event_train = data_event_train.astype(bool)
    data_event_test = data_event_test.astype(bool)
    data_time_train = data_time_train

# ==================================================================================
#                             Compute Accuracy measures
# ==================================================================================       
    survival_train=np.dtype([('event',data_event_train.dtype),('surv_time',data_time_train.dtype)])
    survival_train=np.empty(len(data_event_train),dtype=survival_train)
    survival_train['event']=data_event_train
    survival_train['surv_time']=data_time_train

    survival_test=np.dtype([('event',data_event_test.dtype),('surv_time',data_time_test.dtype)])
    survival_test=np.empty(len(data_event_test),dtype=survival_test)
    survival_test['event']=data_event_test
    survival_test['surv_time']=data_time_test


    auc,mean_auc=cumulative_dynamic_auc(survival_train, survival_test, cif_test, eval_time) ## Time-dependent Area under the ROC
    surv=pd.DataFrame(np.transpose(sp_test))  
    surv=surv.set_index([eval_time])         
    ev = EvalSurv(surv, np.array(data_time_test), np.array(data_event_test), censor_surv='km')
    cindex=ev.concordance_td() ## time-dependent cindex
    brier=ev.integrated_brier_score(eval_time) ## integrated brier score

# ==================================================================================
#                             Compute fairnes measures
# ================================================================================== 

    ## individual fairness measures
    F_ind = individual_fairness(sp_test,data_X_test,len(eval_time), scale_fairness)
    ## Censoring individual fairness measures    
    F_cen_ind= censoring_individual_fairness(sp_test_uncen, sp_test_cen, X_test_uncen, X_test_cen, test_time_uncen, test_time_cen, len(eval_time),scale_fairness)
    
    ## Cesnoring Group fairness measures  
    if X_test_cen.shape[0]==0:
        F_cen_group=0.0
    else:
        F_cen_group=censoring_group_fairness(sp_test_uncen, sp_test_cen, X_test_uncen, X_test_cen, test_time_uncen, test_time_cen, len(eval_time),scale_fairness, dataset_name) 
        
    ## Group fairness measures     
    if len(np.unique(S_protected_attribute_1))==1:
        F_group_protected_attribute_1 = 0.0
    else:    
        #%% group fairness measures - protected attribute 1
        F_group_protected_attribute_1 = group_fairness(sp_test, S_protected_attribute_1, len(eval_time))

    if len(np.unique(S_protected_attribute_2))==1:
        F_group_protected_attribute_2 = 0.0
    else:    
        #%% group fairness measures - protected attribute 1
        F_group_protected_attribute_2 = group_fairness(sp_test, S_protected_attribute_2, len(eval_time))        

    return cindex, brier, mean_auc, F_ind, F_cen_ind, F_cen_group, F_group_protected_attribute_1, F_group_protected_attribute_2 


# ==================================================================================
#                             Compute accuracy and fairnes measure using FIPNAM model 
# ================================================================================== 

def FIPNAM_Evaluation(model, data_X_test, X_test_uncen,  X_test_cen, data_time_train, train_time_uncen, train_time_cen, data_time_test, test_time_cen,  test_time_uncen, data_event_train, data_event_test, test_event_uncen, test_event_cen, S_protected_attribute_1, S_protected_attribute_2, eval_time, scale_fairness, dataset_name):

    """
    Compute accuracy and fairnes measures using FIPNAM model 
    Arguments:
    model -- FIPNAM model
    data_X_test--Covariates in test data
    X_test_uncen--Covariates in uncensored test data
    X_test_cen--Covariates in censored test data
    data_time_train--Observed time in train data
    train_time_uncen--Observed time in uncensored train data
    train_time_cen--Observed time in censored train data
    data_time_test---Observed time in test data
    test_time_cen--Observed time in censored test data
    test_time_uncen--Observed time in uncensored test data
    data_event_train--Event status in train data
    data_event_test--Event status in test data
    test_event_uncen--Event status in uncensored test data
    test_event_cen--Event status in censored test data
    S_protected_attribute_1--Protected attribute 1 (e.g. age)
    S_protected_attribute_2--Protected attribute 2 (e.g. gender/race)
    eval_time--Pre-specified evaluation time points
    scale_fairness--scale parameter
    dataset_name--Name of the dataset (e.g., SEER/SUPPORT/FLChain)
    Returns:
    Accuracy and fairnes measures
    """        
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp_test,_ = model(torch.tensor(np.array(data_X_test, dtype='float32')).to(device)) ## Predict survival probabilities for test data
    sp_test=sp_test.cpu().detach().numpy()
    cif_test = 1-sp_test   

    sp_test_uncen,_ = model(torch.tensor(np.array(X_test_uncen, dtype='float32')).to(device)) ## Predict survival probabilities for uncensored test data
    sp_test_uncen=sp_test_uncen.cpu().detach().numpy()
    cif_test_uncen = 1-sp_test_uncen    
    
    sp_test_cen,_ = model(torch.tensor(np.array(X_test_cen, dtype='float32')).to(device)) ## Predict survival probabilities for censored test data
    sp_test_cen=sp_test_cen.cpu().detach().numpy()
    cif_test_cen = 1-sp_test_cen     
    
    data_event_train = data_event_train.astype(bool)
    data_event_test = data_event_test.astype(bool)
    data_time_train = data_time_train

# ==================================================================================
#                             Compute Accuracy measures
# ==================================================================================     
    survival_train=np.dtype([('event',data_event_train.dtype),('surv_time',data_time_train.dtype)])
    survival_train=np.empty(len(data_event_train),dtype=survival_train)
    survival_train['event']=data_event_train
    survival_train['surv_time']=data_time_train

    survival_test=np.dtype([('event',data_event_test.dtype),('surv_time',data_time_test.dtype)])
    survival_test=np.empty(len(data_event_test),dtype=survival_test)
    survival_test['event']=data_event_test
    survival_test['surv_time']=data_time_test

    auc,mean_auc=cumulative_dynamic_auc(survival_train, survival_test, cif_test, eval_time) ### Time-dependent Area under the ROC
    surv=pd.DataFrame(np.transpose(sp_test))  
    surv=surv.set_index([eval_time])         
    ev = EvalSurv(surv, np.array(data_time_test), np.array(data_event_test), censor_surv='km')
    cindex=ev.concordance_td() ## time-dependent cindex
    brier=ev.integrated_brier_score(eval_time) ## integrated brier score

# ==================================================================================
#                             Compute fairnes measures
# ================================================================================== 

    ## individual fairness measures
    F_ind = individual_fairness(sp_test,data_X_test,len(eval_time), scale_fairness)
    ## Censoring individual fairness measures    
    F_cen_ind= censoring_individual_fairness(sp_test_uncen, sp_test_cen, X_test_uncen, X_test_cen, test_time_uncen, test_time_cen, len(eval_time),scale_fairness)
    
    ## Cesnoring Group fairness measures  
    if X_test_cen.shape[0]==0:
        F_cen_group=0.0
    else:
        F_cen_group=censoring_group_fairness(sp_test_uncen, sp_test_cen, X_test_uncen, X_test_cen, test_time_uncen, test_time_cen, len(eval_time),scale_fairness, dataset_name)   
    
    ## Group fairness measures 
    if len(np.unique(S_protected_attribute_1))==1:
        F_group_protected_attribute_1 = 0.0
    else:    
        #%% group fairness measures - protected attribute 1
        F_group_protected_attribute_1 = group_fairness(sp_test, S_protected_attribute_1, len(eval_time))

    if len(np.unique(S_protected_attribute_2))==1:
        F_group_protected_attribute_2 = 0.0
    else:    
        #%% group fairness measures - protected attribute 1
        F_group_protected_attribute_2 = group_fairness(sp_test, S_protected_attribute_2, len(eval_time))        
        

    return cindex, brier, mean_auc, F_ind, F_cen_ind, F_cen_group, F_group_protected_attribute_1, F_group_protected_attribute_2 


# ==================================================================================
#                             Compute accuracy and fairnes measure using Cox model 
# ================================================================================== 

def Cox_Evaluation(model, data_X_test, X_test_uncen,  X_test_cen, data_time_train, train_time_uncen, train_time_cen, data_time_test, test_time_cen, test_time_uncen, data_event_train, data_event_test, test_event_uncen, test_event_cen, S_protected_attribute_1, S_protected_attribute_2, eval_time, scale_fairness, dataset_name):
    
    """
    Compute accuracy and fairnes measures using Cox model 
    Arguments:
    model -- Cox model
    data_X_test--Covariates in test data
    X_test_uncen--Covariates in uncensored test data
    X_test_cen--Covariates in censored test data
    data_time_train--Observed time in train data
    train_time_uncen--Observed time in uncensored train data
    train_time_cen--Observed time in censored train data
    data_time_test---Observed time in test data
    test_time_cen--Observed time in censored test data
    test_time_uncen--Observed time in uncensored test data
    data_event_train--Event status in train data
    data_event_test--Event status in test data
    test_event_uncen--Event status in uncensored test data
    test_event_cen--Event status in censored test data
    S_protected_attribute_1--Protected attribute 1 (e.g. age)
    S_protected_attribute_2--Protected attribute 2 (e.g. gender/race)
    eval_time--Pre-specified evaluation time points
    scale_fairness--scale parameter
    dataset_name--Name of the dataset (e.g., SEER/SUPPORT/FLChain)
    Returns:
    Accuracy and fairnes measures
    """     
    
    # Convert inputs to numpy arrays if they are DataFrames
    if isinstance(data_X_test, pd.DataFrame):
        data_X_test = data_X_test.values
    if isinstance(X_test_uncen, pd.DataFrame):
        X_test_uncen = X_test_uncen.values
    if isinstance(X_test_cen, pd.DataFrame):
        X_test_cen = X_test_cen.values
    
    # Predict survival functions for test data
    surv_funcs_test = model.predict_survival_function(data_X_test, return_array=False)
    
    # Extract survival probabilities at eval_time points
    sp_test = np.zeros((len(data_X_test), len(eval_time)))
    for i, surv_func in enumerate(surv_funcs_test):
        for j, t in enumerate(eval_time):
            # Get survival probability at time t
            # surv_func is a function that takes time and returns survival probability
            sp_test[i, j] = surv_func(t)
    
    cif_test = 1 - sp_test
    
    # Predict for uncensored test data
    if len(X_test_uncen) > 0:
        surv_funcs_test_uncen = model.predict_survival_function(X_test_uncen, return_array=False)
        sp_test_uncen = np.zeros((len(X_test_uncen), len(eval_time)))
        for i, surv_func in enumerate(surv_funcs_test_uncen):
            for j, t in enumerate(eval_time):
                sp_test_uncen[i, j] = surv_func(t)
        cif_test_uncen = 1 - sp_test_uncen
    else:
        sp_test_uncen = np.zeros((0, len(eval_time)))
        cif_test_uncen = np.zeros((0, len(eval_time)))
    
    # Predict for censored test data
    if len(X_test_cen) > 0:
        surv_funcs_test_cen = model.predict_survival_function(X_test_cen, return_array=False)
        sp_test_cen = np.zeros((len(X_test_cen), len(eval_time)))
        for i, surv_func in enumerate(surv_funcs_test_cen):
            for j, t in enumerate(eval_time):
                sp_test_cen[i, j] = surv_func(t)
        cif_test_cen = 1 - sp_test_cen
    else:
        sp_test_cen = np.zeros((0, len(eval_time)))
        cif_test_cen = np.zeros((0, len(eval_time)))
    
    data_event_train = data_event_train.astype(bool)
    data_event_test = data_event_test.astype(bool)
    data_time_train = data_time_train

# ==================================================================================
#                             Compute Accuracy measures
# ==================================================================================       
    survival_train=np.dtype([('event',data_event_train.dtype),('surv_time',data_time_train.dtype)])
    survival_train=np.empty(len(data_event_train),dtype=survival_train)
    survival_train['event']=data_event_train
    survival_train['surv_time']=data_time_train

    survival_test=np.dtype([('event',data_event_test.dtype),('surv_time',data_time_test.dtype)])
    survival_test=np.empty(len(data_event_test),dtype=survival_test)
    survival_test['event']=data_event_test
    survival_test['surv_time']=data_time_test

    auc,mean_auc=cumulative_dynamic_auc(survival_train, survival_test, cif_test, eval_time) ## Time-dependent Area under the ROC
    surv=pd.DataFrame(np.transpose(sp_test))  
    surv=surv.set_index([eval_time])         
    ev = EvalSurv(surv, np.array(data_time_test), np.array(data_event_test), censor_surv='km')
    cindex=ev.concordance_td() ## time-dependent cindex
    brier=ev.integrated_brier_score(eval_time) ## integrated brier score

# ==================================================================================
#                             Compute fairnes measures
# ================================================================================== 

    ## individual fairness measures
    F_ind = individual_fairness(sp_test,data_X_test,len(eval_time), scale_fairness)
    ## Censoring individual fairness measures    
    if len(X_test_uncen) > 0 and len(X_test_cen) > 0:
        F_cen_ind= censoring_individual_fairness(sp_test_uncen, sp_test_cen, X_test_uncen, X_test_cen, test_time_uncen, test_time_cen, len(eval_time),scale_fairness)
    else:
        F_cen_ind = 0.0
    
    ## Cesnoring Group fairness measures  
    if X_test_cen.shape[0]==0:
        F_cen_group=0.0
    else:
        F_cen_group=censoring_group_fairness(sp_test_uncen, sp_test_cen, X_test_uncen, X_test_cen, test_time_uncen, test_time_cen, len(eval_time),scale_fairness, dataset_name) 
        
    ## Group fairness measures     
    if len(np.unique(S_protected_attribute_1))==1:
        F_group_protected_attribute_1 = 0.0
    else:    
        #%% group fairness measures - protected attribute 1
        F_group_protected_attribute_1 = group_fairness(sp_test, S_protected_attribute_1, len(eval_time))

    if len(np.unique(S_protected_attribute_2))==1:
        F_group_protected_attribute_2 = 0.0
    else:    
        #%% group fairness measures - protected attribute 2
        F_group_protected_attribute_2 = group_fairness(sp_test, S_protected_attribute_2, len(eval_time))        

    return cindex, brier, mean_auc, F_ind, F_cen_ind, F_cen_group, F_group_protected_attribute_1, F_group_protected_attribute_2 


# ==================================================================================
#                             Concordance Index metric 
# ================================================================================== 

def DP_Concordance(model,x,durations,events, evaltime):
    """
    Compute the time-dependent concordance index
    Arguments:
    model -- FIDP model
    x -- Covariates
    durations--Observed times
    events--Event status
    evaltime--Pre-specified evaluation time points
    Returns:
    Time-dependent concordance index
    """     
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    x=x.to(device)
    # Use forward_prob to get probabilities in [0,1] range for evaluation
    if hasattr(model, 'forward_prob'):
        surv = model.forward_prob(x)
    else:
        # Fallback: apply sigmoid if method doesn't exist
        surv = torch.sigmoid(model(x))
    y_pred = pd.DataFrame(np.transpose(surv.cpu().detach().numpy()))
    y_pred = y_pred.set_index([evaltime])
    ev     = EvalSurv(y_pred, durations, events)
    cindex = ev.concordance_td()
    return cindex

def PNAM_Concordance(model,x,durations,events, evaltime):
    """
    Compute the time-dependent concordance index
    Arguments:
    model -- FIPNAM model
    x -- Covariates
    durations--Observed times
    events--Event status
    evaltime--Pre-specified evaluation time points
    Returns:
    Time-dependent concordance index
    """     
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    x=x.to(device)
    surv,_ = model(x)
    y_pred = pd.DataFrame(np.transpose(surv.cpu().detach().numpy()))
    y_pred = y_pred.set_index([evaltime])
    ev     = EvalSurv(y_pred, durations, events)
    cindex = ev.concordance_td()
    return cindex

def Cox_Concordance(model, x, durations, events, evaltime):
    """
    Compute the time-dependent concordance index
    Arguments:
    model -- Cox model
    x -- Covariates (numpy array)
    durations--Observed times
    events--Event status
    evaltime--Pre-specified evaluation time points
    Returns:
    Time-dependent concordance index
    """     
    # Convert x to numpy if it's a tensor
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    # Predict survival functions
    surv_funcs = model.predict_survival_function(x, return_array=False)
    
    # Extract survival probabilities at evaltime points
    surv = np.zeros((len(x), len(evaltime)))
    for i, surv_func in enumerate(surv_funcs):
        for j, t in enumerate(evaltime):
            surv[i, j] = surv_func(t)
    
    y_pred = pd.DataFrame(np.transpose(surv))
    y_pred = y_pred.set_index([evaltime])
    ev = EvalSurv(y_pred, durations, events)
    cindex = ev.concordance_td()
    return cindex

# ==================================================================================
#                             Pseudo value based loss function
# ================================================================================== 


def pseudo_loss(output, target):
    """
    Compute loss using Pseudo value based loss function
    Arguments:
    output -- Predicted survival probability
    target -- Jackknife pseudo values
    Returns:
    Loss
    """     
    loss   = torch.mean((target*(1-2*output)+(output**2)))
    return loss



# ==================================================================================
#                             Faster Tensor Data Loader
# ================================================================================== 


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
