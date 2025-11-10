import logging
import copy
import argparse
import torch
import os
import pandas as pd
import numpy as np
from model import *
from fairness_measure import *
from utils import *
from training import *
from data_preprocess import *


def set_random_seed(state=1):
    """Set the random seed for numpy and torch.
    Arguments:
        state: seed.
    Returns:
        fixed random seed
    """   
    np.random.seed(state)
    torch.manual_seed(state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(state)
        torch.cuda.manual_seed_all(state)  # For multi-GPU setups

def run_experiment(fn_csv, path_name, model_name, dataset_name, batch_size, lr, epochs):
    # Configure logging to output to console
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
    
    print(f"Starting experiment: {model_name} on {dataset_name}")
    print(f"Data file: {fn_csv}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    print("-" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception as e:
            print(f"Warning: Could not set CUDA device: {e}. Continuing with default device.")
        
    RANDOM_STATE = 1
    set_random_seed(RANDOM_STATE) # Set random seed
    
    print("Loading and preprocessing data...")
    ## Load the preprocessed attributes
    eval_time, test_data, data_X_train, data_X_val, data_X_test, data_X_test_uncen, data_X_test_cen, train_pseudo, val_pseudo, test_pseudo, data_time_train, data_time_train_uncen, data_time_train_cen, data_time_val, data_time_test, data_time_test_uncen,data_time_test_cen, data_event_train, data_event_val, data_event_test, data_event_test_uncen, data_event_test_cen, protected_X_test, protected_event_test, protected_time_test, protected_X_test_uncen, protected_X_test_cen, protected_time_train_uncen, protected_time_train_cen, protected_time_test_uncen,protected_time_test_cen, protected_event_test_uncen, protected_event_test_cen=data_preprocess(fn_csv,dataset_name)

    if dataset_name=='SUPPORT':
        protected_group=["race_1","race_2"] 
    elif dataset_name=='SEER':
        protected_group=["Race_ord_1","Race_ord_2","Race_ord_3","Race_ord_4"]        
    elif dataset_name=='FLChain':
        protected_group=["sex_1","sex_0"]
    elif dataset_name=='SIMULATED':
        protected_group=["A_0","A_1"]  
        
    ## Loading data using DataLoader
    train_loader = FastTensorDataLoader(torch.from_numpy(data_X_train), torch.from_numpy(train_pseudo), batch_size=batch_size, shuffle=True)
    validate_loader = FastTensorDataLoader(torch.from_numpy(data_X_val), torch.from_numpy(val_pseudo), batch_size=batch_size, shuffle=True)   

    ## Training the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_features = data_X_train.shape[1]
    out_features = len(eval_time)

    if model_name=='FIDP': 
        model = FIDP(in_features, out_features).to(device)
    elif model_name=='FIPNAM':  
        
        config = defaults()  #Default settings for PseudoNAM model
        config.regression=True

        model = FIPNAM(
              config=config,
              name="PseudoNAM",
              num_inputs=np.array(data_X_train).shape[1],
              num_units=get_num_units(config, torch.tensor(np.array(data_X_train))),
              num_output=len(eval_time)
            )
        model = model.to(device)
      
    loss_fn = pseudo_loss #Pseudo value based loss function
    learning_rate=lr

    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    Epochs = epochs
    patience=5
    best_val_loss=10000000.0
    
    scale=torch.tensor(0.01).to(device) ## Scale parameter
    lamda=torch.tensor(0.1).to(device)  ## Trade-off parameter between accuracy and fairness 

# ==============================================================================
#                             Training FIDP model
# ==============================================================================    
     
    if model_name=='FIDP':        
        train_losses=[]
        val_losses=[]
        cindex=[]
        for epoch in range(Epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")        
            train_loss = FIDP_train(train_loader, model, loss_fn, optimizer, len(eval_time),scale, lamda)
            train_losses.append(train_loss)
            logging.info(f"epoch {epoch} | train | {train_loss}")
            print('Epoch:',epoch,'train loss:', train_loss)

            val_loss = FIDP_evaluate(validate_loader, model, loss_fn, len(eval_time),scale, lamda)
            val_losses.append(val_loss)
            logging.info(f"epoch {epoch} | validate |{val_loss}")
            print('Epoch:',epoch, 'validation loss:', val_loss)

            metrics=DP_Concordance(model,torch.tensor(data_X_val),np.array(data_time_val),np.array(data_event_val),eval_time)
            cindex.append(metrics)
            print('Epoch:',epoch, 'validation C-index:', metrics)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                es = 0

                torch.save(model.state_dict(), '{}/Trained_models/model_{}_{}.pt'.format(path_name, model_name, dataset_name))
            else:
                es += 1
                print("Counter {} of {}".format(es,patience))

                if es > patience:
                    print("Early stopping with best_val_loss: ", best_val_loss)
                    break
        print("Done!")
        
        
        ## Evaluation
        model.load_state_dict(torch.load('{}/Trained_models/model_{}_{}.pt'.format(path_name, model_name, dataset_name), map_location=device))
        model.eval()       

        scale_fairness = 0.01 ## Scale parameter

        cindex_all, brier_all, mean_auc_all, F_ind_all, F_cen_ind_all, F_cen_group_all, F_group_prot_1_all, F_group_prot_2_all =  FIDP_Evaluation(model, data_X_test, data_X_test_uncen,  data_X_test_cen, data_time_train, data_time_train_uncen, data_time_train_cen, data_time_test, data_time_test_cen,  data_time_test_uncen, data_event_train, data_event_test, data_event_test_uncen, data_event_test_cen, np.array(test_data['protected_group1']).astype(int), np.array(test_data['protected_group2']).astype(int), eval_time, scale_fairness, dataset_name)  ## Compute the accuracy and fairness measures

        cindex={}
        brier={}
        mean_auc={}
        F_ind={}
        F_cen_ind={}
        F_cen_group={}
        F_g_prot_1={}
        F_g_prot_2={}
        for group in protected_group:
            cindex[group], brier[group], mean_auc[group], F_ind[group], F_cen_ind[group], F_cen_group[group], F_g_prot_1[group], F_g_prot_2[group] =  FIDP_Evaluation(model, protected_X_test[group], protected_X_test_uncen[group],  protected_X_test_cen[group], data_time_train, protected_time_train_uncen[group], protected_time_train_cen[group], protected_time_test[group], protected_time_test_cen[group],  protected_time_test_uncen[group], data_event_train, protected_event_test[group], protected_event_test_uncen[group], protected_event_test_cen[group], np.array(test_data[test_data[group]==1]['protected_group1']).astype(int), np.array(test_data[test_data[group]==1]['protected_group2']).astype(int), eval_time, scale_fairness, dataset_name) ## Compute the accuracy and fairness measures for protected groups
        

# ==============================================================================
#                             Training FIPNAM model
# ==============================================================================             
            
    elif model_name=='FIPNAM':        
        train_losses=[]
        val_losses=[]
        cindex=[]
        for epoch in range(Epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")                
            train_loss = FIPNAM_train(train_loader, model, loss_fn, optimizer, len(eval_time),scale, lamda)
            train_losses.append(train_loss)
            logging.info(f"epoch {epoch} | train | {train_loss}")
            print('Epoch:',epoch,'train loss:', train_loss)
            val_loss = FIPNAM_evaluate(validate_loader, model, loss_fn, len(eval_time),scale, lamda)
            val_losses.append(val_loss)
            logging.info(f"epoch {epoch} | validate |{val_loss}")
            print('Epoch:',epoch, 'validation loss:', val_loss)
            metrics=PNAM_Concordance(model,torch.tensor(data_X_val),np.array(data_time_val),np.array(data_event_val),eval_time)
            cindex.append(metrics)
            print('Epoch:',epoch, 'validation C-index:', metrics)
        
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                es = 0

                torch.save(model.state_dict(), '{}/Trained_models/model_{}_{}.pt'.format(path_name, model_name, dataset_name))
            else:
                es += 1
                print("Counter {} of {}".format(es,patience))

                if es > patience:
                    print("Early stopping with best_val_loss: ", best_val_loss)
                    break          

        print("Done!")
        
        
         ## Evaluation
        model.load_state_dict(torch.load('{}/Trained_models/model_{}_{}.pt'.format(path_name, model_name, dataset_name), map_location=device))
        model.eval()       

        scale_fairness = 0.01 ## Scale parameter

        cindex_all, brier_all, mean_auc_all, F_ind_all, F_cen_ind_all, F_cen_group_all, F_group_prot_1_all, F_group_prot_2_all =  FIPNAM_Evaluation(model, data_X_test, data_X_test_uncen,  data_X_test_cen, data_time_train, data_time_train_uncen, data_time_train_cen, data_time_test, data_time_test_cen,  data_time_test_uncen, data_event_train, data_event_test, data_event_test_uncen, data_event_test_cen, np.array(test_data['protected_group1']).astype(int), np.array(test_data['protected_group2']).astype(int), eval_time, scale_fairness, dataset_name) ## Compute the accuracy and fairness measures


        cindex={}
        brier={}
        mean_auc={}
        F_ind={}
        F_cen_ind={}
        F_cen_group={}
        F_g_prot_1={}
        F_g_prot_2={}
        for group in protected_group:
            cindex[group], brier[group], mean_auc[group], F_ind[group], F_cen_ind[group], F_cen_group[group], F_g_prot_1[group], F_g_prot_2[group] =  FIPNAM_Evaluation(model, protected_X_test[group], protected_X_test_uncen[group],  protected_X_test_cen[group], data_time_train, protected_time_train_uncen[group], protected_time_train_cen[group], protected_time_test[group], protected_time_test_cen[group],  protected_time_test_uncen[group], data_event_train, protected_event_test[group], protected_event_test_uncen[group], protected_event_test_cen[group], np.array(test_data[test_data[group]==1]['protected_group1']).astype(int), np.array(test_data[test_data[group]==1]['protected_group2']).astype(int), eval_time, scale_fairness, dataset_name) ## Compute the accuracy and fairness measures for protected groups
        

# ==============================================================================
#                             Save the results in CSV format
# ==============================================================================                
        
        
    # Create a dictionary to store results
    results_dict = {
        'Dataset': [dataset_name],
        'Cindex': [cindex_all],
        'Brier': [brier_all],
        'AUC': [mean_auc_all],
        'F_I': [F_ind_all],
        'F_CI': [F_cen_ind_all],
        'F_CG': [F_cen_group_all],
        'F_G_Prot_1': [F_group_prot_1_all],
        'F_G_Prot_2': [F_group_prot_2_all]
    }
    
    # Add protected group results
    for m, group in enumerate(protected_group):
        results_dict[f'{group}_Cindex'] = [cindex[group]]
        results_dict[f'{group}_Brier'] = [brier[group]]
        results_dict[f'{group}_AUC'] = [mean_auc[group]]
        results_dict[f'{group}_F_I'] = [F_ind[group]]
        results_dict[f'{group}_F_CI'] = [F_cen_ind[group]]
        results_dict[f'{group}_F_CG'] = [F_cen_group[group]]
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results_dict)
    csv_path = '{}/Results/Results_{}_{}.csv'.format(path_name, model_name, dataset_name)
    df_results.to_csv(csv_path, index=False)
    print('Your result is ready!!!')
    
    return


# ==============================================================================
#                             Run the experiments
# ==============================================================================   

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception as e:
        print(f"Warning: Could not set CUDA device: {e}. Continuing with default device.")

import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
    scipy.integrate.simps = scipy.integrate.simpson

def main(args):

    
    fn_csv             = args.i  ## CSV file of the dataset
    path_name          = args.path_name
    model_name         = args.model_name  ## Name of the model (e.g., FIDP/FIPNAM)
    dataset_name       = args.dataset_name ## Name of the dataset (e.g., SEER/SUPPORT/FLChain)
    batch_size         = args.batch_size
    lr                 = args.lr
    epochs             = args.epochs

    
    ## Run the experiments
    run_experiment(fn_csv,
        path_name,           
        model_name, 
        dataset_name,
        batch_size, 
        lr, 
        epochs)  


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fair and interpretable survival models")
    parser.add_argument("-i", help='input data in csv format')
    parser.add_argument("-p", '--path-name', help='Name of the directory')
    parser.add_argument("-m", '--model-name', type=str, default='FIDP', help='neural network used in training')    
    parser.add_argument("-d", "--dataset-name", default="SUPPORT", type=str)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-lr", "--lr", default=0.01, type=float)    
    parser.add_argument("-e", "--epochs", default=100, type=int)  
    
    args = parser.parse_args()

    main(args)    

