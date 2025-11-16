import logging
import copy
import argparse
import torch
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from pycox.evaluation import EvalSurv
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
    # Python random module
    random.seed(state)
    
    # NumPy random
    np.random.seed(state)
    
    # PyTorch random
    torch.manual_seed(state)
    
    # CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(state)
        torch.cuda.manual_seed_all(state)  # For multi-GPU setups
        # Set deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Python hash seed for dictionary ordering
    os.environ['PYTHONHASHSEED'] = str(state)

def compute_metrics(model, X_data, time_data, event_data, time_train, event_train, eval_time, model_name='FIDP'):
    """
    Compute c-index, brier score, and AUC for training/validation data.
    
    Arguments:
        model: Trained model (FIDP or FIPNAM)
        X_data: Input features (numpy array)
        time_data: Observed time (numpy array)
        event_data: Event status (numpy array)
        time_train: Training time (for AUC calculation)
        event_train: Training event status (for AUC calculation)
        eval_time: Evaluation time points
        model_name: Model name ('FIDP' or 'FIPNAM')
    
    Returns:
        cindex: Concordance index
        brier: Integrated Brier score
        mean_auc: Mean AUC
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Save model training state
    was_training = model.training
    model.eval()
    
    try:
        with torch.no_grad():
            # Get predictions - reuse logic from FIDP_Evaluation and FIPNAM_Evaluation
            if model_name == 'FIPNAM':
                if hasattr(model, 'forward_prob'):
                    sp_pred = model.forward_prob(torch.tensor(np.array(X_data, dtype='float32')).to(device)).cpu().detach().numpy()
                else:
                    # Fallback: apply sigmoid if method doesn't exist (same as FIPNAM_Evaluation)
                    sp_pred, _ = model(torch.tensor(np.array(X_data, dtype='float32')).to(device))
                    sp_pred = torch.sigmoid(sp_pred).cpu().detach().numpy()
            else:  # FIDP
                if hasattr(model, 'forward_prob'):
                    sp_pred = model.forward_prob(torch.tensor(np.array(X_data, dtype='float32')).to(device)).cpu().detach().numpy()
                else:
                    # Fallback: apply sigmoid if method doesn't exist (same as FIDP_Evaluation)
                    sp_pred = torch.sigmoid(model(torch.tensor(np.array(X_data, dtype='float32')).to(device))).cpu().detach().numpy()
        
        cif_pred = 1 - sp_pred
        
        # Prepare survival data structures
        event_data_bool = event_data.astype(bool)
        event_train_bool = event_train.astype(bool)
        
        survival_train = np.dtype([('event', event_train_bool.dtype), ('surv_time', time_train.dtype)])
        survival_train = np.empty(len(event_train_bool), dtype=survival_train)
        survival_train['event'] = event_train_bool
        survival_train['surv_time'] = time_train
        
        survival_data = np.dtype([('event', event_data_bool.dtype), ('surv_time', time_data.dtype)])
        survival_data = np.empty(len(event_data_bool), dtype=survival_data)
        survival_data['event'] = event_data_bool
        survival_data['surv_time'] = time_data
        
        # Compute AUC
        auc, mean_auc = cumulative_dynamic_auc(survival_train, survival_data, cif_pred, eval_time)
        
        # Compute c-index and Brier score
        surv = pd.DataFrame(np.transpose(sp_pred))
        surv = surv.set_index([eval_time])
        ev = EvalSurv(surv, np.array(time_data), np.array(event_data), censor_surv='km')
        cindex = ev.concordance_td()
        brier = ev.integrated_brier_score(eval_time)
        
        return cindex, brier, mean_auc
    finally:
        # Restore model training state
        if was_training:
            model.train()
        else:
            model.eval()

def run_experiment(fn_csv, path_name, model_name, dataset_name, batch_size, lr, epochs, lamda_param=0.01):
    # Configure logging to output to console
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
    
    # Re-import os in function scope to avoid issues with import * potentially shadowing it
    # This ensures os module is available even if import * from other modules overwrote it
    import os
    
    # data_I.csv -> SIMULATED_I, data_II.csv -> SIMULATED_II
    dataset_identifier = dataset_name
    if dataset_name == 'SIMULATED':
        basename = os.path.basename(fn_csv)
        if 'data_II' in basename:
            dataset_identifier = 'SIMULATED_II'
        elif 'data_I' in basename:
            dataset_identifier = 'SIMULATED_I'
    
    print(f"Starting experiment: {model_name} on {dataset_identifier}")
    print(f"Data file: {fn_csv}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    print(f"Lambda (fairness-accuracy trade-off): {lamda_param}")
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
    elif model_name=='Cox':
        # Cox model doesn't need device or features setup
        model = CoxModel(alpha=0.1)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: FIDP, FIPNAM, Cox")
      
    # Only set up optimizer and loss for neural network models
    if model_name in ['FIDP', 'FIPNAM']:
        loss_fn = pseudo_loss #Pseudo value based loss function
        learning_rate=lr
        optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)  
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    else:
        loss_fn = None
        optimizer = None
        scheduler = None

    Epochs = epochs
    patience=10
    best_val_loss=10000000.0
    
    # Fairness parameters - unified for training and evaluation
    SCALE_PARAM = 0.01  ## Scale parameter for Lipschitz constraint (used in individual fairness, censoring-based individual fairness, and censoring-based group fairness)
    
    # Only set lambda and scale tensors for neural network models
    if model_name in ['FIDP', 'FIPNAM']:
        LAMDA_PARAM = lamda_param   ## Trade-off parameter between accuracy and fairness
        print(f"Fairness parameters - Scale: {SCALE_PARAM} (FIXED), Lambda: {LAMDA_PARAM}")
        scale=torch.tensor(SCALE_PARAM).to(device) ## Scale parameter (used in training)
        lamda=torch.tensor(LAMDA_PARAM).to(device)  ## Trade-off parameter between accuracy and fairness 
    else:
        print(f"Fairness parameters - Scale: {SCALE_PARAM} (FIXED, used for evaluation only)")

    # Initialize loss lists for plotting
    train_losses = []
    val_losses = []

# ==============================================================================
#                             Training FIDP model
# ==============================================================================    
     
    if model_name=='FIDP':        
        cindex=[]
        for epoch in range(Epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")        
            train_loss = FIDP_train(train_loader, model, loss_fn, optimizer, len(eval_time),scale, lamda)
            train_losses.append(train_loss)
            logging.info(f"epoch {epoch} | train | {train_loss}")
            
            val_loss = FIDP_evaluate(validate_loader, model, loss_fn, len(eval_time),scale, lamda)
            val_losses.append(val_loss)
            logging.info(f"epoch {epoch} | validate |{val_loss}")
            
            # Compute metrics for training set
            train_cindex, train_brier, train_auc = compute_metrics(
                model, data_X_train, data_time_train, data_event_train, 
                data_time_train, data_event_train, eval_time, model_name='FIDP'
            )
            
            # Compute metrics for validation set
            val_cindex, val_brier, val_auc = compute_metrics(
                model, data_X_val, data_time_val, data_event_val,
                data_time_train, data_event_train, eval_time, model_name='FIDP'
            )
            
            cindex.append(val_cindex)
            
            # Print all metrics
            print(f'Epoch {epoch}:')
            print(f'  Train   - Loss: {train_loss:.6f}, C-index: {train_cindex:.6f}, Brier: {train_brier:.6f}, AUC: {train_auc:.6f}')
            print(f'  Valid   - Loss: {val_loss:.6f}, C-index: {val_cindex:.6f}, Brier: {val_brier:.6f}, AUC: {val_auc:.6f}')

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                es = 0

                # Format lambda for filename (replace . with _)
                lamda_str = str(lamda_param).replace('.', '_')
                model_path = '{}/Trained_models/model_{}_{}_lambda_{}.pt'.format(path_name, model_name, dataset_identifier, lamda_str)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
            else:
                es += 1
                print("Counter {} of {}".format(es,patience))

                if es > patience:
                    print("Early stopping with best_val_loss: ", best_val_loss)
                    break
        print("Done!")
        
        
        ## Evaluation
        lamda_str = str(lamda_param).replace('.', '_')
        model.load_state_dict(torch.load('{}/Trained_models/model_{}_{}_lambda_{}.pt'.format(path_name, model_name, dataset_identifier, lamda_str), map_location=device))
        model.eval()       

        scale_fairness = SCALE_PARAM  ## Scale parameter (should match training scale)

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
        cindex=[]
        for epoch in range(Epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")                
            train_loss = FIPNAM_train(train_loader, model, loss_fn, optimizer, len(eval_time),scale, lamda)
            train_losses.append(train_loss)
            logging.info(f"epoch {epoch} | train | {train_loss}")
            
            val_loss = FIPNAM_evaluate(validate_loader, model, loss_fn, len(eval_time),scale, lamda)
            val_losses.append(val_loss)
            logging.info(f"epoch {epoch} | validate |{val_loss}")
            
            # Compute metrics for training set
            train_cindex, train_brier, train_auc = compute_metrics(
                model, data_X_train, data_time_train, data_event_train, 
                data_time_train, data_event_train, eval_time, model_name='FIPNAM'
            )
            
            # Compute metrics for validation set
            val_cindex, val_brier, val_auc = compute_metrics(
                model, data_X_val, data_time_val, data_event_val,
                data_time_train, data_event_train, eval_time, model_name='FIPNAM'
            )
            
            cindex.append(val_cindex)
            
            # Print all metrics
            print(f'Epoch {epoch}:')
            print(f'  Train   - Loss: {train_loss:.6f}, C-index: {train_cindex:.6f}, Brier: {train_brier:.6f}, AUC: {train_auc:.6f}')
            print(f'  Valid   - Loss: {val_loss:.6f}, C-index: {val_cindex:.6f}, Brier: {val_brier:.6f}, AUC: {val_auc:.6f}')
        
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                es = 0

                # Format lambda for filename (replace . with _)
                lamda_str = str(lamda_param).replace('.', '_')
                model_path = '{}/Trained_models/model_{}_{}_lambda_{}.pt'.format(path_name, model_name, dataset_identifier, lamda_str)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
            else:
                es += 1
                print("Counter {} of {}".format(es,patience))

                if es > patience:
                    print("Early stopping with best_val_loss: ", best_val_loss)
                    break          

        print("Done!")
        
        
         ## Evaluation
        lamda_str = str(lamda_param).replace('.', '_')
        model.load_state_dict(torch.load('{}/Trained_models/model_{}_{}_lambda_{}.pt'.format(path_name, model_name, dataset_identifier, lamda_str), map_location=device))
        model.eval()       

        scale_fairness = SCALE_PARAM  ## Scale parameter (should match training scale)

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
#                             Training Cox model
# ==============================================================================             
            
    elif model_name=='Cox':
        print("Training Cox model...")
        
        # Prepare survival data in sksurv format
        y_train = Surv.from_arrays(event=data_event_train.astype(bool), time=data_time_train)
        y_val = Surv.from_arrays(event=data_event_val.astype(bool), time=data_time_val)
        
        # Fit the Cox model
        print("Fitting Cox model on training data...")
        train_loss = Cox_train(data_X_train, y_train, model)
        train_losses.append(train_loss)
        print("Cox model fitted successfully!")
        
        # Evaluate on validation set (for compatibility, though Cox doesn't use the same loss)
        val_loss = Cox_evaluate(data_X_val, y_val, model)
        val_losses.append(val_loss)
        
        # Compute validation C-index
        metrics = Cox_Concordance(model, data_X_val, np.array(data_time_val), np.array(data_event_val), eval_time)
        print('Validation C-index:', metrics)
        
        # Save the model
        model_path = '{}/Trained_models/model_{}_{}.pkl'.format(path_name, model_name, dataset_identifier)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model.state_dict(), f)
        print(f"Model saved to: {model_path}")
        
        print("Done!")
        
        ## Evaluation
        # Load model (for consistency, though we just saved it)
        model_path = '{}/Trained_models/model_{}_{}.pkl'.format(path_name, model_name, dataset_identifier)
        with open(model_path, 'rb') as f:
            model_state = pickle.load(f)
        model.load_state_dict(model_state)
        model.eval()
        
        scale_fairness = SCALE_PARAM  ## Scale parameter (should match training scale)

        cindex_all, brier_all, mean_auc_all, F_ind_all, F_cen_ind_all, F_cen_group_all, F_group_prot_1_all, F_group_prot_2_all = Cox_Evaluation(model, data_X_test, data_X_test_uncen, data_X_test_cen, data_time_train, data_time_train_uncen, data_time_train_cen, data_time_test, data_time_test_cen, data_time_test_uncen, data_event_train, data_event_test, data_event_test_uncen, data_event_test_cen, np.array(test_data['protected_group1']).astype(int), np.array(test_data['protected_group2']).astype(int), eval_time, scale_fairness, dataset_name)  ## Compute the accuracy and fairness measures

        cindex={}
        brier={}
        mean_auc={}
        F_ind={}
        F_cen_ind={}
        F_cen_group={}
        F_g_prot_1={}
        F_g_prot_2={}
        for group in protected_group:
            # Convert protected group data to proper format
            # For Cox model, we need to preserve DataFrame structure for fairness measures
            # (especially for FLChain dataset where column names are critical)
            if isinstance(protected_X_test[group], pd.DataFrame):
                prot_X_test = protected_X_test[group].values
            else:
                prot_X_test = protected_X_test[group]
                
            # Preserve DataFrame structure for X_test_uncen and X_test_cen to maintain column names
            # This is critical for censoring_group_fairness which needs column names like 'sex_1', 'sex_0'
            if isinstance(protected_X_test_uncen[group], pd.DataFrame):
                prot_X_test_uncen = protected_X_test_uncen[group]  # Keep as DataFrame
            else:
                prot_X_test_uncen = protected_X_test_uncen[group]
                
            if isinstance(protected_X_test_cen[group], pd.DataFrame):
                prot_X_test_cen = protected_X_test_cen[group]  # Keep as DataFrame
            else:
                prot_X_test_cen = protected_X_test_cen[group]
            
            cindex[group], brier[group], mean_auc[group], F_ind[group], F_cen_ind[group], F_cen_group[group], F_g_prot_1[group], F_g_prot_2[group] = Cox_Evaluation(model, prot_X_test, prot_X_test_uncen, prot_X_test_cen, data_time_train, protected_time_train_uncen[group], protected_time_train_cen[group], protected_time_test[group], protected_time_test_cen[group], protected_time_test_uncen[group], data_event_train, protected_event_test[group], protected_event_test_uncen[group], protected_event_test_cen[group], np.array(test_data[test_data[group]==1]['protected_group1']).astype(int), np.array(test_data[test_data[group]==1]['protected_group2']).astype(int), eval_time, scale_fairness, dataset_name) ## Compute the accuracy and fairness measures for protected groups
        

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
    
    # Ensure Results directory exists
    results_dir = '{}/Results'.format(path_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Build filename based on model type (Cox doesn't have lambda)
    if model_name == 'Cox':
        csv_path = '{}/Results/Results_{}_{}.csv'.format(path_name, model_name, dataset_identifier)
    else:
        lamda_str = str(lamda_param).replace('.', '_')
        csv_path = '{}/Results/Results_{}_{}_lambda_{}.csv'.format(path_name, model_name, dataset_identifier, lamda_str)
    
    df_results.to_csv(csv_path, index=False)
    print(f'Your result is ready!!! Saved to: {csv_path}')
    
    # Plot training and validation loss curves
    # Only plot loss for neural network models (FIDP, FIPNAM), not for Cox
    # Cox model doesn't use the same loss function and returns 0.0, so plotting is not meaningful
    if model_name != 'Cox' and train_losses and val_losses and len(train_losses) == len(val_losses) and len(train_losses) > 0:
        # Check if losses are meaningful (not all zeros)
        if any(loss != 0.0 for loss in train_losses) or any(loss != 0.0 for loss in val_losses):
            plt.figure(figsize=(10, 6))
            epochs_range = range(1, len(train_losses) + 1)
            plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
            plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training and Validation Loss - {model_name} on {dataset_identifier} (λ={lamda_param})', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot with same name as CSV but with .png extension
            plot_path = csv_path.replace('.csv', '.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Loss plot saved to: {plot_path}')
        else:
            print('Skipping loss plot: All loss values are zero (not meaningful).')
    elif model_name == 'Cox':
        print('Skipping loss plot: Cox model does not use the same loss function as neural network models.')
    elif train_losses or val_losses:
        print('Warning: Loss lists have different lengths or are empty. Skipping plot generation.')
    
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
    lamda_param        = args.lamda 

    
    ## Run the experiments
    run_experiment(fn_csv,
        path_name,           
        model_name, 
        dataset_name,
        batch_size, 
        lr, 
        epochs,
        lamda_param)  


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Fair and interpretable survival models")
    parser.add_argument("-i", help='input data in csv format')
    parser.add_argument("-p", '--path-name', help='Name of the directory')
    parser.add_argument("-m", '--model-name', type=str, default='FIDP', help='neural network used in training')    
    parser.add_argument("-d", "--dataset-name", default="SUPPORT", type=str)
    parser.add_argument("-b", "--batch-size", default=64, type=int)
    parser.add_argument("-lr", "--lr", default=0.01, type=float)    
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-l", "--lamda", default=0.01, type=float)  
    
    args = parser.parse_args()

    main(args)    

