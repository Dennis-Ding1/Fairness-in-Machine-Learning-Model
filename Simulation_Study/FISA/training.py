import torch
import numpy as np
from sksurv.util import Surv
from model import *
from fairness_measure import *
from utils import *


# ==============================================================================
#                             Train the FIDP model
# ==============================================================================    

def FIDP_train(dataloader, model, loss_fn, optimizer, ntimes, scale, lamda):
    """
    Train the FIDP model
    Arguments:
    dataloader -- Training dataloader
    model -- FIDP model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    optimizer--Optimizer (e.g. Adam optimizer)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    Updated FIDP model 
    """ 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    total_pseudo_loss = 0
    total_R_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        pseudo_loss_val = loss_fn(pred, y)
        target_fairness = torch.tensor(0.0).to(device)
        IFloss=criterionHinge() ## Fairness penalty constraint
        R_loss_val = IFloss(target_fairness,pred,X, ntimes, scale)
        # calculate loss
        loss = pseudo_loss_val + lamda*R_loss_val        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Check for gradient issues (only on first batch)
        if batch == 0:
            grad_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param_count += 1
            grad_norm = grad_norm ** 0.5
            if grad_norm < 1e-8:
                print(f"  WARNING: Very small gradient norm: {grad_norm:.2e}")
        
        optimizer.step()
        total_loss += float(loss.item())
        total_pseudo_loss += float(pseudo_loss_val.item())
        total_R_loss += float(R_loss_val.item())
    total_loss /= num_batches
    total_pseudo_loss /= num_batches
    total_R_loss /= num_batches
    
    # Print detailed loss breakdown (only occasionally to avoid spam)
    if num_batches > 0:
        print(f"  Loss breakdown - Pseudo: {total_pseudo_loss:.6f}, Fairness: {total_R_loss:.6f} (λ={lamda.item():.3f}), Total: {total_loss:.6f}")
    
    return total_loss

# ==============================================================================
#                             Evaluate the FIDP model
# ==============================================================================

def FIDP_evaluate(dataloader, model, loss_fn, ntimes, scale, lamda):
    """
    Evaluate the FIDP model
    Arguments:
    dataloader -- Training dataloader
    model -- FIDP model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    FIDP model evaluation
    """     
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pseudo_loss = loss_fn(pred, y)
            target_fairness = torch.tensor(0.0).to(device)
            IFloss=criterionHinge() ## Fairness penalty constraint
            R_loss = IFloss(target_fairness,pred,X, ntimes, scale)
            # calculate loss
            loss = pseudo_loss + lamda*R_loss           
            test_loss += float(loss.item())

    test_loss /= num_batches

    return test_loss

# ==============================================================================
#                             Train the FIPNAM model
# ==============================================================================  

def FIPNAM_train(dataloader, model, loss_fn, optimizer, ntimes, scale, lamda):
    
    """
    Train the FIPNAM model
    Arguments:
    dataloader -- Training dataloader
    model -- FIPNAM model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    optimizer--Optimizer (e.g. Adam optimizer)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    Updated FIPNAM model 
    """     
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    total_pseudo_loss = 0
    total_R_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        # Use raw output for training (without sigmoid), like FIDP
        pred, _ = model(X)
        pseudo_loss_val = loss_fn(pred, y)
        target_fairness = torch.tensor(0.0).to(device)
        IFloss=criterionHinge() ## Fairness penalty constraint
        R_loss_val = IFloss(target_fairness,pred,X, ntimes, scale)
        # calculate loss
        loss = pseudo_loss_val + lamda*R_loss_val        
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Check for gradient issues (only on first batch)
        if batch == 0:
            grad_norm = 0.0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
                    param_count += 1
            grad_norm = grad_norm ** 0.5
            if grad_norm < 1e-8:
                print(f"  WARNING: Very small gradient norm: {grad_norm:.2e}")
        
        optimizer.step()
        total_loss += float(loss.item())
        total_pseudo_loss += float(pseudo_loss_val.item())
        total_R_loss += float(R_loss_val.item())
    total_loss /= num_batches
    total_pseudo_loss /= num_batches
    total_R_loss /= num_batches
    
    # Print detailed loss breakdown
    if num_batches > 0:
        print(f"  Loss breakdown - Pseudo: {total_pseudo_loss:.6f}, Fairness: {total_R_loss:.6f} (λ={lamda.item():.3f}), Total: {total_loss:.6f}")
    
    return total_loss
        
# ==============================================================================
#                             Evaluate the FIPNAM model
# ==============================================================================

def FIPNAM_evaluate(dataloader, model, loss_fn, ntimes, scale, lamda):
    """
    Evaluate the FIPNAM model
    Arguments:
    dataloader -- Training dataloader
    model -- FIPNAM model 
    loss_fn--Objective function (e.g. Pseudo value based loss function)
    ntimes--Number of pre-specified time points
    scale--scale parameter
    lambda--accuracy-fairness trade-off parameter
    Returns:
    FIPNAM model evaluation
    """       
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Use raw output for evaluation loss calculation (without sigmoid), like FIDP
            pred, _ = model(X)
            pseudo_loss = loss_fn(pred, y)
            target_fairness = torch.tensor(0.0).to(device)
            IFloss=criterionHinge() ## Fairness penalty constraint
            R_loss = IFloss(target_fairness,pred,X, ntimes, scale)
            # calculate loss
            loss = pseudo_loss + lamda*R_loss           
            test_loss += float(loss.item())

    test_loss /= num_batches

    return test_loss

# ==============================================================================
#                             Train the Cox model
# ==============================================================================  

def Cox_train(X_train, y_train, model):
    """
    Train (fit) the Cox model
    Arguments:
    X_train -- Training covariates (numpy array)
    y_train -- Training survival data (structured array from sksurv.util)
    model -- Cox model 
    Returns:
    Fitted Cox model (loss is 0.0 as Cox doesn't use the same loss function)
    """     
    # Fit the Cox model
    model.fit(X_train, y_train)
    
    # Return a dummy loss value for compatibility
    # Cox model doesn't use the same loss function as neural networks
    return 0.0

# ==============================================================================
#                             Evaluate the Cox model
# ==============================================================================

def Cox_evaluate(X_val, y_val, model):
    """
    Evaluate the Cox model
    Arguments:
    X_val -- Validation covariates (numpy array)
    y_val -- Validation survival data (structured array from sksurv.util)
    model -- Cox model 
    Returns:
    Cox model evaluation (loss is 0.0 as Cox doesn't use the same loss function)
    """     
    # For Cox model, we don't compute the same loss as neural networks
    # Return a dummy loss value for compatibility
    return 0.0

