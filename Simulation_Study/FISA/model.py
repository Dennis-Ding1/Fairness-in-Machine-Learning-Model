import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from typing import Sequence
from typing import Tuple
from types import SimpleNamespace
from typing import List
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis

# =============================================================================
#                             FIDP Model (A deep feed-forward neural network)
# =============================================================================
class FIDP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.surv_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.SELU(),nn.Dropout(p=0.4),       
            nn.Linear(128, 64), nn.SELU(),nn.Dropout(p=0.4),                      
            nn.Linear(64, 64), nn.SELU(),nn.Dropout(p=0.4),
            nn.Linear(64, 32), nn.SELU(),nn.Dropout(p=0.4),
            nn.Linear(32, 32), nn.SELU(),nn.Dropout(p=0.4),
            nn.Linear(32, out_features)  # Removed Sigmoid to allow output beyond [0,1] for pseudo values
        )

    def forward(self, input):
        output = self.surv_net(input)
        return output
    
    def forward_prob(self, input):
        """Forward pass with sigmoid activation for probability outputs (used in evaluation)"""
        output = self.surv_net(input)
        return torch.sigmoid(output)
    


# =======================================================================================
#                             FIPNAM Model 
# =======================================================================================
# Neural Additive Model implementation is borrowed from https://github.com/AmrMKayid/nam.
# =======================================================================================

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def get_num_units(
    config,
    features: torch.Tensor,
) -> List:
    # Convert to CPU for numpy operations, handling both CPU and GPU tensors
    if features.is_cuda:
        features = features.cpu()
    elif not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    num_unique_vals = [len(np.unique(features[:, i])) for i in range(features.shape[1])]

    num_units = [min(config.num_basis_functions, i * config.units_multiplier) for i in num_unique_vals]

    return num_units


class Config(SimpleNamespace):
    """Wrapper around SimpleNamespace, allows dot notation attribute access."""

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return Config(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)
    

def defaults() -> Config:
    config = Config(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=2021,

        ## Data Path
        data_path="data/GALLUP.csv",
        experiment_name="NAM",
        regression=False,

        ## training
        num_epochs=1,
        lr=3e-4,
        batch_size=1024,

        ## logs
        logdir="output",
        wandb=True,

        ## Hidden size for layers
        hidden_sizes=[64, 32],

        ## Activation choice
        activation='exu',  ## Either `ExU` or `Relu`
        optimizer='adam',

        ## regularization_techniques
        dropout=0.5,
        feature_dropout=0.5,
        decay_rate=0.995,
        l2_regularization=0.5,
        output_regularization=0.5,

        ## Num units for FeatureNN
        num_basis_functions=1000,
        units_multiplier=2,
        shuffle=True,

        ## Folded
        cross_val=False,
        num_folds=5,
        num_splits=3,
        fold_num=1,

        ## Models
        num_models=1,

        ## for dataloaders
        num_workers=16,

        ## saver
        save_model_frequency=2,
        save_top_k=3,

        ## Early stopping
        use_dnn=False,
        early_stopping_patience=50,  ## For early stopping
    )

    return config



class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=4.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'



class LinReLU(torch.nn.Module):
    __constants__ = ['bias']

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(LinReLU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = (inputs - self.bias) @ self.weights
        output = F.relu(output)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'


class Model(torch.nn.Module):

    def __init__(self, config, name):
        super(Model, self).__init__()
        self._config = config
        self._name = name

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__}(name={self._name})'

    @property
    def config(self):
        return self._config

    @property
    def name(self):
        return self._name

    
class FeatureNN(Model):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        config,
        name,
        *,
        input_shape: int,
        num_units: int,
        feature_num: int = 0,
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__(config, name)
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self.dropout = nn.Dropout(p=self.config.dropout)

        hidden_sizes = [self._num_units] + self.config.hidden_sizes

        layers = []

        ## First layer is ExU
        if self.config.activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=1))

        self.model = nn.ModuleList(layers)
        # self.apply(init_weights)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs


    
    
class FIPNAM(Model):

    def __init__(
        self,
        config,
        name,
        *,
        num_inputs: int,
        num_units: int,
        num_output:int,
    ) -> None:
        super(FIPNAM, self).__init__(config, name)

        self._num_inputs = num_inputs
        self._num_output = num_output
        self.dropout = nn.Dropout(p=self.config.dropout)

        if isinstance(num_units, list):
            assert len(num_units) == num_inputs
            self._num_units = num_units
        elif isinstance(num_units, int):
            self._num_units = [num_units for _ in range(self._num_inputs)]

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(config=config, name=f'FeatureNN_{i}', input_shape=1, num_units=self._num_units[i], feature_num=i)
            for i in range(num_inputs)
        ])

#         self._bias = torch.nn.Parameter(data=torch.zeros(1))
        self.linear = nn.Linear(in_features=num_inputs, out_features=num_output,bias=False)
    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self._num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout(conc_out)

        out = self.linear(dropout_out)
        # FIPNAM uses sigmoid in forward (unlike FIDP) because it only has a simple linear layer at the end
        # Without sigmoid, the linear layer output can be unbounded, causing training instability
        return torch.sigmoid(out), dropout_out
    
    def forward_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with sigmoid activation for probability outputs (used in evaluation)"""
        # forward already applies sigmoid, so just return the first element
        out, _ = self.forward(inputs)
        return out
    

# =======================================================================================
#                             Cox Proportional Hazards Model
# =======================================================================================

class CoxModel:
    """
    Wrapper class for Cox Proportional Hazards model using scikit-survival.
    This is a baseline model for comparison with FIDP and FIPNAM.
    """
    def __init__(self, alpha=0.1):
        """
        Initialize Cox model.
        Arguments:
            alpha: Regularization strength (L2 penalty). Default 0.1.
        """
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the Cox model.
        Arguments:
            X: Covariates (numpy array or pandas DataFrame)
            y: Structured array with survival data (from sksurv.util)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict_survival_function(self, X, return_array=False):
        """
        Predict survival function.
        Arguments:
            X: Covariates
            return_array: If True, return as numpy array; if False, return as DataFrame
        Returns:
            Survival function predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_survival_function(X, return_array=return_array)
    
    def predict_partial_hazard(self, X):
        """
        Predict partial hazard (risk score).
        Arguments:
            X: Covariates
        Returns:
            Partial hazard predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def state_dict(self):
        """Return model state for saving (compatibility with PyTorch models)."""
        # For Cox model, we don't have state_dict in the same way
        # Return a dictionary with the fitted model
        return {'model': self.model, 'is_fitted': self.is_fitted}
    
    def load_state_dict(self, state_dict):
        """Load model state (compatibility with PyTorch models)."""
        self.model = state_dict['model']
        self.is_fitted = state_dict['is_fitted']
    
    def eval(self):
        """Set model to evaluation mode (compatibility with PyTorch models)."""
        pass  # Cox model doesn't have train/eval modes
    
    def train(self):
        """Set model to training mode (compatibility with PyTorch models)."""
        pass  # Cox model doesn't have train/eval modes

