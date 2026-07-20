from .baselines import train_logistic_regression, train_gradient_boosting
from .neural_models import GRUSmall, GRUDeep, LSTMDeep, Conv1DSmall, Conv1DDeep, Conv1DMultiScale

def get_model_registry():
    return [
        {
            "name": "logistic_regression",
            "family": "baseline",
            "train_fn": train_logistic_regression
        },
        {
            "name": "gradient_boosting",
            "family": "baseline",
            "train_fn": train_gradient_boosting
        },
        {
            "name": "gru_small",
            "family": "recurrent",
            "model_class": GRUSmall
        },
        {
            "name": "gru_deep",
            "family": "recurrent",
            "model_class": GRUDeep
        },
        {
            "name": "lstm_deep",
            "family": "recurrent",
            "model_class": LSTMDeep
        },
        {
            "name": "conv1d_small",
            "family": "cnn",
            "model_class": Conv1DSmall
        },
        {
            "name": "conv1d_deep",
            "family": "cnn",
            "model_class": Conv1DDeep
        },
        {
            "name": "conv1d_multiscale",
            "family": "cnn",
            "model_class": Conv1DMultiScale
        }
    ]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
