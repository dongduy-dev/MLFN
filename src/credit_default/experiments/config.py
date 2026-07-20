GLOBAL_SEED = 42

TRAINING_CONFIG = {
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 256,
    "maximum_epochs": 50,
    "early_stopping_patience": 7,
    "early_stopping_min_delta": 0.0001,
    "gradient_clipping_max_norm": 5.0
}

FIXED_THRESHOLD = 0.5
