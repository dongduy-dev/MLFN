import numpy as np

THRESHOLD_GRID = np.arange(0.050, 0.955, 0.005) # Includes 0.050 to 0.950 inclusive
FIXED_THRESHOLD = 0.500

# The three expected models
EXPECTED_MODELS = ["logistic_regression", "gru_deep", "conv1d_multiscale"]
EXPECTED_SHAS = {
    "logistic_regression": "cff5103be4b14f6e189c100a55a6e1c82827ec7558bfc2fe70da9f33dbaba13f",
    "gru_deep": "80ad92cae0c4cf49e749ba22b9938c306e3e377a3bc91be21a4ced16805c42e9",
    "conv1d_multiscale": "862c633eb8c64e732faadc8bf2bcc85ce8e8c2b8214f7a7357fcbdb6ec716c53"
}
EXPECTED_MANIFEST_SHA = "860e1578ed82fd9cb87aeed49a64c621c401d4f41376f1544f72cea93389cdd1"
