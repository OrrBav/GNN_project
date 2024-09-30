import os, sys
import pandas as pd

if __name__ == "__main__":

    SRC_DIR = "cvc"
    assert os.path.isdir(SRC_DIR), f"Cannot find src dir: {SRC_DIR}"
    sys.path.append(SRC_DIR)

    from cvc import model_utils

    from lab_notebooks.utils import SC_TRANSFORMER, TRANSFORMER, DEVICE
    MODEL_DIR = os.path.join(SRC_DIR, "models")
    sys.path.append(MODEL_DIR)

    FILT_EDIT_DIST = True
    print("Hello")