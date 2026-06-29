import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---- pasted from compute_gatedriver_stats.py, computed on TRAIN xlsx only ----
GATEDRIVER_FEATURE_COLS = ['vdd_driver', 'tj_c', 'vgs_min', 'vgs_max', 'vgs_mean', 'vgs_std', 'vgs_final', 'vgs_plateau_mean', 'vgs_rise_slope', 'vgs_fall_slope', 'vds_min', 'vds_max', 'vds_mean', 'vds_std', 'vds_off_mean', 'vds_on_mean', 'vds_overshoot', 'vds_settling_std', 'id_min', 'id_max', 'id_mean', 'id_std', 'id_plateau_mean', 'current_slew_max', 'psw_min', 'psw_max', 'psw_mean', 'psw_std', 'switching_energy_proxy', 'turnon_energy_proxy', 'turnoff_energy_proxy', 'psw_peak_to_mean_ratio']

GATEDRIVER_MEAN = np.array([
    14.951402, 77.166761, 0.239059, 13.600144, 6.309939, 5.663953, 0.395923,
    11.871683, 4.422081, -6.786797, 69.754990, 602.324969, 346.851305,
    239.477728, 588.200407, 120.126865, 14.124562, 33.116559, 1.571812,
    58.385694, 26.773035, 21.299409, 45.079021, 32.421638, 0.209976,
    16973.532397, 3127.383719, 4609.402226, 3127.383719, 4384.885342,
    4419.400140, 9.119003
])

GATEDRIVER_STD = np.array([
    2.095700, 34.986929, 1.486267, 4.409655, 2.359680, 1.997573, 1.450874,
    4.924007, 2.658906, 3.022991, 198.958050, 74.114154, 112.739777,
    93.549263, 68.487144, 216.070954, 19.984426, 61.966523, 9.543193,
    37.347750, 14.438075, 10.098969, 22.282267, 15.755586, 3.002720,
    11839.505057, 4932.910799, 3931.257421, 4932.910799, 2607.622019,
    10040.340254, 5.065664
])

GATEDRIVER_LABELS = ['bootstrap_droop', 'desat_short_circuit', 'gate_overshoot', 'gate_undershoot', 'high_didt', 'high_dvdt', 'miller_plateau_extension', 'normal', 'ovlo', 'pcb_parasitic_ringing', 'propagation_delay', 'reverse_recovery', 'slow_turn_off', 'slow_turn_on', 'thermal_shutdown', 'uvlo']
# ---- end pasted constants ----

LABEL_TO_IDX = {name: i for i, name in enumerate(GATEDRIVER_LABELS)}


class GateDriverDataset(Dataset):
    """
    Loads a gate driver feature Excel file (train/val/test) and applies the
    train-fitted per-column mean/std defined above. Never refit mean/std
    on this class - those constants must come only from the train Excel file,
    via compute_gatedriver_stats.py.
    """

    def __init__(self, xlsx_path):
        df = pd.read_excel(xlsx_path)

        missing = set(GATEDRIVER_FEATURE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Excel file missing expected feature columns: {missing}")

        unknown_labels = set(df["label"].unique()) - set(GATEDRIVER_LABELS)
        if unknown_labels:
            raise ValueError(
                f"Excel file contains labels not seen in GATEDRIVER_LABELS: {unknown_labels}"
            )

        X = df[GATEDRIVER_FEATURE_COLS].values.astype(np.float32)
        X = (X - GATEDRIVER_MEAN.astype(np.float32)) / GATEDRIVER_STD.astype(np.float32)
        self.X = torch.from_numpy(X)

        self.y = torch.tensor(
            [LABEL_TO_IDX[label] for label in df["label"]], dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]