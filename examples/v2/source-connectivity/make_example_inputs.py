"""Generate the small generic inputs used by project.example.yaml."""
from pathlib import Path

import mne
import numpy as np
from scipy.io import savemat


HERE = Path(__file__).resolve().parent
SFREQ = 256.0
SAMPLES = 512
EPOCHS = 6

time = np.arange(SAMPLES, dtype=np.float64) / SFREQ
data = np.empty((EPOCHS, 3, SAMPLES), dtype=np.float64)
for epoch in range(EPOCHS):
    data[epoch, 0] = np.sin(2 * np.pi * 10 * time + epoch / 30)
    data[epoch, 1] = np.sin(2 * np.pi * 10 * time + np.pi / 4 + epoch / 30)
    data[epoch, 2] = np.sin(2 * np.pi * 22 * time + np.pi / 7)

info = mne.create_info(["Fz", "Cz", "Pz"], SFREQ, "eeg")
events = np.column_stack((np.arange(EPOCHS), np.zeros(EPOCHS, dtype=int), np.ones(EPOCHS, dtype=int)))
mne.EpochsArray(data, info, events=events, tmin=0.0, verbose="ERROR").save(
    HERE / "clean-evoked-epo.fif",
    overwrite=True,
    verbose="ERROR",
)
savemat(
    HERE / "sensor-to-scout.mat",
    {
        "SensorToScoutOperator": np.asarray(
            [
                [0.55, 0.35, 0.10],
                [0.10, 0.35, 0.55],
            ],
            dtype=np.float64,
        )
    },
)
print(f"Wrote generic example inputs under {HERE}")
