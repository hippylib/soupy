import subprocess
import sys
import os

# List of mixture component counts to run
n_mix_list = [3, 5, 7, 9, 11, 15]
# Approximation order and direction for GMM decomposition
approximation = "linear"  # or "quadratic"
direction = "hep"  # or "kle"

# obtain the path to the driver script
script_dir = os.path.dirname(os.path.abspath(__file__))
driver_path = os.path.join(script_dir, "driver_helmholtz_mixture_taylor.py")

for n_mix in n_mix_list:
    cmd = [
        sys.executable, driver_path,
        "-a", approximation,
        "--n-mix", str(n_mix),
        "--direction", direction
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Finished n_mix={n_mix}\n{'='*40}")
