# Load necessary modules or source scripts
source /projects/wind_uq/gyalla/src/python/loadtcf.sh
which python3

# Define the input parameters
input_file=./Example/MedWS_LowTI_Pulse_A2_St0p3_6D_45_emgauss.yaml
save_file_root=FLORIS_Results
spacing=6
AWC_amp=2
num_bins=30
num_cores=112
use_timeseries=0
farm_angle=225

time python3 AEP_AWC_Optimizer.py "$input_file" "$save_file_root" "$spacing" "$AWC_amp" "$farm_angle" "$num_bins" "$num_cores" "$use_timeseries"
