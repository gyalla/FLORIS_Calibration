import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np 
import pathlib
import pandas as pd
import scipy as sp
import pickle
import sys
import itertools

import floris.layout_visualization as layoutviz
from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
    WindTIRose,
)
from floris.flow_visualization import visualize_cut_plane
from concurrent.futures import ProcessPoolExecutor

library_path = '../'
if library_path not in sys.path:
    sys.path.append(library_path)
import calibrate_floris as cf
import floris
print(f"The source file for the floris library is located at: {floris.__file__}",flush=True)

def rotation_matrix(theta):
    theta_rad = np.radians(theta)  
    return np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                     [np.sin(theta_rad), np.cos(theta_rad)]])

def calculate_wind_farm_coordinates(x1, y1, D, theta, spacing):
    # Convert angle from degrees to radians
    theta_rad = np.radians(theta)

    # Define the spacing
    spacing = spacing * D

    # Initialize a list to hold the coordinates
    coordinates_x = np.zeros((3,3))
    coordinates_y = np.zeros((3,3))

    rot = rotation_matrix(theta)
    # Calculate the coordinates for a 3x3 wind farm aligned with the wind direction
    for i in range(3):  # For rows
        for j in range(3):  # For columns
            x = (x1 - spacing) + j*spacing
            y = (y1 - spacing) + i*spacing
            vec = np.asarray([x,y])
            rot_vec = np.dot(rot,vec)
            coordinates_x[i,j] = rot_vec[0]
            coordinates_y[i,j] = rot_vec[1]

    return (coordinates_x,coordinates_y)

def setup_floris_model(input_file,farm_angle,spacing,amp,D=240):
    #Define list of cases 
    cases = [
        "MedWS_LowTI_AWC_" + str(float(spacing)) + "D_" + str(float(amp)) + "_" + str(float(farm_angle)),
    ]

    #Specify setup files to be generated for each case
    setup_files = {}
    setup_files[cases[0]] = cases[0] + '.yaml'

    # Get the coordinates of the wind farm
    farm_center_x  = 0.0
    farm_center_y = 0.0
    coordinates_x, coordinates_y = calculate_wind_farm_coordinates(farm_center_x,farm_center_y, D,farm_angle, spacing)
    coordinates_x = coordinates_x.flatten().tolist()
    coordinates_y = coordinates_y.flatten().tolist()

    setup_params  = {
        'flow_field.air_density': 1.2456,
        'flow_field.wind_veer': 8.94,
        'flow_field.wind_shear': 0.16,
        'farm.turbine_type': ['iea_15MW_calibrate',],
        'farm.turbine_library_path': '/ascldap/users/gyalla/GPFS/Advanced_Controls/Floris/turbine_library',
        'farm.layout_x': coordinates_x,
        'farm.layout_y': coordinates_y,
        'wake.wake_velocity_parameters.empirical_gauss.mixing_gain_velocity': 3.308139775170343,
        'wake.wake_turbulence_parameters.wake_induced_mixing.atmospheric_ti_gain': 0.1,
        'wake.wake_velocity_parameters.empirical_gauss.wake_expansion_rates': [0.0036464646727800824,0.02584939232511248,0.6850233768241475],
        'wake.wake_velocity_parameters.empirical_gauss.breakpoints_D': [4.1085781527711225,11.669334552973915],
    }

    #do not need this correct?
    #floris_models = {}
    for case in cases:
        cf.setup_floris_yaml(input_file,setup_params,output_file=setup_files[case])
        print("Setup file: ",setup_files[case])
    return cases, setup_files


# Define a function to run the model for a given index
def run_model(iiter, i , input_file, wd, ws, ti, awc_amp,wind_speed_factor=1.0):
    fmodel = FlorisModel(input_file)
    awc_amplitudes = [awc_amp,] * fmodel.n_turbines
    baseline_settings = ['baseline']*fmodel.n_turbines
    if ws > 15 or ti > 0.10:
        return (baseline_settings,baseline_settings)
    best_result       = float('-inf')
    best_result_3awc  = float('-inf')
    best_awc_settings = None
    best_3awc_settings = None
    for awc_settings in itertools.product(['baseline', 'helix'], repeat=fmodel.n_turbines):
        check_3_helix  = awc_settings.count('helix') <= 3
        # Create the awc_settings array for the current combination
        fmodel.set_operation_model("awc")
        fmodel.set(wind_directions=np.array([wd]), 
               wind_speeds=[ws*wind_speed_factor], 
               turbulence_intensities=np.array([ti]),
               awc_amplitudes=np.array([awc_amplitudes]),
               awc_modes=np.array([awc_settings])
        )
        fmodel.run()
        farm_power = fmodel.get_farm_power() 

        if farm_power > best_result:
            best_awc_settings = np.array(awc_settings)
            best_result = farm_power

        if check_3_helix and farm_power > best_result_3awc:
            best_3awc_settings = np.array(awc_settings)
            best_result_3awc = farm_power

        if farm_power == 0: #below cut in 
            #return (best_awc_settings,best_3awc_settings)
            return (baseline_settings,baseline_settings)

    return (best_awc_settings,best_3awc_settings)

def optimize_awc_settings(wd_data,ti_data,ws_data,n_findex,non_zero_indices,awc_amp,cases,setup_files,num_cores,savefilename,wind_speed_factor=1.0):
    # ### Wind Rose
    best_awc_settings = []
    best_3awc_settings = []

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(run_model, iiter, non_zero_indices[iiter], setup_files[cases[0]], wd_data[i], ws_data[i], ti_data[i],awc_amp,wind_speed_factor)
            for iiter, i in enumerate(non_zero_indices)
        ]
        
        for future in futures:
            best_awc_settings.append(future.result()[0])   # Append results to the list
            best_3awc_settings.append(future.result()[1])  # Append results to the list

    # Convert the list of results to a NumPy array
    best_awc_settings = np.array(best_awc_settings)
    best_3awc_settings = np.array(best_3awc_settings)

    # Output the shape of the best_awc_settings to verify
    print("Shape of best_awc_settings:", best_awc_settings.shape,flush=True)
    return (best_awc_settings,best_3awc_settings)

def run_aep_estimation(wd_data,ti_data,ws_data,n_findex,non_zero_indices,awc_amp,cases,setup_files,awc_settings,wind_speed_factor=1.0,return_farm_powers=False):

    for case in cases:
        fmodel = FlorisModel(setup_files[case])
        awc_amplitudes = [awc_amp,] * fmodel.n_turbines
        awc_amps = np.array([awc_amplitudes]*n_findex)

        time_series = TimeSeries(wind_directions = wd_data, wind_speeds=ws_data*wind_speed_factor, turbulence_intensities=ti_data)
        fmodel.set_operation_model("awc")
        fmodel.set(
            wind_data=time_series,
            awc_modes=awc_settings,
            awc_amplitudes=awc_amps)
        fmodel.run()
        farm_power = fmodel.get_farm_power() 
        hours_per_year = 8760
        aep = np.sum(farm_power) / n_findex * hours_per_year
        print("case: ",case,flush=True)
        print(f"--> AEP: {aep/1E9:.3f} (GWh)",flush=True)
        print()
    if return_farm_powers:

        return aep, farm_power

    else:
        return aep 

def aep_estimate(cases,setup_files,awc_amp,wind_ti_rose,best_awc_settings,best_3awc_settings):
   
    n_wd = len(wind_ti_rose.wind_directions)
    n_ws = len(wind_ti_rose.wind_speeds)
    n_ti = len(wind_ti_rose.turbulence_intensities)
    mask_non_zero = wind_ti_rose.freq_table.flatten() != 0
    n_findex = np.sum(mask_non_zero)
    non_zero_indices = np.where(mask_non_zero)[0]  
    baseline_array = ['baseline',] * 9 
    awc_amplitudes = [awc_amp,] * 9

    #TODO: add in wind speed factor

    aeps = {}
    for case in cases:
        print("case: ",case,flush=True)
        baseline_settings = np.array([baseline_array]*n_findex)
        awc_amps = np.array([awc_amplitudes]*n_findex)

        fmodel = FlorisModel(setup_files[case])
        fmodel.set_operation_model("awc")
        fmodel.set(
            wind_data=wind_ti_rose,
            awc_modes=best_awc_settings,
            awc_amplitudes=awc_amps)
        fmodel.run()
        aep_awc = fmodel.get_farm_AEP()
        print(f"AEP AWC: {aep_awc/1E9:.3f} (GWh)")

        fmodel = FlorisModel(setup_files[case])
        fmodel.set_operation_model("awc")
        fmodel.set(
            wind_data=wind_ti_rose,
            awc_modes=best_3awc_settings,
            awc_amplitudes=awc_amps)
        fmodel.run()
        aep_3awc = fmodel.get_farm_AEP()
        print(f"AEP 3AWC: {aep_3awc/1E9:.3f} (GWh)")

        fmodel = FlorisModel(setup_files[case])
        fmodel.set_operation_model("awc")
        fmodel.set(
            wind_data=wind_ti_rose,
            awc_modes=baseline_settings,
            awc_amplitudes=awc_amps)
        fmodel.run()
        aep_baseline = fmodel.get_farm_AEP()
        print(f"AEP Baseline: {aep_baseline/1E9:.3f} (GWh)")

    return (aep_baseline,aep_awc,aep_3awc)

def get_wind_rose(num_bins,wind_speed_factor=1.0,return_timeseries=False):
    with open('./Example/3D_weibull_data.pkl', 'rb') as f:
        wd_data, ti_data, ws_data = pickle.load(f)

    sample = np.column_stack((wd_data,ws_data,ti_data))
    hist , edges = np.histogramdd(sample, bins=num_bins, range=((0,360),(0,30),(0,0.5)))
    wind_directions = edges[0][1:]
    wind_speeds = edges[1][1:]
    turbulence_intensities = edges[2][1:]
    # Uniform value
    value_table = np.ones((len(wind_directions), len(wind_speeds), len(turbulence_intensities)))

    wind_ti_rose = WindTIRose(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds * wind_speed_factor,
        turbulence_intensities=turbulence_intensities,
        freq_table=hist,
        value_table=value_table,
    )

    if return_timeseries==False:
        return wind_ti_rose
    else:
        return wind_ti_rose, wd_data, ti_data, ws_data

def main(argv):
    input_file = str(argv[0])
    save_file  = str(argv[1])
    spacing    = float(argv[2])
    awc_amp    = float(argv[3])
    farm_angle = float(argv[4]) #add option to test all wind angles
    num_bins   = int(argv[5]) #add option to test all wind angles
    num_cores  = int(argv[6])
    use_timeseries = int(argv[7])

    if use_timeseries == 0:
        use_timeseries = True
    else:
        use_timeseries = False

    #savefilename = 'best_awc_settings_ws_10_all_ti.npy'
    print("\n------------------------------------",flush=True)
    print("Input file: ",input_file,flush=True)
    print("Save file: ",save_file,flush=True)
    print("Spacing: ",spacing,flush=True)
    print("AWC Amp: ",awc_amp,flush=True)
    print("farm_angle: ",farm_angle,flush=True)
    print("num_bins: ",num_bins,flush=True)
    print("num_cores: ",num_cores,flush=True)
    print("------------------------------------\n\n",flush=True)

    #get wind rose object
    print("Creating wind rose object",flush=True)
    print("------------------------------------",flush=True)
    if use_timeseries:
        wind_ti_rose, wd_data, ti_data, ws_data  = get_wind_rose(num_bins,wind_speed_factor=1.0,return_timeseries=use_timeseries)
        n_findex = len(wd_data)
        non_zero_indices = np.arange(0,len(wd_data))
        #print("Wind Speeds: ",ws_data)
        #print("Wind Directions: ",wd_data)
        #print("TI: ",ti_data)
        print("n_findex: ",n_findex)
    else:
        wind_ti_rose = get_wind_rose(num_bins,wind_speed_factor=1.0,return_timeseries=use_timeseries)
        flat_table = wind_ti_rose.freq_table.flatten()
        table_3D = wind_ti_rose.freq_table
        wd_grid,ws_grid,ti_grid = np.meshgrid(wind_ti_rose.wind_directions,wind_ti_rose.wind_speeds,wind_ti_rose.turbulence_intensities,indexing='ij')
        wd_data = wd_grid.flatten()
        ws_data = ws_grid.flatten()
        ti_data = ti_grid.flatten()
        mask_non_zero = wind_ti_rose.freq_table.flatten() != 0
        n_findex = np.sum(mask_non_zero)
        non_zero_indices = np.where(mask_non_zero)[0]  # The [0] is to get the first element of the tuple returned by np.where

    print("------------------------------------\n\n",flush=True)

    if farm_angle == -1:
        farm_angles = wind_ti_rose.wind_directions
    else:
        farm_angles = [farm_angle,]

    for farm_angle in farm_angles:
        print("Working on farm angle: ",farm_angle,"\n\n",flush=True)
        #setup floris model files
        print("Generating setup files",flush=True)
        print("------------------------------------",flush=True)
        cases, setup_files = setup_floris_model(input_file,farm_angle,spacing,awc_amp)
        print("------------------------------------\n\n",flush=True)

        #optimize with given wind_angle
        print("Optimizing AWC Settings",flush=True)
        print("------------------------------------",flush=True)
        wind_speed_factor = 0.9760704004728211
        if awc_amp > 0:
            (best_awc_settings,best_3awc_settings) = optimize_awc_settings(wd_data,ti_data,ws_data,n_findex,non_zero_indices,awc_amp,cases,setup_files,num_cores,save_file,wind_speed_factor=wind_speed_factor)
        else:
            baseline_array = ['baseline',] * 9
            baseline_settings = np.array([baseline_array]*n_findex)

            best_awc_settings = baseline_settings
            best_3awc_settings = baseline_settings

        print("------------------------------------\n\n",flush=True)

        print("Running AEP Estimation",flush=True)
        print("------------------------------------",flush=True)

        if use_timeseries:
            baseline_array = ['baseline',] * 9
            baseline_settings = np.array([baseline_array]*n_findex)

            aep_baseline, farm_powers_baseline = run_aep_estimation(wd_data,ti_data,ws_data,n_findex,non_zero_indices,awc_amp,cases,setup_files,baseline_settings,wind_speed_factor=wind_speed_factor,return_farm_powers=True)
            aep_awc , farm_powers_awc  = run_aep_estimation(wd_data,ti_data,ws_data,n_findex,non_zero_indices,awc_amp,cases,setup_files,best_awc_settings,wind_speed_factor=wind_speed_factor,return_farm_powers=True)
            aep_3awc , farm_powers_3awc = run_aep_estimation(wd_data,ti_data,ws_data,n_findex,non_zero_indices,awc_amp,cases,setup_files,best_3awc_settings,wind_speed_factor=wind_speed_factor,return_farm_powers=True)
            #aep_awc  = 0
            #aep_3awc = 0
        else:
            wind_ti_rose = get_wind_rose(num_bins,wind_speed_factor = wind_speed_factor)
            (aep_baseline, aep_awc, aep_3awc) = aep_estimate(cases,setup_files,awc_amp,wind_ti_rose,best_awc_settings,best_3awc_settings)

        print()
        print(f"AEP change: {100 * (aep_awc- aep_baseline)/aep_baseline:.2f}%")
        print(f"3AEP change: {100 * (aep_3awc- aep_baseline)/aep_baseline:.2f}%")
        print("------------------------------------\n\n",flush=True)

        #np.save(
        # Create a dictionary to hold all the data
        data_to_save = {
            'cases': cases,
            'setup_files': setup_files,
            'spacing': spacing,
            'awc_amp': awc_amp,
            'farm_angle': farm_angle,
            'num_bins': num_bins,
            'best_awc_settings': best_awc_settings,
            'best_3awc_settings': best_3awc_settings,
            'AEP_Baseline': aep_baseline,
            'AEP_AWC':aep_awc,
            'AEP_3AWC':aep_3awc,
            'Farm_Power_Baseline': farm_powers_baseline,
            'Farm_Power_AWC':farm_powers_awc,
            'Farm_Power_3AWC':farm_powers_3awc
        }

        # Write the data to a pickle file
        savefilename = save_file + '_' + cases[0] + '.pkl'
        with open(savefilename, 'wb') as f:
            pickle.dump(data_to_save, f)

        #with open(savefilename, 'rb') as f:
        #    loaded_data = pickle.load(f)
        #print(loaded_data)

if __name__=='__main__':
    main(sys.argv[1:])

