import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import NonlinearConstraint
from floris import FlorisModel
import yaml

def setup_floris_yaml(input_file,setup_params,output_file = None):
    # Load the existing YAML file
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    def set_nested_value(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        if value is None:
            # Remove the key if value is None
            d.pop(keys[-1], None)
        else:
            d[keys[-1]] = value
    # Apply modifications
    for path, value in setup_params.items():
        keys = path.split('.')
        set_nested_value(data, keys, value)

    if output_file == None: output_file=input_file

    # Write the modified data back to a new YAML file
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    return

class FLORIS_Optimizer:

    def __init__(self,floris_models,calibration_params,target_data,calibration_bounds=None,turbine_calibration_params = None,turbine_calibration_bounds = None):

        self.iter = 0
        self.llhood = []

        self.floris_models = floris_models
        self.calibration_params = calibration_params
        self.turbine_calibration_params = turbine_calibration_params
        self.cases = floris_models.keys()
        self.num_cases = len(self.cases)
        self.target_data_dict = target_data 
        self.get_target_data()

        self.operation_model = {}
        self.awc_modes = {}
        self.awc_amplitudes = {}
        self.num_turbines = {}
        self.power_curve  = {}
        self.thrust_curve = {}

        for case_iter, case in enumerate(self.cases):
            floris_model = self.floris_models[case]
            #TODO: THIS MAY FAIL! 
            self.num_turbines[case] = len(floris_model.core.farm.layout_x)
            turbine_types = floris_model.core.farm.turbine_type
            #print(floris_model.core.farm)
            self.operation_model[case] = []
            self.power_curve[case] = []
            self.thrust_curve[case] = []
            try:
                #Standard
                for turb_def in floris_model.core.farm.turbine_definitions:
                    self.operation_model[case].append(turb_def['operation_model'])
                    self.power_curve[case].append(turb_def['power_thrust_table']['power'])
                    self.thrust_curve[case].append(turb_def['power_thrust_table']['thrust_coefficient'])
            except:
                #AWC
                try:
                    for turbine_type in turbine_types:
                        self.operation_model[case].append(turbine_type['operation_model'])
                        self.power_curve[case].append(turbine_type['power_thrust_table']['power'])
                        self.thrust_curve[case].append(turbine_type['power_thrust_table']['thrust_coefficient'])
                except:
                    print("Operation mode has changed data structures. Add to FLORIS_Optimizer constructor")

            self.awc_modes[case] = floris_model.core.farm.awc_modes
            self.awc_amplitudes[case] = floris_model.core.farm.awc_amplitudes
        
        if self.turbine_calibration_params != None:
            self.turbine_files = {}
            for t in self.turbine_calibration_params.keys():
                self.turbine_files[t] = None
                for case_iter, case in enumerate(self.cases):
                    floris_model = self.floris_models[case]
                    internal_fn = (floris_model.core.farm.internal_turbine_library / t).with_suffix(".yaml")
                    external_fn = (floris_model.core.farm.turbine_library_path / t).with_suffix(".yaml")
                    in_internal = internal_fn.exists()
                    in_external = external_fn.exists()
                    if in_internal:
                        self.turbine_files[t] = internal_fn
                    elif in_external:
                        self.turbine_files[t] = external_fn
                if self.turbine_files[t] == None:
                    print("Error: Turbine file for type ",t," not found.")
                    sys.exit()
                else:
                    print("Turbine file: ",self.turbine_files[t])

        self.x0 = np.asarray(self.calibration_dict_to_array(calibration_params))
        self.num_params = []
        if calibration_bounds == None:
            self.bounds = []
            for i in range(len(self.x0)):
                self.bounds.append((None,None))
        else:
            self.bounds = self.calibration_dict_to_array(calibration_bounds)
        self.num_params.append(len(self.x0)) #number of wake parameters
        
        if self.turbine_calibration_params != None:
            for titer, t in enumerate(self.turbine_calibration_params.keys()):
                x0 = np.asarray(self.calibration_dict_to_array(self.turbine_calibration_params[t]))
                self.num_params.append(len(x0) + self.num_params[-1])
                self.x0 = np.concatenate((self.x0, x0))

                if turbine_calibration_bounds == None or turbine_calibration_bounds[t] == None:
                    for i in range(len(self.x0)-self.num_params[0]):
                        self.bounds.append((None,None))
                else:
                    self.bounds += self.calibration_dict_to_array(turbine_calibration_bounds[t])

        return

    def optimize(self,maxiter=1000,globalSolve=False,constraints=(),savecsv=None):
        if not globalSolve: 
            MLE = sp.optimize.minimize(
                self.cost_function,
                self.x0,
                bounds=self.bounds,
                constraints=constraints,
                options={"disp": True, "maxiter": maxiter},
            )
        else: 
            minimizer_kwargs = {}
            minimizer_kwargs['bounds']  = self.bounds
            minimizer_kwargs['constraints']  = constraints
            MLE = sp.optimize.basinhopping(self.cost_function, self.x0, minimizer_kwargs=minimizer_kwargs,niter=maxiter)

        if not savecsv==None:
            df = pd.DataFrame(self.llhood)
            df.to_csv(savecsv, index=False, header=False)  # Set index=False to avoid writing row indices
        return MLE

    def cost_function(self,x):
        self.iter += 1
        self.calibration_params = self.calibration_array_to_dict(self.calibration_params,x[0:self.num_params[0]])
        if self.turbine_calibration_params != None:
            for titer, t in enumerate(self.turbine_calibration_params.keys()):
                self.turbine_calibration_params[t] = self.calibration_array_to_dict(self.turbine_calibration_params[t],x[self.num_params[titer]:self.num_params[titer+1]])

        obs = self.run_floris_models()
        llhood = 0
        scaling = 1.0/1000.0
        for case_iter , case in enumerate(self.cases):
            if 'awc' in self.operation_model[case]:
                floris_model = self.floris_models[case]
                baseline_array = ['baseline']*self.num_turbines[case]
                floris_model.set(awc_modes=np.array([baseline_array]),awc_amplitudes=np.array(self.awc_amplitudes[case]))
                floris_model.run()
                baseline_results = floris_model.get_turbine_powers()
                #print()
                #print("x",x)
                #print("floris awc: ",scaling*(obs[case]))
                #print("floris baseline: ",scaling*(baseline_results))
                #print("floris deltas: ",scaling*(obs[case]-baseline_results))
                #print("LES deltas: ",self.target_data[case])
                #print()
                llhood += 0.5 * np.sum(((obs[case]-baseline_results)*scaling - self.target_data[case]) ** 2)
            else:
                llhood += 0.5 * np.sum((obs[case]*scaling - self.target_data[case]) ** 2)
        print("cost function: ",llhood, ", iteration: ", self.iter)
        self.llhood.append(llhood)
        return llhood

    def get_target_data(self):
        self.target_data = {}
        for case in self.cases:
            file = self.target_data_dict[case][1]
            floris_model = self.floris_models[case]
            target_data = np.zeros((1,floris_model.n_turbines))
            df = np.asarray(pd.read_csv(file, header=None))
            self.target_data[case] = df
        return

    def run_floris_models(self):
        results = {}
        if self.turbine_calibration_params != None:
            for t in self.turbine_calibration_params.keys():
                self.modify_yaml(self.turbine_files[t],self.turbine_calibration_params[t])

        for case_iter, case in enumerate(self.cases):
            floris_model = self.floris_models[case]
            self.modify_yaml(floris_model.configuration,self.calibration_params)
            floris_model = FlorisModel(floris_model.configuration)
            floris_model.set_operation_model(self.operation_model[case])
            floris_model.set(awc_modes=np.array(self.awc_modes[case]),awc_amplitudes=np.array(self.awc_amplitudes[case]))
            floris_model.run()
            qoi = self.target_data_dict[case][0]
            if qoi == 'turbine_power':
                results[case] = floris_model.get_turbine_powers()
        return results

    def modify_yaml(self,input_file,params,output_file = None):
        # Load the existing YAML file
        with open(input_file, 'r') as file:
            data = yaml.safe_load(file)

        def set_nested_value(d, keys, value):
            for key in keys[:-1]:
                if ('turbine_type' in key):
                    print("HERE: ",key,value)
                d = d.setdefault(key, {})
            if value is None:
                # Remove the key if value is None
                d.pop(keys[-1], None)
            else:
                d[keys[-1]] = value
        # Apply modifications
        for path, value in params.items():
            keys = path.split('.')
            set_nested_value(data, keys, value)

        if output_file == None: output_file=input_file
        # Write the modified data back to a new YAML file
        with open(output_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        return

    def calibration_dict_to_array(self,params):
        params_vec = []
        for key, value in params.items():
            if isinstance(value, list):
                params_vec.extend(value)  # Add all elements from the list
            elif isinstance(value, (int, float, tuple)):
                params_vec.append(value)  # Add the single number
            elif value == None:
                params_vec.append(value)
        #params_vec = np.asarray(params_vec)
        return params_vec

    def calibration_array_to_dict(self,params,values):
        counter = 0
        for key, value in params.items():
            if isinstance(value, list):
                for i in range(len(value)): 
                    params[key][i] = float(values[counter])
                    counter += 1
            elif isinstance(value, (int, float)):
                params[key] = float(values[counter])
                counter += 1
        return params

