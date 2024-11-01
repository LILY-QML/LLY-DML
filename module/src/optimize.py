from module.helper import qubit 
import importlib
import json
import os
import logging
import re

class Optimizer:
    def __init__(self, optimizer, config_path='var'):
        self.Qubit_Object = {}
        self.current_job = None
        self.optimizer = optimizer
        self.optimizer_class = None
        self.config_path = config_path
        self.dict_params_current_job = None

    def check_prerequisites(self):
        """
        Check optimizer and generate qubits.
        """
        Error_loading_optimizer = False

        optmizer_file_name = self.optimizer.split("Optimizer")[0].lower()+"_optimizer"        
        optimizer_module_name = f"module.optimizers.{optmizer_file_name}"
        optimizer_class = None
        try:
            optimizer_module = importlib.import_module(optimizer_module_name)
            optimizer_class = getattr(optimizer_module, self.optimizer, None)
        except Exception as e:
            Error_loading_optimizer = True

        if optimizer_class is None:
            Error_loading_optimizer = True

        if Error_loading_optimizer == True:
            return {"Error Code": 1111, "Message": "Optimizer not found."}

        self.optimizer_class = optimizer_class

        ###Create qubits
        path_data_json=os.path.join(self.config_path, 'data.json')
        try:
            with open(path_data_json, 'r') as config_file:
                jsonload = json.load(config_file)
        except Exception as e:
            return {"Error Code": 1112, "Message": "Data file not found."}

        num_qubits = jsonload['qubits']
        for i in range(num_qubits):
            qubit_object = qubit.Qubit(qubit_number=i)
            self.Qubit_Object[i] = qubit_object
            #print(f"Qubit {i} created")

    
    def check_data_structure(self):
        """
        Check if the data structure is correct.
        """
        re_pattern=r"\(\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\) Qubit_(\d+) \(\s*(\d+)\:\s*(\d+);\s*(\d):\s*(\d+)\) \(\s*S:\s*([0-1]+)\)"
        ret=re.match(re_pattern, self.current_job)

        if ret is None:
            return {"Error Code": 1119, "Message": "Datastructure is not consistent."}  
        else:

            self.dict_params_current_job = {
                "T1_0":ret.group(1),
                "T1_1":ret.group(2),
                "T1_2":ret.group(3),
                "num_qubit":ret.group(4),
                "dist_0_0":ret.group(5),
                "vdist_0_1":ret.group(6),
                "dist_1_0":ret.group(7),
                "dist_1_1":ret.group(8),
                "state":ret.group(9)
            }         

            #print(self.dict_params_current_job)

    def evaluate(self, current_job):
        """
        Evaluate the current job by checking the data structure and loading the state into the corresponding qubit.
        
        Parameters:
        current_job (str): The current job string to be evaluated.
        """
        self.current_job = current_job
        ret=self.check_data_structure()
        if ret is not None:
            num_qubit = self.dict_params_current_job["num_qubit"]
            self.Qubit_Object[num_qubit].load_state(self.dict_params_current_job["state"])

    def execute(self, current_job):
        """
        Execute the optimization process for the current job.
        
        Parameters:
        current_job (str): The current job string to be executed.
        
        Returns:
        str: The new job string after optimization.
        """
        self.evaluate(current_job)
        optimizer=self.optimizer_class(None, None, self.dict_params_current_job["state"])
        optimizer.evaluate()
        loss=optimizer.calculate_loss()
        optimizer.compute_gradient()        
        tuning_parameters, optimization_steps = optimizer.optimize()
        num_qubit = self.dict_params_current_job["num_qubit"]
        self.Qubit_Object[num_qubit].load_function(loss)       
        new_job="({}, {}, {} Qubit_{} ({}:{}; {}:{}) (S:{})".format(tuning_parameters[0], tuning_parameters[1], tuning_parameters[2], 
                                                                    self.dict_params_current_job["num_qubit"], self.dict_params_current_job["dist_0_0"], 
                                                                    self.dict_params_current_job["vdist_0_1"], self.dict_params_current_job["dist_1_0"],
                                                                     self.dict_params_current_job["dist_1_1"], self.dict_params_current_job["state"])
        return new_job