# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Joan Pujol (@supercabb)
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++


from module.helper import qubit 
import importlib
import json
import os
import logging
import re
import datetime
import numpy as np


class Optimizer:
    def __init__(self, config_path='var'):
        self.Qubit_Object = {}
        self.current_job = None
        self.optimizer = None
        self.optimizer_class = None
        self.config_path = config_path
        self.dict_params_current_job = None
        self.target_state = None
        self.train_json_file_path = os.path.join("var", "train.json")
        self.train_json_data = None
        self.logger = logging.getLogger()
        self.data_json = None

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
                self.data_json = json.load(config_file)
        except Exception as e:
            return {"Error Code": 1112, "Message": "Data file not found."}

        num_qubits = self.data_json['qubits']
        self.initialize_qubits(num_qubits)

    
    def initialize_qubits(self, num_qubits):
        """
        Initialize the qubits.
        """
        self.Qubit_Object = {}

        for i in range(num_qubits):
            qubit_object = qubit.Qubit(qubit_number=i)
            self.Qubit_Object[i] = qubit_object
            print(f"Qubit {i} created")

    def extract_fields_from_job(self, job):
        fields=None
        
        re_pattern=r"\((.*)\) Qubit_(\d+) \(\s*(\d+)\:\s*(\d+);\s*(\d+):\s*(\d+)\) \(\s*S:\s*([0-1]+)\)"

        ####VERIFY####
        #If we want to restric the state to unique 0 or 1 value instead of many qubit state like S:100
        #but this would give error in the actual optimizer.evaluate() method
        #re_pattern=r"\((.*)\) Qubit_(\d+) \(\s*(\d+)\:\s*(\d+);\s*(\d+):\s*(\d+)\) \(\s*S:\s*([0-1])\)"

        ret=re.match(re_pattern, job)

        if ret is not None:
            fields = {
                "matrix_row_str":ret.group(1),
                "num_qubit":ret.group(2),
                "dist_0_0":ret.group(3),
                "vdist_0_1":ret.group(4),
                "dist_1_0":ret.group(5),
                "dist_1_1":ret.group(6),
                "state":ret.group(7)
            } 

        return fields

    def extract_matrix_from_string(self, matrix_str):
        matrix_elements = None

        #Check matrix_row_str
        re_matrix_str=r"([^,()\s]+),?"
        ret=re.findall(re_matrix_str, matrix_str)

        if ret is not None:
            matrix_elements = []
            for m in ret:
                try:
                    val = float(m)
                except ValueError:
                    self.logger.error("Error converting matrix element to float."+str(ret.group(i)))
                    return None

                matrix_elements.append(val)        

        return matrix_elements

    def check_data_structure(self):
        """
        Check if the data structure is correct.
        """
        self.dict_params_current_job = self.extract_fields_from_job(self.current_job)

        if self.dict_params_current_job is None:
            error = {"Error Code": 1119, "Message": "Datastructure is not consistent."}
            self.logger.error(error)  
            return error
         
        matrix_elements = self.extract_matrix_from_string(self.dict_params_current_job["matrix_row_str"])

        if matrix_elements is None:
            error = {"Error Code": 1119, "Message": "Datastructure is not consistent."}
            self.logger.error(error)
            return error
        
        self.dict_params_current_job["matrix_elements"] = matrix_elements

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
        optimizer=self.optimizer_class(self.data_json, self.dict_params_current_job["matrix_elements"], self.dict_params_current_job["state"])
        optimizer.evaluate()
        loss=optimizer.calculate_loss()
        optimizer.compute_gradient()        
        tuning_parameters, optimization_steps = optimizer.optimize()
        num_qubit = int(self.dict_params_current_job["num_qubit"])
        self.Qubit_Object[num_qubit].load_function(loss)   

        new_job="("
        for t in tuning_parameters:
            #new_job+=str(round(t,5))+","
            new_job+=str(t)+","
        new_job=new_job[:-1]+")"


        new_job=new_job+" Qubit_{} ({}:{}; {}:{}) (S:{})".format(self.dict_params_current_job["num_qubit"], self.dict_params_current_job["dist_0_0"], 
                                                                    self.dict_params_current_job["vdist_0_1"], self.dict_params_current_job["dist_1_0"],
                                                                     self.dict_params_current_job["dist_1_1"], self.dict_params_current_job["state"])
        
        self.logger.info("result: "+new_job)
        return new_job



    def start(self, optimizer, target_state):
        self.optimizer = optimizer
        self.target_state = target_state

        optmizer_file_name = self.optimizer.split("Optimizer")[0].lower()+"_optimizer.py"
        optimizer_file_path = os.path.join(os.path.join("module","optimizers"), optmizer_file_name)

        if not os.path.isfile(optimizer_file_path):
            error = {"Error Code": 1072, "Message": "Optimizer not found."}
            self.logger.error(error)
            return error

        ret = self.check_prerequisites()

        if ret is not None:
            return ret

        if(len(target_state)!=len(self.Qubit_Object)):
            error = {"Error Code": 1071, "Message": "Target state has incorrect formatting."}
            self.logger.error(error)
            return error
        
        self.logger.info({"Succes Code": 2071, "Message": "Target state and optimizer successfully validated and initialized."})
        
        if not os.path.exists(self.train_json_file_path):
            error = {"Error Code": 1070, "Message": "train.json not found."}
            self.logger.error(error)
            return error
        
        with open(self.train_json_file_path, 'r') as config_file:
            self.train_json_data = json.load(config_file)
        
        self.logger.info({"Succes Code": 2070, "Message": "train.json successfully found and loaded."})
        
        self.logger.info("Optimizer "+self.optimizer+" validated and loaded.")
        self.logger.info("Target State "+self.target_state+" validated and loaded.")
        self.logger.info("Starting optimization process at "+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        
    ####VERIFY####
    #I changed the proptotye adding training_matrix as parameter ans returning the new training matrix
    def optimize(self, measurement, training_matrix):
        self.logger.info("Starting optimization process.")
        self.logger.info({"Succes Code": 2073, "Message": "Successful data collection at the beginning of optimization."})

        qubits_measurement = self.encode_measurements(measurement)

        if(qubits_measurement is None):
            error = {"Error Code": 1074, "Message": "Inconsistent data structure after assigning measurement values."}
            self.logger.error(error)
            return None            


        #Training matrix i guess that is in the shape:
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  ....
        #  [0, 0, 0, 0]]
        
        if(len(training_matrix)!=len(self.Qubit_Object)):
            error = {"Error Code": 1075, "Message": "Inconsistent matrix due to incorrect number of rows."}
            self.logger.error(error)
            return None

        for index, row in enumerate(training_matrix):
            qubit_matrix_str="("
            for element in row:
                qubit_matrix_str+=str(element)+","
            qubit_matrix_str=qubit_matrix_str[:-1]+")"
            

            self.Qubit_Object[index].load_training_matrix(qubit_matrix_str)
            self.Qubit_Object[index].load_actual_distribution(qubits_measurement[index])


        try:
            new_training_matrix = []
            for num_qubit,qubit in self.Qubit_Object.items():
                ####VERIFY####
                #Maybe I need to extract the state for this qubit from the target_state, for example if target state is 1001
                #qubit 0 need S:1, qubit 1 need S:0, qubit 2 need S:0, qubit 3 need S:1?
                #In that case the currint job should be like:
                #current_job = qubit.read_training_matrix()+" "+"Qubit_"+str(qubit.read_qubit_number())+" "+qubit.read_actual_distribution()+" (S:"+self.target_state[num_qubit]+")"
                #but optimizer.evaluate() is giving error with targer_state lesser of size 3...
                
                current_job = qubit.read_training_matrix()+" "+"Qubit_"+str(qubit.read_qubit_number())+" "+qubit.read_actual_distribution()+" (S:"+self.target_state+")"
                new_job = self.execute(current_job)
                extracted_fields = self.extract_fields_from_job(new_job)
                extracted_matrix = self.extract_matrix_from_string(extracted_fields["matrix_row_str"])
                new_training_matrix.append(extracted_matrix)
                self.logger.info("Updates matrix for qubit "+str(qubit.read_qubit_number())+" with new values.")
                
        except Exception as e:
            error = {"Error Code": 1077, "Message": "Optimization error while writing the training matrix."}
            self.logger.error(error)
            return None

        self.logger.info({"Succes Code": 2077, "Message": "Optimization successfully completed and training matrix updated."})
        self.logger.info("Ending optimization process.")

        #print(new_training_matrix)
        return new_training_matrix

    

    def encode_measurements(self, measurement):
        qubits_measurement=[]
        qubits_measurement_count = np.zeros((len(self.Qubit_Object), 2), dtype=int)

        self.logger.info({"Starting encode measurements."})

        if(len(next(iter(measurement)))!=len(self.Qubit_Object)):
            error = {"Error Code": 1073, "Message": "Inconsistent data due to incorrect number of qubits."}
            self.logger.error(error)
            return None

        for key, value in measurement.items():
            for index, c in enumerate(key):
                qubits_measurement_count[index][int(c)] += value

        for i in range(len(self.Qubit_Object)):
            qubit_measurement = "("
            qubit_measurement += "1:"+str(qubits_measurement_count[i][1])+ "; "
            qubit_measurement += "0:"+str(qubits_measurement_count[i][0]) + ")"
            
            qubits_measurement.append(qubit_measurement)


        self.logger.info("Encoded measurements: "+str(qubits_measurement))

        return qubits_measurement



