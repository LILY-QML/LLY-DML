from module.src.reader import Reader

class Start:
    def __init__(self, train_file_flag: bool):
        # Instantiate the Reader class
        self.reader = Reader()
        # Set the train_file_flag value
        self.train_file_flag = train_file_flag

    def run(self):
        # First, perform the fileCheck
        file_check_result = self.reader.fileCheck()

        # Check the result of fileCheck and handle the possible Error/Success Codes
        if file_check_result == 1001:
            print("Error Code: 1001 - 'var' folder is missing.")
            return
        elif file_check_result == 1002:
            print("Error Code: 1002 - 'data.json' file is missing in 'var' folder.")
            return
        elif file_check_result == 1003:
            print("Error Code: 1003 - 'config.json' file is missing in 'var' folder.")
            return
        elif file_check_result == 2001:
            print("Success Code: 2001 - All files are present. 'train.json' exists.")
            self.train_file_flag = True
        elif file_check_result == 2002:
            print("Success Code: 2002 - All files are present. 'train.json' is missing.")
            self.train_file_flag = False

        # Then, perform the checkLog
        self.reader.checkLog()
        print("Log check completed.")
        
        # Lastly, perform the dataConsistency check
        self.reader.dataConsistency()
        print("Data consistency check completed.")

# Example of creating an instance and calling the run method
# start_instance = Start(train_file_flag=True)
# start_instance.run()
