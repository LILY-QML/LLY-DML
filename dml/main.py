# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

from module.src.console import Console

class DML:
    def __init__(self):
        self.console = Console()

    def start(self):
        result = self.console.run()
        if result == "Train_all_optimizers":
            self.train_all_optimizers()
        elif result == "Train_specific_optimizers":
            self.train_specific_optimizers()
        elif result == "Start_testing_write_to_test_data":
            self.start_testing_write_to_test_data()
        elif result == "Create_train_report":
            self.create_train_report()
        elif result == "Create_test_report":
            self.create_test_report()
        elif result == "Create_test_and_train_report":
            self.create_test_and_train_report()
        elif result == "Exit_program":
            self.exit_program()

    # Define methods corresponding to each action
    def train_all_optimizers(self):
        print("train_all_optimizers method executed")

    def train_specific_optimizers(self):
        print("train_specific_optimizers method executed")

    def start_testing_write_to_test_data(self):
        print("start_testing_write_to_test_data method executed")

    def create_train_report(self):
        print("create_train_report method executed")

    def create_test_report(self):
        print("create_test_report method executed")

    def create_test_and_train_report(self):
        print("create_test_and_train_report method executed")

    def exit_program(self):
        print("exit_program method executed")

# Usage
if __name__ == "__main__":
    dml = DML()
    dml.start()
