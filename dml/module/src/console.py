# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 2.0.0 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# Contributors:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

class Console:
    banner = (
        "\033[1;34m"
        " ____   __  __   _\n"
        "|  _ \\ |  \\/  | | |\n"
        "| | | || |\\/| | | |\n"
        "| |_| || |  | | | |___\n"
        "|____/ |_|  |_| |_____|\n"
        "\033[0m"
    )

    info = (
        "\033[1;32m"
        "LLY-DML - Part of the LILY Project - Version 1.6 Beta - info@lilyqml.de - lilyqml.de"
        "\033[0m"
    )

    def display_main_menu(self):
        os.system('clear')
        print(self.banner)
        print(self.info)
        print("\n\033[1;33mChoose an option:\033[0m")
        print("\033[1;36m1.\033[0m \033[0;37mTraining - Train a new Model\033[0m")
        print("\033[1;36m2.\033[0m \033[0;37mTest - Test an existing Model\033[0m")
        print("\033[1;36m3.\033[0m \033[0;37mCreate Report - Create a Report\033[0m")
        print("\033[1;36m4.\033[0m \033[0;37mExit\033[0m\n")

    def display_training_menu(self):
        os.system('clear')
        print("\033[1;33m" + "="*77 + "\033[0m")
        print(" " * 24 + "TRAINING - OVERVIEW")
        print("\033[1;33m" + "="*77 + "\033[0m\n")
        print("\033[1;36m1.\033[0m \033[0;37mTrain All Optimizers - Train all optimizers for a comparative analysis.\033[0m")
        print("\033[1;36m2.\033[0m \033[0;37mTrain Specific Optimizer(s) - Train one or more selected optimizers.\033[0m")
        print("\033[1;36m3.\033[0m \033[0;37mExit\033[0m\n")

    def display_test_menu(self):
        os.system('clear')
        print("\033[1;33m" + "="*77 + "\033[0m")
        print(" " * 24 + "TEST - OVERVIEW")
        print("\033[1;33m" + "="*77 + "\033[0m\n")
        print("test.json available           train.json available           data.json available\n")
        print("\033[1;36m1.\033[0m \033[0;37mStart Testing and Write results in test_data.json\033[0m")
        print("\033[1;36m2.\033[0m \033[0;37mExit\033[0m\n")

    def display_report_menu(self):
        os.system('clear')
        print("\033[1;33m" + "="*77 + "\033[0m")
        print(" " * 24 + "REPORT - OVERVIEW")
        print("\033[1;33m" + "="*77 + "\033[0m\n")
        print("\033[1;36m1.\033[0m \033[0;37mCreate a Training Report\033[0m")
        print("\033[1;36m2.\033[0m \033[0;37mCreate a Test Report\033[0m")
        print("\033[1;36m3.\033[0m \033[0;37mCreate Test and Training Report\033[0m")
        print("\033[1;36m4.\033[0m \033[0;37mExit\033[0m\n")

    def get_user_choice(self):
        choice = input("\033[1;34m@xleonplayz ➜ /workspaces/LLY-DML/module/src (main) $ \033[0m")
        return choice

    def run(self):
        while True:
            self.display_main_menu()
            choice = self.get_user_choice()
            if choice == "1":
                self.display_training_menu()
                training_choice = self.get_user_choice()
                if training_choice == "1":
                    return "Train_all_optimizers"
                elif training_choice == "2":
                    return "Train_specific_optimizers"
                elif training_choice == "3":
                    continue  # Go back to the main menu

            elif choice == "2":
                self.display_test_menu()
                test_choice = self.get_user_choice()
                if test_choice == "1":
                    return "Start_testing_write_to_test_data"
                elif test_choice == "2":
                    continue  # Go back to the main menu

            elif choice == "3":
                self.display_report_menu()
                report_choice = self.get_user_choice()
                if report_choice == "1":
                    return "Create_train_report"
                elif report_choice == "2":
                    return "Create_test_report"
                elif report_choice == "3":
                    return "Create_test_and_train_report"
                elif report_choice == "4":
                    continue  # Go back to the main menu

            elif choice == "4":
                return "Exit_program"
