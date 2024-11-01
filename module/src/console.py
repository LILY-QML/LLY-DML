# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Project: LILY-QML
# Version: 1.6 LLY-DML
# Author: Leon Kaiser
# Contact: info@lilyqml.de
# Website: www.lilyqml.de
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import logging

# Set up logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'lily_qml_console.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

class Colors:
    RED = '\033[91m'
    RESET = '\033[0m'

class Console:
    def __init__(self, train_exists, train_content, data_exists, data_content, optimizers, missing_optimizers):
        """
        Initialisiert die Console-Klasse.

        :param train_exists: Boolean, ob train.json vorhanden ist.
        :param train_content: Inhalt von train.json.
        :param data_exists: Boolean, ob data.json vorhanden ist.
        :param data_content: Inhalt von data.json.
        :param optimizers: Liste der verfügbaren Optimierer.
        :param missing_optimizers: Liste der fehlenden Optimierer.
        """
        self.train_exists = train_exists
        self.train_content = train_content
        self.data_exists = data_exists
        self.data_content = data_content
        self.optimizers = optimizers
        self.missing_optimizers = missing_optimizers

        logger.info("Console initialized with data and train file statuses.")

    def clear_console(self):
        """Löscht die Konsole."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_banner(self):
        banner = r"""
 ____   __  __   _
|  _ \ |  \/  | | |
| | | || |\/| | | |
| |_| || |  | | | |___
|____/ |_|  |_| |_____|

        """
        info = """
LLY-DML - Teil des LILY Projekts - Version 1.6 Beta - info@lilyqml.de - lilyqml.de
        """
        print(banner)
        print(info)
        logger.info("Displayed LILY-QML banner.")

    def run_main_menu(self):
        """Zeigt das Hauptmenü an und gibt die Auswahl zurück."""
        while True:
            self.clear_console()
            self.show_banner()
            if not self.train_exists:
                logger.warning("train.json file not found.")
                print(f"{Colors.RED}train.json wurde nicht gefunden. Nur neues Training möglich.{Colors.RESET}\n")

            print("\n---\n")
            print("Wähle eine Option:")
            print("1. Informationen - Zeige mehr Informationen über die Referenzdateien")
            print("2. Training - Wähle eine Trainingsmethode")
            if self.train_exists:
                print("3. Bericht - Berichtoptionen")
                print("4. Beenden - Beende LLY-DML")
            else:
                print(f"{Colors.RED}3. Bericht - Berichtoptionen (nicht verfügbar){Colors.RESET}")
                print("4. Beenden - Beende LLY-DML")

            choice = input("Bitte wähle eine Option (1-4): ")

            if choice == '1':
                logger.info("User selected to view information.")
                selection = self.run_information_menu()
                if selection:
                    return selection
            elif choice == '2':
                logger.info("User selected training menu.")
                selection = self.run_training_menu()
                if selection:
                    return selection
            elif choice == '3':
                if self.train_exists:
                    logger.info("User selected to view report options.")
                    selection = self.run_report_menu()
                    if selection:
                        return selection
                else:
                    logger.warning("Report option not available since train.json is missing.")
                    print(f"{Colors.RED}Option 3 ist derzeit nicht verfügbar.{Colors.RESET}")
                    input("Drücke Enter, um fortzufahren.")
            elif choice == '4':
                logger.info("User exited the program.")
                print("Programm wird beendet. Auf Wiedersehen!")
                sys.exit()
            else:
                logger.warning("Invalid input in main menu.")
                print("Ungültige Eingabe. Bitte versuche es erneut.")
                input("Drücke Enter, um fortzufahren.")

    def run_training_menu(self):
        """Zeigt das Trainingsmenü an und gibt die Auswahl zurück."""
        while True:
            self.clear_console()
            print("\n---\n")
            print("TRAINING")
            print("Wähle eine der folgenden Optionen:")
            print("1. Training aller Optimierer")
            print("2. Training spezifischer Optimierer")
            if self.train_exists:
                print("3. Training verbleibender Optimierer")
                print("4. Zurück zum Hauptmenü")
            else:
                print(f"{Colors.RED}3. Training verbleibender Optimierer (nicht verfügbar){Colors.RESET}")
                print("4. Zurück zum Hauptmenü")

            choice = input("Bitte wähle eine Option (1-4): ")

            if choice == '1':
                logger.info("User selected to train all optimizers.")
                return {"action": "train_all_optimizers"}
            elif choice == '2':
                logger.info("User selected specific optimizers to train.")
                selected = self.select_from_list(self.optimizers, "verfügbare Optimierer")
                if selected:
                    return {"action": "train_specific_optimizers", "optimizers": selected}
            elif choice == '3':
                if self.train_exists:
                    logger.info("User selected to train missing optimizers.")
                    selected = self.select_from_list(self.missing_optimizers, "fehlende Optimierer")
                    if selected:
                        return {"action": "train_missing_optimizers", "optimizers": selected}
                else:
                    logger.warning("Missing optimizers option not available due to missing train.json.")
                    print(f"{Colors.RED}Option 3 ist derzeit nicht verfügbar.{Colors.RESET}")
                    input("Drücke Enter, um fortzufahren.")
            elif choice == '4':
                logger.info("User returned to main menu.")
                return None  # Zurück zum Hauptmenü
            else:
                logger.warning("Invalid input in training menu.")
                print("Ungültige Eingabe. Bitte versuche es erneut.")
                input("Drücke Enter, um fortzufahren.")

    def show_data_info(self):
        """Zeigt Informationen aus data.json an."""
        self.clear_console()
        print("\n---\n")
        if self.data_exists:
            logger.info("Displaying data.json content.")
            print("- data.json gefunden")
            print(f"Inhalt von data.json:\n{self.data_content}")
        else:
            logger.warning("data.json file not found.")
            print("- data.json nicht gefunden.")
        input("\nDrücke Enter, um fortzufahren.")

    def show_train_info(self):
        """Zeigt Informationen aus train.json an."""
        self.clear_console()
        print("\n---\n")
        if self.train_exists:
            logger.info("Displaying train.json content.")
            print("- train.json gefunden")
            print(f"Inhalt von train.json:\n{self.train_content}")
        else:
            logger.warning("train.json file not found.")
            print("- train.json nicht gefunden.")
        input("\nDrücke Enter, um fortzufahren.")

    def run_report_menu(self):
        """Zeigt das Berichtsmenü an und gibt die Auswahl zurück."""
        while True:
            self.clear_console()
            print("\n---\n")
            print("BERICHT")
            print("1. Bericht basierend auf bestehenden Daten erstellen")
            print("2. Zurück zum Hauptmenü")
            choice = input("Bitte wähle eine Option (1-2): ")

            if choice == '1':
                logger.info("User selected to create a report.")
                return {"action": "create_report"}
            elif choice == '2':
                logger.info("User returned to main menu from report menu.")
                return None  # Zurück zum Hauptmenü
            else:
                logger.warning("Invalid input in report menu.")
                print("Ungültige Eingabe. Bitte versuche es erneut.")
                input("Drücke Enter, um fortzufahren.")

    def select_from_list(self, items, description):
        """
        Zeigt eine Liste von Items an und ermöglicht dem Benutzer, eine Auswahl zu treffen.

        :param items: Liste der verfügbaren Items.
        :param description: Beschreibung der Items (für die Anzeige).
        :return: Liste der ausgewählten Items.
        """
        if not items:
            logger.info(f"No {description} available for selection.")
            print(f"Keine {description} verfügbar.")
            input("Drücke Enter, um fortzufahren.")
            return None

        while True:
            self.clear_console()
            print(f"\n---\n")
            print(f"Wähle aus den folgenden {description}:")
            for idx, item in enumerate(items, start=1):
                print(f"{idx}. {item}")
            print(f"{len(items)+1}. Zurück zum Trainingsmenü")

            choices = input(f"Bitte wähle eine oder mehrere Optionen (z.B. 1,3): ")
            selected = choices.split(',')
            selected_items = []
            try:
                for choice in selected:
                    idx = int(choice.strip())
                    if 1 <= idx <= len(items):
                        selected_items.append(items[idx - 1])
                    elif idx == len(items) + 1:
                        logger.info("User returned to training menu without selecting items.")
                        return None  # Zurück zum Trainingsmenü
                    else:
                        logger.warning(f"Invalid option {choice} in selection list.")
                        print(f"Ungültige Option: {choice}")
                        input("Drücke Enter, um fortzufahren.")
                        break
                else:
                    if selected_items:
                        logger.info(f"User selected items: {selected_items}")
                        return selected_items
                    else:
                        logger.warning("No valid optimizers selected.")
                        print("Keine gültigen Optimierer ausgewählt.")
                        input("Drücke Enter, um fortzufahren.")
            except ValueError:
                logger.warning("Invalid input in list selection.")
                print("Ungültige Eingabe. Bitte gib Zahlen durch Kommas getrennt ein.")
                input("Drücke Enter, um fortzufahren.")
