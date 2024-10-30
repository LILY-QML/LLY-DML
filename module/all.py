class All:
    def __init__(self, optimizers, data, train_data):
        """
        Initializes the All class with the list of optimizers and required data.
        
        :param optimizers: List of optimizers from the JSON file.
        :param data: The data loaded from data.json.
        :param train_data: The data loaded from train.json.
        """
        self.optimizers = optimizers
        self.data = data
        self.train_data = train_data

    def run(self):
        """
        Run method to execute training on all optimizers.
        """
        print(f"Running training for all optimizers: {', '.join(self.optimizers)}")
        for optimizer in self.optimizers:
            # Here you can add the logic to run training for each optimizer.
            print(f"Training optimizer: {optimizer}")
            # You can fetch specific configurations from data and train_data if needed
            # optimizer.train(self.data, self.train_data)  # Example call
        print("Training for all optimizers completed.")