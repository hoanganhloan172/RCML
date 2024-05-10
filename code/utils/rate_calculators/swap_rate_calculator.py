class SwapRateCalculator:
    def __init__(self, start, end, epochs):
        self.start = start
        self.end = end
        self.epochs = epochs
        self.static_epoch_perc = 0.1 # For the last 10% of epochs, rate is set to 1.
        self.static_epochs = [self.epochs-epoch for epoch in range(int(self.epochs * self.static_epoch_perc))]
        self.swap_gap = self.end - self.start
        self.dynamic_epochs = self.epochs - len(self.static_epochs)

    def calculate_swap_rate(self, epoch):
        '''
        Returns the swap rate being between start and end
        '''
        if epoch in self.static_epochs:
            return self.end / 100

        # epoch-1 because epochs start with 1 in run_together.py
        return int(self.start + ((epoch-1) * (self.swap_gap / self.epochs))) / 100
