import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, mode='min', delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.mode = mode

        if self.mode == 'min':
            self.criterion = np.less
            self.delta = - delta
            self.best_score = np.Inf

        elif self.mode == 'max':
            self.criterion = np.greater
            self.delta = delta
            self.best_score = np.NINF

        else:
            raise ValueError("mode only takes as value in input 'min' or 'max'")

    def __call__(self, score, model, filename):
        """Determines if the score is the best and saves the model if so.
           Also manages early stopping.
        Arguments:
            score (float): Value of the metric or loss.
        """
        if np.isinf(self.best_score):
            self.best_score = score
            torch.save(model.state_dict(), filename)

        elif self.criterion(score, self.best_score + self.delta):
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), filename)

        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print('Early stopping counter is exceeded')
