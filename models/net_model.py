import torch
import typer
from pathlib import Path
import torch.nn as nn
import click_spinner


class NetModel(nn.Module):
    """
    NetModel class for the neural network. inherits from nn.Module.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size):
        """
        Initialize the model.
        :param input_size:
        :param output_size:
        :param hidden_size:
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        super(NetModel, self, ).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        """
        Forward pass of the model.
        :param x:
        :return: logits
        """
        # x = self.flatten(x)
        return self.linear_relu_stack(x)

    def save(self, path):
        """
        Save the model to a file.
        :param path:
        :return: None
        """
        path = Path(path)
        with click_spinner.spinner('Saving model to {}'.format(path)):
            with path.open('wb') as f:
                torch.save(self, f)
        typer.secho(f'NetModel saved', fg='green')
        return None

    @classmethod
    def load(cls, path):
        """
        Load a model from a file.
        :param path: path to the model file (str)
        :return: NetModel instance
        """
        path = Path(path)
        with click_spinner.spinner('Loading model from {}'.format(path)):
            with path.open('rb') as f:
                model = torch.load(f)
        typer.secho(f'NetModel loaded', fg='green')
        return model
