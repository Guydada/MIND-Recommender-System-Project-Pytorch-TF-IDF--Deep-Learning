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
        super(NetModel, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
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
        return self.stack(x)

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
        typer.secho(f'{self.__class__.__name__} saved', fg='green')
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
        # get the class name from the model
        class_name = model.__class__.__name__
        typer.secho(f'{class_name} loaded', fg='green')
        return model


class LinearModel(nn.Module):
    """
    NetModel class for the neural network. inherits from NetModel.
    """

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size):
        """
        Initialize the model.
        :param input_size:
        :param output_size:
        """
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 50)
        self.fc4 = nn.PReLU()
        self.fc5 = nn.Linear(50, output_size)
        self.out = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.
        :param x:
        :return: logits
        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return self.out(x)

    def save(self,
             filename: str):
        """
        Save the model to a file.
        :param filename:
        :return: None
        """
        path = Path(filename)
        with click_spinner.spinner('Saving model to {}'.format(path)):
            with path.open('wb') as f:
                torch.save(self, f)
        typer.secho(f'{self.__class__.__name__} saved', fg='green')
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
        name = model.__class__.__name__
        typer.secho(f'{name} loaded', fg='green')
        return model
