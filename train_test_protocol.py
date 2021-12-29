import torch
import typer
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# Train the net model
###############################################################################
def train_net_model(train_dataloader,
                    classifier,
                    criterion,
                    optimizer,
                    epochs,
                    file,
                    limit,
                    device):
    """
    Trains the net model
    :param train_dataloader:
    :param classifier:
    :param criterion:
    :param optimizer:
    :param epochs:
    :return:
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss_df = pd.DataFrame(columns=['loss', 'epoch', 'batch', 'running_loss'])
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('\nEpoch: {}'.format(epoch))
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_dataloader, 0)):
            if limit is not None and i == limit:
                break
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = classifier(inputs)  # forward, backward, optimize
            outputs = outputs.mean(dim=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_df = loss_df.append(pd.DataFrame([[loss.item(),
                                                    epoch,
                                                    i,
                                                    running_loss]], columns=['loss',
                                                                             'epoch',
                                                                             'batch',
                                                                             'running_loss']))
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    end.record()
    classifier.save(file)
    # Waits for everything to finish running
    torch.cuda.synchronize(device=device)
    typer.secho('Done training', fg=typer.colors.GREEN)
    typer.secho('Time taken: {}'.format(start.elapsed_time(end)), fg='cyan')
    return loss_df
