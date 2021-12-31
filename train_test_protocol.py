import torch
import typer
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader


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
    :param device:
    :param limit:
    :param file:
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
        for i in tqdm(range(limit)):
            inputs, labels = next(iter(train_dataloader))  # get the inputs; data is a list of [inputs, labels]
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
            i += 1
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize(device=device)
    typer.secho('Done training', fg=typer.colors.GREEN)
    tim = start.elapsed_time(end)
    # format tim to hh:mm:ss
    tim = str(tim // 3600) + ':' + str((tim % 3600) // 60) + ':' + str(tim % 60)
    typer.secho(f'Time taken: {tim}', fg='cyan')
    return loss_df


def test_proc(test_dataloader,
              classifier,
              evaluator,
              # criterion,
              limit,
              device):
    """
    Test the net model.
    :param test_dataloader:
    :param classifier:
    :param evaluator:
    :param criterion:
    :param limit:
    :param device:
    :return:
    """
    with torch.no_grad():
        for i in tqdm(range(limit)):
            test_features, test_labels = next(iter(test_dataloader))
            test_features, test_labels = test_features.to(device), test_labels.to(device)
            preds = classifier(test_features)
            # loss = criterion(preds,
            #                  test_labels)
            evaluator.update(i,
                             preds,
                             test_labels,
                             # loss,
                             prints=True)
            i += 1
    return None
