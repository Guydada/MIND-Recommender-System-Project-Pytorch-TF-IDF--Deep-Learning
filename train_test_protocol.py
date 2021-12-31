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
    train_loss_df = pd.DataFrame(columns=['loss', 'epoch', 'batch', 'running_loss'])
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('\nEpoch: {}'.format(epoch))
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_dataloader)):
            if i == limit and limit is not None:
                break
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = classifier.forward(inputs)  # forward, backward, optimize
            outputs = outputs.squeeze().mean(axis=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loss_df = train_loss_df.append(pd.DataFrame([[loss.item(),
                                                                epoch,
                                                                i,
                                                                running_loss]], columns=['loss',
                                                                                         'epoch',
                                                                                         'batch',
                                                                                         'running_loss']))
            if i % 2000 == 1999:
                print(f'\nLoss: {loss.item()}')
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
    return train_loss_df


def test_proc(test_dataloader,
              classifier,
              evaluator,
              criterion,
              limit,
              device,
              prints):
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
        for i, data in tqdm(enumerate(test_dataloader)):
            if i == limit and limit is not None:
                break
            test_features, test_labels = data
            test_features, test_labels = test_features.float().to(device), test_labels.float().to(device)

            preds = classifier(test_features)
            output = preds.squeeze().mean(axis=1).to(device)
            test_loss = criterion(output, test_labels)

            top_classes = torch.exp(output).squeeze().int().to(device)
            evaluator.update(i=i,
                             device=device,
                             pred=top_classes,
                             target=test_labels.int(),
                             loss=test_loss,
                             prints=prints)
            i += 1
    return None
