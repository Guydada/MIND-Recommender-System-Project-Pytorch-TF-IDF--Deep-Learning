import torch
import torchmetrics
import typer
import pandas as pd
from tqdm import tqdm
import torch.autograd


# Train the net model
###############################################################################
def train_net_model(train_dataloader,
                    model,
                    criterion,
                    optimizer,
                    epochs,
                    limit,
                    mode,
                    device):
    """
    Trains the net model
    :param mode:
    :param device:
    :param limit:
    :param train_dataloader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epochs:
    :return:
    """
    train_loss_df = pd.DataFrame(columns=['loss', 'epoch', 'batch', 'running_loss', 'acc'])
    torch.set_grad_enabled(True)
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('\nEpoch: {}'.format(epoch))
        running_loss = 0.0
        running_acc = 0.0
        for i, data in tqdm(enumerate(train_dataloader)):
            if i == limit and limit is not None:
                break
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            # check if inputs is all zero
            if torch.all(torch.eq(inputs, torch.zeros(inputs.shape, device=device))):  # skip the batch if all zero
                continue
            for param in model.parameters():
                param.grad = None  # zero the parameter gradients

            outputs = model(inputs)  # forward, backward, optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.round().int()
            acc = (preds == labels).float().sum() / len(labels)

            running_loss += loss.item()
            running_acc += acc.item()

            train_loss_df = train_loss_df.append(pd.DataFrame([[loss.item(),
                                                                epoch,
                                                                i,
                                                                loss.item(),
                                                                acc.item()]], columns=['loss',
                                                                                       'epoch',
                                                                                       'batch',
                                                                                       'running_loss',
                                                                                       'acc']))
            if i % 1000 == 0:
                print('\n({}, {}) Loss: {:.4f} | Acc: {:.4f}'.format(epoch,
                                                                     i + 1,
                                                                     running_loss / (i + 1),
                                                                     running_acc / (i + 1)))
            le = i + 1
        print(f'Epoch number:{epoch} | '
              f'Loss: {running_loss / le} | Acc: {running_acc / le}')

    # Waits for everything to finish running
    torch.cuda.synchronize(device=device)
    typer.secho('Done training', fg=typer.colors.GREEN)
    # format tim to hh:mm:ss
    return train_loss_df


def test_proc(test_dataloader,
              model,
              evaluator,
              criterion,
              limit,
              device,
              mode,
              prints):
    """
    Test the net model.
    :param test_dataloader:
    :param model:
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
            outputs = model(test_features)
            test_loss = criterion(outputs, test_labels)
            pred_labels = torch.sigmoid(outputs)
            pred_labels = torch.round(pred_labels).int()
            evaluator.update(i=i,
                             device=device,
                             pred=pred_labels,
                             target=test_labels.int(),
                             loss=test_loss,
                             prints=prints)
            i += 1
    return None
