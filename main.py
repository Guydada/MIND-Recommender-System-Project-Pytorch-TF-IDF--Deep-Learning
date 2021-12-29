import pandas as pd
import typer
from train_test_protocol import train_net_model
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import models.net_model
from mind import Mind
import torch
import click
from models.net_model import Model
import torchmetrics
import random

###############################################################################
# Torch Parameters
###############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cuda = torch.cuda.is_available()  # sets cuda availability for model and PyTorch tensors
torch.set_default_dtype(torch.float32)  # sets default dtype for PyTorch tensors - to avoid casting issues
###############################################################################
# Miind full parameters - train and test
###############################################################################
train_parameters = dict(data_path='data',  # path to data
                        max_features=None,  # for tfidf
                        cols='title',  # for tfidf
                        ngram_range=(1, 2),  # for tfidf
                        filename=None,  # for saving. if None, sets default name
                        device=device,  # for torch
                        cuda=cuda,  # for torch
                        data_type='train',  # if set to 'test' no undersampling is done
                        undersample=True,  # manually set to False if you don't want undersampling
                        save=True,  # if set to False, no data is saved
                        pkl_path='pkl',
                        overwrite=True)

test_parameters = dict(data_path='data',  # path to data
                       max_features=None,  # for tfidf
                       cols='title',  # for tfidf
                       ngram_range=(1, 2),  # for tfidf
                       filename=None,  # for saving. if None, sets default name
                       device=device,  # for torch
                       cuda=cuda,  # for torch
                       data_type='test',  # if set to 'test' no undersampling is done
                       undersample=False,  # manually set to False if you don't want undersampling
                       save=True,  # if set to False, no data is saved
                       pkl_path='pkl',
                       overwrite=True)


###############################################################################
# Main Function
###############################################################################
@click.command()
def main(train_params: dict = train_parameters,
         test_params: dict = test_parameters,
         mode: str = 'tfidf',
         train_load: bool = True,  # TODO: change to False
         train_file: str = 'pkl/MIND_train_size_463060.pkl',
         test_load: bool = False,
         test_limit: int = 100,
         test_file: str = 'pkl/MIND_test_size_463060.pkl',
         train_model: bool = True,
         train_limit: int = 1000,
         save_model: bool = True,
         test_model: bool = True,
         model_load: bool = False,
         model_file: str = 'models/model_mind_tfidf_463060.pkl',
         min_df: float = 0.0005,
         k: int = 5,
         sessions: bool = False,
         data_rows: int = None,
         batch_size: int = 1,  # TODO: change to 64
         epochs: int = 50,  # TODO: validate
         lr: float = 0.001,  # TODO: validate
         tfidf_limit: int = 1):  # TODO: change to number of iterations of batch_size
    """
    Main function for the MIND model.
    :param train_params: a dictionary with the parameters for the train data MIND class
    :param test_params: a dictionary with the parameters for the test data MIND class
    :param mode: 'tfidf' or 'model'
    :param train_load: load train data from pkl file (True) or create new data (False)
    :param train_file: path to pkl file with train data
    :param test_load: load test data from pkl file (True) or create new data (False)
    :param test_limit: limit the number of test data iterations (for debugging)
    :param test_file: path to pkl file with test data
    :param train_model: do training (True) or not (False)
    :param train_limit: define the number of train data iterations (for debugging)
    :param save_model: boolean to save the model (True) or not (False)
    :param test_model: boolean to test the model (True) or not (False)
    :param model_load: boolean to load the model (True) or not (False)
    :param model_file: path to the model file
    :param min_df: set the minimum document frequency for the tfidf vectorizer (float, 0 < min_df < 1)
    :param k: top k recommendations to return (int), k=5 is the default
    :param sessions: boolean to use sessions (True) or not (False). If set to True, user's history isn't grouped
    :param data_rows: limit the number of data rows. Affects only Mind.behave and not Mind.news
    :param batch_size: define the batch size for the training, default is 64
    :param epochs: set the number of epochs for the training, default is 50
    :param lr: set the learning rate for the training, default is 0.001
    :param tfidf_limit: define the number of tfidf iterations to use
    :return: None
    """
    if train_params is None:
        train_params = train_parameters
    if train_load:
        train = Mind.load(train_file)
    else:
        train = Mind(mode=mode,
                     min_df=min_df,
                     sessions=sessions,
                     data_rows=data_rows,
                     tfidf_k=k,
                     **train_params)

    # Model mode
    ###############################################################################
    if mode == 'model':
        train_dataloader = DataLoader(train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      drop_last=True)

        if model_load:
            classifier = Model.load(model_file)
        else:
            classifier = Model(input_size=len(train.tfidf.vocabulary_),
                               hidden_size=batch_size,
                               output_size=0)
        classifier.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        train_model = typer.prompt('Do you want to train the model?', default=train_model)
        if train_model:
            loss_df = train_net_model(train_dataloader,
                                      classifier,
                                      criterion,
                                      optimizer,
                                      epochs,
                                      model_file,
                                      train_limit,
                                      device)
            if save_model:
                classifier.save(model_file)
            # export loss df to csv
            loss_df.to_csv(f'loss_df_{train.filename}.csv', index=True)
            typer.echo(f'Loss DataFrame saved to {loss_df}')
        if test_model:
            if test_load:
                test = Mind.load(test_file)
            else:
                test = Mind(mode=mode,
                            min_df=min_df,
                            sessions=sessions,
                            data_rows=data_rows,
                            tfidf_k=k,
                            **test_params)
            metric1 = torchmetrics.F1(average='macro')
            metric2 = torchmetrics.Accuracy()
            metric3 = torchmetrics.Recall(average='macro')
            metric4 = torchmetrics.Precision(average='macro')
            metric5 = torchmetrics.AUC()
            cols = ['batch_num', 'loss', 'f1', 'accuracy', 'recall', 'precision', 'AUC']
            score = pd.DataFrame(columns=cols)
            test_dataloader = DataLoader(test,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=0,
                                         drop_last=True)
            for data, i in tqdm(enumerate(test_dataloader), total=test_limit):
                test_features, test_labels = data
                test_features, test_labels = test_features.to(device), test_labels.to(device)
                preds = classifier(test_features)
                loss = criterion(preds, test_labels)
                metric1.update(preds, test_labels)
                metric2.update(preds, test_labels)
                metric3.update(preds, test_labels)
                metric4.update(preds, test_labels)
                metric5.update(preds, test_labels)
                score.loc[i] = [i, loss.item(), metric1.compute(), metric2.compute(), metric3.compute(),
                                metric4.compute(), metric5.compute()]
            score.to_csv(f'score_{train.filename}.csv', index=False)
            typer.echo(f'Test DataFrame saved to {score}')
        return

    # TFIDF Mode
    ###############################################################################
    elif mode == 'tfidf':
        cols = ['user_rankings',
                'impression_click_rank',  # this is also the impression's id in mind.news df
                'impression_rank_tfidf']
        rank_df = pd.DataFrame(columns=[cols])
        if tfidf_limit is None:
            tfidf_limit = len(train)
        # tfidf_loader = DataLoader(train,
        #                           batch_size=batch_size,
        #                           shuffle=True,
        #                           num_workers=0) # TODO: I found a major problem with this. I will update this later.

        # for i, data in tqdm(enumerate(tfidf_loader, 0), total=tfidf_limit):
        #     if i == tfidf_limit:
        #         break
            user_rankings, impression_click_rank, impression_rank_tfidf = data
            # rank_df = rank_df.append(pd.DataFrame({'user_rankings': user_rankings,
            #                                        'impression_click_rank': impression_click_rank,
            #                                        'impression_rank_tfidf': impression_rank_tfidf},
            #                                       index=[i]))
        # rank_df.to_csv('rank_df.csv', index=False)  # TODO: add metrics to this
        return


if __name__ == '__main__':
    typer.secho('MIND Recommender Engine, by Guy & Guy. Running: ', fg=typer.colors.BLUE)
    main()
    ans = typer.prompt('Finished!, press enter to exit')
    exit()
