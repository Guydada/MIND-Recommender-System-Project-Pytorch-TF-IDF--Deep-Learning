import train_test_protocol as ttp
import torch.nn as nn
import mind
import torch
import models.net_model as mod
import utils.evaluate as ev
import datetime
import typer
from torch.utils.data import DataLoader
import torch.autograd
import torch.backends.cudnn

VERSION = "0.1"

###############################################################################
# File paths
model_path = "models/model.pkl"
train_file = "pkl/MindTensorDot_train_cols_title_size_463060.pt"
test_file = "pkl/"
tfidf_file = "pkl/MindTF_IDF_train_size_463060.pkl"
# crete a formatted time stamp of the current date and time
time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
###############################################################################
# Torch Parameters
###############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cuda = torch.cuda.is_available()  # sets cuda availability for model and PyTorch tensors
torch.set_default_dtype(torch.float32)  # sets default dtype for PyTorch tensors - to avoid casting issues
lr = 0.001  # learning rate
hidden_layer = 64  # number of hidden nodes in the network. This is chosen arbitrarily.
###############################################################################
# Mind full parameters - train and test
###############################################################################
tfidf_params = dict(data_path='data',  # path to data
                    max_features=None,  # max number of features to use
                    cols=['title', 'category', 'abstract'],  # for tfidf. can add: ['title, 'abstract', 'category', 'subcategory']
                    ngram_range=(1, 2),  # for tfidf
                    filename=None,  # for saving. if None, sets default name
                    data_type='train',  # if set to 'test' no undersampling is done
                    undersample=False,  # manually set to False if you don't want undersampling
                    save=False,  # if set to False, no data is saved
                    overwrite=False,
                    data_rows=None,  # if set to None, all rows are used
                    tfidf_k=5,
                    sessions=False,
                    pkl_path='pkl',  # limit the number of rows read from the behaviour.tsv file only
                    min_df=0.0005,
                    group=True)  # for tfidf

train_params = dict(data_path='data',  # path to data
                    max_features=2500,  # for tfidf
                    cols='title',  # for tfidf
                    ngram_range=(1, 2),  # for tfidf vectorizer, ['title, 'abstract', 'category', 'subcategory']
                    filename=None,  # for saving. if None, sets default name
                    device=device,  # for torch
                    cuda=cuda,  # for torch
                    data_rows=None,
                    data_type='train',  # if set to 'test' no undersampling is done
                    undersample=True,  # manually set to False if you don't want undersampling
                    save=True,
                    pkl_path='pkl',  # if set to False, no data is saved
                    overwrite=True,
                    min_df=0.0005,
                    sessions=False,
                    group=False)

test_params = train_params.copy()
test_params['filename'] = None
test_params['data_type'] = 'test'
test_params['undersample'] = False
test_params['save'] = False
test_params.pop('max_features')


###############################################################################
# Main Function
###############################################################################


def main(
        epochs: int = typer.Option(50, '--epochs', '-e', help='number of epochs'),
        tr_load: bool = typer.Option(False, '--tr_load', '-t', help='if True, loads train data', show_default=True,
                                     flag_value=True),
        te_load: bool = typer.Option(False, '--te_load', '-e', help='if True, loads test data'),
        mo_load: bool = typer.Option(False, '--mo_load', '-m', help='if True, loads model'),
        hidden: int = typer.Option(hidden_layer, '--hidden', '-h', help='hidden layer size'),
        tf_load: bool = typer.Option(False, '--tf_load', '-f', help='if True, loads tfidf'),
        min_df: float = typer.Option(0.005, '--min_df', '-d', help='min_df for tfidf'),
        train_model: bool = typer.Option(True, '--train_model', '-m', help='if True, trains model'),
        batch: int = typer.Option(64, '--batch', '-b', help='batch size, default 64'),
        mode: str = typer.Option(..., prompt='Mode', help='tfidf or model', case_sensitive=False),
        model_type: str = typer.Option('linear', help='model type: [\'linear\', \'net\']', case_sensitive=False),
        limit: int = typer.Option(None, '--limit', '-l', help='limits tfidf/train/test data iterations'),
        data_rows: int = typer.Option(None, '--rows', '-n', help='Limit dataset (behave) length')) -> \
        None:
    """
    Main Project Function
    """
    # Model mode
    ###############################################################################
    if data_rows is not None:
        train_params['data_rows'] = data_rows
        test_params['data_rows'] = data_rows
        tfidf_params['data_rows'] = data_rows
    if mode == 'model':
        mind_type = mind.MindTensorMul if model_type == 'net' else mind.MindTensorDot
        if tr_load:
            train = mind_type.load(train_file, mode='torch')
        else:
            train = mind_type(**train_params)
        train_dataloader = DataLoader(train,
                                      batch_size=batch,
                                      shuffle=True,
                                      num_workers=0,
                                      drop_last=True)
        feat_num = train.vocab_size
        if model_type is None:
            model_type = typer.prompt('Choose a model: ', show_choices=True)
        if model_type == 'net':
            model = mod.NetModel(input_size=feat_num,
                                 hidden_size=hidden,
                                 output_size=2)
            criterion = nn.BCEWithLogitsLoss()
        elif model_type == 'linear':
            model = mod.LinearModel(input_size=1,
                                    output_size=1,
                                    hidden_size=25)
            criterion = nn.BCELoss()
        if mo_load:
            model = model.load(model_path)
        model.to(device)
        if cuda:
            torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        loss_df = ttp.train_net_model(train_dataloader=train_dataloader, model=model, criterion=criterion,
                                      optimizer=optimizer, epochs=epochs, limit=limit, mode=model_type, device=device)
        model.save(f'pkl/model_{train.filename}_{time_stamp}.pkl')
        loss_df.to_csv(f'results/loss_df_{train.filename}_{time_stamp}.csv', index=True)
        typer.secho(f'Loss DataFrame saved to results/loss_df', fg=typer.colors.GREEN)
        if te_load:
            test = mind_type.load(test_file,
                                  mode='torch')
        else:
            test = mind_type(max_features=feat_num,
                             **test_params)
        test_dataloader = DataLoader(test,
                                     batch_size=batch,
                                     shuffle=True,
                                     num_workers=0,
                                     drop_last=True)
        evaluator = ev.NetEvaluator()
        model.eval()
        ttp.test_proc(test_dataloader, model, evaluator, criterion, limit=limit, device=device, mode=model_type,
                      prints=False)
        evaluator.save(ts=time_stamp,
                       filename=test.filename)
        typer.secho('Done with evaluation of test set', fg=typer.colors.GREEN)
        return

    # TFIDF Mode
    ###############################################################################
    elif mode == 'tfidf':
        if tf_load:
            tfidf = mind.MindTF_IDF.load(tfidf_file,
                                         mode='pickle')
        else:
            tfidf = mind.MindTF_IDF(**tfidf_params)
        tfidf.run(limit=limit,  # How many users to plot
                  shuffle=False)


if __name__ == '__main__':
    typer.secho(f'MIND Recommender Engine, by Guy & Guy, V-{VERSION} Running: ', fg=typer.colors.BLUE)
    typer.run(main)
    ans = typer.prompt('Finished!, press enter to exit')
    exit()
