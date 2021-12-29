import os
from tqdm import tqdm
from collections import OrderedDict
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from utils.load_preprocess import *
import torch
from scipy.sparse import vstack, csr_matrix
from typing import Union
import numpy as np
import click_spinner
import typer
import pickle
from utils.tensorize import *


class Mind:
    """
    Mind dataset class. Inherits from torch.utils.data.Dataset.
    """

    def __init__(self,
                 mode: str,
                 data_path: str = 'data',
                 max_features: int = None,
                 cols: list or str = 'title',
                 ngram_range: tuple = (1, 2),
                 min_df: float = 0.5,
                 tfidf_k: int = 5,
                 filename: str = None,
                 device: str = 'cpu',
                 cuda: bool = False,
                 data_type: str = 'train',
                 undersample: bool = True,
                 save: bool = True,
                 sessions: bool = False,
                 pkl_path: str = 'pkl',
                 overwrite: bool = False,
                 data_rows: int = None):
        """
        Initialize Mind dataset
        :param mode: can be 'tfidf' or 'model'
        :param data_path: data path
        :param max_features: features to use in tfidf
        :param cols: cols to use in tfidf
        :param ngram_range: for tfidf vectorizer
        :param min_df: for tfidf vectorizer
        :param tfidf_k: select how much news to recommend using tfidf for each user
        :param filename: leave 'None' if you want to use default filename, or select your own
        :param device: device to use, default is 'cpu'
        :param cuda: default is 'False'
        :param data_type: 'train' or 'test'
        :param undersample: set to 'True' if you want to undersample the dataset. Train set will not be undersampled
        :param save: set to 'True' if you want to save the dataset. default is 'True'
        :param sessions: set to 'True' if you want to use sessions. default is 'False'
        :param pkl_path: Path to save the Mind Class object
        """
        with typer.progressbar(f'Loading {data_type} data',
                               length=1000,
                               label='Loading MIND Class',
                               show_eta=True,
                               show_percent=True) as bar:
            inter = 200
            bar.label = 'Setting attributes'
            self.pkl_path = pkl_path
            self.cuda = cuda
            self.mode = mode
            if self.mode not in ['tfidf', 'model']:
                typer.echo(f'\n{self.mode} is not a valid mode')
                typer.echo('Please choose between tfidf and model')
                typer.echo('Exiting...')
                exit()
            self.text_cols = cols
            self.data_path = data_path
            self.data_type = data_type
            self.ngram_range = ngram_range
            self.max_features = max_features
            self.k = tfidf_k
            self.overwrite = overwrite
            if self.data_type not in ['train', 'test']:
                raise ValueError('data_type must be train or test')
            self.device = device
            self.sessions = sessions
            # Validate data_type
            if self.data_type == 'test' and undersample:
                self.undersample = False
                typer.echo(f'Test data, undersample changed to False', color=typer.colors.YELLOW)
            else:
                self.undersample = undersample
            bar.update(inter)
            bar.label = "Loading data"
            # Load behave data
            self.behave, self.labels, self.clicks = load_behave(undersample=self.undersample,
                                                                data_path=self.data_path,
                                                                data_type=self.data_type,
                                                                sessions=self.sessions,
                                                                data_rows=data_rows)
            # Load news data
            self.news = load_news(cols=self.text_cols,
                                  data_path=self.data_path,
                                  data_type=self.data_type,
                                  preprocess=True)
            self.new_num = len(self.news)
            self._rearrange_news()
            bar.update(inter)
            bar.label = "Running TF-IDF vectorizer"
            # Initialize TF-IDF and apply to news data
            self.tfidf = TfidfVectorizer(stop_words='english',
                                         ngram_range=self.ngram_range,
                                         max_features=self.max_features,
                                         lowercase=False,
                                         use_idf=True,
                                         strip_accents='ascii',
                                         min_df=min_df)
            self.matrix = self._tfidf_news(self.news)
            self.dic = self.news.reset_index().set_index('news_id')['index'].to_dict()
            # Re-index news data
            self._re_index()
            bar.update(inter)
            # delete unwanted attributes
            del self.dic, self.clicks
            # Set filename
            if filename is None:
                self.filename = f'MIND_{self.data_type}_size_{len(self)}.pkl'
            else:
                self.filename = filename
            if save:
                bar.update(inter)
                bar.label = "Saving MIND Class object"
                self.save()
            else:
                bar.update(inter)
                bar.label = "Finalizing"
            bar.label = "Done"
            bar.update(inter)

    def __len__(self) -> int:
        """
        Return length of dataset - number of rows in behave data
        :return:
        """
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Get item from dataset: for tfidf mode returns the place of the impression in user ranking,
        for model mode returns tensor multiplication of impressions and user_vector
        :param index:
        :return: if mode is tfidf: (user_ranking, impressions_id(as news_id))
                 if mode is model: (user_vector * impression_vector, impression label)
        """
        # Used only for model mode
        if self.mode == 'model':
            user_vector = torch.tensor(csr_matrix.todense(self.get_user_vector(index,
                                                                               csr=True)),
                                       device=self.device)
            news_vector = torch.tensor(csr_matrix.todense(self.get_impression_vector(index,
                                                                                     csr=True)),
                                       device=self.device)
            features = torch.mul(news_vector, user_vector) # for readability purposes, hence the naming of the variable
            labels = self.labels.iloc[index] # same as above, to fit naming in model
            labels = torch.tensor(labels,
                                  device=self.device)
            return features, labels

        # This part will be used for TFIDF mode only
        elif self.mode == 'tfidf':
            user_vector = self.get_user_vector(index, csr=True)
            user_ranking = self.get_user_ranking(user_vector, k=None)  # this k is not the self.k, if set - only k
            # ranked news will be returned
            # get the k first indices in user_ranking
            top_k = list(user_ranking.keys())[:self.k]
            impression_index = self.behave.iloc[index]['impressions'] # the index is also the click rank
            impression_rank_tfidf = list(user_ranking.keys()).index(impression_index)
            typer.secho(
                f'\nUser: \t {self.behave.iloc[index]["user_id"]}, Impression index: \t {impression_index}\n'
                f'Rank of news by clicks:           \t {impression_index}\t/\t{self.new_num}\n'
                f'Rank by TF-IDF Cosine Similarity: \t {impression_rank_tfidf}\t/\t{self.new_num}',
                color=typer.colors.BRIGHT_CYAN)
            return user_ranking, impression_index, impression_rank_tfidf
        else:
            raise ValueError(f'Wrong mode: {self.mode}')

    def _rearrange_news(self):
        """
        Rearrange self.news by clicking popularity
        :return:
        """
        self.news['clicks'] = self.clicks['label']
        self.news = self.news.fillna(0)
        self.news = self.news.sort_values(by='clicks', ascending=False).reset_index(drop=True)

    def _tfidf_news(self,
                    df: pd.DataFrame,
                    col_name: str = 'text') -> pd.DataFrame:
        """
        Apply TF-IDF to news data
        :param df: self.news
        :param col_name: 'text' by default - can be changed to custom column name
        :return: tfidf_matrix (sparse_matrix)
        """
        tfidf_matrix = self.tfidf.fit_transform(df[col_name])
        return tfidf_matrix

    def _re_index(self):
        """
        Re-index self.behave by self.dic
        :return:
        """
        self.behave['impressions'] = self.behave['impressions'].map(self.dic)
        self.behave['history'] = self.behave['history'].apply(lambda x: [self.dic[i] for i in x])

    def get_user_vector(self,
                        index: int,
                        csr: bool = True) -> Union[csr_matrix, ndarray]:
        """
        Get the session vectors for a given user, return a sparse matrix
        :param csr: whether to return a csr_matrix or a numpy array
        :param index: self.behave index for iterator
        :return: csr_matrix
        """
        hist_ind = self.behave.iloc[index]['history']
        user_mat = self.matrix[hist_ind]
        user_mat = np.mean(user_mat, axis=0)
        if csr:
            return csr_matrix(user_mat)
        else:
            return user_mat

    def get_impression_vector(self,
                              index: int,
                              csr: bool = True) -> Union[csr_matrix, ndarray]:
        """
        Get the impression vectors for a given user, return a sparse matrix.
        Receives index from self.behave['impressions']
        :param csr:
        :param index: self.behave index for iterator
        :return: csr_matrix
        """
        ind = self.behave.iloc[index]['impressions']
        news_mat = self.matrix[[ind]]
        if csr:
            return csr_matrix(news_mat)
        else:
            return news_mat

    def get_user_ranking(self,
                         user_vec: csr_matrix,
                         k: int = None) -> OrderedDict:
        """
        Get user k sorted rankings by index in self.behave. if k is not specified, return all sorted
        :return:
        """
        tfidf_matrix_with_user = vstack((self.matrix, user_vec))
        cosine_sim = cosine_similarity(tfidf_matrix_with_user, tfidf_matrix_with_user)
        sim_scores = list(enumerate(cosine_sim[len(cosine_sim) - 1]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: k]
        user_ranking = OrderedDict(sim_scores)
        return user_ranking

    def get_rank_in_user(self,
                         index: int) -> int:
        """
        Get the rank of the impression in the user's ranking
        :param index: self.behave index for iterator
        :return:
        """
        pass

    def get_base_line(self,
                      percentile: float = 0.5) -> list:
        """
        Get the baseline recommendation for all users
        :return:
        """
        pass  # TODO: use value_counts to get the most common value in the dataset

    def save(self) -> None:
        """
        Save the MIND object to disk
        :return: None
        """
        path = Path(f'{self.pkl_path}/{self.filename}')
        if os.path.exists(path) and not self.overwrite:
            beep()
            typer.echo(f'{path} already exists')
            ans = typer.confirm(f'Overwrite?', default=False)
            if ans:
                typer.echo(f'Overwriting')
            else:
                typer.echo('Skipping save')
                return None
        with open(path, 'wb') as f:
            torch.save(obj=self, f=f)
        return None

    @classmethod
    def load(cls,
             filename: str) -> 'Mind':
        """
        Load a MIND object from disk directly
        :param filename:
        :return:
        """
        with click_spinner.spinner(f'Loading MIND {filename}'):
            path = Path(filename)
            with open(path, 'rb') as f:
                mind = torch.load(f)
            typer.secho(f'MIND {filename} loaded', color=typer.colors.BRIGHT_GREEN)
        return mind


class Mind_tensor(Mind, Dataset): # TODO: implement
    """
    MIND class for Pytorch
    """
    def __init__(self):
        super().__init__()

    def save(self) -> None:
        """
        Save the MIND object to disk
        :return: None
        """
        path = Path(f'{self.pkl_path}/{self.filename}')
        if os.path.exists(path) and not self.overwrite:
            beep()
            typer.echo(f'{path} already exists')
            ans = typer.confirm(f'Overwrite?', default=False)
            if ans:
                typer.echo(f'Overwriting')
            else:
                typer.echo('Skipping save')
                return None
        with open(path, 'wb') as f:
            torch.save(obj=self, f=f)
        return None

    @classmethod
    def load(cls,
             filename: str) -> 'Mind':
        """
        Load a MIND_tensor object from disk directly
        :param filename:
        :return:
        """
        with click_spinner.spinner(f'Loading MIND_tensor {filename}'):
            path = Path(filename)
            with open(path, 'rb') as f:
                mind = torch.load(f)
            typer.secho(f'MIND {filename} loaded', color=typer.colors.BRIGHT_GREEN)
        return mind