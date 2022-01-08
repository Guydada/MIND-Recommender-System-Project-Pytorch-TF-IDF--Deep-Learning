import os
import pickle
from collections import OrderedDict
from typing import Union
import click_spinner
import torch
import typer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import utils.load_preprocess as lp
import utils.tensorize as tz


class Mind:
    """
    Mind dataset class. Uses for loading and preprocessing data.
    """

    def __init__(self,
                 data_path: str = 'data',
                 max_features: int = None,
                 cols: list or str = 'title',
                 ngram_range: tuple = (1, 2),
                 filename: str = None,
                 data_type: str = 'train',
                 undersample: bool = True,
                 save: bool = True,
                 overwrite: bool = False,
                 data_rows: int = None,
                 pkl_path: str = 'pkl',
                 min_df: float = 0.005,
                 sessions: bool = False,
                 group: bool = False):
        """
        Initialize Mind dataset
        :param data_path: data path
        :param max_features: features to use in tfidf
        :param cols: cols to use in tfidf
        :param ngram_range: for tfidf vectorizer
        :param min_df: for tfidf vectorizer
        :param filename: leave 'None' if you want to use default filename, or select your own
        :param data_type: 'train' or 'test'
        :param undersample: set to 'True' if you want to undersample the dataset. Train set will not be undersampled
        :param save: set to 'True' if you want to save the dataset. default is 'True'
        :param sessions: set to 'True' if you want to use sessions. default is 'False'
        """
        label = self.__class__.__name__
        inter = 200
        with typer.progressbar(f'Loading {label} data',
                               length=1000,
                               label=f'Creating {label} Class',
                               show_eta=True,
                               show_percent=True) as bar:
            bar.label = 'Setting attributes'
            self.group = group
            self.pkl_path = pkl_path
            self.text_cols = cols
            self.data_path = data_path
            self.data_type = data_type
            self.ngram_range = ngram_range
            self.max_features = max_features
            self.overwrite = overwrite
            self.mode = 'pickle' if label in ['Mind', 'MindTF_IDF'] else 'torch'
            if self.data_type not in ['train', 'test']:
                raise ValueError('data_type must be train or test')
            self.sessions = sessions
            if self.data_type == 'test' and undersample:
                self.undersample = False
                typer.echo(f'\nTest data chosen, undersample changed to False', color=typer.colors.YELLOW)
            else:
                self.undersample = undersample
            bar.update(inter)
            bar.label = "Loading data"
            self.behave, self.labels, self.clicks = lp.load_behave(undersample=self.undersample,
                                                                   data_path=self.data_path,
                                                                   data_type=self.data_type,
                                                                   sessions=self.sessions,
                                                                   data_rows=data_rows,
                                                                   group=self.group)
            self.news = lp.load_news(cols=self.text_cols,
                                     data_path=self.data_path,
                                     data_type=self.data_type,
                                     preprocess=True)
            self.new_num = len(self.news)
            self._rearrange_news()
            bar.update(inter)
            bar.label = "Running TF-IDF vectorizer"
            self.tfidf = TfidfVectorizer(stop_words='english',
                                         ngram_range=self.ngram_range,
                                         max_features=self.max_features,
                                         lowercase=False,
                                         use_idf=True,
                                         strip_accents='ascii',
                                         min_df=min_df)
            self.matrix = self._tfidf_news(self.news)
            self.dic = self.news.reset_index().set_index('news_id')['index'].to_dict()
            self._re_index()
            bar.update(inter)
            del self.dic, self.clicks
            if self.data_type == 'train' and not self.group:
                bar.label = f"Dropping {data_type} {label} duplicates from behave"
                self.behave.drop_duplicates(subset=['user_id', 'impressions'], inplace=True)
                self.labels = self.labels.loc[self.behave.index]
                self.behave.reset_index(drop=True, inplace=True)
                self.labels.reset_index(drop=True, inplace=True)
            if filename is None:
                self.ext = 'pkl' if self.mode == 'pickle' else 'pt'
                self.filename = f'{label}_{self.data_type}_cols_{self.text_cols}_size_{len(self)}'
            else:
                self.filename = filename
            self.path = Path(f'{self.pkl_path}/{self.filename}.{self.ext}')
            self.vocab_size = len(self.tfidf.vocabulary_)

            if save:
                bar.label = f"Saving {label} Class object"
                bar.update(inter)
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
        if self.group:
            self.behave['impressions'] = self.behave['impressions'].apply(lambda x: [self.dic[i] for i in x])
        else:
            self.behave['impressions'] = self.behave['impressions'].map(self.dic)
        self.behave['history'] = self.behave['history'].apply(lambda x: [self.dic[i] for i in x])

    def get_user_vector(self,
                        index: int,
                        csr: bool = True) -> Union[sp.csr_matrix, np.ndarray]:
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
            return sp.csr_matrix(user_mat)
        else:
            return user_mat

    def get_impression_vector(self,
                              index: int,
                              csr: bool = True) -> Union[sp.csr_matrix, np.ndarray]:
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
            return sp.csr_matrix(news_mat)
        else:
            return news_mat

    def save(self) -> None:
        """
        Save the MIND object to disk as a pickle file, or as Tensor (for use the child class)
        :return: None
        """
        if os.path.exists(self.path) and not self.overwrite:
            lp.beep()
            typer.echo(f'{self.path} already exists')
            ans = typer.confirm(f'Overwrite?', default=False)
            if ans:
                typer.echo(f'Overwriting')
            else:
                typer.echo('Skipping save')
                return None
        if self.mode == 'pickle':
            with open(self.path, 'wb') as f:
                pickle.dump(self, f)
        elif self.mode == 'torch':
            torch.save(self, self.path)
        return None

    @classmethod
    def load(cls,
             filename: str,
             mode: str = 'pickle') -> 'Mind':
        """
        Load a MIND object from disk directly
        :param device:
        :param mode: 'pickle' or 'tensor'
        :param filename: path to file (str)
        :return:
        """
        typer.echo(f'Loading {filename}')
        if mode not in ['pickle', 'torch']:
            raise ValueError(f"Mode {mode} not supported, use 'pickle' or 'tensor'")
        with click_spinner.spinner(f'Loading MIND {filename}'):
            path = Path(filename)
            with open(path, 'rb') as f:
                if mode == 'pickle':
                    if filename[-2:] == 'pt':
                        raise ValueError('Loading a torch file is not supported with pickle mode, change mode to torch')
                    else:
                        mind = pickle.load(f)
                elif mode == 'torch':
                    if filename[-3:] == 'pkl':
                        raise ValueError(
                            'Loading a pickle file is not supported with torch mode, change mode to pickle')
                    else:
                        mind = torch.load(f)
            typer.secho(f'MIND {filename} loaded', color=typer.colors.BRIGHT_GREEN)
        return mind


class MindTF_IDF(Mind):
    """
    This class is for the use of TF-IDF Recommendation.
    """

    def __init__(self,
                 data_path: str = 'data',
                 max_features: int = None,
                 cols: list or str = 'title',
                 ngram_range: tuple = (1, 2),
                 filename: str = None,
                 data_type: str = 'train',
                 undersample: bool = False,
                 save: bool = True,
                 overwrite: bool = False,
                 data_rows: int = None,
                 tfidf_k: int = 5,
                 sessions: bool = False,
                 pkl_path: str = 'pkl',
                 result_path: str = 'results',
                 min_df: float = 0.005,
                 group: bool = True):
        """
        Initialize the MIND TF-IDF object
        :param data_path:
        :param max_features:
        :param cols:
        :param ngram_range:
        :param min_df:
        :param tfidf_k:
        :param filename:
        :param data_type:
        :param undersample:
        :param save:
        :param sessions:
        :param overwrite:
        :param data_rows:
        """
        super().__init__(data_path=data_path,
                         max_features=max_features,
                         cols=cols,
                         ngram_range=ngram_range,
                         filename=filename,
                         data_type=data_type,
                         undersample=undersample,
                         save=save,
                         overwrite=overwrite,
                         data_rows=data_rows,
                         sessions=sessions,
                         pkl_path=pkl_path,
                         min_df=min_df,
                         group=group)
        if undersample:
            raise ValueError('Undersampling not supported for TF-IDF')
        self.tfidf_k = tfidf_k
        self.df_scores = None
        self.result_path = result_path

    def run(self,
            limit=None,
            shuffle=True) -> pd.DataFrame:
        """
        Run the TF-IDF algorithm and evaluate the results, return a df
        :param shuffle:
        :param limit: Number of users to evaluate (int)
        :return: df with results
        """
        if limit is None:
            limit = len(self.behave)
        if shuffle:
            indices = self.behave.sample(n=limit, replace=False).index.tolist()
        else:
            indices = self.behave.index.tolist()[:limit]
        self.df_scores = pd.DataFrame(columns=['behave_index',
                                               'user_id',
                                               'nDCG - baseline',
                                               'nDCG - tfidf',
                                               'nDCG%5 - baseline',
                                               'nDCG%5 - tfidf',
                                               'nDCG%10 - baseline',
                                               'nDCG%10 - tfidf'])
        for i in tqdm(indices, desc='Running TF-IDF on users', total=limit):
            labels = self.labels.iloc[i]
            user_id = self.behave.iloc[i]["user_id"]
            user_vector = self.get_user_vector(i, csr=True)
            user_ranking = self.get_user_ranking(i, user_vector, labels)
            impression_index = self.behave.iloc[i]['impressions']
            true_score = [np.array(labels)]

            rank_base = pd.DataFrame(impression_index, columns=['impression_ind'])
            rank_base["labels"] = np.array(labels)
            rank_base.sort_values(by=['impression_ind'], ascending=False, inplace=True)
            pred_baseline = [np.array(rank_base["labels"])]
            pred_tfidf = [np.array(user_ranking["labels"])]
            score_baseline = ndcg_score(true_score, pred_baseline)
            score_tfidf = ndcg_score(true_score, pred_tfidf)

            score_baseline5 = ndcg_score(true_score, pred_baseline, k=5)
            score_tfidf5 = ndcg_score(true_score, pred_tfidf, k=5)
            score_baseline10 = ndcg_score(true_score, pred_baseline, k=10)
            score_tfidf10 = ndcg_score(true_score, pred_tfidf, k=10)

            self.df_scores.loc[i] = [i,
                                     user_id,
                                     score_baseline,
                                     score_tfidf,
                                     score_baseline5,
                                     score_tfidf5,
                                     score_baseline10,
                                     score_tfidf10]

        self.df_scores.to_csv(f'{self.result_path}/tfidf_scores_{self.filename}.csv', index=False)
        return self.df_scores

    def get_user_ranking(self,
                         index: int,
                         user_vec: sp.csr_matrix,
                         labels: list) -> OrderedDict:
        """
        Get the user ranking based on the TF-IDF vector
        :return:
        """
        impression_ind = self.behave.iloc[index]['impressions']
        user_impression_matrix = self.matrix[impression_ind]
        impressions_tfidf_matrix_with_user_vec = sp.vstack((user_impression_matrix, user_vec),
                                                           format='csr')
        cosine_sim = cosine_similarity(impressions_tfidf_matrix_with_user_vec,
                                       impressions_tfidf_matrix_with_user_vec)
        sim_scores = list(enumerate(cosine_sim[len(cosine_sim) - 1]))
        rank_df = pd.DataFrame(sim_scores, columns=['mat_ind', 'cosine_sim'])
        rank_df = rank_df[:-1]
        rank_df["labels"] = np.array(labels)
        rank_df.sort_values(by=['cosine_sim'], ascending=False, inplace=True)
        return rank_df


class MindTensorMul(Dataset, Mind):
    """
    MIND class for Pytorch
    """

    def __init__(self,
                 data_path: str = 'data',
                 max_features: int = None,
                 cols: list or str = 'title',
                 ngram_range: tuple = (1, 2),
                 filename: str = None,
                 data_type: str = 'train',
                 undersample: bool = True,
                 save: bool = True,
                 overwrite: bool = False,
                 data_rows: int = None,
                 sessions: bool = False,
                 pkl_path: str = 'pkl',
                 min_df: float = 0.005,
                 group: bool = False,
                 device: str = 'cpu',
                 cuda: bool = False):
        """
        Initialize the MIND object for Pytorch
        :param data_path:
        :param max_features:
        :param cols:
        :param ngram_range:
        :param min_df:
        :param filename:
        :param data_type:
        :param undersample:
        :param save:
        :param sessions:
        :param overwrite:
        :param data_rows:
        :param device: device to use, default is 'cpu'
        :param cuda: default is 'False'
        """
        super().__init__(data_path=data_path,
                         max_features=max_features,
                         cols=cols,
                         ngram_range=ngram_range,
                         filename=filename,
                         data_type=data_type,
                         undersample=undersample,
                         save=save,
                         overwrite=overwrite,
                         data_rows=data_rows,
                         sessions=sessions,
                         pkl_path=pkl_path,
                         min_df=min_df,
                         group=group)
        if self.group:
            raise NotImplementedError('Grouping not implemented for Pytorch MIND - Tensor Multiply')
        self.device = device
        self.cuda = cuda
        self.mode = 'torch'

    def tensorize_index(self,
                        index: int):
        """
        Tensorize the index features
        :param index:
        :return:
        """
        user_vector = torch.tensor(sp.csr_matrix.todense(self.get_user_vector(index, csr=True)),
                                   device=self.device, dtype=torch.float32)
        news_vector = torch.tensor(sp.csr_matrix.todense(self.get_impression_vector(index, csr=True)),
                                   device=self.device, dtype=torch.float32)
        labels = self.labels.iloc[index]
        labels = torch.tensor(labels, device=self.device, dtype=torch.int8)
        return user_vector, news_vector, labels

    def __getitem__(self, index: int):
        """
        Get the item from the dataset. The main function of the MIND Tensor class.
        """
        user_vector, news_vector, labels = self.tensorize_index(index)
        features = torch.mul(news_vector, user_vector)
        return features, labels


class MindTensorDot(MindTensorMul):
    def __init__(self,
                 data_path: str = 'data',
                 max_features: int = None,
                 cols: list or str = 'title',
                 ngram_range: tuple = (1, 2),
                 filename: str = None,
                 data_type: str = 'train',
                 undersample: bool = True,
                 save: bool = True,
                 overwrite: bool = False,
                 data_rows: int = None,
                 sessions: bool = False,
                 pkl_path: str = 'pkl',
                 min_df: float = 0.005,
                 group: bool = False,
                 device: str = 'cpu',
                 cuda: bool = False):
        """
        Initialize the MIND object for Pytorch
        :param data_path:
        :param max_features:
        :param cols:
        :param ngram_range:
        :param min_df:
        :param filename:
        :param data_type:
        :param undersample:
        :param save:
        :param sessions:
        :param overwrite:
        :param data_rows:
        :param device: device to use, default is 'cpu'
        :param cuda: default is 'False'
        """
        super().__init__(data_path=data_path,
                         max_features=max_features,
                         cols=cols,
                         ngram_range=ngram_range,
                         filename=filename,
                         data_type=data_type,
                         undersample=undersample,
                         save=save,
                         overwrite=overwrite,
                         data_rows=data_rows,
                         sessions=sessions,
                         pkl_path=pkl_path,
                         min_df=min_df,
                         group=group,
                         device=device,
                         cuda=cuda)

    def __getitem__(self, index: int):
        """
        Get the item from the dataset. The main function of the MIND Tensor class.
        """
        user_vector, news_vector, labels = self.tensorize_index(index)
        features = torch.tensordot(news_vector, user_vector).unsqueeze(-1)
        labels = labels.float().unsqueeze(-1)
        return features, labels
