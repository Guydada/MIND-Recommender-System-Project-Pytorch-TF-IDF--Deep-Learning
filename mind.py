import os
from tqdm import tqdm
from collections import OrderedDict
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import Dataset
from utils.load_preprocess import *
from scipy.sparse import vstack, csr_matrix
from typing import Union
import numpy as np
import click_spinner
import typer
import pickle
from utils.tensorize import *


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
        :param tfidf_k: select how much news to recommend using tfidf for each user
        :param filename: leave 'None' if you want to use default filename, or select your own
        :param data_type: 'train' or 'test'
        :param undersample: set to 'True' if you want to undersample the dataset. Train set will not be undersampled
        :param save: set to 'True' if you want to save the dataset. default is 'True'
        :param sessions: set to 'True' if you want to use sessions. default is 'False'
        """
        # get the name of the class
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
            # Load behave data
            self.behave, self.labels, self.clicks = load_behave(undersample=self.undersample,
                                                                data_path=self.data_path,
                                                                data_type=self.data_type,
                                                                sessions=self.sessions,
                                                                data_rows=data_rows,
                                                                group=self.group)
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
                ext = 'pkl' if self.mode == 'pickle' else 'pt'
                self.filename = f'{label}_{self.data_type}_size_{len(self)}.{ext}'
            else:
                self.filename = filename
            self.path = Path(f'{self.pkl_path}/{self.filename}')
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

    def save(self) -> None:
        """
        Save the MIND object to disk as a pickle file, or as Tensor (for use the child class)
        :return: None
        """
        if os.path.exists(self.path) and not self.overwrite:
            beep()
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
        # cl = cls().__class__.__name__
        with click_spinner.spinner(f'Loading MIND {filename}'):
            path = Path(filename)
            with open(path, 'rb') as f:
                if mode == 'pickle':
                    mind = pickle.load(f)
                elif mode == 'torch':
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
            # top_k: int = 5,
            limit=None,
            shuffle=True,
            prints=True): # todo: this function needs to be fitted to new data structure - impression as a list
        """
        Run the TF-IDF algorithm and evaluate the results, return a df
        :param top_k:
        :param prints:
        :param shuffle:
        :param limit: Number of users to evaluate (int)
        :return:
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
                                               'nDCG - tfidf'])
                                               # f'nDCG-top:{top_k} Score'])
        for i in tqdm(indices, desc='Running TF-IDF on users', total=limit): # todo: this needs to be remodified
            labels = self.labels.iloc[i]
            user_id = self.behave.iloc[i]["user_id"]
            user_vector = self.get_user_vector(i, csr=True)
            # user_ranking = self.get_user_ranking(i, user_vector, k=None)
            user_ranking = self.get_user_ranking(i, user_vector, labels)
            impression_index = self.behave.iloc[i]['impressions']  # the index is also the click rank
            news_id = self.news.loc[impression_index, 'news_id']
            true = [np.array(labels)]

            rank_base = pd.DataFrame(impression_index, columns=['impre_ind'])
            rank_base["labels"] = np.array(labels)
            rank_base.sort_values(by=['impre_ind'], ascending=False, inplace=True)

            pred_basline = [np.array(rank_base["labels"])]
            pred_tfidf = [np.array(user_ranking["labels"])]
            score_basline = ndcg_score(true, pred_basline)
            score_tfidf = ndcg_score(true, pred_tfidf)
            # add line to df
            self.df_scores.loc[i] = [i, user_id, score_basline, score_tfidf]
            # if prints:
            #     typer.secho(
            #         f'\nUser: \t {user_id}, Impression News ID: \t {news_id}\n'
            #         f'Rank of news by clicks:           \t {impression_index}\t/\t{self.new_num}\n'
            #         f'Rank by TF-IDF Cosine Similarity: \t {impression_rank_tfidf}\t/\t{self.new_num}',
            #         color=typer.colors.BRIGHT_CYAN)
            #     # format score to 3 decimal places
            #     typer.secho(f'nDCG Score: \t {score:.3f}', color=typer.colors.BRIGHT_CYAN)


            # user_id = self.behave.iloc[i]["user_id"]
            # user_vector = self.get_user_vector(i, csr=True)
            # user_ranking = self.get_user_ranking(user_vector, k=None)
            # impression_index = self.behave.iloc[i]['impressions']  # the index is also the click rank
            # impression_rank_tfidf = list(user_ranking.keys()).index(impression_index)
            # news_id = self.news.loc[impression_index, 'news_id']
            # true = np.asarray([self.news.index][:top_k])
            # pred = np.asarray([list(user_ranking)][:top_k])
            # score = ndcg_score(true, pred)
            # # add line to df
            # self.df_scores.loc[i] = [i, user_id, impression_index, score]
            # if prints:
            #     typer.secho(
            #         f'\nUser: \t {user_id}, Impression News ID: \t {news_id}\n'
            #         f'Rank of news by clicks:           \t {impression_index}\t/\t{self.new_num}\n'
            #         f'Rank by TF-IDF Cosine Similarity: \t {impression_rank_tfidf}\t/\t{self.new_num}',
            #         color=typer.colors.BRIGHT_CYAN)
            #     # format score to 3 decimal places
            #     typer.secho(f'nDCG Score: \t {score:.3f}', color=typer.colors.BRIGHT_CYAN)
        self.df_scores.to_csv(f'{self.result_path}/tfidf_scores_{self.filename}.csv', index=False)
        return self.df_scores

    def get_user_ranking(self,
                         index: int,
                         user_vec: csr_matrix,
                         labels: list) -> OrderedDict:
        """
        Get user k sorted rankings by index in self.behave. if k is not specified, return all sorted
        :return:
        """
        impression_ind = self.behave.iloc[index]['impressions']
        user_impression_matrix = self.matrix[impression_ind]
        impressions_tfidf_matrix_with_user_vec = vstack((user_impression_matrix, user_vec),
                                                           format='csr')
        cosine_sim = cosine_similarity(impressions_tfidf_matrix_with_user_vec, impressions_tfidf_matrix_with_user_vec)
        sim_scores = list(enumerate(cosine_sim[len(cosine_sim) - 1]))

        rank_df = pd.DataFrame(sim_scores, columns=['mat_ind', 'cosine_sim'])
        rank_df = rank_df[:-1]
        rank_df["labels"] = np.array(labels)
        rank_df.sort_values(by=['cosine_sim'], ascending=False, inplace=True)
        return rank_df

    # def get_user_ranking(self,
    #                      user_vec: csr_matrix,
    #                      k: int = None) -> OrderedDict:
    #     """
    #     Get user k sorted rankings by index in self.behave. if k is not specified, return all sorted
    #     :return:
    #     """
    #     tfidf_matrix_with_user = vstack((self.matrix, user_vec))
    #     cosine_sim = cosine_similarity(tfidf_matrix_with_user, tfidf_matrix_with_user)
    #     sim_scores = list(enumerate(cosine_sim[len(cosine_sim) - 1]))
    #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: k]
    #     user_ranking = OrderedDict(sim_scores)
    #     return user_ranking


class MindTensor(Dataset, Mind):
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
        :param tfidf_k:
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
            raise NotImplementedError('Grouping not implemented for Pytorch MIND')
        self.device = device
        self.cuda = cuda
        self.mode = 'torch'

    def __getitem__(self, index: int):
        """
        Get the item from the dataset. The main function of the MIND Tensor class.
        """
        user_vector = torch.tensor(csr_matrix.todense(self.get_user_vector(index, csr=True)),
                                   device=self.device, dtype=torch.float32)
        news_vector = torch.tensor(csr_matrix.todense(self.get_impression_vector(index, csr=True)),
                                   device=self.device, dtype=torch.float32)
        features = torch.mul(news_vector, user_vector)  # for readability purposes, hence the naming of the variable
        labels = self.labels.iloc[index]  # same as above, to fit naming in model
        labels = torch.tensor(labels, device=self.device, dtype=torch.int8)
        return features, labels
