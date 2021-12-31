import nltk as nltk
import typer
import pandas as pd
from pathlib import Path
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from string import punctuation
from imblearn.under_sampling import RandomUnderSampler
import win32api


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')


def load_news(data_type: str,
              data_path: str = 'data',
              preprocess: bool = True,
              cols: str or list = 'title') -> pd.DataFrame:
    """
    Loads the news data from the data folder.
    :param cols:
    :param preprocess:
    :param data_type:
    :param data_path:
    :return:
    """
    file = Path(f'{data_path}/{data_type}/news.tsv')
    news_df = pd.read_csv(file,
                          sep='\t',
                          header=None)
    news_df.drop(news_df.columns[5:8],
                 axis=1,
                 inplace=True)
    news_df = news_df.rename(columns={0: 'news_id',
                                      1: 'category',
                                      2: 'subcategory',
                                      3: 'title',
                                      4: 'abstract'})
    if isinstance(cols, str):
        cols = [cols]
    # apply preprocess
    if preprocess:
        news_df = preprocess_news(news_df, cols)
    return news_df


def load_behave(sessions: bool,
                undersample: bool,
                data_type: str,
                data_path: str = 'data',
                data_rows: int = None,
                group: bool = False) -> (pd.DataFrame,
                                         pd.DataFrame,
                                         pd.DataFrame):
    """
    Loads the news data from the data folder.
    :param group:
    :param data_rows:
    :param sessions:
    :param data_type:
    :param data_path:
    :param undersample:
    :return:
    """
    file = Path(f'{data_path}/{data_type}/behaviors.tsv')
    behave_df = pd.read_table(file,
                              sep='\t',
                              header=None,
                              nrows=data_rows)
    behave_df.drop([0, 2],
                   axis=1,
                   inplace=True)
    behave_df.columns = ['user_id',
                         'history',
                         'impressions']
    # drop rows with empty lists only for train
    behave_df = behave_df.dropna()
    # drop history lines with empty lists (not na but empty)
    behave_df = behave_df[behave_df['history'].apply(lambda f: len(f) > 1)]
    # split impressions into list
    behave_df['impressions'] = behave_df['impressions'].apply(lambda f: f.split(' '))
    # clean duplicates in impressions
    behave_df['impressions'] = behave_df['impressions'].apply(lambda f: list(set(f)))
    # split history lines into list
    behave_df['history'] = behave_df['history'].apply(lambda f: f.split(' '))
    # clean duplicates in history
    behave_df['history'] = behave_df['history'].apply(lambda f: list(set(f)))
    if not sessions:
        behave_df = behave_df.groupby('user_id').agg(sum).reset_index()
    # distribute impressions by user
    res = behave_df.copy()
    res = res.explode('impressions')
    # remove labels from impressions and add them to labels column
    res['label'] = res['impressions'].apply(lambda f: int(f[-1]))
    res['impressions'] = res['impressions'].apply(lambda f: f[:-2])
    # return popularity rank of impressions # This is done purposefully before undersampling!!!
    click_df = res[['impressions', 'label']].copy()
    # count how many clicks each impression has
    click_df = click_df.groupby('impressions').sum().reset_index()
    click_df = click_df.sort_values(by='label', ascending=False).reset_index(drop=True)
    if not group: # used for model training
        res.drop_duplicates(subset=['user_id', 'impressions'])
        res.reset_index(drop=True, inplace=True)  # TT
        x, y = res.drop('label', axis=1), res['label']
        if undersample and data_type == 'train':
            x, y = under_sample(x, y)
        elif undersample and data_type == 'test':
            typer.echo(f'{data_type} data is not undersampled because it is test data')
        return x, y, click_df
    else: # used for TF-IDF
        behave_df['labels'] = behave_df['impressions'].apply(lambda f: [int(i[-1]) for i in f])
        behave_df['impressions'] = behave_df['impressions'].apply(lambda f: [i[:-2] for i in f])
        x, y = behave_df.drop('labels', axis=1), behave_df['labels']
        return x, y, click_df


def preprocess_news(news_df: pd.DataFrame,
                    cols: str or list = 'title') -> pd.DataFrame:
    """
    Preprocesses the news data. apply stemming and lemmatization, and remove stopwords.
    This function applies the common methods for working with text before TF-IDF.
    :param news_df:
    :param cols:
    :return:
    """
    if isinstance(cols, str):
        cols = [cols]
    # convert all columns to string
    news_df = news_df.astype(str)
    # combine all columns in cols into one column 'text'
    news_df['text'] = news_df[cols].apply(lambda x: ' '.join(x), axis=1)
    # apply word_tokenize to column_name
    news_df['text'] = news_df['text'].apply(word_tokenize)
    # remove digits
    news_df['text'] = news_df['text'].apply(lambda x: [word for word in x if not str(word).isdigit()])
    # lowercase
    news_df['text'] = news_df['text'].apply(lambda x: [word.lower() for word in x])
    # remove punctuation
    news_df['text'] = news_df['text'].apply(lambda x: [word for word in x if word not in punctuation])
    # remove stopwords
    stopwords_set = set(stopwords.words("english"))
    news_df['text'] = news_df['text'].apply(lambda x: [word for word in x if word not in stopwords_set])
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    news_df['text'] = news_df['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    # remove duplicates in text column
    news_df['text'] = news_df['text'].apply(lambda x: list(set(x)))
    # convert to string
    news_df['text'] = news_df['text'].apply(lambda x: ' '.join(x))
    # remove rows with empty lists
    news_df = news_df[news_df[cols[0]].apply(lambda x: len(x) > 0)]
    return news_df[['news_id', 'text']]


def under_sample(x, y):
    """
    Under-sample the data by randomly selecting a row from the minority class.
    :param x:
    :param y:
    :return:
    """
    undersample = RandomUnderSampler(sampling_strategy='majority')
    x, y = undersample.fit_resample(x, y)
    return x, y


def word_tokenize(sent: str) -> list:
    """
    Tokenize a sentence into words.
    :param sent:
    :return: List of words
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def beep():
    """
    Play a beep sound to notify the user.
    :return:
    """
    win32api.MessageBeep()
    pass
