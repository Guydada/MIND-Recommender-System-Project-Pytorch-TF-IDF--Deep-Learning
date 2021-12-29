from tabulate import tabulate
import pandas as pd


def display(data,
            rows=5,
            headers: str = 'keys',
            tablefmt: str = "github") -> None:
    """
    Display a dataframe in a tabular format.
    :param tablefmt:
    :param headers:
    :param rows:
    :param data:
    :return: returns a tabulated dataframe
    """
    # check if df is a pandas series, else convert for display
    if isinstance(data, pd.Series):
        data = data.to_frame().reset_index()
    elif data is None:
        return None
    tab = tabulate(data.head(rows),
                   headers=headers,
                   tablefmt=tablefmt)
    print(tab)
    return None
