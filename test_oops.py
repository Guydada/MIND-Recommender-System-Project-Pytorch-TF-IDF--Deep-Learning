from collections.abc import Container

import numpy as np
import pandas as pd
from datasets import Dataset as ds


class out(ds):
    def __init__(self):
        # create a a random df of shape (10, 10)
        df = pd.DataFrame(np.random.randint(0, 100 ,size=(100, 4)), columns=list('ABCD'))
        self.df = df

    def __getitem__(self, index):
        return self.avg(index)

    def __len__(self):
        return len(self.df)

    def avg(self, index):
        res = self.df.iloc[index, :] = self.df.iloc[index, :].mean()
        return res


class inner(ds):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
    def __repr__(self):
        return f'{self.name}'

x = out()
print(x)
