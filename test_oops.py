from collections.abc import Container

import numpy as np
import pandas as pd

class out:
    def __init__(self, a, b, c):
        print(self.__class__.__name__)
        self.a = a
        self.b = b
        self.c = c
        self.hidden = 'hidden'
        print(f'doing something {self.hidden}')

    def mul(self):
        return self.a * self.b * self.c

    @classmethod
    def add(cls, x):
        return   x * 2

    def __repr__(self):
        return 'out(a={}, b={}, c={})'.format(self.a, self.b, self.c)


class inner(out):
    def __init__(self, a, b, c, d):
        super().__init__()
        print(self.__class__.__name__)
        self.hidden = 'plain'
        self.d = d

    def mul(self):
        return self.a * self.b * self.c * self.d

    def __repr__(self):
        return 'inner(a={}, b={}, c={}, d={})'.format(self.a, self.b, self.c, self.d)

x = out(1, 2, 3)
y = inner(1, 2, 3, 4)

