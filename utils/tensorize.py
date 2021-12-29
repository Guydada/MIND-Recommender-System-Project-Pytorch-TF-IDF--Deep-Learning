import numpy as np
import torch
from scipy.sparse import csr_matrix


def sparse_to_tensor(spa_matrix: csr_matrix,
                     device: str = 'cpu',
                     cuda: bool = False):
    return torch.tensor(csr_matrix.todense(spa_matrix), device=device, dtype=torch.float64)


def numpy_to_torch(data: np.ndarray,
                   device: str ='cpu',
                   dtype = torch.float64) -> torch.Tensor:
    """
    Converts a numpy array to a torch tensor.
    :param data:
    :param device:
    :param dtype:
    :return:
    """
    return torch.from_numpy(data).type(torch.FloatTensor).to(device, dtype=dtype)


def tfidf_to_tensor(tfidf_matrix: csr_matrix,
                    device: str = 'cpu'):
    """
    Converts a tfidf matrix to a torch tensor.
    :param tfidf_matrix:
    :param device:
    :return:
    """
    return torch.tensor(csr_matrix.todense(tfidf_matrix)).to(device, dtype=torch.float64)

