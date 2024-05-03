import torch
print(torch.cuda.is_available())
import tempfile
from argparse import Namespace
from torch import Tensor
from typing import Any, Dict, Tuple
import numpy as np
import torch
import abc
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator
import pandas as pd
import seaborn as sns
import inspect
import sys
import json
import random
import string
import torch.nn as nn
from numpy import ndarray
from typing import Tuple, Union
import torch.nn.functional as F
import torch.optim as optim
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import multiprocessing
from IPython.core.display import HTML
import math


class BaseModel(BaseEstimator):

    HISTORY_KEYS = ("train loss", "train theta hat", "test loss", "test theta hat")

    @staticmethod
    @abc.abstractmethod
    def name():
        raise NotImplementedError

    def __init__(self, **kwargs):
        super().__init__()

        self.params = kwargs
        self.history = None
        self.reset_history()

    def __str__(self):
        return self.__class__.__name

    @classmethod
    @abc.abstractmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def restart(self):
        raise NotImplementedError

    def reset_history(self):
        self.history = {k: [] for k in self.HISTORY_KEYS}

    @abc.abstractmethod
    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        raise NotImplementedError

def est_theta_numpy(y: ndarray, d: ndarray, m_hat: ndarray, l_hat: ndarray) -> Tuple[float, float]:
    v_hat = d - m_hat
    mean_v_hat_2 = np.mean(v_hat * v_hat)
    theta_hat = np.mean(v_hat * (y - l_hat)) / mean_v_hat_2
    return theta_hat.item(), mean_v_hat_2.item()


def est_theta_torch(y: Tensor, d: Tensor, m_hat: Tensor, l_hat: Tensor) -> Tuple[float, float]:
    v_hat = d - m_hat
    mean_v_hat_2 = torch.mean(v_hat * v_hat)
    theta_hat = torch.mean(v_hat * (y - l_hat)) / mean_v_hat_2
    return theta_hat.item(), mean_v_hat_2.item()


def pearson_correlation_numpy(x: np.ndarray, y: np.ndarray):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    coeff = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return coeff


def pearson_correlation_torch(x: Tensor, y: Tensor):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    coeff = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return coeff.item()


def est_theta(
    y: Union[ndarray, Tensor], d: Union[ndarray, Tensor], m_hat: Union[ndarray, Tensor], l_hat: Union[ndarray, Tensor]
) -> Tuple[float, float]:
    if all(isinstance(i, Tensor) for i in (y, d, m_hat, l_hat)):
        return est_theta_torch(y, d, m_hat, l_hat)
    if all(isinstance(i, ndarray) for i in (y, d, m_hat, l_hat)):
        return est_theta_numpy(y, d, m_hat, l_hat)
    raise TypeError


def pearson_correlation(x: Union[ndarray, Tensor], y: Union[ndarray, Tensor]) -> Tuple[float, float]:
    if all(isinstance(i, Tensor) for i in (x, y)):
        return pearson_correlation_torch(x, y)
    if all(isinstance(i, ndarray) for i in (x, y)):
        return pearson_correlation_numpy(x, y)
    raise TypeError
    


def gen_random_string(length: int):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def dump_opts_to_json(opts: Namespace, path: Path):
    params = vars(opts)
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def calc_cv_optimal_gamma_for_experiment(df: pd.DataFrame, thresh: float, sort_by: str = "y_res.2") -> float:
    df_x = df[~np.isnan(df["gamma"])]
    ref_df_sorted = df_x.sort_values(by=sort_by, ascending=True)
    optimal_gamma = ref_df_sorted.iloc[0]["gamma"].item()

    return optimal_gamma


class GammaScheduler:
    @classmethod
    def init_from_opts(cls, opts: Namespace):
        return cls(
            epochs=opts.sync_dml_epochs,
            warmup_epochs=opts.sync_dml_warmup_epochs,
            start_gamma=opts.sync_dml_start_gamma,
            end_gamma=opts.sync_dml_end_gamma,
        )

    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        # self.desc = "{} (epochs={}, warmup_epochs={}, start_gamma={:.3f}, end_gamma={:.3f})".format(
        #     self.__class__.__name__, epochs, warmup_epochs, start_gamma, end_gamma
        # )
        self.desc = "{} ({},{},{:.3f},{:.3f})".format(
            self.__class__.__name__, epochs, warmup_epochs, start_gamma, end_gamma
        )

    @abc.abstractmethod
    def __call__(self, epoch: int) -> float:
        raise NotImplementedError

    def __str__(self):
        return self.desc


class FixedGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.gamma = start_gamma
        self.desc = "{} (gamma={:.2f})".format(self.__class__.__name__, start_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gamma


class LinearGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.gammas = np.linspace(start_gamma, end_gamma, epochs, endpoint=True, dtype=np.float32)
        self.desc = "{} (start_gamma={:.2f}, end_gamma={:.2f})".format(self.__class__.__name__, start_gamma, end_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


class GeomGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 1e-6, end_gamma: float = 1.0):
        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        assert start_gamma > 0.0
        self.gammas = np.geomspace(start_gamma, end_gamma, epochs, endpoint=True, dtype=np.float32)
        self.desc = "{} (start_gamma={:.2f}, end_gamma={:.2f})".format(self.__class__.__name__, start_gamma, end_gamma)

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


class StepGamma(GammaScheduler):
    def __init__(self, epochs: int, warmup_epochs: int = 0, start_gamma: float = 0.0, end_gamma: float = 1.0):

        super().__init__(epochs, warmup_epochs, start_gamma, end_gamma)

        self.start = start_gamma * np.ones(shape=(warmup_epochs,), dtype=np.float32,)
        self.end = end_gamma * np.ones(shape=(epochs - warmup_epochs,), dtype=np.float32,)
        self.gammas = np.concatenate((self.start, self.end))
        self.desc = "{} (warmup_epochs={}, start_gamma={:.2f}, end_gamma={:.2f})".format(
            self.__class__.__name__, warmup_epochs, start_gamma, end_gamma
        )

    def __call__(self, epoch: int) -> float:
        return self.gammas[epoch]


GAMMA_SCHEDULERS = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}




class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)

    def __post_init__(self):
        if torch.cuda.is_available():
            self.net_m = self.net_m.cuda()
            self.net_l = self.net_l.cuda()

        self.net_m = self.net_m.type(torch.float64)
        self.net_l = self.net_l.type(torch.float64)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        m_x = self.net_m(x).squeeze()
        l_x = self.net_l(x).squeeze()
        return m_x, l_x


class LinearNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Linear(in_features=in_features, out_features=1)
        self.net_l = nn.Linear(in_features=in_features, out_features=1)

        self.__post_init__()


class NonLinearNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 2, out_features=in_features // 4),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features // 4, out_features=1),
        )

        self.__post_init__()


############################################################


class AttentionModule(nn.Module):
    def __init__(self,in_features):
        super(AttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
        nn.Linear(in_features, int(in_features / 4)),
        nn.ReLU(inplace=True),
        nn.Linear(int(in_features / 4), in_features)
        )

    def forward(self, x):
        attention = self.channel_attention(x)
        return attention

class SoftshareNet(nn.Module):
    def __init__(self, in_features):
        super(SoftshareNet, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
        )
        self.shared_layers=self.shared_layers.cuda()
        self.shared_layers=self.shared_layers.type(torch.float64)
        
        self.attention_m = AttentionModule(in_features).cuda().type(torch.float64)
        self.attention_l = AttentionModule(in_features).cuda().type(torch.float64)
        
        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_m=self.net_m.cuda()

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_l=self.net_l.cuda()
        
        self.net_m=self.net_m.type(torch.float64)
        self.net_l=self.net_l.type(torch.float64)

    def forward(self, x):
        #x=self.attention_l(x)
        shared = self.attention_m(x)
        shared=shared+x
        shared = self.shared_layers(x)
        attended_m = self.attention_m(shared)
        attended_m+=shared
        attended_l = self.attention_l(shared)
        attended_l+=shared
        out_m = self.net_m(attended_m).squeeze()
        out_l = self.net_l(attended_l).squeeze()
        return out_m, out_l
    
########################################################################################
class SelfAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttentionModule, self).__init__()


    def forward(self, x):
        _,N=x.shape
        self.query = nn.Linear(N, N).cuda().type(torch.float64)
        self.key = nn.Linear(N, N).cuda().type(torch.float64)
        self.value = nn.Linear(N, N).cuda().type(torch.float64)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, V)
        return attended

class SoftShareNet(nn.Module):
    def __init__(self, in_features):
        super(SoftShareNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
        )
        self.shared_layers=self.shared_layers.cuda()
        self.shared_layers=self.shared_layers.type(torch.float64)
        
        self.attention_m = SelfAttentionModule(in_features).cuda()
        self.attention_l = SelfAttentionModule(in_features).cuda()
        self.attention_l=self.attention_l.type(torch.float64)
        self.attention_m=self.attention_m.type(torch.float64)
        
        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_m=self.net_m.cuda()
        
        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_l=self.net_l.cuda()
        self.net_m=self.net_m.type(torch.float64)
        self.net_l=self.net_l.type(torch.float64)
        

    def forward(self, x):
        shared = self.attention_l(x)
        #shared = self.shared_layers(shared)
        attended_m = self.attention_m(shared)
        attended_l = self.attention_l(shared)
        
        out_m = self.net_m(attended_m).squeeze()
        out_l = self.net_l(attended_l).squeeze()
        
        return out_m, out_l
    
    
############################################################################
    
class ISelfAttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(ISelfAttentionModule, self).__init__()


    def forward(self, x):
        N,_=x.shape
        self.query = nn.Linear(N, N).cuda().type(torch.float64)
        self.key = nn.Linear(N, N).cuda().type(torch.float64)
        self.value = nn.Linear(N, N).cuda().type(torch.float64)
        
        x1=x.permute(1, 0)
        Q = self.query(x1)
        K = self.key(x1)
        V = self.value(x1)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        attended = attended.permute(1, 0)
        return attended

class ISoftShareNet(nn.Module):
    def __init__(self, in_features):
        super(ISoftShareNet, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features),
        )
        self.shared_layers=self.shared_layers.cuda()
        self.shared_layers=self.shared_layers.type(torch.float64)
        
        self.attention_m = ISelfAttentionModule(in_features).cuda()
        self.attention_l = ISelfAttentionModule(in_features).cuda()
        self.attention_l=self.attention_l.type(torch.float64)
        self.attention_m=self.attention_m.type(torch.float64)
        
        self.net_m = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_m=self.net_m.cuda()
        
        self.net_l = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )
        self.net_l=self.net_l.cuda()
        self.net_m=self.net_m.type(torch.float64)
        self.net_l=self.net_l.type(torch.float64)
        

    def forward(self, x):
        shared = self.attention_l(x)
        #shared = self.shared_layers(shared)
        attended_m = self.attention_m(shared)
        attended_l = self.attention_l(shared)
        
        out_m = self.net_m(attended_m).squeeze()
        out_l = self.net_l(attended_l).squeeze()
        
        return out_m, out_l
############################################################

class ExpressiveNet(BaseNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net_m = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )

        self.net_l = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_features, out_features=1),
        )

        self.__post_init__()


############################################################


class SharedNet(nn.Module):
    def __init__(self):
        super().__init__()

    def __post_init__(self):
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.net = self.net.type(torch.float64)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.net(x).squeeze()
        m_x, l_x = pred[:, 0], pred[:, 1]
        return m_x, l_x


class SharedLinearNet(SharedNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net = nn.Linear(in_features=in_features, out_features=2)

        self.__post_init__()


class SharedNonLinearNet(SharedNet):
    def __init__(self, in_features: int):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features=in_features // 2, out_features=2),
        )

        self.__post_init__()


Nets = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}


class CorrelationLoss:
    def __init__(self, gamma_scheduler: GammaScheduler, dml_stats: Dict[str, float]):
        self.gamma_scheduler = gamma_scheduler
        self.corr_abs = dml_stats["corr.abs"]
        self.res_m_2 = dml_stats["res_m.2"]
        self.res_l_2 = dml_stats["res_l.2"]

    def __call__(self, d: Tensor, y: Tensor, m_hat: Tensor, l_hat: Tensor, **kwargs):
        res_m = d - m_hat
        res_l = y - l_hat
        res_m_2 = torch.mean(res_m ** 2)
        res_l_2 = torch.mean(res_l ** 2)
        res_corr_abs = torch.absolute(torch.mean(res_m * res_l))
        res_m_2 = res_m_2 / self.res_m_2
        res_l_2 = res_l_2 / self.res_l_2
        res_corr_abs = res_corr_abs / self.corr_abs
        gamma = self.gamma_scheduler(**kwargs)
        loss = res_m_2 + res_l_2 + gamma * res_corr_abs
        return loss


LOSSES = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if obj.__module__ is __name__
}


############################################################
OPTIMIZERS_ = optim.__dict__
OPTIMIZERS = {o: OPTIMIZERS_[o] for o in OPTIMIZERS_ if not o.startswith("_") and inspect.isclass(OPTIMIZERS_[o])}
############################################################


############################################################
OPTIMIZERS_PARAMS = {k: {} for k in OPTIMIZERS.keys()}

OPTIMIZERS_PARAMS["SGD"] = dict(momentum=0.9)
OPTIMIZERS_PARAMS["Adam"] = dict(betas=(0.9, 0.999))
############################################################


def add_common_args(parser) -> argparse.ArgumentParser:
    """
    Parse shared args.
    :param parser: argparser object.
    :return: argparser object.
    """
    test = parser.add_argument_group("Test Parameters")
    test.add_argument("--n-processes", type=int, default=(mp.cpu_count() - 1) // 2, help="number of processes to launch")
    test.add_argument("--n-exp", type=int, default=500, help="number of experiments to run")
    test.add_argument("--seed", type=int, default=None, help="random seed")
    test.add_argument("--output-dir", type=Path, default=Path("results"))
    test.add_argument("--name", type=str, default=gen_random_string(5), help="experiment name")
    test.add_argument("--real-theta", type=str, default="0.0", help="true value of theta")

    return parser


def add_data_args(parser) -> argparse.ArgumentParser:
    """
    Parse shared data args.
    :param parser: argparser object.
    :return: argparser object.
    """
    parser.add_argument("--nb-features", type=int, default=20, help="number of high-dimensional features")
    parser.add_argument("--nb-observations", type=int, default=2000, help="number of observations")
    parser.add_argument("--sigma-v", type=float, default=1.0, help="V ~ N(0,sigma)")
    parser.add_argument("--sigma-u", type=float, default=1.0, help="U ~ N(0,sigma)")

    syn_data = parser.add_argument_group("Synthetic Data Parameters")
    syn_data.add_argument("--ar-rho", type=float, default=0.8, help="AutoRegressive(rho) coefficient")
    syn_data.add_argument("--majority-s", type=float, default=0.75, help="majority split value")
    syn_data.add_argument("--m0-g0-setup", type=str, default="s1", choices=("s1", "s2", "s3", "s4", "s5"))

    return parser


def add_double_ml_args(parser) -> argparse.ArgumentParser:
    dml = parser.add_argument_group("Double Machine Learning Parameters")
    dml.add_argument("--dml-net", type=str, default="NonLinearNet", choices=Nets.keys())
    dml.add_argument("--dml-lr", type=float, default=0.01, help="learning rate")
    dml.add_argument("--dml-clip-grad-norm", type=float, default=3.0)
    dml.add_argument("--dml-epochs", type=int, default=2000)
    dml.add_argument("--dml-optimizer", type=str, default="SGD", help="torch.optim.Optimizer name")
    return parser


def add_sync_dml_args(parser) -> argparse.ArgumentParser:
    sync_dml = parser.add_argument_group("SYNChronized (Double) Machine Learning Parameters")
    sync_dml.add_argument("--sync-dml-warmup-with-dml", action="store_true", default=False)
    sync_dml.add_argument("--sync-dml-net", type=str, default="NonLinearNet", choices=Nets.keys())
    sync_dml.add_argument("--sync-dml-loss", type=str, default="CorrelationLoss", choices=LOSSES.keys())
    sync_dml.add_argument("--sync-dml-lr", type=float, default=0.01, help="learning rate")
    sync_dml.add_argument("--sync-dml-clip-grad-norm", type=float, default=3.0)
    sync_dml.add_argument("--sync-dml-epochs", type=int, default=2000)
    sync_dml.add_argument("--sync-dml-optimizer", type=str, default="SGD", help="torch.optim.Optimizer name")

    sync_dml.add_argument("--sync-dml-gamma-scheduler", type=str, default="FixedGamma", choices=GAMMA_SCHEDULERS.keys())
    sync_dml.add_argument("--sync-dml-warmup-epochs", type=int, default=1000)
    sync_dml.add_argument("--sync-dml-start-gamma", type=float, default=1.0)
    sync_dml.add_argument("--sync-dml-end-gamma", type=float, default=1.0)

    return parser


def set_seed(seed: int):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class Parser(object):
    @staticmethod
    def double_ml() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_double_ml_args(parser)
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.real_theta = float(opt.real_theta)

        return opt

    @staticmethod
    def sync_dml() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_sync_dml_args(parser)
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.real_theta = float(opt.real_theta)

        return opt

    @staticmethod
    def compare() -> argparse.Namespace:
        """
        Parse command-line arguments
        :return: argparser object with user opts.
        """
        parser = argparse.ArgumentParser()
        parser = add_common_args(parser)
        parser = add_data_args(parser)
        parser = add_double_ml_args(parser)
        parser = add_sync_dml_args(parser)

        regression = parser.add_argument_group("Regression Parameters")
        regression.add_argument(
            "--thetas", type=str, nargs="+", default=[" 0.0", " 10.0"],
        )
        regression.add_argument(
            "--gammas",
            type=str,
            nargs="+",
            default=[
                " 0.000",
                " 0.001",
                " 0.01",
                " 0.1",
                " 1.0",
                " 10.",
                " 100.",
                " 1000.",
            ],
        )
        opt = parser.parse_args(args=[])

        set_seed(opt.seed)
        opt.output_dir = opt.output_dir.expanduser()
        opt.output_dir.mkdir(parents=True, exist_ok=True)
        opt.thetas = [float(theta) for theta in opt.thetas]
        opt.n_thetas = len(opt.thetas)
        opt.gammas = [float(gamma) for gamma in opt.gammas]
        opt.n_gammas = len(opt.gammas)

        return opt


class DoubleMachineLearningPyTorch(BaseModel):

    HISTORY_KEYS = (
        "train loss",
        "train theta hat",
        "train delta_m^2",
        "train delta_l^2",
        "test loss",
        "test theta hat",
        "test delta_m^2",
        "test delta_l^2",
    )

    @staticmethod
    def name():
        return "Double Machine Learning"

    def __init__(self, net_type: str, in_features: int, **kwargs):
        """
        :param net_type: Net type to use.
        :param num_features: Number of features in X.
        """
        super().__init__(**kwargs)

        self.net_type = net_type
        self.in_features = in_features
        self.net = Nets[net_type](in_features=in_features)

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(net_type=opts.dml_net, in_features=opts.nb_features, **kwargs)

    def fit_params(self, opts: Namespace, **kwargs) -> Dict[str, Any]:
        return dict(
            learning_rate=opts.dml_lr,
            max_epochs=opts.dml_epochs,
            optimizer=opts.dml_optimizer,
            clip_grad_norm=opts.dml_clip_grad_norm,
        )

    def restart(self):
        self.net = Nets[self.net_type](in_features=self.in_features)
        self.reset_history()

    @staticmethod
    def _set_fit_params(**kwargs):
        return dict(
            learning_rate=kwargs.get("learning_rate", 0.001),
            max_epochs=kwargs.get("max_epochs", 1000),
            optimizer=kwargs.get("optimizer", "Adam"),
            clip_grad_norm=kwargs.get("clip_grad_norm", None),
        )

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        params = self._set_fit_params(**kwargs)

        loss_fn = torch.nn.MSELoss()
        optimizer_name = params["optimizer"]

        optimizer = OPTIMIZERS[optimizer_name](
            self.net.parameters(), lr=params["learning_rate"], **OPTIMIZERS_PARAMS[optimizer_name]
        )

        custom_temp_dir = "E:/C"
        
        tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", dir=custom_temp_dir, delete=False)
        #tmpfile = tempfile.NamedTemporaryFile(suffix=".pt")
        torch.save(self.net, tmpfile.name)

        test_min_loss = None
        for epoch in range(params["max_epochs"]):
            optimizer.zero_grad()
            m_pred, l_pred = self.net(x=train["x"])
            m_loss = loss_fn(train["d"], m_pred)
            l_loss = loss_fn(train["y"], l_pred)
            loss = m_loss + l_loss
            loss.backward()

            if params["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), params["clip_grad_norm"])
            optimizer.step()
            theta_hat, _ = est_theta_torch(train["y"].detach(), train["d"].detach(), m_pred.detach(), l_pred.detach())

            with torch.no_grad():
                self.net.eval()
                m_hat, l_hat = self.net(x=test["x"])
                m_loss = loss_fn(test["d"], m_hat).item()
                l_loss = loss_fn(test["y"], l_hat).item()
                test_loss = m_loss + l_loss
                theta_hat, _ = est_theta_torch(
                    train["y"].detach(), train["d"].detach(), m_pred.detach(), l_pred.detach()
                )
                self.net.train()
            
            if test_min_loss is None or test_loss < test_min_loss:
                test_min_loss = test_loss
                torch.save(self.net, tmpfile.name)

        self.net = torch.load(tmpfile.name)
        tmpfile.close()

        return self

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        with torch.no_grad():
            m_hat, l_hat = self.net(x=x)
        return m_hat, l_hat






class SYNChronizedDoubleMachineLearning(BaseModel):

    HISTORY_KEYS = (
        "train loss",
        "train theta hat",
        "train residuals m.2",
        "train residuals l.2",
        "train residuals correlation",
        "test loss",
        "test theta hat",
        "test residuals m.2",
        "test residuals l.2",
        "test residuals correlation",
    )

    @staticmethod
    def name():
        return "Synchronized Double Machine Learning"

    def __init__(self, net_type: str, in_features: int, **kwargs):
        """
        :param net_type: Net type to use.
        :param num_features: Number of features in X.
        """
        super().__init__(**kwargs)

        self.net_type = net_type
        self.in_features = in_features
        self.net = Nets[net_type](in_features=in_features)

    @classmethod
    def init_from_opts(cls, opts: Namespace, **kwargs):
        return cls(net_type=opts.sync_dml_net, in_features=opts.nb_features, **kwargs)

    def fit_params(self, opts: Namespace, dml_stats: Dict[str, float], **kwargs) -> Dict[str, Any]:
        gamma_scheduler = GAMMA_SCHEDULERS[opts.sync_dml_gamma_scheduler].init_from_opts(opts)
        loss_fn = LOSSES[opts.sync_dml_loss](gamma_scheduler=gamma_scheduler, dml_stats=dml_stats)
        return dict(
            loss_fn=loss_fn,
            learning_rate=opts.sync_dml_lr,
            max_epochs=opts.sync_dml_epochs,
            optimizer=opts.sync_dml_optimizer,
            clip_grad_norm=opts.sync_dml_clip_grad_norm,
        )

    def restart(self):
        self.net = Nets[self.net_type](in_features=self.in_features)
        self.reset_history()

    def _set_fit_params(self, **kwargs):
        return dict(
            loss_fn=kwargs.get("loss_fn", None),
            learning_rate=kwargs.get("learning_rate", 0.0001),
            max_epochs=kwargs.get("max_epochs", 1000),
            optimizer=kwargs.get("optimizer", "Adam"),
            clip_grad_norm=kwargs.get("clip_grad_norm", None),
        )

    def fit(self, train: Dict[str, Tensor], test: Dict[str, Tensor], **kwargs):
        """
        Fit the model to the data.
        :param train: a dictionary with train data.
        :param test: a dictionary with test data.
        """
        params = self._set_fit_params(**kwargs)

        loss_fn = params["loss_fn"]
        optimizer_name = params["optimizer"]

        optimizer = OPTIMIZERS[optimizer_name](
            self.net.parameters(), lr=params["learning_rate"], **OPTIMIZERS_PARAMS[optimizer_name]
        )
        
        custom_temp_dir = "E:/C"
        tmpfile = tempfile.NamedTemporaryFile(suffix=".pt", dir=custom_temp_dir, delete=False)
        #tmpfile = tempfile.NamedTemporaryFile(suffix=".pt")
        torch.save(self.net, tmpfile.name)

        test_min_loss = None
        for epoch in range(params["max_epochs"]):
            optimizer.zero_grad()
            m_hat, l_hat = self.net(x=train["x"])
            loss = loss_fn(train["d"], train["y"], m_hat, l_hat, epoch=epoch)
            loss.backward()

            if params["clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), params["clip_grad_norm"])
            optimizer.step()

            theta_hat, _ = est_theta_torch(train["y"].detach(), train["d"].detach(), m_hat.detach(), l_hat.detach())

            res_m = train["d"].detach() - m_hat.detach()
            res_l = train["y"].detach() - l_hat.detach()

            with torch.no_grad():
                self.net.eval()
                m_hat, l_hat = self.net(x=test["x"])
                test_loss = loss_fn(test["d"].detach(), test["y"].detach(), m_hat.detach(), l_hat.detach(), epoch=epoch)
                theta_hat, _ = est_theta_torch(test["y"].detach(), test["d"].detach(), m_hat.detach(), l_hat.detach())
                self.net.train()

            res_m = test["d"].detach() - m_hat.detach()
            res_l = test["y"].detach() - l_hat.detach()

            if test_min_loss is None or test_loss < test_min_loss:
                test_min_loss = test_loss
                torch.save(self.net, tmpfile.name)

        self.net = torch.load(tmpfile.name)
        tmpfile.close()

        return self

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predicts (m(x), l(x)) for a given input x.
        :param x: a tensor of shape (num_samples, num_features).
        :return: m(x) and l(x), each of shape (num_samples, ).
        """
        with torch.no_grad():
            m_hat, l_hat = self.net(x=x)
        return m_hat, l_hat



def dml_theta_estimator(y: torch.Tensor, d: torch.Tensor, m_hat: torch.Tensor, l_hat: torch.Tensor) -> float:

    v_hat = d - m_hat
    u_hat = y - l_hat
    v_hat=v_hat.cpu()
    u_hat=u_hat.cpu()
    x_array = v_hat.numpy()
    y_array = u_hat.numpy()
    x_array_with_constant = x_array
    model = sm.OLS(y_array, x_array).fit()
    
    theta_hat = torch.mean(v_hat * u_hat) / torch.mean(v_hat * v_hat)
    print(theta_hat)
    return theta_hat.item(),model


def run_dml(opts, D1, D2, D2_a, D2_b) -> Tuple[Dict, torch.Tensor, Dict]:
    preds = dict(theta=opts.real_theta, gamma=np.nan,  method="DML")

    double_ml = DoubleMachineLearningPyTorch.init_from_opts(opts=opts)
    params = double_ml.fit_params(opts=opts)
    double_ml = double_ml.fit(train=D1, test=D2, **params)

    m_hat, l_hat = double_ml.predict(D2["x"])
    theta_init ,model1= dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
    u_hat, v_hat = D2["y"] - l_hat, D2["d"] - m_hat
    
    stats = {
        "corr.abs": torch.mean(torch.absolute(u_hat * v_hat)).item(),
        "res_m.2": torch.mean(v_hat ** 2).item(),
        "res_l.2": torch.mean(u_hat ** 2).item()
    }
    

    g_hat = l_hat - theta_init * m_hat
    dml_theta_for_cv ,model2= dml_theta_estimator(
        y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)

    dml_y_hat = g_hat + dml_theta_for_cv * D2["d"]
    preds["y_res.2"] = torch.mean((D2["y"] - dml_y_hat) ** 2).item()
    
    dml_theta,model3 = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
    preds['model3']=model3.summary()
    preds["U_res.2"] = torch.mean((D2["y"] - l_hat) ** 2).item()
    preds["v_res.2"] = torch.mean((D2["d"] - m_hat) ** 2).item()
    
    
    
    preds["theta_hat"] = dml_theta
    preds["bias"] = dml_theta - opts.real_theta
    

    return preds, g_hat, stats


def run_cdml_modified(opts: argparse.Namespace,gamma, D1, D2, D2_a, D2_b, g_hat, dml_stats) -> Dict:
    results = []
    gamma=gamma
    opts.sync_dml_start_gamma = gamma

    sync_dml = SYNChronizedDoubleMachineLearning.init_from_opts(opts=opts)
    params = sync_dml.fit_params(opts=opts, dml_stats=dml_stats)

    preds = dict(theta=opts.real_theta, gamma=gamma, method="C-DML")

    sync_dml = sync_dml.fit(train=D1, test=D2, **params)

    m_hat, l_hat = sync_dml.predict(D2["x"])
    sync_dml_theta_for_cv,model1 = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)

    g_hat = l_hat - sync_dml_theta_for_cv * m_hat
    sync_dml_y_hat= g_hat + sync_dml_theta_for_cv * D2["d"]
    preds["y_res.2"] = torch.mean((D2["y"] - sync_dml_y_hat) ** 2).item()

    sync_dml_theta,model2 = dml_theta_estimator(y=D2["y"], d=D2["d"], m_hat=m_hat, l_hat=l_hat)
    preds["U_res.2"] = torch.mean((D2["y"] - l_hat) ** 2).item()
    preds["v_res.2"] = torch.mean((D2["d"] - m_hat) ** 2).item()
    preds["Y_res.2"] = torch.sum((D2["y"] - l_hat) ** 2)
    preds["gamma"]=gamma
    print(model2.summary())
    preds["model"]=model2.summary()
    preds["theta_hat"] = sync_dml_theta
    preds["bias"] = sync_dml_theta - opts.real_theta
    results.append(preds)

    results = pd.DataFrame(results)
    results = results.sort_values(by="y_res.2", ascending=True)
    preds = results.iloc[0].squeeze().to_dict()

    return preds


def simulated_annealing(opts, D1, D2, D2_a, D2_b, g_hat, dml_stats):
    def objective_function(gamma):
        preds = run_cdml_modified(opts, gamma, D1, D2, D2_a, D2_b, g_hat,dml_stats)
        return preds["y_res.2"], preds

    def new_solution(current_gamma, T):
        # 定义gamma和corr_threshold的最大最小变化范围
        gamma_min, gamma_max = 0, 1
    
        # 定义基于温度T的最大变化范围
        max_gamma_change = (gamma_max - gamma_min) * 0.1 * T  # 根据T调整最大变化幅度
    
        # 随机选择一个整数倍数，用于调整变化幅度，确保在1到10之间
        multiplier_gamma = random.randint(0,1)
    
        # 应用扰动，确保变化在最大变化量的一半的整数倍内
        gamma_change = (max_gamma_change) * multiplier_gamma * random.choice([-1, 1])  # 随机选择增加或减少

        # 计算新的gamma和corr_threshold值，并确保它们在允许的范围内
        new_gamma = max(gamma_min, min(gamma_max, current_gamma + gamma_change))
    
        return new_gamma

    T = 1.0
    T_min = 0.01
    alpha = 0.5
    current_gamma = random.choice([0,0.5,1,1.5,2])
    print('current_gamma:',current_gamma)
    current_cost, current_preds = objective_function(current_gamma)

    no_improve_iter_inner = 0
    no_improve_iter_outer = 0
    no_improve_limit = 4  
    while T > T_min:
        print(T)
        improved_this_temperature = False
        no_improve_iter_inner = 0
        for i in range(20):  
            new_gamma= new_solution(current_gamma, T)
            print('new_gamma:',new_gamma)
            new_cost, new_preds= objective_function(new_gamma)
            cost_diff = new_cost - current_cost

            if cost_diff < 0 or random.uniform(0, 1) < math.exp(-cost_diff / T):
                current_gamma = new_gamma
                current_cost = new_cost
                current_preds = new_preds
                improved_this_temperature = True
                no_improve_iter_inner = 0  
            else:
                no_improve_iter_inner += 1
                if no_improve_iter_inner >= no_improve_limit:
                    break  
        
        if improved_this_temperature:
            no_improve_iter_outer = 0  
        else:
            no_improve_iter_outer += 1
            if no_improve_iter_outer >= no_improve_limit:
                break  
        T *= alpha

    return current_preds


def run_cdml(opts: argparse.Namespace, D1, D2, D2_a, D2_b, g_hat, dml_stats) -> Dict:
    best_preds = simulated_annealing(opts, D1, D2, D2_a, D2_b, g_hat, dml_stats)
    return best_preds

class DataSimulator:
    def __init__(self, n, k, l, n_good, n_bad, n_mediatory, effect_X_to_D=0.2, effect_X_to_Y=0.8):
        self.n = n
        self.k = k
        self.l = l
        self.n_good = n_good
        self.n_bad = n_bad
        self.n_mediatory = n_mediatory
        self.effect_X_to_D = effect_X_to_D
        self.effect_X_to_Y = effect_X_to_Y

    def simulate_data(self):
        X = np.random.randn(self.n, self.k, self.l)
        D = np.random.randn(self.n, self.l)
        Y = np.random.randn(self.n, self.l)
        
        if self.n_bad==0 and self.n_mediatory ==0 :
            for t in range(self.l):
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y 
                Y[:, t] +=D[:, t]*t
                
        if self.n_bad > 0:
            for t in range(self.l):
                for i in range(self.n_bad):
                    X[:, i , t] += D[:, t]*self.effect_X_to_D + Y[:, t]*self.effect_X_to_Y
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y 
                Y[:, t] +=D[:, t]*t
        if self.n_mediatory > 0:
            for t in range(self.l):
                for i in range(self.n_good):
                    D[:,t]+=X[:, i , t]*self.effect_X_to_D
                    Y[:,t]+=X[:, i , t]*self.effect_X_to_Y             
                for i in range(self.n_mediatory):
                    X[:, self.n_good+i , t] += D[:, t]*self.effect_X_to_D
                for i in range(self.n_mediatory):
                    Y[:,t]+=X[:, self.n_good + i , t]*self.effect_X_to_Y 
                Y[:, t] +=D[:, t]*t

        return X, D, Y

simulator = DataSimulator(n=1000, k=20, l=11, n_good=15, n_bad=5, n_mediatory=0)
X1, D1, Y1 = simulator.simulate_data()

X=X1[:,:,1].squeeze()
D=D1[:,1].squeeze()
Y=Y1[:,1].squeeze()

def run_experiment(opts: argparse.Namespace) -> Dict[str, Dict]:
    data = pd.read_excel("my_data.xlsx")
    df=data
    df_2006_2011=df
    Y = df_2006_2011['net_tfa'] +  np.random.randn(9915) #Here you can either change the number of samples from Excel or use a random function
    D = df_2006_2011['e401']
    X = df_2006_2011.drop(['e401','net_tfa'], axis=1) 
    X_numpy = X.values
    Y_numpy = Y.values
    D_numpy = D.values
    #if you want to use X,Y,D please uncomment the following and remove the X,Y and D from the top marks
    #X_numpy=X
    #Y_numpy=Y
    #D_numpy=D
    X_numpy = X_numpy.astype(np.float64)
    Y_numpy = Y_numpy.astype(np.float64)
    D_numpy = D_numpy.astype(np.float64)
    X_tensor = torch.from_numpy(X_numpy).to('cuda:0')
    Y_tensor = torch.from_numpy(Y_numpy).to('cuda:0')
    D_tensor = torch.from_numpy(D_numpy).to('cuda:0')
    X_tensor = torch.tensor(X_tensor)
    D_tensor = torch.tensor(D_tensor)
    Y_tensor = torch.tensor(Y_tensor)
    (
            x_train,
            x_test,
            d_train,
            d_test,
            y_train,
            y_test,
            ) = train_test_split(X_tensor, D_tensor,Y_tensor, train_size=0.5, random_state=42, shuffle=True)
    D1 = {"x": x_train, "d": d_train, "y": y_train}
    D2 = {"x": x_test, "d": d_test, "y": y_test}
    (
            x_train1,
            x_test1,
            d_train1,
            d_test1,
            y_train1,
            y_test1,
            ) = train_test_split(D2["x"],D2["d"], D2["y"], train_size=0.5, random_state=42, shuffle=True)
    D2_a = {"x": x_train1, "d": d_train1, "y": y_train1}
    D2_b = {"x": x_test1, "d": d_test1, "y": y_test1}
    dml_results, g_hat, dml_stats = run_dml(opts, D1, D2, D2_a, D2_b)
    cdml_results = run_cdml(opts, D1, D2, D2_a, D2_b, g_hat, dml_stats)
    return {
        "dml_results": dml_results,
        "cdml_results": cdml_results,
    }
def run_cv(opts: argparse.Namespace) -> pd.DataFrame:
    pbar = tqdm(total=1, desc=f"running C-DML")
    def _update(*a):
        pbar.update()
    results=run_experiment(opts)
    print(results)
    results = [result["dml_results"] for result in results] + [result["cdml_results"] for result in results]
    return pd.DataFrame(results)

HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
x = sns.color_palette("Set1")
palette = {"DML": x[0], "C-DML": x[1]}

if __name__ == '__main__':
    HTML("""<style>
    .output_png {
        display: table-cell;
        text-align: center;
        vertical-align: middle;
    }
    </style>""")
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    x = sns.color_palette("Set1")
    palette = {"DML": x[0], "C-DML": x[1]}
    opts = Parser.compare()
    #ExpressiveNet，NonLinearNet，SharedNonLinearNet,SoftshareNet,SoftShareNet,ISoftShareNet
    opts.dml_net = 'NonLinearNet'
    opts.sync_dml_net = 'SoftshareNet'
    opts.dml_lr=0.05
    opts.sync_dml_lr = 0.05
    opts.n_exp = 1
    opts.nb_features=10
    opts.real_theta = 10
    '''
    In fact, the gamma value here is only used for pre-training, and you can adjust the gamma range to line 1173. 
    You can save yourself a lot of time by finding the interval with the lowest y_res2 before simulated annealing,
    narrowing it down to a smaller number of iterations, and then performing simulated annealing.
    '''
    opts.gammas = [0,0.1,0.5,1.0,1.5,2]
    #opts.gammas = [0.0, 0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4 ,1.5,1.6,1.7,1.8,1.9,2,5]
    opts.n_gammas = len(opts.gammas)
    results0_df = pd.DataFrame()
    opts_copy = copy.deepcopy(opts)
    results0_df_rho = run_cv(opts_copy)



