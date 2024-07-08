import itertools as itools
import json
import os
import os.path
import random
from dataclasses import dataclass
from typing import Sequence, Union
from copy import deepcopy

"""
This file contains the code for expanding certain configuration options.
"""


@dataclass(eq=True, frozen=True)
class FrequencyConfig():
    freq: int
    prevalence: float = 1
    cell_reselection_priority_intra: int = 0
    cell_reselection_priority_inter: int = 0
    q_rx_lev_min_intra: int = -80
    q_rx_lev_min_inter: int = -80
    max_users: int = float('inf')


@dataclass
class FrequencyConfigRange():
    freq: Union[int, Sequence[int]]
    prevalence: Union[float, Sequence[float]] = 1
    cell_reselection_priority_intra: Union[int, Sequence[int]] = 0
    cell_reselection_priority_inter: Union[int, Sequence[int]] = 0
    q_rx_lev_min_intra: Union[int, Sequence[int]] = -80
    q_rx_lev_min_inter: Union[int, Sequence[int]] = -80
    max_users: Union[int, Sequence[int]] = float('inf')

    def fix_inter_to_infra(self):
        self.cell_reselection_priority_inter = self.cell_reselection_priority_intra
        self.q_rx_lev_min_inter = self.q_rx_lev_min_intra

    def expand(self, fix_inter_to_infra=False) -> Sequence[FrequencyConfig]:
        for key, item in self.__dict__.items():
            if not isinstance(item, Sequence):
                self.__dict__[key] = [item]
        args = itools.product(*(item for key, item in self.__dict__.items()))
        if fix_inter_to_infra:
            res = [FrequencyConfig(
                freq=x[0],
                prevalence=x[1],
                cell_reselection_priority_intra=x[2],
                cell_reselection_priority_inter=x[2],
                q_rx_lev_min_intra=x[4],
                q_rx_lev_min_inter=x[4],
                max_users=x[6])
                for x in args]
        else:
            res = [FrequencyConfig(*x) for x in args]
        if not res:
            return [FrequencyConfig(it for key, it in self.__dict__.items())]
        return res


def generate_sib(frequency_configs: Sequence[FrequencyConfig], q_hyst=0):
    sib = deepcopy(BASIC_SIB)
    sib['q_hyst'] = q_hyst
    for conf in frequency_configs:
        sib['freq_list'].append(
            {**BASIC_FREQ,
             **conf.__dict__
             })
    return sib


def generate_sib_file(frequency_configs: Sequence[FrequencyConfig], q_hyst=0):
    sib = generate_sib(frequency_configs, q_hyst=q_hyst)
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    rand = random.randint(1000000, 9999999)
    filename = os.path.join('.tmp', f'sib_{rand}.json')
    with open(filename, 'w') as f:
        json.dump(sib, f)
    return filename


BASIC_SIB = {
    "freq": 0,
    "q_hyst": 0,
    "q_hyst_sf": {
        "medium": 0,
        "high": 0
    },
    "mobility": {
        "t_evaluation": 60,
        "t_max_hyst": 180,
        "n_medium": 1,
        "n_high": 3
    },
    "freq_list": []
}
BASIC_FREQ = {
    "freq": 1000,
    "s_intra_search": 15,
    "s_non_intra_search": 15,
    "cell_reselection_priority_intra": 5,
    "q_rx_lev_min_intra": -80,
    "t_reselection_sf": {
        "medium": 1,
        "high": 1
    },
    "q_rx_lev_min_inter": -80,
    "p_max": 23,
    "t_reselection": 1,
    "thresh_serving_low": 0,
    "thresh_high": 0,
    "thresh_low": 0,
    "cell_reselection_priority_inter": 0,
    "q_offset_freq": 0,
    "neigh_cell_list": [{
        "phys_cell_id": 1,
        "q_offset_cell": 0
    }],
    "black_cell_list": []
}
