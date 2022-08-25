import pytest
from utils.augment_data import augment_data
import numpy as np

signal_emg_3_patients = [[[11],[12],[13],[14]],[[21],[22],[23],[24]], [[31],[32],[33],[34]]]

all_possible_permutations_len = len(signal_emg_3_patients)**2 # perm of 2

@pytest.mark.parametrize("test_input, expected", 
                        [(signal_emg_3_patients, all_possible_permutations_len)])

def test_preprocess(test_input, expected) -> None:
    assert len(augment_data(test_input))== expected
