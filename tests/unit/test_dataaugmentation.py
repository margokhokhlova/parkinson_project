import pytest
from utils.augment_data import get_augmentation_indexes, augment_data
import numpy as np

signal_emg_2_patients = [[[11,111],[12, 122,],[13,133],[14,144]],[[21,211],[22,222],[23,233],[24,244]]]
signal_emg_3_patients = [[[11],[12],[13],[14]],[[21],[22],[23],[24]], [[31],[32],[33],[34]]]
signal_emg_5_patients = [[[11],[12],[13],[14]],[[21],[22],[23],[24]], [[31],[32],[33],[34]],
                        [[41],[42],[43],[44]], [[51],[52],[53],[54]]]


all_possible_permutations_len = len(signal_emg_3_patients)**2 # perm of 2

@pytest.mark.parametrize("test_input, expected", 
                        [(signal_emg_3_patients, all_possible_permutations_len), (signal_emg_2_patients, len(signal_emg_2_patients)**2),
                        (signal_emg_5_patients, len(signal_emg_5_patients)**2)])

def test_augmentation_indexes(test_input, expected) -> None:
    assert len(get_augmentation_indexes(test_input))== expected

def test_augmentation():
    ind = get_augmentation_indexes(signal_emg_2_patients)
    augmented = augment_data(signal_emg_2_patients, ind)
    print('Resulting augmented data ',augmented) #visual evaluation

