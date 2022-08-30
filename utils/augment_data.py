import itertools
import numpy as np


def get_augmentation_indexes(X, model_type=4):
    """Function takes the data for one group of patients
    and all possible permutation indexes which are then used for
    data augmentaion.
    """
    if model_type == 4:
        num_patients = len(X)
        patient_indexes = list(range(num_patients))
        # In mathematics, a Cartesian product (or product set) is the direct product of two sets
        # which makes here a permutation with repetition
        possible_indexes = [p for p in itertools.product(patient_indexes, repeat=2)]
    # possible_indexes = list(combinations(patient_indexes, 2))
    else:
        print("Currently, only model 4 is implemented!")
        raise NameError
    return possible_indexes

def augment_data(X, augmentation_indexes):
    ''' Perform the data augmentation using augmentation indexes'''
    augmented_x = []
    for new_pair in augmentation_indexes:
        i =  new_pair[0]
        j = new_pair[1]
        new_patient = np.zeros((len(X[0]), len(X[0][0])))
        print(new_patient.shape)
        new_patient[0,:] = X[i][0] # left hand
        new_patient[2,:] = X[i][2] # left leg
        new_patient[1,:] = X[j][1] # right hand
        new_patient[3,:] = X[j][3] # right leg
        augmented_x.append(new_patient)
    return augmented_x



if __name__ == "__main__":
    L = get_augmentation_indexes(20)
    print(len(L))
    print(L)
