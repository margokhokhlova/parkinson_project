import itertools


def augment_data(X, model_type=4):
    """Function takes the data for one group of patients
    and augments them following this rule:
    take all
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


if __name__ == "__main__":
    L = augment_data(20)
    print(len(L))
    print(L)
