from random import uniform
import numpy as np
import re

def print_augm_matrix(inputMatrix, size):
    for r in range(size):
        print("[ ", end="")
        for c in range(size):
            print(" ", inputMatrix[r][c], " ", end="")
        print("| ", inputMatrix[r][size], " ]")

def random_aug_matrix(inputMatrix, size):
    for r in range(size):
        for c in range(size):
            inputMatrix[r][c] = uniform(0, 10)
        inputMatrix[r][size] = uniform(0, 10)

def import_matrix_from_file(fileName):
    file = open(fileName, "r")

    line = file.readline()
    offset = len(line)
    line = re.sub("\n", "", line)
    array1d = np.array(re.split(" ", line, maxsplit=0))
    
    file.seek(offset)
    print(array1d)

    r = 0
    for line in file:
        line = re.sub("\n", "", line)
        array = re.split(" ", line, maxsplit=0)
        print(array)
        np.vstack((array1d, array))
        # array1d = np.expand_dims(array, axis=r)
        # array1d = np.append(array1d, array, axis=r)
        r = r + 1
    # print(array1d)

    file.close()
