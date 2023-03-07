from random import uniform
import numpy as np
import re

""" Matrix handler functions """

def print_augm_matrix(inputMatrix, size):
    for r in range(size):
        print("[ ", end="")
        for c in range(size):
            print(" ", inputMatrix[r][c], " ", end="")
        print("| ", inputMatrix[r][size], " ]")
    print("")

def random_aug_matrix(inputMatrix, size):
    for r in range(size):
        for c in range(size):
            inputMatrix[r][c] = uniform(0, 10)
        inputMatrix[r][size] = uniform(0, 10)

def import_matrix_from_file(fileName):
    file = open(fileName, "r")

    # Read one line to deduce augmented matrix size
    line = file.readline()
    file.seek(0)
    matrixSize = len(np.array(re.split(" ", line, maxsplit=0)))
    array2d = np.zeros(matrixSize)
    print(array2d)

    #  Iterating through all the rest
    counter = 0
    for line in file:
        line = re.sub("\n", "", line)
        newArray = re.split(" ", line, maxsplit=0)
        newArray = [float(i) for i in newArray]
        if counter == 0:
            array2d = newArray
        else:
            array2d = np.vstack((array2d, newArray))
        counter = counter + 1

    file.close()
    importedMatrix = np.asarray(array2d)
    return [importedMatrix, matrixSize-1]
