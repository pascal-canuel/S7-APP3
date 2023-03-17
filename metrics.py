# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(x,y):
    # Calcul de la distance d'édition

    # Create a 2D array of size (len(a)+1) x (len(b)+1)
    distance = np.zeros((len(x) + 1, len(y) + 1))

    # Initialize the first row and column to 0, 1, 2, 3, ...
    for i in range(len(x) + 1):
        distance[i, 0] = i

    for j in range(len(y) + 1):
        distance[0, j] = j

    # Loop over the array and fill it with the correct values
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            distance[i, j] = min(
                distance[i - 1, j - 1] if x[i - 1] == y[j - 1] else distance[i - 1, j - 1] + 1,
                distance[i - 1, j] + 1,
                distance[i, j - 1] + 1
            )

    # Return the value in the bottom right corner of the array
    return distance[len(x), len(y)]

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion

    # À compléter

    return None
