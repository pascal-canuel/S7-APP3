# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *
from dataset import *
from metrics import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    n_epochs = 0
    batch_size = 64  # Taille des lots pour l'entraînement
    train_val_split = 0.7  # Proportion d'échantillons

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')
    # dataset_test = HandwrittenWords('data_testval.p')

    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samples = int(len(dataset) * train_val_split)
    n_val_samples = len(dataset) - n_train_samples

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samples, n_val_samples])
   

    # Instanciation des dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)
    # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)


    # Instanciation du model
    # À compléter


    # Initialisation des variables
    # À compléter

    if trainning:

        # Fonction de coût et optimizateur
        # À compléter

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            # À compléter
            
            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter
        
        # Affichage de la matrice de confusion
        # À compléter

        pass