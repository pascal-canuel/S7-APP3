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
    force_cpu = False  # Forcer a utiliser le cpu?
    trainning = True  # Entrainement?
    test = True  # Test?
    learning_curves = False  # Affichage des courbes d'entrainement?
    gen_test_images = True  # Génération images test?
    seed = 1  # Pour répétabilité
    n_workers = 0  # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    # n_epochs = 20  # Nombre d'iteration sur l'ensemble de donnees
    n_epochs = 1
    lr = 0.01  # Taux d'apprentissage pour l'optimizateur
    batch_size = 64  # Taille des lots pour l'entraînement
    train_val_split = 0.7  # Proportion d'échantillons

    hidden_dim = 20  # Nombre de neurones caches par couche
    n_layers = 2  # Nombre de de couches

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords('data_trainval.p')
    dataset_test = HandwrittenWords('data_trainval.p') # TODO: change before validation
    # dataset_test = HandwrittenWords('data_testval.p')

    # Séparation de l'ensemble de données (entraînement et validation)
    n_train_samples = int(len(dataset) * train_val_split)
    n_val_samples = len(dataset) - n_train_samples

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samples, n_val_samples])

    # Instanciation des dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)

    # Instanciation du model
    model = trajectory2seq(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        symb2int=dataset.symb2int,
        int2symb=dataset.int2symb,
        dict_size=dataset.dict_size,
        device=device,
        maxlen=dataset.max_len
    )

    # Initialisation des variables
    # À compléter

    if trainning:

        # Initialisation affichage
        if learning_curves:
            train_dist = []  # Historique des distances
            train_loss = []  # Historique des coûts

            val_dist = []  # Historique des distances
            val_loss = []  # Historique des coûts

            fig, (ax1, ax2) = plt.subplots(1, 2)

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist_train = 0

            running_loss_val = 0
            dist_val = 0

            for batch_idx, data in enumerate(train_loader):
                # Formatage des données
                handwritten_seq, target_seq = data
                # handwritten_seq = handwritten_seq.to(device).double()
                # target_seq = target_seq.to(device).long()
                handwritten_seq = handwritten_seq.type(torch.LongTensor).to(device).float()
                target_seq = target_seq.type(torch.LongTensor).to(device)

                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(handwritten_seq)  # Passage avant
                loss = criterion(output.view((-1, model.dict_size['word'])), target_seq.view(-1))

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist_train += edit_distance(a[:Ma], b[:Mb]) / batch_size

                    # Affichage pendant l'entraînement
                    print(
                        'Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                            epoch, n_epochs, batch_idx * batch_size, len(train_loader.dataset),
                                             100. * batch_idx * batch_size / len(train_loader.dataset),
                                             running_loss_train / (batch_idx + 1),
                                             dist_train / len(train_loader)), end='\r')

            print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                epoch, n_epochs, (batch_idx + 1) * batch_size, len(train_loader.dataset),
                                 100. * (batch_idx + 1) * batch_size / len(train_loader.dataset),
                                 running_loss_train / (batch_idx + 1),
                                 dist_train / len(train_loader)), end='\r')
            print('\n')

            # Validation
            for batch_idx, data in enumerate(val_loader):
                # Formatage des données
                handwritten_seq, target_seq = data
                # handwritten_seq = handwritten_seq.to(device).double()
                # target_seq = target_seq.to(device).long()
                handwritten_seq = handwritten_seq.type(torch.LongTensor).to(device).float()
                target_seq = target_seq.type(torch.LongTensor).to(device)

                output, hidden, attn = model(handwritten_seq)  # Passage avant
                loss = criterion(output.view((-1, model.dict_size['word'])), target_seq.view(-1))

                running_loss_val += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(M):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist_val += edit_distance(a[:Ma], b[:Mb]) / batch_size

                    # Affichage pendant l'entraînement
                    print(
                        'Validation - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                            epoch, n_epochs, batch_idx * batch_size, len(val_loader.dataset),
                                             100. * batch_idx * batch_size / len(val_loader.dataset),
                                             running_loss_val / (batch_idx + 1),
                                             dist_val / len(val_loader)), end='\r')

            print(
                'Validation - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}'.format(
                    epoch, n_epochs, (batch_idx + 1) * batch_size, len(val_loader.dataset),
                                     100. * (batch_idx + 1) * batch_size / len(val_loader.dataset),
                                     running_loss_val / (batch_idx + 1),
                                     dist_val / len(val_loader)), end='\r')
            print('\n')

            # Ajouter les loss aux listes
            train_loss.append(running_loss_train / len(train_loader))
            train_dist.append(dist_train / len(train_loader))

            val_loss.append(running_loss_val / len(val_loader))
            val_dist.append(dist_val / len(val_loader))

            # Enregistrer les poids
            torch.save(model, 'model.pt')

            # Affichage
            if learning_curves:
                # visualization
                ax1.cla()
                ax1.plot(train_loss, label='training loss')
                ax1.plot(train_dist, label='training distance')
                ax1.legend()
                ax2.cla()
                ax2.plot(val_loss, label='validation loss')
                ax2.plot(val_dist, label='validation distance')
                ax2.legend()
                plt.draw()
                plt.pause(0.01)

        if learning_curves:
            plt.show()
            plt.close('all')

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
