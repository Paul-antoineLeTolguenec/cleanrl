#!/bin/bash

# Définition du chemin du projet
PROJECT_PATH=$(pwd)
EXPORT_LINE="export PYTHONPATH=\"${PROJECT_PATH}:\$PYTHONPATH\""

# Vérification de la présence de la ligne dans .bashrc
if grep -qF "$EXPORT_LINE" ~/.bashrc; then
    echo "La commande est déjà dans le .bashrc."
else
    echo "$EXPORT_LINE" >> ~/.bashrc
    echo "La commande a été ajoutée à .bashrc."
fi

# Rechargement du fichier .bashrc
source ~/.bashrc