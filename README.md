# Projet de Prédiction Énergétique - Ville de Seattle

## 📋 Vue d'ensemble
Ce projet vise à prédire la consommation d'énergie et les émissions de CO2 des bâtiments non résidentiels de Seattle. L'objectif est de soutenir l'ambition de la ville d'atteindre la neutralité carbone d'ici 2050.

## 🎯 Objectifs
- Analyser les données de consommation énergétique des bâtiments
- Développer un modèle prédictif de consommation d'énergie
- Créer une API permettant aux propriétaires d'obtenir des prédictions en temps réel

## 🏗️ Structure du Projet

### Partie 1 : Modélisation Prédictive

#### 1. Analyse Exploratoire
- Préparation de l'environnement Python
- Analyse des types de bâtiments pertinents
- Identification des valeurs aberrantes
- Sélection de la variable cible
- Visualisation des relations entre variables

#### 2. Feature Engineering
- Création de nouvelles features basées sur :
  - La localisation
  - La temporalité
  - La structure du bâtiment
  - Les types d'usage
- Attention particulière au data leakage

#### 3. Préparation des Features
- Traitement des valeurs aberrantes
- Encodage des variables catégorielles
- Analyse des corrélations
- Scaling des features

#### 4. Modélisation
- Séparation train-test
- Validation croisée
- Comparaison de différents modèles
- Optimisation des hyperparamètres
- Analyse des features importance

### Partie 2 : Développement API

#### 1. Création de l'API
- Sauvegarde du modèle avec BentoML
- Développement des endpoints
- Implémentation de la validation des données (Pydantic/Pandera)
- Tests locaux

#### 2. Déploiement
- Configuration du bentofile.yaml
- Création de l'image Docker
- Déploiement sur plateforme Cloud
- Tests de l'API déployée

## 🛠️ Technologies Utilisées
- Python
- Scikit-learn
- BentoML
- Docker
- Cloud Platform (AWS/GCP/Azure)
- Pydantic/Pandera

## 📦 Installation

1. Créer un environnement virtuel :
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# ou
.\env\Scripts\activate  # Windows
```

2. Installer les dépendances :
```bash
pip install poetry
poetry install
```

## 🚀 Utilisation

### Développement Local
```bash
bentoml serve service:svc
```

### Tests API
L'API peut être testée via :
- Requêtes HTTP
- Interface Swagger
- Scripts Python

## ⚠️ Points de Vigilance
- Éviter le data leakage dans le feature engineering
- Limiter la suppression des valeurs manquantes
- Maintenir un équilibre entre nettoyage des données et taille du dataset
- Limiter le nombre de combinaisons dans la GridSearch (~500 max)
- Arrêter les ressources Cloud après les tests

## 📝 Documentation
Pour plus d'informations sur :
- BentoML : [Documentation officielle](https://docs.bentoml.org/)
- Déploiement Cloud : [Documentation Google Cloud RUN](https://cloud.google.com/run/docs?hl=fr)
