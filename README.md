# My Torch

Projet de réseau neuronal en Python et NumPy pour analyser des positions
d'échecs représentées en FEN.

Deux exécutables sont disponibles :

- `my_torch_generator` : crée des réseaux vierges à partir d'une configuration JSON.
- `my_torch_analyzer` : charge un réseau et calcule une prédiction par position.

**État actuel :** la génération et l'inférence sont implémentées. Les réseaux
générés possèdent des poids aléatoires et ne sont pas entraînés : leurs sorties
permettent de tester le fonctionnement du programme, sans démontrer une capacité
à reconnaître correctement les positions. L'apprentissage reste à implémenter.

## Installation

Prérequis : Python 3.11 ou ultérieur, son module `venv`, `pip` et `make`.
Les commandes ci-dessous s'exécutent depuis la racine du dépôt, sous Linux.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r tests/requirements-dev.txt
make
```

Le fichier de dépendances installe NumPy et l'outil de couverture `coverage`.
`make` construit les deux exécutables sous forme d'archives Python zipapp.
Les dépendances externes ne sont pas incluses dans les archives : conserve
l'environnement virtuel activé pour les lancer.

Après ouverture d'un nouveau terminal, réactive l'environnement :

```bash
source .venv/bin/activate
```

## Premier lancement

### 1. Générer trois réseaux

Une configuration d'exemple est fournie dans
[`tests/conf_example/basic_network.conf`](tests/conf_example/basic_network.conf).

```bash
./my_torch_generator tests/conf_example/basic_network.conf 3
```

Cette commande écrit dans le répertoire courant :

```text
basic_network_1.nn
basic_network_2.nn
basic_network_3.nn
```

Les trois réseaux ont la même architecture, avec des graines d'initialisation
différentes. Les fichiers existants ne sont pas écrasés : pour refaire un essai,
choisis un autre `name` dans une copie de la configuration ou lance le générateur
depuis un autre répertoire en adaptant les chemins.

### 2. Analyser les positions d'exemple

```bash
./my_torch_analyzer --predict basic_network_1.nn \
    tests/dataset_example/checkmate/10_pieces.txt
```

Le programme affiche une ligne par position, dans l'ordre du fichier, contenant
l'indice de la position (à partir de zéro), le label prédit et son score :

```text
FEN game <indice>: Prediction: <label>, Confidence: <score>
```

Les labels actuels sont `Checkmate`, `Check` et `Nothing`. La valeur affichée sous
le nom `Confidence` est le score sigmoid du neurone retenu ; les trois scores
ne constituent pas une distribution de probabilités normalisée.

Les fichiers suivants sont aussi disponibles :

```text
tests/dataset_example/check/10_pieces.txt
tests/dataset_example/nothing/10_pieces.txt
```

## Configuration du générateur

Les fichiers `.conf` utilisent JSON. Exemple compatible avec l'encodeur actuel :

```json
{
  "name": "basic_network",
  "layers": [838, 64, 3],
  "seed": 42,
  "init": 0.1,
  "learning_rate": 0.01,
  "labels": ["Checkmate", "Check", "Nothing"]
}
```

| Champ | Utilisation |
| --- | --- |
| `name` | Préfixe des fichiers produits, suivi de `_1.nn`, `_2.nn`, etc. |
| `layers` | Tailles successives : entrées, couches cachées, sorties. |
| `seed` | Graine du premier réseau ; elle augmente d'une unité pour chaque réseau suivant. |
| `init` | Poids tirés uniformément entre `-init` et `init`. Les biais commencent à zéro. |
| `learning_rate` | Valeur conservée dans le modèle pour le futur entraînement. |
| `labels` | Correspondance entre les indices de sortie et les labels. |

Pour l'analyse des FEN, conserve actuellement **838 entrées** et **3 sorties**
avec cet ordre de labels. L'encodeur produit 768 valeurs pour les pièces, une
pour le joueur au trait, 64 pour la prise en passant, quatre pour les droits de
roque et une pour le compteur de demi-coups.

Les fichiers `.nn` sont également du JSON. Ils contiennent l'architecture, le
taux d'apprentissage, les labels, les poids et les biais. La convention des
matrices de poids est `(taille_sortie, taille_entree)` ; chaque vecteur de biais
possède `taille_sortie` valeurs.

## Format des positions

Chaque ligne contient les six champs FEN, puis éventuellement un label :

```text
4k3/8/8/8/8/8/4R3/4K3 b - - 0 12
4k3/8/8/8/8/8/4R3/4K3 b - - 0 12 Check
```

Le label fourni dans le fichier ne fait pas partie des entrées du réseau.
L'analyseur calcule sa propre prédiction.

## Aide et modes disponibles

```bash
./my_torch_generator --help
./my_torch_analyzer --help
```

Syntaxe du générateur, avec un ou plusieurs couples configuration/nombre :

```text
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2 ...]
```

Syntaxe déclarée de l'analyseur :

```text
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

**L'entraînement n'est pas encore disponible.** `--train` appelle désormais une
fonction dédiée, qui signale cette absence sur stderr et termine avec le code 84.
Aucune prédiction ni sauvegarde n'est effectuée dans ce mode. La destination
prévue est `SAVEFILE` si `--save` est fourni, sinon `LOADFILE`.

## Organisation de l'analyseur

| Fichier | Responsabilité |
| --- | --- |
| `src/analyzer/__init__.py` | Parse les arguments et traduit les erreurs en stderr/code 84. |
| `src/analyzer/application.py` | Charge le réseau et les positions une seule fois, puis sélectionne le mode. |
| `src/analyzer/prediction.py` | Encode les positions et affiche les prédictions dans l'ordre. |
| `src/analyzer/training.py` | Point d'entrée du futur apprentissage : réseau, positions et destination de sauvegarde. |

Le choix du mode est fait avant la boucle de prédiction. L'apprentissage pourra
ainsi gérer le jeu de données et plusieurs époques dans son propre parcours.

## Tests et couverture

```bash
make tests_run
```

Cette cible prépare automatiquement `.venv-tests`, installe les dépendances,
lance la suite `unittest`, affiche la couverture du code de `src/` (branches
comprises) et produit le rapport HTML `htmlcov/index.html`.
Une erreur dans la suite fait échouer la cible.

Pour lancer uniquement les tests de l'encodeur après préparation des dépendances :

```bash
PYTHONPATH=src .venv-tests/bin/python -m unittest tests.test_chessencoder -v
```

## Compilation et nettoyage

| Commande | Action |
| --- | --- |
| `make` | Construit les deux exécutables. |
| `make re` | Nettoie les fichiers générés puis reconstruit les exécutables. |
| `make clean` | Supprime les caches Python. |
| `make fclean` | Supprime aussi les exécutables, les données de couverture et le rapport HTML. |

Après toute modification de `src/`, relance `make` : les exécutables contiennent
une copie des sources. Les réseaux `.nn` et les environnements virtuels sont
conservés par les cibles de nettoyage.

## Problèmes courants

- **`No module named numpy`** : active `.venv` et installe les dépendances dans
  cet environnement. Le Python utilisé par l'exécutable est celui trouvé dans le `PATH`.
- **Le générateur refuse un fichier existant** : change le préfixe `name` ou le
  répertoire de sortie ; il protège les fichiers déjà présents.
- **Une modification du code ne change pas le résultat** : reconstruis les
  exécutables avec `make`.
- **Erreur de dimensions lors de l'analyse** : utilise un réseau généré avec
  838 entrées et des paramètres compatibles avec son architecture.
- **Les prédictions d'un réseau vierge sont incorrectes** : les poids aléatoires
  permettent de vérifier le parcours d'inférence ; il faudra un modèle entraîné
  pour évaluer la reconnaissance des états d'échecs.
