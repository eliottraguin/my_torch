# My Torch

Projet de réseau neuronal en Python et NumPy pour analyser des positions
d'échecs représentées en FEN.

Deux exécutables sont disponibles :

- `my_torch_generator` : crée des réseaux vierges à partir d'une configuration JSON.
- `my_torch_analyzer` : charge un réseau et calcule une prédiction par position.

**État actuel :** la génération et l'inférence sont implémentées. Les réseaux
générés possèdent des poids aléatoires et ne sont pas entraînés : leurs sorties
permettent de tester le fonctionnement du programme, sans démontrer une capacité
à reconnaître correctement les positions. Le mode entraînement ajuste ces poids
à partir de positions étiquetées.

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

Si des positions portent un label attendu, un bilan est affiché à la fin sur
stderr, par exemple `Prediction accuracy: 42/50 correct (84.00%); 0 unlabeled
position(s) excluded.` Le taux est le nombre de bonnes prédictions divisé par
le nombre de positions étiquetées. Les positions sans label sont exclues du
calcul ; un suffixe comme `White` ou `Black` ne participe pas à la comparaison
des trois classes actuelles. Sans label attendu, aucun taux n'est affiché.

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
  "max_epochs": 200,
  "patience": 10,
  "min_delta": 0.000001,
  "hidden_activation": "relu",
  "output_activation": "sigmoid",
  "labels": ["Checkmate", "Check", "Nothing"]
}
```

| Champ | Utilisation |
| --- | --- |
| `name` | Préfixe des fichiers produits, suivi de `_1.nn`, `_2.nn`, etc. |
| `layers` | Tailles successives : entrées, couches cachées, sorties. |
| `seed` | Graine du premier réseau ; elle augmente d'une unité pour chaque réseau suivant. |
| `init` | Poids tirés uniformément entre `-init` et `init`. Les biais commencent à zéro. |
| `learning_rate` | Taux utilisé pour les mises à jour pendant l'entraînement. |
| `max_epochs` | Nombre maximal d'époques, entier positif. Défaut : `200`. |
| `patience` | Nombre d'époques sans progrès significatif avant l'arrêt, entier positif. Défaut : `10`. |
| `min_delta` | Baisse de perte à dépasser pour remettre la patience à zéro, nombre fini supérieur ou égal à zéro. Défaut : `0.000001`. |
| `hidden_activation` | Activation des couches cachées : `sigmoid`, `relu` ou `silu`. Défaut : `sigmoid`. |
| `output_activation` | Activation de sortie. Conserver `sigmoid` pour comparer les couches cachées. Défaut : `sigmoid`. |
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

Pour entraîner un réseau depuis des positions étiquetées :

```bash
./my_torch_analyzer --train --save my_torch_network.nn basic_network_1.nn \
    tests/dataset_example/checkmate/10_pieces.txt
```

La destination est `SAVEFILE` si `--save` est fourni, sinon `LOADFILE`. Les messages
d'apprentissage vont sur stderr. Le modèle sauvegardé conserve ses activations.
Un checkpoint est écrit avant les mises à jour puis après chaque époque terminée.
Avec `Ctrl+C`, le programme restaure et sauvegarde la dernière époque terminée,
puis quitte avec le code 130. Une époque interrompue n'est pas conservée.
L'écriture utilise un fichier temporaire puis un remplacement atomique pour
préserver le précédent `.nn` si la sauvegarde échoue. Un arrêt forcé (`kill -9`)
ou une coupure ne permet pas d'exécuter la sauvegarde finale ; le dernier
checkpoint déjà écrit reste disponible.
Les anciens modèles sans champs d'activation utilisent sigmoid sur chaque couche.
Pour comparer ReLU et SiLU, génère deux modèles à partir de configurations ayant
respectivement `"hidden_activation": "relu"` et `"hidden_activation": "silu"`.
La propagation avant et la rétropropagation utilisent les mêmes activations et
leurs dérivées correspondantes ; la loss reste l'erreur quadratique actuelle.

L'entraînement réserve environ 20 % des groupes de positions à la validation,
avec une graine fixe de 42 et une répartition par classe quand les effectifs le
permettent. Deux entrées encodées identiques restent dans le même ensemble ;
des labels contradictoires pour une même entrée sont refusés. Ce découpage ne
regroupe pas automatiquement les positions différentes issues d'une même partie.
Pour une évaluation finale, garde également un fichier de test indépendant.

Après chaque époque, la perte de validation est calculée sans modifier les poids.
Par défaut, l'entraînement s'arrête après 10 époques sans amélioration supérieure
à `1e-6`, avec un maximum de 200 époques. Les meilleurs poids et biais sont restaurés avant
la sauvegarde, même si le meilleur modèle était celui du départ (époque 0).
Les pertes sont affichées en notation scientifique pour distinguer les petites
valeurs de zéro. La patience, le seuil, la fraction de validation et la graine
sont des arguments nommés de `run_training` ; la syntaxe CLI reste inchangée.
Les champs `max_epochs`, `patience` et `min_delta` du `.conf` sont écrits dans le
`.nn`, chargés et conservés à chaque sauvegarde. L'entraînement CLI utilise ces
valeurs ; les anciens modèles sans ces champs gardent les valeurs par défaut.
Les arguments explicites de `run_training` et les réglages choisis dans le
dashboard prennent priorité et sont conservés dans le modèle entraîné.
Modifier un `.conf` n'altère pas un `.nn` existant : régénère un réseau ou adapte
ces trois champs dans le JSON du modèle existant pour conserver ses poids.

Si le dataset est trop petit pour fournir une validation distincte, un avertissement
annonce que l'arrêt surveille uniquement la perte d'entraînement, sans mesure de
généralisation. Un dataset contenant une seule classe provoque aussi un avertissement :
une faible perte sur ce fichier ne prouve pas la reconnaissance des autres classes.

## Organisation de l'analyseur

| Fichier | Responsabilité |
| --- | --- |
| `src/analyzer/__init__.py` | Parse les arguments et traduit les erreurs en stderr/code 84. |
| `src/analyzer/application.py` | Charge le réseau et les positions une seule fois, puis sélectionne le mode. |
| `src/analyzer/prediction.py` | Encode les positions et affiche les prédictions dans l'ordre. |
| `src/analyzer/training.py` | Prépare les données ; `train_network` entraîne une époque, `evaluate_loss` évalue à poids fixes et `_run_epochs` gère les sauvegardes et interruptions. |
| `src/analyzer/early_stopping.py` | Suit la meilleure perte, compte les époques sans progrès significatif et conserve une copie des meilleurs poids. |

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

## Préparation de données équilibrées

Le script [`scripts/generate_dataset.py`](scripts/generate_dataset.py) prépare
les fichiers FEN existants pour ton réseau à cinq classes. Il vérifie les
positions, recalcule les labels, retire les doublons et équilibre les classes
ainsi que les groupes de matériel. Il produit des ensembles d'entraînement,
de validation et de test, avec un manifeste reproductible.

```bash
python -m pip install -r scripts/requirements-dataset.txt
python scripts/generate_dataset.py --input dataset/fusion.txt --output dataset/generated
```

Par défaut : 60 000 positions, dont 48 000 pour l'entraînement, 6 000 pour la
validation externe et 6 000 pour le test. L'analyseur conserve son découpage
interne de validation à partir du fichier d'entraînement. Les règles de
séparation, les options et les commandes sont détaillées dans
[`scripts/README.md`](scripts/README.md).

## Dashboard d'entraînement et de prédiction

```bash
make dashboard
```

Ouvre **http://127.0.0.1:8501**. Le dashboard permet de lancer et arrêter un
entraînement, suivre les courbes de loss et d'accuracy, analyser les prédictions
avec une matrice de confusion et comparer plusieurs expériences. Les graphiques
s'exportent en HTML autonome, les métriques et prédictions en CSV. Chaque session
conserve un modèle indépendant et son historique dans `.dashboard_runs/`.

Les dépendances graphiques sont installées séparément dans `.venv-dashboard`.
Les commandes CLI restent disponibles. Voir
[`bonus/dashboard/README.md`](bonus/dashboard/README.md) pour les métriques,
la validation séparée, les sauvegardes et les tests `make dashboard_tests`.
