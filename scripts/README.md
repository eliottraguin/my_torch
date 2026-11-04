# Préparer les datasets d'entraînement

`generate_dataset.py` construit de nouveaux fichiers d'entraînement à partir de
fichiers FEN existants, étiquetés ou non. Il utilise `chess` uniquement pour la
préparation et la vérification des données ; l'analyseur reste un réseau NumPy.

Depuis la racine du dépôt, dans ton environnement virtuel :

```bash
python -m pip install -r scripts/requirements-dataset.txt
python scripts/generate_dataset.py --input dataset/fusion.txt --output dataset/generated
```

Avec les paramètres par défaut et des sources suffisamment variées :

| Fichier | Positions | Utilisation |
| --- | ---: | --- |
| `dataset/generated/train.txt` | 48 000 | Apprentissage ; l'analyseur en réserve encore une partie à sa validation interne. |
| `dataset/generated/validation.txt` | 6 000 | Comparaison de modèles et de réglages avec un outil d'évaluation. |
| `dataset/generated/test.txt` | 6 000 | Évaluation finale après le choix du modèle. |
| `dataset/generated/manifest.json` | — | Audit, paramètres, statistiques et empreintes SHA-256. |

`--per-class 12000` signifie 12 000 positions **par classe, tous ensembles
confondus**. Avec cinq classes, cela produit 60 000 positions. Chaque classe
contient autant de positions avec 2–10, 11–20 et 21–32 pièces. Les fichiers sont
mélangés et leurs labels utilisent l'ordre sémantique de ta configuration actuelle :

```text
Checkmate White
Checkmate Black
Check White
Check Black
Nothing
```

La couleur désigne le camp qui donne l'échec ou le mat, donc l'opposé du joueur au
trait. Un pat est classé `Nothing`, conformément aux classes actuelles. Les labels
sont recalculés : des annotations source incorrectes sont corrigées et comptées.
Les positions échouant aux contrôles de validité de python-chess sont rejetées.
Ces contrôles ne démontrent pas qu'une position arbitraire est accessible depuis
la position initiale par une partie légale complète.

## Lancer l'apprentissage

```bash
make
mkdir -p dataset/generated/initial
(cd dataset/generated/initial && ../../../my_torch_generator \
    ../../../tests/conf_example/chess_network.conf 1)
./my_torch_analyzer --train --save my_torch_network_balanced.nn \
    dataset/generated/initial/my_torch_network_1.nn dataset/generated/train.txt
./my_torch_analyzer --predict my_torch_network_balanced.nn \
    dataset/generated/test.txt > dataset/generated/test_predictions.txt
```

Ne donne pas `test.txt` à `--train`. La redirection ci-dessus enregistre les
prédictions ; elle ne calcule pas automatiquement une accuracy. Le fichier de
test contient les réponses attendues pour qu'un évaluateur puisse les comparer.

L'analyseur actuel choisit sa validation **dans le fichier passé à `--train`**.
Il ne charge pas automatiquement `validation.txt` : ce fichier supplémentaire
est disponible pour tes comparaisons externes. Ne mélange pas les trois fichiers.
Pour comparer des architectures, pars de réseaux vierges et conserve les mêmes
sources, graines et paramètres de préparation.
Un modèle déjà entraîné sur `fusion.txt` a potentiellement vu les futures positions
de test : réutiliser ce modèle ne fournirait pas une évaluation indépendante.

## Autres commandes

```bash
# Essai plus petit : 15 000 positions au total pour cinq classes.
python scripts/generate_dataset.py --input dataset/fusion.txt \
    --output dataset/small --per-class 3000 --seed 42

# Utiliser directement les neuf fichiers, sans fusion préalable.
python scripts/generate_dataset.py \
    --input dataset/check/*.txt dataset/checkmate/*.txt dataset/nothing/*.txt \
    --output dataset/from_sources

# Compatibilité avec un réseau à trois sorties.
python scripts/generate_dataset.py --input dataset/fusion.txt \
    --output dataset/three_classes --classes 3

# Sources contenant peu de positions dans certains groupes de matériel.
python scripts/generate_dataset.py --input mes_positions.txt \
    --output dataset/custom --per-class 1000 --no-material-balance
```

Si un groupe ne contient pas assez de positions, la commande échoue avec le code
84 et précise le groupe manquant. Elle ne duplique pas artificiellement les
positions pour atteindre le quota. Un dossier de sortie existant est refusé :
choisis un autre `--output`. La publication se fait seulement lorsque tous les
fichiers sont prêts.

## Reproductibilité et limites de la séparation

- Même liste de sources, mêmes contenus, même version du script/de `chess` et
  mêmes paramètres : mêmes fichiers de positions. Le manifeste conserve les
  empreintes, les chemins sources et les réglages ; aucun horodatage aléatoire
  n'est introduit.
- Les répétitions ne différant que par les compteurs FEN sont supprimées.
- Le placement et le joueur au trait déterminent une famille ; son miroir avec
  couleurs inversées appartient à la même famille. Une famille entière reste
  dans un seul ensemble, même si ses droits de roque ou sa case de prise en
  passant diffèrent. Le script n'ajoute pas automatiquement de positions miroirs.
- La sélection par réservoir donne à chaque position unique d'un même groupe
  une chance égale d'être retenue, sans conserver toutes les positions en mémoire.
- Un fichier FEN n'indique pas l'identifiant de la partie d'origine. Des positions
  différentes provenant d'une même partie peuvent donc traverser les ensembles.
  Pour mesurer la généralisation à de nouvelles parties, réserve des parties
  entièrement distinctes pour un test externe, absent des sources du script.
- Les ensembles équilibrés servent aux expériences contrôlées. Leur répartition
  n'est pas une estimation de la fréquence des classes dans des parties réelles.

Conserve les sources : le sujet demande de pouvoir reproduire l'entraînement.
Les données sont ignorées par Git, mais le script et ses dépendances sont suivis.

## Tests

```bash
make tests_run
# Tests ciblés, une fois les dépendances installées :
PYTHONPATH=src python -m unittest tests.test_dataset_generation -v
```
