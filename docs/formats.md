# Formats et conventions

## Configuration `.conf`

Les extensions `.conf` et `.nn` contiennent du JSON propre à ce projet. Il ne s'agit pas
d'un format universel compatible avec n'importe quel logiciel de réseau neuronal.

```json
{
  "name": "my_torch_network_demo",
  "layers": [838, 256, 5],
  "seed": 42,
  "init": 0.1,
  "learning_rate": 0.01,
  "hidden_activation": "relu",
  "output_activation": "softmax",
  "labels": ["Checkmate White", "Checkmate Black", "Check White", "Check Black", "Nothing"],
  "max_epochs": 40,
  "patience": 8,
  "min_delta": 0.00001
}
```

| Champ | Sémantique |
|---|---|
| `name` | Préfixe de fichier ; la génération ajoute `_1.nn`, `_2.nn`, etc. |
| `layers` | Tailles positives des couches, entrées et sorties comprises. Pour les FEN : entrée 838. |
| `seed` | Graine du tirage initial ; incrémentée pour chaque réseau du même couple de génération. |
| `init` | Poids uniformes dans `[-init, init)` ; tous les biais commencent à zéro. Pas une borne après entraînement. |
| `learning_rate` | Facteur de mise à jour SGD. Exemple mesuré : 0.01. |
| `hidden_activation` | `sigmoid`, `relu`, `silu` ou `softmax` dans le registre actuel. Sigmoid si absent. Softmax cachée n'a pas été évaluée ici. |
| `output_activation` | Même registre ; softmax est utilisée pour les cinq classes exclusives. Sigmoid si absent. |
| `labels` | Ordre exact des classes, une par sortie. L'argmax utilise cet ordre. |
| `max_epochs` | Entier positif, hors booléens. Défaut si absent : 200. |
| `patience` | Entier positif, hors booléens. Défaut : 10. |
| `min_delta` | Nombre fini ≥ 0. Défaut : 1e-6. Baisse significative de perte, pas objectif de perte finale. |

Les dimensions, labels et types indiqués sont le contrat à respecter ; le chargeur ne
valide pas encore exhaustivement toutes les combinaisons malformées. Les trois paramètres
d'arrêt disposent d'une validation commune dans `src/training_config.py`.

`basic_network.conf` utilise trois sorties et, faute de champs explicites, sigmoid.
Les configurations d'échecs récentes utilisent cinq sorties, ReLU et softmax.
Un changement du `.conf` ne modifie pas les `.nn` déjà générés.

## Modèle `.nn`

Le modèle conserve `layers`, `learning_rate`, `labels`, les deux activations, les trois
paramètres d'arrêt, `weights` et `biases`. Il ne conserve ni le préfixe `name`, ni la graine
initiale, ni `init`, ni les sources de données ou l'historique des époques. Archiver le `.conf`
et le manifeste d'expérience à côté du modèle pour pouvoir le reproduire.

Exemple minimal complet de structure **générique à deux entrées**, non compatible avec l'encodeur FEN :

```json
{
  "layers": [2, 2],
  "learning_rate": 0.01,
  "labels": ["classe_0", "classe_1"],
  "hidden_activation": "relu",
  "output_activation": "softmax",
  "max_epochs": 20,
  "patience": 5,
  "min_delta": 0.00001,
  "weights": [[[0.1, -0.2], [0.3, 0.4]]],
  "biases": [[0.0, 0.0]]
}
```

Pour une transition de `n` vers `m` neurones : `weights[i].shape == (m, n)` et
`biases[i].shape == (m,)`. Il faut exactement `len(layers)-1` matrices et vecteurs.
Les nombres sont stockés comme listes JSON ; NumPy les reconstruit au chargement.
Les anciens fichiers sans paramètres d'arrêt utilisent 200/10/1e-6, et sans activations sigmoid.
Aucun numéro de version de l'encodage n'est enregistré : un modèle à 838 entrées doit aussi
respecter **l'ordre** décrit ci-dessous, pas uniquement le nombre d'entrées.

## FEN et labels

```text
<placement> <trait> <roques> <en-passant> <demi-coups> <numéro-coup> [label]
4k3/8/8/8/8/8/4R3/4K3 b - - 0 12 Check White
```

Les rangées vont de 8 à 1, les colonnes de a à h. Les majuscules sont les pièces blanches,
les minuscules les noires ; `P N B R Q K` signifient pion, cavalier, fou, tour, dame, roi.
Un chiffre compte les cases vides consécutives. Chaque rangée doit développer exactement huit cases.

- Trait : `w` ou `b`.
- Roques : lettres parmi `KQkq`, sans doublons, ou `-`.
- En passant : case cible ou `-` ; ce champ ne signifie pas qu'une capture a déjà eu lieu.
- Demi-coups : compteur depuis la dernière capture ou le dernier mouvement de pion.
- Numéro du coup : incrémenté après le coup des Noirs ; ne détermine pas à lui seul si la partie est finie.
- Label : suffixe facultatif en prédiction, nécessaire pour un exemple d'entraînement utilisable.

L'annotation `Check White` signifie que les Blancs donnent échec au roi noir. `Checkmate Black`
signifie que les Noirs donnent mat. Avec cinq sorties, garder les deux mots et leur casse exacte.
L'entraînement compare le label complet : `Check` n'est pas équivalent à `Check White`.
Le dashboard peut réduire les annotations colorées pour évaluer un modèle à trois sorties ;
cette adaptation n'existe pas automatiquement dans l'entraînement CLI.

## Vecteur d'entrée de 838 valeurs

| Indices inclus, base zéro | Nombre | Contenu |
|---|---:|---|
| 0–767 | 768 | 64 cases × 12 pièces, aplatissement de `(8, 8, 12)` ; cases a8…h1 et canaux `PNBRQKpnbrqk`. |
| 768 | 1 | Trait : blanc = 0, noir = 1. |
| 769–832 | 64 | Case en passant en one-hot, ordre a8…h1 ; tout zéro si `-`. |
| 833–836 | 4 | Droits de roque dans l'ordre `K Q k q`, indicateurs indépendants. |
| 837 | 1 | Compteur de demi-coups brut, non normalisé. |

Le vecteur est de type `float32`, de forme exacte `(838,)`. Le numéro du coup n'est pas
encodé et le label n'est jamais inclus. Les pièces et les droits de roque ne décrivent pas
l'historique des répétitions : le réseau ne peut pas en déduire la règle des répétitions.

L'encodeur contrôle la structure, le trait et certains champs. Il n'est pas un validateur
complet de légalité d'échecs. Les données préparées utilisent en plus `chess.Board.is_valid()` ;
même ce contrôle ne prouve pas la provenance d'une partie légale complète.
Un demi-coup démesuré peut actuellement dépasser `float32` et devenir infini : défaut connu.
