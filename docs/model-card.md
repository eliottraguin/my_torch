# Fiche du modèle livré

Inspection au **5 novembre 2026**. Cette fiche décrit le fichier racine, indépendamment
des modèles expérimentaux de benchmark.

| Propriété | Valeur |
|---|---|
| Fichier | `my_torch_network_1.nn` |
| Présence Git | Suivi par Git ; le préfixe respecte le sujet. |
| Taille | 3,392,230 octets |
| SHA-256 | `b63970b67706b1be4743a74f520a6c674909bdacf600b475c72582a9d17dc19d` |
| Architecture | 838 → 128 → 64 → 5 |
| Activations | ReLU cachée, softmax en sortie |
| Learning rate enregistré | 0.001 |
| Paramètres d'arrêt enregistrés | max_epochs=200, patience=20, min_delta=1e-10 |
| Labels | Checkmate White, Checkmate Black, Check White, Check Black, Nothing |

Le fichier est chargeable par une zipapp reconstruite depuis les sources actuelles,
y compris depuis un autre répertoire sans `PYTHONPATH`. Sur la position initiale, il renvoie
`Nothing`, code 0. Cette vérification établit la compatibilité fonctionnelle, pas l'accuracy générale.

## Provenance et performances

Le `.nn` n'enregistre pas sa graine initiale, les empreintes des datasets, le nombre d'époques
réellement effectuées ou le checkpoint choisi. Ces valeurs **ne peuvent pas être déduites**
de `max_epochs` ou du nom du fichier. Aucun manifeste attaché à ce fichier ne permet ici
d'attester son historique complet d'entraînement.

Les benchmarks à 55,90 % d'accuracy et 54,22 % de F1 macro concernent des réseaux **256**
créés à neuf lors des expériences, pas ce réseau **128 → 64**. Ils ne doivent pas être
attribués au modèle racine. Un modèle ayant déjà appris sur `fusion.txt` peut avoir vu
les positions de `dataset/generated/test.txt` ; son score sur ce jeu ne serait pas indépendant.

Pour une fiche de livraison entièrement reproductible, associer au modèle final :

- configuration initiale et graine ;
- version du code et dépendances ;
- sources, manifeste et empreintes des datasets ;
- commande exacte ou configuration dashboard ;
- historique, critère d'arrêt et époque restaurée ;
- mesures sur un jeu réservé et SHA-256 du fichier évalué.

Le dashboard conserve une partie de ces éléments dans `.dashboard_runs/<expérience>/`.
Le benchmark conserve ses réglages et résultats dans son dossier de sortie. Ne pas présenter
ces archives comme la provenance de ce modèle sans vérifier que son empreinte correspond.

## Usage et limites

Ce modèle classe des positions avec l'encodage 838 documenté dans [formats.md](formats.md).
Il ne génère pas de coups, ne garantit pas une décision correcte, ne distingue pas le pat et
ne traite pas l'historique des répétitions. Les scores softmax ne sont pas calibrés.

Le fichier est déjà suivi malgré la règle `*.nn` du `.gitignore`. Pour livrer ultérieurement
un **nouveau** modèle, il faudra vérifier explicitement son suivi Git. La présente tâche
n'a ni entraîné, ni remplacé, ni ajouté de poids à la place du modèle livré.
