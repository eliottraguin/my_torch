# Comparaison des réglages

Dossier recréé à la racine du dépôt le **6 novembre 2026**, à la demande de l'équipe.
Les fichiers JSON ci-joints proviennent d'une nouvelle exécution du moteur du projet.
Aucune mesure manquante n'a été reconstituée à partir des seuls tableaux historiques.

## Protocole

- Réseaux neufs, SGD par position, perte quadratique, code de calcul du projet inchangé.
- Présélection : 3000 positions train, 2000 validation,
  au plus 20 époques, graine 42.
- Confirmation : 6000 train, 2000 validation,
  au plus 40 époques, graines 42, 43.
- Patience 8, min_delta 1e-05. Le checkpoint retenu
  minimise la perte de validation ; les configurations sont classées par F1 macro moyenne.
- Même échantillonnage par classe entre candidats, graine 20261105.
- Séparation des familles de placement/trait et de leurs miroirs vérifiée entre train,
  validation et test. Pas de garantie de séparation par partie faute d'identifiants de parties.
- Test sur 6 000 positions après choix du gagnant. Ce jeu avait déjà été consulté lors
  des expériences précédentes ; ce n'est pas un nouveau test externe.
- BLAS limité à un thread. Les durées incluent les époques et évaluations de validation,
  hors encodage, génération et sauvegardes JSON. Elles dépendent de la machine et de sa charge.

## Présélection

Toutes les architectures ont 838 entrées et 5 sorties. Les valeurs ci-dessous sont celles
restaurées au minimum de perte, pas nécessairement au maximum de F1.

| Variante | Couches cachées | Activation | LR | Init | F1 validation | Meilleure époque |
|---|---|---|---|---|---|---|
| relu256 | 256 | relu | 0.01 | 0.1 | 54.26 % | 18 |
| relu128_64 | 128 → 64 | relu | 0.01 | 0.1 | 51.00 % | 12 |
| relu64 | 64 | relu | 0.01 | 0.1 | 50.61 % | 12 |
| relu256_lr0003 | 256 | relu | 0.003 | 0.1 | 49.93 % | 18 |
| relu128_64_init02 | 128 → 64 | relu | 0.01 | 0.2 | 48.25 % | 8 |
| silu128_64_lr003 | 128 → 64 | silu | 0.03 | 0.1 | 47.74 % | 5 |
| relu128_64_lr003 | 128 → 64 | relu | 0.03 | 0.1 | 47.41 % | 5 |
| relu128_64_lr0003 | 128 → 64 | relu | 0.003 | 0.1 | 44.14 % | 19 |
| relu128_64_lr0001 | 128 → 64 | relu | 0.001 | 0.1 | 31.69 % | 20 |

## Confirmation

| Variante | F1 moyenne validation | Écart-type descriptif | Paramètres | Temps moyen |
|---|---|---|---|---|
| relu256 | 53.29 % | 0.49 points | 216069 | 37.6 s |
| relu128_64 | 52.81 % | 1.09 points | 115973 | 24.0 s |

**Gagnant mesuré : `relu256`.** Les écarts-types ne constituent pas une preuve de significativité.

## Test du gagnant

| Graine | Accuracy | F1 macro |
|---|---|---|
| 42 | 55.52 % | 54.03 % |
| 43 | 56.03 % | 53.84 % |

Moyennes : **55.77 % d'accuracy**, **53.93 % de F1 macro**.

## Limites

Il s'agit d'une recherche limitée sur des sous-ensembles, pas d'un optimum global.
Les paramètres ne sont pas optimisés séparément pour chaque architecture. Les faibles taux
peuvent nécessiter plus d'époques. Les scores ne décrivent pas le modèle livré à la racine.
Le nombre de paramètres ou la profondeur ne suffisent pas à garantir la qualité.
Le test déjà utilisé ne doit pas servir à poursuivre le choix des hyperparamètres.

## Reproduction

Depuis la racine du dépôt, avec `dataset/generated/` disponible et les dépendances
`tests/requirements-dev.txt` installées dans `.venv-tests` :

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv-tests/bin/python config_benchmark/benchmark.py \
  --output /tmp/mytorch-settings-new
```

Choisir un **nouveau répertoire** : un dossier existant est refusé. Le script conserve les
configurations effectives, les modèles, les historiques et les métriques dans ce dossier.
Il n'entraîne ni ne remplace les modèles existants du dépôt. Le fichier `chess_recommended.conf`
produit dans la sortie sert à générer un nouveau modèle, qui devra ensuite être entraîné.

Les mesures complètes, versions et empreintes des sources/données sont dans [results.json](results.json).

## Graphiques et documentation

[Rapport illustré](../docs/benchmarks.md) · [Architectures](creative.md) · [Figures](figures/).

```bash
python -m pip install -r config_benchmark/requirements-plots.txt
python config_benchmark/plot_results.py
```

Cette dernière commande produit les PNG/SVG depuis `creative_results.json`, sans entraînement.
Le manifeste des figures indique l'empreinte exacte des résultats utilisés.
