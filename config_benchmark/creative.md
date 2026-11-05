# Comparaison des architectures

Dossier recréé à la racine du dépôt le **6 novembre 2026**, à la demande de l'équipe.
Les fichiers JSON ci-joints proviennent d'une nouvelle exécution du moteur du projet.
Aucune mesure manquante n'a été reconstituée à partir des seuls tableaux historiques.

## Protocole

- Réseaux neufs, SGD par position, perte quadratique, code de calcul du projet inchangé.
- Présélection : 3000 positions train, 2000 validation,
  au plus 40 époques, graine 42.
- Confirmation : 6000 train, 2000 validation,
  au plus 60 époques, graines 42, 43, 44.
- Patience 12, min_delta 1e-05. Le checkpoint retenu
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
| [baseline256](../tests/conf_example/architectures/baseline256.conf) | 256 | relu | 0.01 | 0.1 | 54.26 % | 18 |
| [deep64](../tests/conf_example/architectures/deep64.conf) | 64 → 64 → 64 → 64 | relu | 0.01 | 0.1 | 53.34 % | 21 |
| [expanding32_64_128](../tests/conf_example/architectures/expanding32_64_128.conf) | 32 → 64 → 128 | relu | 0.01 | 0.1 | 51.76 % | 14 |
| [classic128_64](../tests/conf_example/architectures/classic128_64.conf) | 128 → 64 | relu | 0.01 | 0.1 | 51.00 % | 12 |
| [diamond32_128_64_32](../tests/conf_example/architectures/diamond32_128_64_32.conf) | 32 → 128 → 64 → 32 | relu | 0.01 | 0.1 | 50.08 % | 25 |
| [wide512_128](../tests/conf_example/architectures/wide512_128.conf) | 512 → 128 | relu | 0.01 | 0.1 | 49.91 % | 8 |
| [bottleneck128_8_128](../tests/conf_example/architectures/bottleneck128_8_128.conf) | 128 → 8 → 128 | relu | 0.01 | 0.1 | 49.51 % | 14 |
| [pyramid256_128_64_32](../tests/conf_example/architectures/pyramid256_128_64_32.conf) | 256 → 128 → 64 → 32 | relu | 0.01 | 0.1 | 48.68 % | 19 |
| [deep128](../tests/conf_example/architectures/deep128.conf) | 128 → 128 → 128 → 128 | relu | 0.01 | 0.1 | 47.77 % | 11 |
| [zigzag86_42_4_82](../tests/conf_example/architectures/zigzag86_42_4_82.conf) | 86 → 42 → 4 → 82 | relu | 0.01 | 0.1 | 44.11 % | 30 |
| [deep64_init02](../tests/conf_example/architectures/deep64_init02.conf) | 64 → 64 → 64 → 64 | relu | 0.01 | 0.2 | 43.37 % | 5 |

## Confirmation

| Variante | F1 moyenne validation | Écart-type descriptif | Paramètres | Temps moyen |
|---|---|---|---|---|
| baseline256 | 53.18 % | 0.43 points | 216069 | 44.9 s |
| deep64 | 50.68 % | 1.63 points | 66501 | 29.3 s |

**Gagnant mesuré : `baseline256`.** Les écarts-types ne constituent pas une preuve de significativité.

## Test du gagnant

| Graine | Accuracy | F1 macro |
|---|---|---|
| 42 | 55.52 % | 54.03 % |
| 43 | 56.03 % | 53.84 % |
| 44 | 56.15 % | 54.80 % |

Moyennes : **55.90 % d'accuracy**, **54.22 % de F1 macro**.

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
  --configs tests/conf_example/architectures \
  --output /tmp/mytorch-creative-new \
  --screen-epochs 40 --final-epochs 60 --patience 12 \
  --final-seeds 42 43 44 --baseline baseline256
```

Choisir un **nouveau répertoire** : un dossier existant est refusé. Le script conserve les
configurations effectives, les modèles, les historiques et les métriques dans ce dossier.
Il n'entraîne ni ne remplace les modèles existants du dépôt. Le fichier `chess_recommended.conf`
produit dans la sortie sert à générer un nouveau modèle, qui devra ensuite être entraîné.

Les mesures complètes, versions et empreintes des sources/données sont dans [creative_results.json](creative_results.json).

## Graphiques et documentation

[Rapport illustré](../docs/benchmarks.md) · [Architectures](creative.md) · [Figures](figures/).

```bash
python -m pip install -r config_benchmark/requirements-plots.txt
python config_benchmark/plot_results.py
```

Cette dernière commande produit les PNG/SVG depuis `creative_results.json`, sans entraînement.
Le manifeste des figures indique l'empreinte exacte des résultats utilisés.
