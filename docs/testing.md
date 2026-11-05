# Validation, couverture et limites connues

| Contrôle | Résultat |
|---|---|
| Unittest principal | 156 tests exécutés, 4 échecs, aucune erreur de suite. |
| Dashboard | 8 tests réussis, dont simulateur Streamlit et entraînement réel court. |
| Couverture | 93 % au total, 675 instructions, 192 branches ; détail JSON fourni. |
| Makefile en copie isolée | `all`, `re`, `clean`, `fclean`, puis `all` réussis. |
| Zipapp depuis `/tmp`, sans `PYTHONPATH` | Aides réussies ; modèle livré chargé et position initiale prédite `Nothing`. |
| CLI sans arguments | Code 84, diagnostic sur stderr. |
| Modèle malformé avec `layers: []` | Code 1 et traceback : défaut connu, non conforme au contrat 84. |
| 11 configurations expérimentales | Génération, propagation et une étape de rétropropagation réussies, sorties finies normalisées. |

Le build isolé utilise une copie de `src/`, `tests/` et du Makefile : aucune suppression
n'est effectuée dans le répertoire utilisateur pour vérifier les règles de nettoyage.
Le test d'une position ne mesure pas la qualité générale du modèle livré.

## Exécuter les tests

```bash
make tests_run
make dashboard_tests
```

`make tests_run` crée `.venv-tests` et installe `tests/requirements-dev.txt`. Le fichier marqueur
`.dependencies-installed` évite de réinstaller à chaque invocation. Si les dépendances sont
incomplètes, réinstaller explicitement :

```bash
.venv-tests/bin/python -m pip install -r tests/requirements-dev.txt
```

Dans l'état actuel, l'échec unittest arrête Make avant `coverage report` et `coverage html`.
Les données `.coverage` ont néanmoins été écrites. Pour consulter les résultats :

```bash
.venv-tests/bin/python -m coverage report
.venv-tests/bin/python -m coverage html
```

Ouvrir `htmlcov/index.html`. `.coveragerc` fixe `source=src`, active les branches, affiche les
lignes manquantes et choisit `htmlcov/`. Le dashboard et les scripts ne sont pas inclus dans
ce pourcentage, même si des tests les exercent. Les sous-processus ne sont pas instrumentés
par cette configuration. Un module couvert à 100 % peut encore contenir un défaut.

## Les quatre échecs constatés

| Test | Diagnostic |
|---|---|
| `tests.test_analyzer_application.PredictionTests.test_predictions_keep_position_order_and_display_format` | Attend encore les bannières `FEN game … Confidence …`, alors que stdout contient désormais les labels seuls. |
| `tests.test_analyzer_application.PredictionTests.test_accuracy_excludes_unlabeled_positions_and_ignores_color_suffix` | Attend un bilan d'accuracy sur stderr retiré de la CLI. |
| `tests.test_zipapp.AnalyzerZipappTests.test_prediction_loads_labeled_records_with_mode_last` | Même attente de bilan retiré. |
| `tests.test_chessencoder.ChessEncoderTests.test_rejects_invalid_halfmove_counters` | Un compteur extrêmement grand est converti en `float32` infini au lieu d'être rejeté. |

Les trois premières assertions doivent être alignées sur le contrat de sortie retenu ; la
quatrième appelle une correction de validation numérique. Cette tâche de documentation
n'a modifié ni les tests ni le moteur pour masquer les échecs.

## Autres limites vérifiables dans le code

- Le chargeur `.nn` ne valide pas exhaustivement nombre de matrices, dimensions, labels et valeurs finies.
  Des données malformées peuvent provoquer une exception non traduite en 84.
- `ConfigFile.load_config` affiche certaines erreurs de JSON sur stdout et retourne `None` ;
  le générateur peut ensuite échouer sans respecter le contrat de sortie.
- Les poids et biais sont parcourus avec `zip` : un fichier incohérent n'est pas forcément rejeté tôt.
- `predict` vérifie la longueur d'entrée, pas toutes les formes possibles ; fournir un vecteur 1D.
- Les contrôles FEN ne démontrent pas la légalité complète d'une position ni son historique.
- Une erreur rencontrée en cours de prédiction peut laisser des lignes de sortie déjà émises.
- Le compteur de demi-coups reste brut, sans normalisation ; le modèle ne représente pas l'historique des répétitions.
- L'environnement cible « dump » n'a pas été vérifié directement dans cette session.

## CI et livraison

Le workflow `.github/workflows/mirror.yml` copie le dépôt sur un push de `main`, seulement
depuis `eliottraguin/my_torch`. Il utilise les secrets `GIT_REPO_URL` et `GIT_SSH_PRIVATE_KEY`.
Il ne lance pas `make tests_run`. Un miroir réussi ne signifie donc pas que le code est validé.
Aucune valeur de secret n'est nécessaire à la documentation et ne doit y être copiée.

## Dépannage

| Symptôme | Action |
|---|---|
| NumPy absent | Activer le bon venv ; installer les dépendances avec son Python. |
| Coverage absent | Passer par `make tests_run` ou installer dans `.venv-tests`. |
| Import zipapp ancien | Refaire `make` ; ne pas lancer une archive construite avant la modification. |
| Label inconnu | Vérifier l'ordre et le texte des labels du `.nn` ; préparer les données avec `--classes 3` si nécessaire. |
| Loss très faible sur une classe | Ne pas conclure à une bonne classification ; mesurer sur plusieurs classes et une validation distincte. |
| Dashboard avec ancien cache | Rafraîchir ; si le processus conserve d'anciens imports après modification de code, le redémarrer. |
| Fichier généré déjà présent | Changer de préfixe ou de répertoire ; ne pas supprimer un modèle utile pour contourner la protection. |
