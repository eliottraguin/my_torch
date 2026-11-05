# My Torch — Chess Lab

Dashboard local en français pour entraîner, évaluer et comparer les réseaux NumPy
existants. Les graphiques sont interactifs et exportables pour la soutenance.

## Démarrage

Depuis la racine du dépôt :

```bash
make dashboard
```

La cible crée `.venv-dashboard`, installe les dépendances facultatives puis lance
l'application sur **http://127.0.0.1:8501**. Le serveur écoute uniquement en local.
L'environnement de l'analyseur reste distinct. Pour changer de port :

```bash
make dashboard DASHBOARD_PORT=8502
```

## Entraînement

- Créer un réseau vierge ou reprendre un modèle `.nn` présent à la racine ou dans
  une expérience antérieure.
- Choisir les données d'entraînement et, si disponible, un fichier de validation
  distinct. Sans ce fichier, le découpage interne habituel est utilisé.
- Régler les couches cachées du réseau vierge, les activations, la graine, le taux
  d'apprentissage, le nombre d'époques, la patience et `min_delta`.
- La limite de lignes vaut zéro par défaut : toutes les lignes sont examinées.
  Choisir une limite pour un essai rapide ; elle s'applique séparément à chaque fichier.
- Le suivi affiche l'époque 0 puis chaque époque : perte quadratique moyenne,
  accuracy, durée, meilleure perte, matrice de confusion et scores par classe.
- Le bouton d'arrêt répond entre les exemples et conserve la dernière époque
  complète. À la fin normale ou avec early stopping, les meilleurs poids sont restaurés.

Les métriques des courbes sont calculées à poids fixes sur les ensembles
correspondants. Leur calcul nécessite un passage supplémentaire sur les données
d'entraînement ; l'instrumentation du dashboard peut donc ralentir l'apprentissage.
Les valeurs détaillées affichées sont celles de la dernière époque évaluée. Le
modèle restauré à la fin peut correspondre à une époque antérieure, indiquée par
`best_epoch`. L'export CSV permet de retrouver les valeurs de cette époque.

La CLI continue d'utiliser son comportement habituel. Les nouvelles options
`observer`, `max_epochs`, `should_stop` et `validation_positions` sont facultatives
sur `run_training`. Les métriques supplémentaires ne sont pas calculées sans observateur.

## Prédiction

Sélectionner un modèle et un fichier FEN, ou saisir une ou plusieurs FEN directement.
Les positions sans labels restent prédictibles ; elles sont exclues des métriques.
Les labels inconnus sont signalés et exclus du score. Un modèle à trois sorties
peut être évalué contre des annotations avec couleur, en comparant leur état
`Check`, `Checkmate` ou `Nothing`. Un modèle à cinq sorties compare le label complet.

Les résultats comprennent :

- Accuracy : bonnes réponses / positions avec label compatible.
- Précision par classe : bonnes prédictions de cette classe / toutes ses prédictions.
- Rappel : bonnes réponses de cette classe / toutes ses positions attendues.
- F1 : moyenne harmonique de la précision et du rappel.
- F1 macro : moyenne des F1 sur toutes les classes du modèle ; une classe absente
  avec F1 nul contribue à cette moyenne.
- Matrice de confusion : **attendus en lignes, prédictions en colonnes**, avec
  normalisation facultative par ligne.
- Tableau des résultats, filtre des erreurs, dessin de l'échiquier et scores de
  chaque sortie pour la position choisie.

Les scores softmax sont normalisés, mais ne constituent pas une mesure de
confiance calibrée. Avec une autre activation, les scores ne sont pas nécessairement
normalisés. Les rapports HTML incorporent Plotly et se consultent sans connexion ;
les prédictions et les historiques sont aussi exportables en CSV.

## Comparaison et données

L'onglet de comparaison superpose les courbes des expériences enregistrées et
présente leur meilleure époque observée. Pour une comparaison pertinente, garder
les mêmes sources, le même découpage et le même protocole. Un modèle arrêté
manuellement conserve la dernière époque complète, qui peut différer de la meilleure
époque observée dans le tableau.

L'explorateur de données affiche les classes, le nombre de pièces, le joueur au
trait et les erreurs d'encodage. Son profil est limité à 10 000 lignes par défaut ;
la limite est affichée et peut être retirée. Ces profils ne constituent pas un
contrôle exhaustif de la légalité des positions : ils utilisent les critères de
l'encodeur de l'analyseur.

## Conservation des expériences

Chaque lancement crée `.dashboard_runs/<date>-<identifiant>/` avec :

```text
initial.nn       modèle de départ indépendant
model.nn         checkpoint, puis meilleur modèle à la fin normale
config.json      réglages, classes, erreurs de lecture et empreintes SHA-256
history.jsonl    métriques par époque et événement de fin
result.json      état final ou erreur
```

Ces fichiers sont ignorés par Git. Télécharger ou copier le modèle final sous un
nom `my_torch_network…` puis le versionner pour le rendu. L'application n'écrase
jamais le fichier réseau sélectionné comme source.

L'historique reste disponible après un redémarrage du serveur. Les courbes ne
peuvent pas être reconstituées rétroactivement pour un entraînement CLI ancien
qui n'avait pas enregistré ses métriques. Une fermeture du navigateur laisse le
travail continuer tant que le serveur tourne. Un arrêt brutal du serveur peut
perdre l'époque partielle ; le dernier checkpoint déjà écrit reste sur disque.

Pour une évaluation finale, réserver un test indépendant, jamais utilisé pour
l'entraînement ou le choix des réglages. Un modèle déjà entraîné sur `fusion.txt`
a potentiellement déjà vu les lignes du dataset généré à partir de ce fichier.

## Vérification

```bash
make dashboard_tests
# Tests d'instrumentation du moteur, dans l'environnement habituel :
PYTHONPATH=src .venv-tests/bin/python -m unittest tests.test_training_monitoring -v
```

Les tests couvrent de vrais entraînements courts, la préservation du modèle
source, les métriques, les exports HTML et les interactions de l'interface avec
le simulateur officiel Streamlit AppTest.

## Réglages persistants et documentation

Les champs `max_epochs`, `patience` et `min_delta` sont conservés dans les `.nn`.
L’inspection applique 200, 10 et 1e-6 aux anciens fichiers qui les omettent.
Les contrôles du formulaire sont préremplis depuis le modèle choisi ; les réglages
explicitement saisis sont ceux utilisés et sauvegardés pour l’expérience.
Voir les [formats](../../docs/formats.md), les [règles de sauvegarde](../../docs/training.md),
le [bilan de tests](../../docs/testing.md) et le [rapport illustré](../../docs/benchmarks.md).
