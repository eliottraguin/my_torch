# Apprentissage, analyse numérique et choix

## Problème supervisé

Chaque exemple associe le vecteur FEN `x` et un label attendu, transformé en cible one-hot.
Pour cinq classes, `Check White` devient `[0, 0, 1, 0, 0]` dans l'ordre des configurations actuelles.
Le réseau prédit une classe, pas une liste de coups ni une stratégie complète de jeu.

À chaque couche : `z = W @ a + b`, puis `a_suivante = activation(z)`.
`z` est la **préactivation**, avant transformation non linéaire ; `a` est l'activation transmise.
Les matrices ont la forme `(sorties, entrées)`. Sans non-linéarité, empiler ces couches ne
produirait qu'une transformation affine équivalente.

## Activations et dérivées implémentées

| Fonction | Calcul | Dérivée par rapport à z |
|---|---|---|
| Sigmoid | `s = 1 / (1 + exp(-z))` | `s * (1-s)` |
| ReLU | `max(0, z)` | 1 si `z > 0`, 0 sinon, y compris en zéro par convention. |
| SiLU | `z * sigmoid(z)` | `s + z*s*(1-s)` |
| Softmax | `p_i = exp(z_i-max(z)) / sum(exp(z-max(z)))` | Matrice `J = diag(p) - outer(p,p)`. |

La sigmoid est calculée avec `exp(-abs(z))` et une branche selon le signe pour limiter
les débordements. Softmax soustrait le maximum avant l'exponentielle. La jacobienne actuelle
est conçue pour un vecteur ; le moteur n'effectue pas de rétropropagation par mini-batch.
Les scores softmax somment à 1, sans garantie de calibration de confiance.

ReLU cachée et softmax en sortie répondent à deux rôles différents : construire des
caractéristiques non linéaires puis répartir les scores entre des classes exclusives.
Ce couple est un choix empirique, pas une preuve qu'il est optimal pour toutes les données.

## Perte et rétropropagation

La perte d'un exemple est `L = 0.5 * sum((prediction - cible)**2)`.
Le programme moyenne cette quantité sur les exemples, **pas sur les neurones de sortie**.
Il s'agit d'une perte quadratique ; la cross-entropy n'est pas implémentée.

1. Le gradient par rapport à la sortie est `prediction - cible`.
2. Pour une activation élément par élément, multiplier par `activation.derivative(z)`.
3. Pour softmax, appliquer `J.T @ gradient_output`.
4. Pour une couche : `dW = outer(delta, activation_précédente)` et `db = delta`.
5. Propager vers la couche précédente avec `W.T @ delta`, avant de modifier cette matrice.
6. Mettre à jour `W -= learning_rate*dW`, `b -= learning_rate*db`.

Le raccourci `delta = prediction - cible` en sortie correspond à softmax avec une perte
cross-entropy adaptée ; il ne remplace pas la jacobienne pour la perte quadratique actuelle.

L'algorithme est **SGD** : une mise à jour par position, ordre mélangé à chaque époque avec
un générateur aléatoire reproductible. Il n'y a pas d'Adam, Momentum, mini-batch, dropout ou L2.
L'initialisation est uniforme avec amplitude `init`, pas Xavier ou He. Ces alternatives
peuvent être explorées, mais elles ne figurent pas dans les résultats actuels.

## Validation et lutte contre le surapprentissage

Par défaut, l'entraînement sépare environ 20 % des groupes d'entrées par classe pour la
validation. Les vecteurs strictement identiques restent ensemble ; des cibles contradictoires
pour le même vecteur sont refusées. Le compteur de demi-coups fait partie du vecteur :
ce split interne n'assure pas le regroupement de toutes les variantes d'une même position.
Le préparateur de données réalise une séparation plus forte par familles et miroirs.

La validation ne reçoit aucune mise à jour de poids. Sa perte est calculée à paramètres fixes.
La perte d'entraînement affichée par la CLI est une moyenne **en ligne**, calculée pendant les
mises à jour ; elle n'est pas exactement la perte du modèle final sur le corpus d'entraînement.
Le dashboard recalcule aussi des métriques à poids fixes, au prix d'un passage supplémentaire.

Le programme prévient lorsqu'une seule classe est présente. Une loss proche de zéro sur
`Nothing` seul ne prouve pas la capacité à reconnaître échecs et mats. Si le dataset est trop
petit pour former une validation, le suivi se rabat sur la perte d'entraînement avec avertissement.

## Arrêt anticipé

- Évaluer le réseau initial, à l'époque 0.
- Chaque **nouveau minimum strict** de perte conserve une copie des poids et biais.
- Une baisse strictement supérieure à `min_delta` par rapport à la dernière référence
  significative remet le compteur d'attente à zéro.
- Sinon, incrémenter ce compteur ; arrêter lorsqu'il atteint `patience`.
- À la fin normale, restaurer le minimum de perte enregistré, même si la meilleure époque est 0.

Exemple avec `min_delta=0.001` : référence 0.2000, puis 0.1996 et 0.1992. Ces deux pertes
améliorent le checkpoint, mais ne réinitialisent pas la patience. À 0.1988, la baisse cumulée
dépasse 0.001 : la référence avance et l'attente repart à zéro. `min_delta` n'est ni un taux
relatif, ni une perte cible. La meilleure F1 peut correspondre à une autre époque que la meilleure perte.

## Sauvegarde et interruption

Un modèle est écrit avant les mises à jour et après chaque époque terminée. La sauvegarde
utilise un temporaire dans le même dossier, puis `os.replace` après flush/fsync.
Un arrêt normal restaure le meilleur checkpoint ; `Ctrl+C` ou un arrêt coopératif restaure
la **dernière époque complète**, qui peut être moins bonne. `Ctrl+C` quitte avec 130.
Un arrêt forcé ou une coupure ne peut pas exécuter la sauvegarde finale.
Les contrôles de fin d'époque rejettent les résultats non finis ; ils ne constituent pas
une validation exhaustive de tous les fichiers modèle fournis en entrée.

## Justification des réglages

La taille 838 reprend l'encodage existant et conserve la compatibilité des modèles.
Le compteur de demi-coups brut peut avoir une échelle très différente des indicateurs binaires ;
il n'a pas été normalisé durant les benchmarks. Modifier ce choix impose de régénérer/réentraîner
les modèles et de préciser la nouvelle convention d'encodage.

Les essais comparent profondeur, largeur, learning rate, amplitude initiale et certaines activations.
Ils retiennent `838 → 256 → 5`, ReLU/softmax, learning_rate=0.01 et init=0.1 dans le budget évalué.
Le modèle profond `64 × 4` réduit les paramètres mais dégrade la F1 moyenne de confirmation.
Voir les [résultats avec graphiques](benchmarks.md), y compris leurs limites.

La première série fixe patience 8 et min_delta 1e-5 ; la série créative utilise patience 12
et le même min_delta, pour laisser davantage de temps aux réseaux profonds. Ces valeurs ont
été choisies pour le protocole, pas prouvées optimales par recherche exhaustive.

Les méthodes effectivement démontrables sont SGD, contrôle de convergence par validation,
comparaison de configurations/graines, courbes et métriques multiples. Le pat distinct,
la génération de coups et l'entraînement parallèle restent hors du périmètre réalisé.
