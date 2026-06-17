Maintenant que votre agent est entraîné, il est temps de mesurer ses performances. Vous allez exécuter un certain nombre d'épisodes d'évaluation (par exemple, 100) en utilisant la Q-table apprise. Cette fois, l'agent ne doit jamais explorer ; il doit toujours choisir la meilleure action qu'il connaît. Calculez son taux de réussite.
 
Prérequis 

    Avoir une Q-table entraînée à l'étape précédente.

Résultat attendu 

Un script qui calcule et affiche le taux de réussite de l'agent sur un grand nombre d'épisodes, par exemple : "Taux de réussite sur 100 épisodes : 72%".

 
Recommandations  

    Créez une nouvelle boucle d'évaluation, similaire à la boucle d'entraînement.

    La grande différence : pour choisir l'action, utilisez toujours action = np.argmax(q_table[state, :]). Il n'y a plus de epsilon.

    Incrémentez un compteur de victoires (total_wins) chaque fois qu'un épisode se termine avec une récompense de 1.0.

 
Points de vigilance  

    Ne modifiez plus la Q-table pendant l'évaluation ! La phase d'apprentissage est terminée.

N'oubliez pas de réinitialiser l'environnement à chaque nouvel épisode d'évaluation.