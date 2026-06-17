Commencez par créer l'environnement "FrozenLake-v1". Contrairement à CartPole, cet environnement a un nombre fini d'états, ce qui le rend idéal pour le Q-Learning. Ensuite, initialisez votre Q-table sous la forme d'un tableau NumPy rempli de zéros, avec une ligne pour chaque état et une colonne pour chaque action possible.
 
Prérequis

Avoir installé les librairies suivantes: 

    Avoir fini le premier exercice.

    Connaître les bases de la bibliothèque NumPy.

Résultat attendu  

    Une Q-table (tableau NumPy) initialisée avec les bonnes dimensions ((16, 4) pour FrozenLake) et remplie de zéros. 

 
Recommandations 

    Utilisez env.observation_space.n et env.action_space.n pour obtenir dynamiquement les dimensions nécessaires pour votre Q-table.

    Utilisez np.zeros() pour créer le tableau.

Outils 

    Numpy

    Gymnasium