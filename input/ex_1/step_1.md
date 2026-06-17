Vous allez d'abord créer une instance de l'environnement classique "CartPole-v1" à l'aide de la bibliothèque Gymnasium. Ensuite, vous explorerez ses propriétés fondamentales :

    l'espace d'observation c’est-à-dire ce que l'agent "voit"
    l'espace d'action c’est-à-dire ce que l'agent "peut faire".

Prérequis 

 

Avoir installé les librairies suivantes : 

    gymnasium pour les environnements,

    stable-baselines3 pour les algorithmes pré-implémentés,

    matplotlib pour la visualisation.

Résultat attendu

    Des cellules de code qui affichent la structure de l'espace d'observation et de l'espace d'action, ainsi que des exemples.

 
Recommandations 

    Utilisez gym.make("NomDeLEnvironnement") pour créer l'environnement.

    Accédez aux attributs .observation_space et  .action_space de l'objet environnement créé.

    Utilisez la méthode .sample() sur ces espaces pour voir à quoi ressemble une observation ou une action aléatoire.

 
Points de vigilance 

Noter le type de chaque espace. CartPole a un espace d'observation Box (continu) et un espace d'action Discrete (discret). 

Cette distinction sera cruciale plus tard.
 
Outils

    Google Colab

    Gymnasium

 
Ressources 

    Le guide de base de Gymnasium sur les environnements (https://gymnasium.farama.org/api/env/)