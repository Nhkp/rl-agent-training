Maintenant que vous comprenez l'environnement, vous allez coder une boucle simple qui fait interagir un agent avec cet environnement. Pour l'instant, l'agent sera très simple : il choisira ses actions de manière complètement aléatoire. Vous ferez tourner cette simulation pour 10 "épisodes" et afficherez la récompense totale pour chaque épisode.

 
Prérequis 

    Avoir créé un environnement et exploré ses espaces.

Résultat attendu 

    Une boucle for qui exécute 10 épisodes complets. 

    Pour chaque épisode, une boucle while interne doit s'exécuter jusqu'à ce que l'épisode se termine, en choisissant et en appliquant une action aléatoire à chaque étape. 

    La récompense totale de chaque épisode doit être affichée.

 
Recommandations 

    Commencez chaque épisode par env.reset(). Cette fonction réinitialise le jeu et vous donne la toute première observation.

    Dans la boucle while, utilisez env.action_space.sample() pour choisir une action aléatoire.

    Passez cette action à env.step(action). Stockez les 5 valeurs qu'elle retourne.

    La boucle while doit continuer tant que l'épisode n'est ni terminated ni truncated.

    N'oubliez pas d'ajouter la reward reçue à une variable total_reward à chaque étape.

    À la fin de votre script, utilisez env.close() pour fermer l'environnement proprement.

 
Points de vigilance 

    Assurez-vous de bien réinitialiser les variables (comme total_reward, terminated, truncated) au début de chaque nouvel épisode dans la boucle for.

Outils 

    Google Colab

    Gymnasium

 
Ressources 

    Exemple d'utilisation de base de Gymnasium(https://gymnasium.farama.org/introduction/basic_usage/)