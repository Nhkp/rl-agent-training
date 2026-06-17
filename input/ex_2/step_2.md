C'est le cœur de l'exercice. Vous allez coder la boucle d'entraînement principale. À chaque pas de temps, votre agent devra décider s'il explore (action aléatoire) ou s'il exploite ses connaissances (meilleure action de la Q-table). Après avoir exécuté l'action, vous mettrez à jour la valeur correspondante dans la Q-table en utilisant la formule de mise à jour du Q-Learning.

 
Prérequis 

    Avoir initialisé la Q-table.

    Avoir défini les hyperparamètres (learning_rate, discount_factor, epsilon, etc.).

Résultat attendu  

Une Q-table remplie de valeurs apprises après des milliers d'épisodes d'entraînement.

 
Recommandations  

    Pour la stratégie epsilon-greedy, générez un nombre aléatoire entre 0 et 1. S'il est inférieur à epsilon, explorez. Sinon, exploitez.

    Pour exploiter, utilisez np.argmax(q_table[state, :]) pour trouver l'index (l'action) de la plus grande valeur Q pour l'état actuel.

    Traduisez la formule de Bellman en code Python : nouvelle_valeur = ancienne_valeur + lr * (recompense + gamma * max_q_futur - ancienne_valeur).

    N'oubliez pas de réduire epsilon à la fin de chaque épisode pour que l'agent explore moins à mesure qu'il apprend.

 
Points de vigilance  

    Assurez-vous de bien utiliser new_state pour trouver la valeur Q future maximale (np.max(q_table[new_state, :])).

    La mise à jour se fait sur la paire (state, action) qui a été utilisée pour la transition.

 
Ressources 

    Tutoriel sur le Q-Learning(https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)
