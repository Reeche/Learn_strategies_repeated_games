This is an on-going repo on the project: Learning dominant and/or extortionate strategies in repeated two-person matrix games.

* env.py contains the environments IPD, Chicken, Stag-Hunt
* actor-critic.py contains the model
* interaction.py contains the function interaction between the Q-learning agent and the fixed agent and the functions
 to plot the Q-values during learning
* interaction_twofixed.py: interaction between two fixed strategies. One of the fixed strategies can be one of the 16
 available strategies
 * helper.py contains helper functions
 * plot_polygraph.py contains the code to plot the polygraph for a fixed strategy vs. a random agent
 * tableQ.py contains the code for the Tabular Q-learning agent
 * test.py contains the tests (todo)
