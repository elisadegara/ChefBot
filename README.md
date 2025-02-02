# ChefBot
Telegram bot which finds recipes for you with a given set of ingredients

How to use the bot?
- Install required libraries: pip install gensim numpy pandas scipy python-telegram-bot 
- Run ChefBot.py on the terminal with: python3 ChefBot.py (or run ChefBotInteractive.py with python3 testGPT.py) 
- In the meantime, open telegram at the link https://t.me/ElisaTommasoChefBot and send the key word ‘/start’ to start the bot. Continue following the instructions and you will get the desired recipes
- Once you have finished the process and you don’t want to ask for new recipes anymore, go back to the terminal and press Ctrl+C to stop the code from running.
 
In this project, Tommaso Giacomello and I built a bot telling us what to eat given some ingredients and dietary metrics.
In order to train a machine learning model to build our bot, we employed the Kaggle Food.com recipes and interaction dataset (https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data).
We decided to employ Doc2Vec, a machine learning model designed to represent entire documents as vectors. In this context, a “document” corresponds to a recipe, and its “words” are the list of ingredients. By training a Doc2Vec model on a collection of recipes, we can encode each recipe into a vector that captures the relationships and patterns among the ingredients. Recipes with similar ingredients are represented by similar vectors. The similarity between recipes can then be calculated by comparing their vectors using cosine similarity, a metric that computes the cosine of the angle between the vectors, which reflects how similar their directions are. Once trained, the Doc2Vec model can be inferred to generate vectors for new sets of ingredients given by the user.
Along with the ingredients, the user can also indicate some characteristics of the recipe, such as calories, proteins (PDV),  time, and many more. If the user does not want to insert the value for a specific metric, it will be set to a default number that corresponds to the median value of the metric in the dataset.

Putting all this together, we created a Telegram Bot, with a nice interface: after the user inserts the ingredients and dietary metrics following the instructions displayed in the chat, a set of recipes is shown based on the provided data.

We created two versions of the bot:
- ChefBot: this version structures the chat in three steps.
   1. Ingredient insertion: the user inserts ingredients (separated by commas), which are then processed by the model and the closest recipes are computed;
   2. Metrics insertion: the metrics are inserted by the user, either by deciding to use the median value (by typing 'n' in the chat) or to employ a custom value. After being standardized withabsolute values applied, the top five closest recipes are chosen in the previous subset using Euclidean distance;
   3. Continuation: if the user decides to ask for other recipes, we go back to point 1, otherwise the session ends.
- InteractiveChefBot: this version is analogue to ChefBot, but we have a nicer interface during the metrics insertion step. The user can click on two different buttons: Use Default or Custom. If the user clicks on the first one, the default median value will be selected. Contrarily, if the user clicks on the second one, he will be allowed to manually set a custom value.
