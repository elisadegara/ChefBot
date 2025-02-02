########################
#     IMPORTS
########################

import numpy as np 
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import ast
import math
from scipy.spatial.distance import cosine
import textwrap
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackContext,
)

import warnings
warnings.simplefilter('ignore')




###########################
#       UTILS
##########################

def similarity_dict(model, n, vec):
    '''
    This function computes the similarity between a vector and multiple recipe vectors.
    '''
    return [1 - cosine(vec, model.dv[f"Recipe_{i}"]) for i in range(n)]


def list_best(sim, rec, n):
    '''
        This function filters the most similar recipes:
        it takes the most similar one and recipes that are at most 10% worse than it.
    '''
    best = max(sim)  # get the best similarity score
    t = 0.1   # tolerance for recipes within 10% of the best match, include recipes with similarity within the threshold
    threshold = best * (1-t)
    ids = rec.index
    # convert indices to recipes IDs:
    l = [ids[i] for i in range(n) if sim[i] >= threshold]
    return l


def recipe_to_string(recipe, recipe_number):
    '''
    This function returns a recipe in readable format as a string with proper alignment.
    '''
    wrapper = textwrap.TextWrapper(width=50)
    output = []

    # Recipe header with Recipe number
    output.append(f"Recipe {recipe_number+1}: {recipe.get('name', 'N/A')}")
    output.append("=" * 50)

    # Ingredients section
    ingredients = recipe.get('ingredients', [])
    output.append("Ingredients:")
    if isinstance(ingredients, str):
        try:
            ingredients = ast.literal_eval(ingredients)
        except (ValueError, SyntaxError):
            output.append("  Error: Ingredients are not properly formatted.")
            ingredients = []
    for ingredient in ingredients:
        output.append(f"  - {ingredient}")

    # Time and tags
    output.append(f"\nTime (min): {recipe.get('minutes', 'N/A')}")
    tags = recipe.get('tags', [])
    output.append("Tags:")
    if isinstance(tags, str):
        try:
            tags = ast.literal_eval(tags)
        except (ValueError, SyntaxError):
            output.append("  Error: Tags are not properly formatted.")
            tags = []
    for tag in tags:
        output.append(f"  - {tag}")

    # Steps section
    output.append("\nSteps:")
    steps = recipe.get('steps', [])
    if isinstance(steps, str):
        try:
            steps = ast.literal_eval(steps)
        except (ValueError, SyntaxError):
            output.append("  Error: Steps are not properly formatted.")
            steps = []
    for i, step in enumerate(steps, 1):
        output.append(f"  Step {i}: {wrapper.fill(step)}")

    # Description section
    description = recipe.get('description', 'N/A')
    if isinstance(description, (float, int)) and math.isnan(description):
        description = 'N/A'
    output.append(f"\nDescription: {wrapper.fill(description)}")

    # Nutritional information
    output.append("\nNutritional Information:")
    output.append(f"  Calories:          {recipe.get('calories', 'N/A')} cal")
    output.append(f"  Total Fat:         {recipe.get('total fat', 'N/A')} PVD")
    output.append(f"  Sugar:             {recipe.get('sugar', 'N/A')} PVD")
    output.append(f"  Sodium:            {recipe.get('sodium', 'N/A')} PVD")
    output.append(f"  Protein:           {recipe.get('protein', 'N/A')} PVD")
    output.append(f"  Saturated Fat:     {recipe.get('saturated fat', 'N/A')} PVD")
    output.append(f"  Carbohydrates:     {recipe.get('carbohydrates', 'N/A')} PVD")

    # Difficulty
    difficulty = recipe.get('difficulty', 0)
    if isinstance(difficulty, np.integer):
        output.append(f"\nDifficulty: {difficulty + 1} (1=easy, 2=hard)")
    else:
        output.append(f"\nDifficulty: N/A")

    output.append("=" * 50)
    output.append("")  # Blank line for spacing

    return "\n".join(output)




#######################
#      DATA
#######################

print('Waiting for data import')

dietary = ['calories', 'total fat', 'sugar', 'sodium', 'protein', 'saturated fat', 'carbohydrates']

# Dataset where we can retrieve each ingredient given its ID
ingredients_dataset = pd.read_pickle('ingr_map.pkl')        
ingredients_dataset.set_index('raw_ingr', inplace=True)

# Create dicts in order to pass from user input to IDs
raw_repl = ingredients_dataset.groupby('raw_ingr')['replaced'].apply(lambda x: list(set(x))[0]).to_dict()
repl_id = ingredients_dataset.groupby('replaced')['id'].apply(lambda x: list(set(x))[0]).to_dict()

rec = pd.read_csv('Recipe_final.csv')        
rec.set_index('id', inplace = True)

# Compute median values of the dietary metrics and of the time of the recipes, which will be the default ones
X_num = rec.iloc[:, -8:]   # select the last 8 columns
med_ns = X_num.median()
X_mean = X_num.mean()
X_std = X_num.std()
X_num = np.abs((X_num - X_mean) / X_std)  # standardize each metric across all recipes
med = X_num.median()    # compute the median of each one of the 8 normalized columns




#########################
#        MODEL
##########################

# load the pre-trained Doc2Vec model
model = Doc2Vec.load('recipe_doc2vec.model')




#######################
#   CODE TELEGRAM BOT
#######################


# Define conversation states
SUGGEST_RECIPE, GET_METRICS, ASK_CONTINUE = range(3)

# Logging configuration
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

metrics = np.zeros(len(dietary)+1)

async def start(update: Update, context: CallbackContext) -> int:
    """
    This function starts the bot and asks for ingredients.
    """
    await update.message.reply_text(
        "Welcome! Send me a list of ingredients separated by commas (e.g. 'ingredient1, ingredient2')."
    )
    return SUGGEST_RECIPE


async def suggest_recipe(update: Update, context: CallbackContext) -> int:
    """
    This function processes the ingredients and prepares to collect metrics.
    """
    ingr = [x.strip().lower() for x in update.message.text.split(',')]
    to_remove = []
    # Process ingredient names to IDs
    for i, ing in enumerate(ingr):
        try:
            ingr[i] = str(repl_id[raw_repl[ing]])  # from raw to replaced to ID
        except KeyError:
            try:
                ingr[i] = str(repl_id[ing]) # if the above does not work from replaced to ID
            except KeyError:
                await update.message.reply_text(f'Ingredient {ing} not found. The software will remove it from the search.')
                to_remove.append(ing)
    
    # remove invalid ingredients
    for ing in to_remove:
        ingr.remove(ing)
    
    if len(ingr) == 0:
        await update.message.reply_text("No valid ingredients inserted. Please provide some ingredients.")
        return SUGGEST_RECIPE

    await update.message.reply_text("Processing ingredients and computing similar recipes...")
    # infer vector for the user input ingredients
    vec = model.infer_vector(ingr)

    # find similar recipes
    n = len(rec)
    sim = similarity_dict(model, n, vec)
    best = list_best(sim, rec, n)

    # save ingredients to context
    context.user_data['ingredients'] = ingr
    context.user_data['similar_recipes'] = best

    await update.message.reply_text("Ingredients processed. Now let's collect nutritional metrics.")
    
    # Start collecting metrics
    context.user_data['metric_index'] = 0
    await update.message.reply_text(f"Insert {dietary[0]} in cal (or type 'n' to use default value {med_ns[0]} cal):")
    return GET_METRICS


async def get_metrics(update: Update, context: CallbackContext) -> int:
    """
    This function collects metrics from the user one by one.
    """
    metric_index = context.user_data.get('metric_index', 0)
    user_input = update.message.text.strip()

    if user_input.lower() == 'n' or not user_input:
        metrics[metric_index] = med[metric_index]  # use default
    else:
        try:
            if float(user_input) < 0:
                await update.message.reply_text(f"Invalid value: it must be non-negative. Using default value")
                metrics[metric_index] = med[metric_index]
            else:
                # standardize metrics
                metrics[metric_index] = np.abs((float(user_input) - X_mean[metric_index])/X_std[metric_index])
        except ValueError:
            await update.message.reply_text("Invalid input. Using the default value.")
            metrics[metric_index] = med[metric_index]

    metric_index += 1
    context.user_data['metric_index'] = metric_index
    if metric_index < len(dietary):
        # prompt for the next metric
        next_metric = dietary[metric_index]
        await update.message.reply_text(f"Insert {next_metric} in PDV (Percentage Daily Value) (or type 'n' to use default value {med_ns[metric_index]} PDV):")
        return GET_METRICS
    elif metric_index == len(dietary):
        # prompt for last metric to collect (time)
        await update.message.reply_text(f"Insert time in minutes (or type 'n' to use default value {med_ns[metric_index]} min):")
        return GET_METRICS
    else:
        # all metrics collected
        await update.message.reply_text(f"All metrics collected: Calculating best recipes...")
        best = context.user_data.get('similar_recipes')
        restr_num = X_num.loc[best]
        distances = np.linalg.norm(restr_num.values - metrics, axis=1)
        closest_indices = np.argsort(distances)[:min(len(distances), 5)]

        await update.message.reply_text("Here are your suggested recipes:")
        for k, i in enumerate(closest_indices):
            closest_recipe = restr_num.iloc[i].name
            printing = recipe_to_string(rec.loc[closest_recipe],k)
            await update.message.reply_text(printing)

        await update.message.reply_text('Would you like to continue with another recipe suggestion? (yes/no)')
        context.user_data['waiting_for_restart'] = True
        return ASK_CONTINUE



async def handle_continue(update: Update, context: CallbackContext) -> int:
    """
    This function handles user's decision to continue or end the recipe suggestion process.
    """
    response = update.message.text.strip().lower()
    if response == 'yes':
        await update.message.reply_text("Send me a new list of ingredients separated by commas (e.g. 'ingredient1, ingredient2').")
        return SUGGEST_RECIPE
    elif response == 'no':
        await update.message.reply_text("Thank you for using ChefBot! Goodbye!")
        return ConversationHandler.END
    else:
        await update.message.reply_text("Please reply with 'yes' or 'no'.")
        return ASK_CONTINUE


def main() -> None:
    """
    Main function to set up and run the bot.
    """
    TOKEN = '8176235673:AAEzYH4DrWkdNj4h2enB2cXu3XkN0KH83nw'
    application = Application.builder().token(TOKEN).build()

    # Create conversation handler
    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SUGGEST_RECIPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, suggest_recipe)],
            GET_METRICS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_metrics)],
            ASK_CONTINUE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_continue)],
        },
        fallbacks=[],
    )

    # Register conversation handler
    application.add_handler(conversation_handler)

    print("Bot is running. Press Ctrl+C to stop.")

    # Start polling for updates
    application.run_polling()


if __name__ == "__main__":
    main()
