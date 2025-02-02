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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    CallbackQueryHandler,
    CallbackContext,
)



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
    best = max(sim)     # get the best similarity score
    t = 0.1   # tolerance for recipes within 10% of the best match, include recipes with similarity within the threshold
    threshold = best * (1 - t)
    ids = rec.index
    return [ids[i] for i in range(n) if sim[i] >= threshold]

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
    output.append(f"\nDifficulty: {difficulty + 1} (1=easy, 2=hard)")
    output.append("=" * 50)
    return "\n".join(output)



#######################
#      DATA
#######################

print('Waiting for data import')

dietary = ['calories', 'total fat', 'sugar', 'sodium', 'protein', 'saturated fat', 'carbohydrates', 'duration']

# Dataset where we can retrieve each ingredient given its ID
ingredients_dataset = pd.read_pickle('ingr_map.pkl')
ingredients_dataset.set_index('raw_ingr', inplace=True)

# Create dicts in order to pass from user input to IDs
raw_repl = ingredients_dataset.groupby('raw_ingr')['replaced'].apply(lambda x: list(set(x))[0]).to_dict()
repl_id = ingredients_dataset.groupby('replaced')['id'].apply(lambda x: list(set(x))[0]).to_dict()

rec = pd.read_csv('Recipe_final.csv')        
rec.set_index('id', inplace=True)

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



# States
SUGGEST_RECIPE, GET_METRICS, ASK_CONTINUE = range(3)

# Configure logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

async def start(update: Update, context: CallbackContext) -> int:
    """
    This function starts the bot and asks for ingredients.
    """
    await update.message.reply_text("Welcome! Send me a list of ingredients separated by commas (e.g. 'ingredient1, ingredient2').")
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
        await update.message.reply_text("No valid ingredients provided. Please insert valid ingredients.")
        return SUGGEST_RECIPE

    await update.message.reply_text("Processing ingredients and computing similar recipes...")
    # infer vector for the user input ingredients
    vec = model.infer_vector(ingr)

    # find similar recipes
    n = len(rec)
    sim = similarity_dict(model, n, vec)
    best = list_best(sim, rec, n)

    # save ingredients to context
    context.user_data['similar_recipes'] = best
    await update.message.reply_text("Ingredients processed. Now let's collect nutritional metrics.")
    return await start_metrics(update, context)

async def start_metrics(update: Update, context: CallbackContext) -> int:
    '''
    This function initializes the metric collection.
    '''
    metric_index = 0
    context.user_data['metric_index'] = metric_index
    context.user_data['metrics'] = np.zeros(len(dietary))
    await send_metric_prompt(update, dietary[metric_index], metric_index)
    return GET_METRICS

async def send_metric_prompt(update: Update, metric_name: str, metric_index: int) -> None:
    '''
    This function asks the user to set a value for a dietary metric.
    '''
    keyboard = [
        [
            InlineKeyboardButton("Use Default", callback_data=f"default_{metric_index}"),
            InlineKeyboardButton("Set Custom", callback_data=f"custom_{metric_index}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"Set value for {metric_name}:",
        reply_markup=reply_markup
    )

async def handle_metric_selection(update: Update, context: CallbackContext) -> int:
    """
    Handler for the user's selection of default/custom metric values.
    """
    query = update.callback_query
    await query.answer()

    data = query.data
    metric_index = int(data.split('_')[1])
    metrics = context.user_data['metrics']

    if "default" in data:
        metrics[metric_index] = med[metric_index]
        unit = get_unit_of_measure(dietary[metric_index])
        await query.edit_message_text(f"{dietary[metric_index]} set to default: {med_ns[metric_index]} {unit}")
        return await process_metric_completion(query, context, metric_index)
    elif "custom" in data:
        unit = get_unit_of_measure(dietary[metric_index])
        await query.edit_message_text(f"Enter custom value for {dietary[metric_index]} in {unit}:")
        context.user_data['awaiting_custom_input'] = metric_index
        return GET_METRICS

def get_unit_of_measure(metric_name: str) -> str:
    """
    This function is used to get the appropriate unit of measure for a metric.
    """
    if metric_name == 'calories':
        return 'cal'
    elif metric_name == 'duration':
        return 'minutes'
    else:
        return 'PDV (Percentage Daily Value)'

async def handle_custom_input(update: Update, context: CallbackContext) -> int:
    """
    This function processes the user's custom input for a dietary metric.
    """
    metric_index = context.user_data.pop('awaiting_custom_input', None)
    if metric_index is None:
        return GET_METRICS

    try:
        custom_value = float(update.message.text.strip())
        if custom_value < 0:
            unit = get_unit_of_measure(dietary[metric_index])
            await update.message.reply_text(f"Invalid value for {dietary[metric_index]}: it must be non-negative. Using default value: {med_ns[metric_index]} {unit}")
            context.user_data['metrics'][metric_index] = med[metric_index]
        else:
            unit = get_unit_of_measure(dietary[metric_index])
            custom_value2 = np.abs((custom_value - X_mean[metric_index])/X_std[metric_index])
            context.user_data['metrics'][metric_index] = custom_value2
            await update.message.reply_text(f"{dietary[metric_index]} set to {custom_value} {unit}.")
    except ValueError:
        unit = get_unit_of_measure(dietary[metric_index])
        await update.message.reply_text(f"Invalid input. Using default value: {med_ns[metric_index]} {unit}")
        context.user_data['metrics'][metric_index] = med[metric_index]

    return await process_metric_completion(update, context, metric_index)

async def process_metric_completion(update, context, metric_index: int) -> int:
    """
    This function determines the next state after a metric has been processed.
    It returns the next conversation state (GET_METRICS or proceeds to recipe calculation).
    """    
    metric_index += 1
    context.user_data['metric_index'] = metric_index

    if metric_index < len(dietary):
        await send_metric_prompt(update, dietary[metric_index], metric_index)
        return GET_METRICS
    else:
        await calculate_and_send_recipes(update, context)
        await update.message.reply_text("Would you like to continue with another recipe suggestion? (yes/no)")
        return ASK_CONTINUE

async def calculate_and_send_recipes(update, context) -> None:
    """
    Helper function that handles the recipe calculation and sending logic.
    """
    await update.message.reply_text("All metrics collected. Calculating recipes...")
    best = context.user_data['similar_recipes']
    metrics = context.user_data['metrics']
    restr_num = X_num.loc[best]
    distances = np.linalg.norm(restr_num.values - metrics, axis=1)
    closest_indices = np.argsort(distances)[:min(len(distances), 5)]

    await update.message.reply_text("Here are your suggested recipes:")
    for k, i in enumerate(closest_indices):
        closest_recipe = restr_num.iloc[i].name
        printing = recipe_to_string(rec.loc[closest_recipe], k)
        await update.message.reply_text(printing)

async def handle_continue(update: Update, context: CallbackContext) -> int:
    """
    Handler for the user's decision to continue or end the recipe suggestion process.
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

    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SUGGEST_RECIPE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, suggest_recipe)
            ],
            GET_METRICS: [
                CallbackQueryHandler(handle_metric_selection),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_input)
            ],
            ASK_CONTINUE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_continue)
            ],  
        },
        fallbacks=[CommandHandler("start", start)],
    )

    # Register conversation handler
    application.add_handler(conversation_handler)

    print("Bot is running. Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()