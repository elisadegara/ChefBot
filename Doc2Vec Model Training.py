########################
#     IMPORTS
########################

import pandas as pd 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re



#######################
#      DATA
#######################

PP = pd.read_csv('PP_recipes.csv')
print('The dataset has been read')

'''
    We now create a list of strings, each one representing a list of ingredient IDs and we preprocess it to use it with Doc2Vec.
'''
def transform(data):
    result = []
    for line in data:
        # split by spaces, group into blocks, and remove spaces within blocks
        transformed = ' '.join([''.join(block.split()) for block in line.split('    ')])
        result.append(transformed)
    return result

def clean_ingredients(text):
    # remove brackets and commas, and normalize spaces
    return re.sub(r'[,\[\]]', '', text).strip()

ingredients_corpus = [" ".join(map(str, row)) for row in PP['ingredient_ids']]
cleaned_ingredients_corpus = [clean_ingredients(doc) for doc in ingredients_corpus]
cleaned_ingredients_corpus = transform(cleaned_ingredients_corpus)

# create TaggedDocument objects
documents = [TaggedDocument(words=text.split(), tags=[f"Recipe_{i}"]) for i, text in enumerate(cleaned_ingredients_corpus)]

print('The dataset has been preprocessed')






'''
    We are now ready to train the model.
    The aim is to assign to each recipe a fixed-size dense vector that captures the semantic meaning of the recipe.
    To do this, we use Doc2Vec, a machine learning model designed to represent entire documents as vectors.
    In this context, a "document" corresponds to a recipe, and its "words" are the list of ingredients.
    By training a Doc2Vec model on a collection of recipes, we can encode each recipe into a dense vector that reflects the relationships and patterns between ingredients.
    Once trained, the Doc2Vec model can generate vectors for both existing and new recipes.
    For a new recipe, the model infers its vector by analyzing its ingredients in the context of the training data.
    Similarities between recipes can then be calculated by comparing these vectors using metrics such as cosine similarity.
    
'''
# train Doc2Vec model
print('Start of training')
model = Doc2Vec(documents, vector_size=50, alpha=0.025, min_alpha = 0.00025, window=2, min_count=1, workers=4, epochs=40, dm=1)
print('End of training')





'''
    We save the model to use it later to create the bot that suggests recipes.
'''
model.save("recipe_doc2vec.model")