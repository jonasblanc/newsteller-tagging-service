import fasttext
from fasttext import load_model

import pandas as pd
import nltk
nltk.download('stopwords') # Added (requested by system)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

from sklearn.model_selection import train_test_split


LR = 1.0
EPOCH = 50
DIM = 100
TEXT_SIZE = 300
SUPPORTED_LANGAGES = ["english", "german", "french", "italian"]

def __preprocess__(sentence, langage, max_words=TEXT_SIZE):
    assert langage in SUPPORTED_LANGAGES, "Langage should be one of: " + " ".join(SUPPORTED_LANGAGES)

    stop_words = stopwords.words(langage)

    sentence = sentence.lower()

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    word_tokens = tokenizer.tokenize(sentence)

    filtered_sentence = [w for w in word_tokens if not w in stop_words][:max_words]
    return " ".join(filtered_sentence)

def __save_datafile__(df, filename):
    with open(filename, 'w') as f:
        for each_text, each_label in zip(df['preproc'], df['category']):
            f.writelines(f'__label__{each_label} {each_text}\n')


def train_save_model(df, categories, langage):
    """Train a fasttext model based on given dataframe and save it.

    Train a fastext model to classify an article in categories using text and title of articles provided in the dataframe. 
    The model will be saved in model/ with name "{given_langage}_model.ftz".
    See SUPPORTED_LANGAGES to know which langages are supported. 
    """

    assert langage in SUPPORTED_LANGAGES, "Langage should be one of: " + " ".join(SUPPORTED_LANGAGES)

    df['category'] = df['url'].apply(lambda x :  x.split('/')[3])
    df = df[df['category'].isin(categories)]

    df['preproc'] = (df['title'] +  " " + df['text']).apply(lambda title_text: __preprocess__(title_text, langage))

    train, test = train_test_split(df, test_size=0.2)

    __save_datafile__(train, 'data/fasttext.train')
    __save_datafile__(test, 'data/fasttext.valid')

    model = fasttext.train_supervised(input= 'data/fasttext.train', lr = LR,epoch=EPOCH, dim=DIM)
    model.test('data/fasttext.train')

    path = "model/" + langage + "_model.ftz"
    model.save_model(path)

def get_article_category(title, text, langage, max_text_size = TEXT_SIZE):
    """Load a previsouly created fasttext model and classify the given article

    Classify a article based on its title and text. "max_text_size" is the maximun number of words taken in account.
    This function should not be called without previous call to "train_save_model" with same langage, otherwise you need to provide a trained model in model/ with name "{given_langage}_model.ftz"..
    See SUPPORTED_LANGAGES to know which langages are supported. 
    """

    assert langage in SUPPORTED_LANGAGES, "Langage should be one of: " + " ".join(SUPPORTED_LANGAGES)

    path = "model/" + langage + "_model.ftz"
    model = load_model(path)
    
    preproc = __preprocess__(title +  " " + text, langage, max_text_size)
    return model.predict(preproc, k =1)[0][0]

def get_article_vector(title, text, langage, max_text_size = TEXT_SIZE):
    """Load a previsouly created fasttext model and return the vector representation of an article

    Return the vector representation of a article based on its title and text. "max_text_size" is the maximun number of words taken in account.
    This function should not be called without previous call to "train_save_model" with same langage, otherwise you need to provide a trained model in model/ with name "{given_langage}_model.ftz".
    See SUPPORTED_LANGAGES to know which langages are supported. 
    """

    assert langage in SUPPORTED_LANGAGES, "Langage should be one of: " + " ".join(SUPPORTED_LANGAGES)

    path = "model/" + langage + "_model.ftz"
    model = load_model(path)
    preproc = __preprocess__(title +  " " + text, langage, max_text_size)

    return model.get_sentence_vector(preproc)
