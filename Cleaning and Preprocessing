import os
import pandas as pd
import numpy as np
import spacy
import unidecode
import contractions as contract
import re
import wordninja
import collections
import pkg_resources
from spellchecker import SpellChecker 
from symspellpy import SymSpell, Verbosity
     

# Load dataset
df = pd.read_csv('data/suicide_detection.csv', index_col=0)
df.reset_index(drop=True, inplace=True)
df.head()
# Defining methods

nlp = spacy.load("en_core_web_sm") 
vocab = collections.Counter()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
"symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


# Remove stopwords list 
deselect_stop_words = ['no', 'not']
    
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False

# Remove extra whitespaces from text
def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())

# Remove accented characters from text, e.g. café
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

# Remove URL 
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# Removing symbols and digits
def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)

# Removing special characters
def remove_special(text):
    return text.replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '')

# Fix word lengthening (characters are wrongly repeated)
def fix_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def text_preprocessing(text, accented_chars=True, contractions=True, convert_num=True, 
                       extra_whitespace=True, lemmatization=True, lowercase=True, 
                       url=True, symbols_digits=True, special_chars=True, 
                       stop_words=True, lengthening=True, spelling=True):
    """preprocess text with default option set to true for all steps"""
    if accented_chars == True: # remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: # expand contractions
        text = contract.fix(text)
    if lowercase == True: # convert all characters to lowercase
        text = text.lower()
    if url == True: # remove URLs before removing symbols 
        text = remove_url(text)
    if symbols_digits == True: # remove symbols and digits
        text = remove_symbols_digits(text)
    if special_chars == True: # remove special characters
        text = remove_special(text)
    if extra_whitespace == True: # remove extra whitespaces
        text = remove_whitespace(text)
    if lengthening == True: # fix word lengthening
        text = fix_lengthening(text)
    if spelling == True: # fix spelling
        text = fix_spelling(text)

    doc = nlp(text) # tokenise text

    clean_text = []

    # return text
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # exclude number words
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            flag = False
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)        
    return " ".join(clean_text)
     

df['cleaned_text'] = df['text'].apply(lambda row: text_preprocessing(row))
     
# Removed anomalous "filler" word 
cleaned_df['cleaned_text'] = cleaned_df['cleaned_text'].str.replace('filler', '')
     
# Remove rows with text length 0
cleaned_df = cleaned_df[cleaned_df['cleaned_text'].apply(lambda x: len(x.split())!=0)]
cleaned_df.reset_index(drop=True, inplace=True)
cleaned_df.head()     

# Check dataset 
cleaned_df.head()
     
# Export cleaned dataset
df.to_csv('data/suicide_final_cleaned.csv', index=False)
