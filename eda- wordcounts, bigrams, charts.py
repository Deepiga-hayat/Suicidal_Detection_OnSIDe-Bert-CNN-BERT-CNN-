
import nltk
from nltk import word_tokenize
from collections import Counter
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])


nltk.download('punkt')
from wordcloud import STOPWORDS
custom_stopwords = set(STOPWORDS)
custom_stopwords.update(['fuck', 'shit', 'okay', 'thing', 'yeah', 'um', 'uh', 'want', 'know', 'like'])  # add more as needed
    


# Load the cleaned dataset
cleaned_df = pd.read_csv('data/suicide_final_cleaned.csv')

# Perform EDA functions
def perform_eda(df):
    # Word count for suicidal and non-suicidal texts
    df_positive = df[df['class'] == 'suicide']
    df_negative = df[df['class'] == 'non-suicide']

    positive_word_counts = df_positive['cleaned_text'].apply(lambda text: len(re.findall(r'\w+', text)))
    negative_word_counts = df_negative['cleaned_text'].apply(lambda text: len(re.findall(r'\w+', text)))

    print("Max suicidal word count:", positive_word_counts.max())
    print("Max non-suicidal word count:", negative_word_counts.max())

    # Plot word count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(positive_word_counts, bins=30, color='blue', alpha=0.5, label='Suicidal Texts')
    sns.histplot(negative_word_counts, bins=30, color='green', alpha=0.5, label='Non-Suicidal Texts')
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(0, 500)  # Limit x-axis to 0–500 words to zoom in on useful data
    plt.show()

    # Bi-grams for suicidal texts and non-suicidal texts
    def get_ngrams(texts, n=2):
        tokens = [word_tokenize(text.lower()) for text in texts]
        ngrams = [nltk.ngrams(token_list, n) for token_list in tokens]
        flattened_ngrams = [item for sublist in ngrams for item in sublist]
        return flattened_ngrams

    def plot_top_ngrams(ngrams, title):
        # Count occurrences of each n-gram
        ngram_counts = Counter(ngrams)
        top_ngrams = ngram_counts.most_common(5)  # Change to top 5 or more as needed

        # Create a DataFrame for displaying in a table format
        df = pd.DataFrame(top_ngrams, columns=['Bigram', 'Count'])

        # Plot as a table
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111, frame_on=False)  # No visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        table = plt.table(cellText=df.values,
                          colLabels=df.columns,
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.2)  # Adjust table size

        plt.title(title)
        plt.show()

    positive_bigrams = get_ngrams(df_positive['cleaned_text'])
    negative_bigrams = get_ngrams(df_negative['cleaned_text'])

    # Plot top bi-grams
    plot_top_ngrams(positive_bigrams, title='Top Bi-grams for Suicidal Texts')
    plot_top_ngrams(negative_bigrams, title='Top Bi-grams for Non-Suicidal Texts')

    # Graph with number of words and text lengths
    posts_lengths = [len(text.split()) for text in df['cleaned_text']]
    plt.figure(figsize=(10, 6))
    plt.hist(posts_lengths, bins=30, color='purple', alpha=0.7)
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
    
    # Word Cloud for Suicidal Texts
    suicidal_text = " ".join(df_positive['cleaned_text'].dropna().astype(str))
    # Apply it to suicidal text
    lemmatized_suicidal_text = lemmatize_text(suicidal_text)

    wordcloud1_suicide = WordCloud(
        width=800, height=400, background_color='white',
        stopwords=custom_stopwords,
        max_words=30
    ).generate(suicidal_text)
    
    wordcloud_suicide = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=custom_stopwords,
        max_words=30
    ).generate(lemmatized_suicidal_text)


    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_suicide, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Keywords in Suicidal Texts')
    plt.tight_layout()
    plt.show()

    # Word Cloud for Non-Suicidal Texts
    non_suicidal_text = " ".join(df_negative['cleaned_text'].dropna().astype(str))
    lemmatized_non_suicidal_text = lemmatize_text(non_suicidal_text)
    wordcloud1_non_suicide = WordCloud(
        width=800, height=400, background_color='white',
        stopwords=custom_stopwords,
        max_words=30
    ).generate(non_suicidal_text)
    
    wordcloud_non_suicide = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=custom_stopwords,
        max_words=30
    ).generate(lemmatized_non_suicidal_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_non_suicide, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Keywords in Non-Suicidal Texts')
    plt.tight_layout()
    plt.show()



# Perform EDA on the cleaned dataset
perform_eda(cleaned_df)
