import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
import re
from collections import Counter

# Load the cleaned dataset
cleaned_df = pd.read_csv('file/suicide_final_cleaned.csv')

# Perform EDA functions
def perform_eda(df):
    # Word count for suicidal and non-suicidal texts
    df_positive = df[df['class'] == 'suicide']
    df_negative = df[df['class'] == 'non-suicide']

    positive_word_counts = df_positive['cleaned_text'].apply(lambda text: len(re.findall(r'\w+', text)))
    negative_word_counts = df_negative['cleaned_text'].apply(lambda text: len(re.findall(r'\w+', text)))

    # Plot word count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(positive_word_counts, bins=30, color='blue', alpha=0.5, label='Suicidal Texts')
    sns.histplot(negative_word_counts, bins=30, color='green', alpha=0.5, label='Non-Suicidal Texts')
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
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

# Perform EDA on the cleaned dataset
perform_eda(cleaned_df)
