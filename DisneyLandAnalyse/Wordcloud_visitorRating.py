import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk import bigrams
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk Ressourcen herunterladen
nltk.download('punkt')
nltk.download('stopwords')

additional_stopwords = {'california', 'disneyland', 'disney'}

# Stemmer initialisieren
stemmer = PorterStemmer()


# Funktion zum Bereinigen und Tokenisieren des Textes und Erstellen von Bigrams
def clean_tokenize_and_bigrams(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english')).union(additional_stopwords)
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming auf jedes Token anwenden
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    bigram_list = list(bigrams(stemmed_tokens))
    bigrams_as_strings = [' '.join(bigram) for bigram in bigram_list]
    return bigrams_as_strings


def main():
    # Einlesen der CSV-Datei
    df = pd.read_csv('Dataset/DisneylandReviews.csv', encoding='latin1')

    # Filtern nach "Disneyland_California"
    filtered_df = df[(df['Branch'] == 'Disneyland_California') & (df['Reviewer_Location'] == 'Switzerland')]

    # Kombinieren aller Review-Texte
    all_text = ' '.join(filtered_df['Review_Text'])

    # Bereinigen, Tokenisieren und Bigrams erstellen
    bigrams = clean_tokenize_and_bigrams(all_text)

    # Zählen der Häufigkeit der Bigrams
    bigram_freq = Counter(bigrams)

    # Erstellen einer Word Cloud für Bigrams
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(bigram_freq)

    # Anzeigen der Word Cloud
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Analyse der Bewertungsverteilung
    filtered_df['Rating'].value_counts().plot(kind='bar')
    plt.title('Verteilung der Bewertungen von Besuchern aus der Schweiz für Disneyland California')
    plt.xlabel('Bewertung')
    plt.ylabel('Anzahl')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
