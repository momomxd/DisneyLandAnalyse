import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Laden Sie die Daten aus Ihrer CSV-Datei
df = pd.read_csv('Dataset/DisneylandReviews.csv', encoding='latin1')

# Filtern Sie die Bewertungen für Disneyland California
df = df[df['Branch'] == 'Disneyland_California']

# Laden des VADER Sentiment Analyzers
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Liste mit bekannten Attraktionen aus Disneyland California
attractions = [
    'Small World', 'Tomorrowland', 'Space Mountain', 'Main Street', 'Castle',
    'Adventureland', 'Fantasyland', 'Star Wars: Galaxy’s Edge',
    'Pirates of the Caribbean', 'Haunted Mansion', 'Jungle Cruise',
    'Big Thunder Mountain', 'Splash Mountain', 'Mickey’s Toontown',
    'Indiana Jones Adventure', 'Matterhorn Bobsleds', 'Peter Pan’s Flight',
    'Sleeping Beauty Castle', 'Star Tours'
]

# Funktion zum Finden und Bewerten von Attraktionen in den Bewertungstexten
def find_attractions(review, attractions_list):
    found_attractions = {}
    for attraction in attractions_list:
        if attraction.lower() in review.lower():
            found_attractions[attraction] = sia.polarity_scores(review)['compound']
    return found_attractions

# Anwenden der Funktion auf jede Bewertung
df['Found_Attractions'] = df['Review_Text'].apply(lambda x: find_attractions(x, attractions))

# Erstellen einer Liste von Tuples für jede Attraktion und ihren Sentiment Score
attraction_sentiments = []
for index, row in df.iterrows():
    for attraction, score in row['Found_Attractions'].items():
        attraction_sentiments.append((attraction, score))

# Umwandeln in DataFrame
attraction_df = pd.DataFrame(attraction_sentiments, columns=['Attraction', 'Sentiment'])

# Berechnung der durchschnittlichen Sentiment-Scores und der Anzahl der Erwähnungen für jede Attraktion
grouped_attraction_df = attraction_df.groupby('Attraction').agg(
    Avg_Sentiment=('Sentiment', 'mean'),
    Mention_Count=('Sentiment', 'count')
).reset_index()

# Erstellen eines Scatter Plots
plt.figure(figsize=(10, 6))  # Angepasste Größe des Plots
sns.scatterplot(data=grouped_attraction_df, x='Mention_Count', y='Avg_Sentiment', hue='Attraction', s=100)
plt.title('Häufigkeit und durchschnittlicher Sentiment Score der Attraktionen in Disneyland California')
plt.xlabel('Anzahl der Erwähnungen')
plt.ylabel('Durchschnittlicher Sentiment Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)  # Angepasste Position der Legende
plt.tight_layout()  # Stellt sicher, dass die Legende im gesamten Layout angezeigt wird
plt.show()
