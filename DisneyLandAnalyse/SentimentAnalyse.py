import pandas as pd
import matplotlib.pyplot as plt

# Einlesen der Daten aus der CSV-Datei
df = pd.read_csv('Dataset/DisneylandReviews.csv', encoding='latin1')

# Sentiment-Kategorien basierend auf dem Rating zuweisen
# Angenommen, Bewertungen mit einem Rating von 4 oder 5 sind positiv, 3 ist neutral, und 1 oder 2 ist negativ
df['Sentiment'] = pd.cut(df['Rating'], bins=[0, 2, 4, 5], labels=['NEGATIV', 'NEUTRAL', 'POSITIV'], include_lowest=True)

# Daten für das Balkendiagramm vorbereiten
sentiment_counts = df.groupby(['Branch', 'Sentiment']).size().unstack(fill_value=0)

# Farben für das Balkendiagramm auswählen (gedecktere Farben)
colors = ['#6baed6', '#bdd7e7', '#bae4b3']  # Blau und Grün in gedeckten Tönen

# Balkendiagramm erstellen
ax = sentiment_counts.plot(kind='bar', stacked=True, color=colors)

# Titel und Achsenbeschriftungen hinzufügen

plt.xlabel('Park Standort')
plt.ylabel('Anzahl der Bewertungen')
plt.xticks(rotation=45)  # Achsenbeschriftungen drehen, um Überlappung zu vermeiden

# Legende hinzufügen
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')

# Layout anpassen und Diagramm anzeigen
plt.tight_layout()
plt.show()
