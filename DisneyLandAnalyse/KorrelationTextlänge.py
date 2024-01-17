import pandas as pd
import matplotlib.pyplot as plt

# Laden der Daten aus der CSV-Datei
df = pd.read_csv('Dataset/DisneylandReviews.csv', encoding='latin1')

# Filtern der Daten, um nur Bewertungen für Disneyland California zu berücksichtigen
df_california = df[df['Branch'] == 'Disneyland_California']

# Hinzufügen einer neuen Spalte für die Länge des Bewertungstextes (gemessen in Anzahl der Wörter)
df_california['Review_Length'] = df_california['Review_Text'].apply(lambda x: len(x.split()))

# Berechnung der durchschnittlichen Textlänge pro Bewertung und Rating
avg_review_length_by_rating = df_california.groupby('Rating')['Review_Length'].mean().reset_index()

# Visualisierung als Liniendiagramm
plt.figure(figsize=(10, 6))
plt.plot(avg_review_length_by_rating['Rating'], avg_review_length_by_rating['Review_Length'], marker='o')
plt.xlabel('Rating')
plt.ylabel('Durchschnittliche Textlänge (Anzahl der Wörter)')
plt.title('Durchschnittliche Textlänge pro Rating für Disneyland California')
plt.grid(True)
plt.show()
