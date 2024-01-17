import pandas as pd
import matplotlib.pyplot as plt


# Funktion zum Erstellen des Diagramms für die Durchschnittsbewertung über die Zeit
def create_average_rating_plot(df):
    # Umwandlung der "Year_Month" Spalte in datetime und Extraktion des Jahres für die Analyse
    df['Year'] = pd.to_datetime(df['Year_Month'], format='%Y-%m', errors='coerce').dt.year

    # Entfernen von Zeilen mit fehlerhaften Daten (wo das Jahr NaT ist)
    df = df.dropna(subset=['Year'])

    # Berechnung des Durchschnitts der Bewertungen pro Jahr und Standort
    avg_ratings_per_year = df.groupby(['Year', 'Branch'])['Rating'].mean().unstack()

    # Erstellen des Plots für den Durchschnitt der Bewertungen
    fig, ax = plt.subplots(figsize=(14, 7))
    avg_ratings_per_year.plot(ax=ax, marker='o')

    ax.set_ylabel('Average Rating', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_title('Average Rating Over Time For Each Park', fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Jedes Jahr anzeigen
    ax.legend(title='Branch', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legende außerhalb des Plots

    # Anzeigen des Plots
    plt.tight_layout()  # Sorgt dafür, dass die Legende im sichtbaren Bereich bleibt
    plt.show()


# Hauptfunktion, die das Skript ausführt
def main():
    # Versuche, die CSV-Datei zu lesen, wobei zunächst von utf-8 Kodierung ausgegangen wird
    try:
        df = pd.read_csv('Dataset/DisneylandReviews.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('Dataset/DisneylandReviews.csv', encoding='ISO-8859-1')  # Versuche alternative Kodierung

    # Erstellen des Durchschnittsbewertungs-Diagramms
    create_average_rating_plot(df)


if __name__ == "__main__":
    main()
