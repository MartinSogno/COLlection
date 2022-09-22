import pandas as pd


def get_all_cols():
    cols = pd.read_csv('cols/fr.csv')
    cols['latitude'] = cols['WGS84 Lat D'].apply(lambda l: float(l.replace(',', '.')))
    cols['longitude'] = cols['WGS84 Lon D'].apply(lambda l: float(l.replace(',', '.')))
    cols['col_name'] = cols['Nom complet']
    return cols
