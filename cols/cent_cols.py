import pandas as pd


def get_all_cols():
    """Get all cols from the centcols list."""
    cols = pd.read_csv('cols/col_list/fr.csv')
    cols['latitude'] = cols['WGS84 Lat D'].apply(lambda l: float(l.replace(',', '.')))
    cols['longitude'] = cols['WGS84 Lon D'].apply(lambda l: float(l.replace(',', '.')))
    cols['elevation'] = cols['Altitude'].apply(int)
    cols['name'] = cols['Nom complet']
    return cols
