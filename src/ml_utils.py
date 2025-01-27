import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pieces_dict = {
    #None values
    'None':0,
    #white pieces
    'wQ': 1,
    'wA1': 2, 'wA2': 2, 'wA3': 2, 
    'wG1': 3, 'wG2': 3, 'wG3': 3,
    'wB1': 4, 'wB2': 4, 
    'wS1': 5, 'wS2': 5,
    'wM': 6,
    'wL': 7,
    'wP': 8,
    #black pieces
    'bQ': -1,
    'bA1': -2, 'bA2': -2, 'bA3': -2,
    'bG1': -3, 'bG2': -3, 'bG3': -3,
    'bB1': -4, 'bB2': -4,
    'bS1': -5, 'bS2': -5,
    'bM': -6,
    'bL': -7,
    'bP': -8
}

color_player_dict = {
    'White': 1,
    'Black': -1
}

def df_preprocessing(df):

    # Replace all NaN values with 0
    df.fillna(0, inplace=True)

    # Use a MinMaxScaler to scale number_of_turn 
    scaler = MinMaxScaler()
    df['number_of_turn'] = scaler.fit_transform(df[['number_of_turn']])

    # Encoding the neighbor cols
    cols = list(df.columns)
    neighbor_cols = [col for col in cols if 'neighbor' in col]
    for col in neighbor_cols:
        df[col] = df[col].replace(pieces_dict)

    # Encoding the player cols
    color_player_cols = ['last_move_played_by', 'current_player_turn', 'result']
    for col in color_player_cols:
        if col in df.columns:
            df[col] = df[col].replace(color_player_dict)

    return df