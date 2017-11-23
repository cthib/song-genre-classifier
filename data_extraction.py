import pandas as pd
from sklearn.model_selection import train_test_split

def balance(df, vals, max=100):
    drop = []
    stop = len(vals) * max
    for i, row in df.iterrows():
        genre = row['genre_top']
        
        if vals[genre] < max:
            vals[genre] += 1
            count = sum(vals.values())
        else:
            drop.append(i)
    return df.drop(drop)
    

features_path = 'dataset/fma/features.csv'
tracks_path = 'dataset/fma/tracks.csv'

genre13_dict = {
    "Rock": 0,
    "Experimental": 0,
    "Electronic": 0,
    "Hip-Hop": 0,
    "Folk": 0,
    "Pop": 0,
    "Instrumental": 0,
    "International": 0,
    "Classical": 0,
    "Jazz": 0,
    "Country": 0,
    "Soul-RnB": 0,
    "Blues": 0,
}

genre10_dict = {
    "Rock": 0,
    "Experimental": 0,
    "Electronic": 0,
    "Hip-Hop": 0,
    "Folk": 0,
    "Pop": 0,
    "Instrumental": 0,
    "International": 0,
    "Classical": 0,
    "Jazz": 0,
}

genre8_dict = {
    "Rock": 0,
    "Experimental": 0,
    "Electronic": 0,
    "Hip-Hop": 0,
    "Folk": 0,
    "Pop": 0,
    "Instrumental": 0,
    "International": 0,
}

print('Starting csv extraction...')
features = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

# Get genre_top list subset
print('- Genre subset')
genre_top = tracks.track.genre_top
without_na = genre_top.dropna(how="any")
genre13_list = without_na.loc[~genre_top.isin(["Spoken", "Easy Listening", "Old-Time / Historic"])]
genre10_list = genre13_list.loc[~genre_top.isin(["Country", "Soul-RnB", "Blues"])]
genre8_list = genre10_list.loc[~genre_top.isin(["Classical", "Jazz"])]

# We want the features for all songs based on 13 top-level genres
genre13_list = genre13_list.to_frame()
genre13_list.columns = ['genre_top']

# We want the features for all songs based on 10 top-level genres
genre10_list = genre10_list.to_frame()
genre10_list.columns = ['genre_top']

# We want the features for all songs based on 8 top-level genres
genre8_list = genre8_list.to_frame()
genre8_list.columns = ['genre_top']

# The audio features we are interested in, joining with genre subset
print('- Feature subset')
feature_list = features[['spectral_bandwidth', 'spectral_centroid', 'spectral_rolloff', 'zcr', 'rmse']]
joined13_features = pd.concat([feature_list, genre13_list], axis=1, join='inner')
joined10_features = pd.concat([feature_list, genre10_list], axis=1, join='inner')
joined8_features = pd.concat([feature_list, genre8_list], axis=1, join='inner')

# Randomizing and splitting data
print('- Randomize split')
mixed13_features = joined13_features.sample(frac=1).reset_index(drop=True)
mixed10_features = joined10_features.sample(frac=1).reset_index(drop=True)
mixed8_features = joined8_features.sample(frac=1).reset_index(drop=True)

bal13 = balance(mixed13_features, genre13_dict)
bal10 = balance(mixed10_features, genre10_dict, max=500)
bal8 = balance(mixed8_features, genre8_dict, max=1350)

bal13_train = bal13.sample(frac=0.8, random_state=200)
bal13_test = bal13.drop(bal13_train.index)

bal10_train = bal10.sample(frac=0.8, random_state=200)
bal10_test = bal10.drop(bal10_train.index)

bal8_train = bal8.sample(frac=0.8, random_state=200)
bal8_test = bal8.drop(bal8_train.index)

unbal13_train = mixed13_features.sample(frac=0.8, random_state=200)
unbal13_test = mixed13_features.drop(bal13_train.index)

# Start writing the CSV files
print('Creating csv files...')
print('- Base feature subset')
joined13_features.to_csv('dataset/feature_subset.csv')

print('- Unbalanced 13 top level genres')
unbal13_train.to_csv('dataset/unbalanced13/train_set.csv', index=False)
unbal13_test.to_csv('dataset/unbalanced13/test_set.csv', index=False)

print('- Balanced 13 top level genres')
bal13_train.to_csv('dataset/balanced13/train_set.csv', index=False)
bal13_test.to_csv('dataset/balanced13/test_set.csv', index=False)

print('- Balanced 10 top level genres')
bal10_train.to_csv('dataset/balanced10/train_set.csv', index=False)
bal10_test.to_csv('dataset/balanced10/test_set.csv', index=False)

print('- Balanced 8 top level genres')
bal8_train.to_csv('dataset/balanced8/train_set.csv', index=False)
bal8_test.to_csv('dataset/balanced8/test_set.csv', index=False)

print('Done.')