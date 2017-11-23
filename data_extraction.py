import pandas as pd

from sklearn.model_selection import train_test_split

features_path = 'dataset/fma/features.csv'
tracks_path = 'dataset/fma/tracks.csv'

print('Starting csv extraction...')
features = pd.read_csv(features_path, index_col=0, header=[0, 1, 2])
tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

# Get genre_top list subset
print('- Genre subset')
genre_top = tracks.track.genre_top
without_na = genre_top.dropna(how="any")
genre_list = without_na.loc[~genre_top.isin(["Spoken", "Easy Listening", "Old-Time / Historic"])]

# We want the features for all songs based on 13 top-level genres
genre_list = genre_list.to_frame()
genre_list.columns = ['genre_top']

# The audio features we are interested in, joining with genre subset
print('- Feature subset')
feature_list = features[['spectral_bandwidth', 'spectral_centroid', 'spectral_rolloff', 'zcr', 'rmse']]
joined_features = pd.concat([feature_list, genre_list], axis=1, join='inner')

# Randomizing and splitting data
print('- Randomize split')
mixed_features = joined_features.sample(frac=1).reset_index(drop=True)
train = mixed_features.sample(frac=0.8, random_state=200)
test = mixed_features.drop(train.index)

print('Creating csv files...')
joined_features.to_csv('dataset/feature_subset.csv')
train.to_csv('dataset/unbalanced_train_set.csv')
test.to_csv('dataset/unbalanced_test_set.csv')
print('Done.')