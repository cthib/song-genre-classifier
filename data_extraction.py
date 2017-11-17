import pandas as pd

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
joined_features = pd.concat([genre_list, feature_list], axis=1, join='inner').drop('genre_top', 1)

print('Creating csv files...')
joined_features.to_csv('dataset/feature_subset.csv')
genre_list.to_csv('dataset/genre_subset.csv')
print('Done.')