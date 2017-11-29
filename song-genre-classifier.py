from __future__ import unicode_literals

import argparse
import os
import sys
import urllib
import ssl

from feature_extraction import compute_features
from genres import Genres
from genre_classification import train_softmax, classify_softmax, train_dnn
from soundcloud_extraction import process_soundcloud


def main():
    """
    song-genre-classifier argument parser.
    Extended version of SoundScrape argparse.
    """

    if sys.platform == "win32":
        os.system("chcp 65001");

    parser = argparse.ArgumentParser(description='SongGenreClassifer. Classify your songs using TensorFlow.\n')
    
    # SoundScrape specific args
    parser.add_argument('artist_url', metavar='U', type=str, nargs='*',
                        help='An artist\'s SoundCloud URL')
    parser.add_argument('-n', '--num-tracks', type=int, default=1,
                        help='The number of tracks to be classified from a soundcloud set')
    parser.add_argument('-p', '--path', type=str, default='',
                        help='Set directory path where soundcloud downloads should be saved to')
    
    # Song-genre-classifier args
    parser.add_argument('-d', '--demo', action="store_true",
                        help='Demo the system')
    parser.add_argument('-tsm13', '--trainsoftmax13', action="store_true",
                        help='Train softmax classifier on 13 genres')
    parser.add_argument('-tsm10', '--trainsoftmax10', action="store_true",
                        help='Train softmax classifier on 10 genres')
    parser.add_argument('-tsm8', '--trainsoftmax8', action="store_true",
                        help='Train softmax classifier on 8 genres')
    parser.add_argument('-tnn13', '--trainneuralnet13', action="store_true",
                        help='Train deep neural net classifier on 13 genres')
    parser.add_argument('-tnn10', '--trainneuralnet10', action="store_true",
                        help='Train deep neural net classifier on 10 genres')
    parser.add_argument('-tnn8', '--trainneuralnet8', action="store_true",
                        help='Train deep neural net classifier on 8 genres')

    args = parser.parse_args()
    vargs = vars(args)

    if vargs['trainsoftmax13']: 
        return train_softmax(save_sess=False)
    elif vargs['trainsoftmax10']:
        return train_softmax(num_classes=10, save_sess=False)
    elif vargs['trainsoftmax8']:
        return train_softmax(num_classes=8, save_sess=False)
    elif vargs['trainneuralnet13']:
        return train_dnn(save_sess=False)
    elif vargs['trainneuralnet10']:
        return train_dnn(num_classes=10, save_sess=False)
    elif vargs['trainneuralnet8']:
        print("Feature unavailable.")
        return train_dnn(num_classes=8, save_sess=False)

    if vargs['demo']:
        vargs['artist_url'] = 'https://soundcloud.com/emerson-mart-nez/a-tribe-called-quest-electric-relaxation-instrumental'
    elif sys.version_info < (3,0,0):
        vargs['artist_url'] = urllib.quote(vargs['artist_url'][0], safe=':/')
    else:
        vargs['artist_url'] = urllib.parse.quote(vargs['artist_url'][0], safe=':/')

    fn = process_soundcloud(vargs)[0]
    features = compute_features(fn)
    train_softmax(save_sess=False, features=features)
    os.remove(fn)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)