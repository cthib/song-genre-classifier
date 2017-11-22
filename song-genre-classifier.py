#! /usr/bin/env python
from __future__ import unicode_literals

import argparse
import os
import sys
import urllib

from genre_classification import train_base
from soundcloud_extraction import process_soundcloud


def main():
    """
    song-genre-classifier argument parser.
    Extended version of SoundScrape argparse.
    """

    if sys.platform == "win32":
        os.system("chcp 65001");

    parser = argparse.ArgumentParser(description='SongGenreClassifer. Classify your songs using TensorFlow.\n')
    parser.add_argument('artist_url', metavar='U', type=str, nargs='*',
                        help='An artist\'s SoundCloud URL')
    parser.add_argument('-a', '--album', action="store_true",
                        help='Classify the genre of an album (Needs a soundcloud set link.)')
    parser.add_argument('-d', '--demo', action="store_true",
                        help='Demo the system')
    parser.add_argument('-n', '--num-tracks', type=int, default=1,
                        help='The number of tracks to be classified from a soundcloud set')
    parser.add_argument('-p', '--path', type=str, default='',
                        help='Set directory path where soundcloud downloads should be saved to')
    parser.add_argument('-t', '--train', action="store_true",
                        help='Train the classifier')

    args = parser.parse_args()
    vargs = vars(args)

    if vargs['train']: 
        train_base()
        return

    if vargs['album']:
        # TODO: Classify a whole album, we dont really need to do this... but if you're bored
        pass

    if vargs['demo']:
        vargs['artist_url'] = 'https://soundcloud.com/swcraig/sebbatical'
    elif sys.version_info < (3,0,0):
        vargs['artist_url'] = urllib.quote(vargs['artist_url'][0], safe=':/')
    else:
        vargs['artist_url'] = urllib.parse.quote(vargs['artist_url'][0], safe=':/')

    process_soundcloud(vargs)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)