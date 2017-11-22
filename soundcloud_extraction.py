"""
SoundScrape functions
"""
import os

from clint.textui import colored, puts, progress
from os.path import dirname, exists, join
from urllib import parse

from soundscrape.soundscrape import download_file, download_track, download_tracks, \
                                    get_client, get_hard_track_url, get_soundcloud_api_playlist_data, \
                                    get_soundcloud_api2_data, get_soundcloud_data, \
                                    puts_safe, sanitize_filename, tag_file


def process_soundcloud(vargs):
    """
    Subset of SoundScrape process_soundcloud call.
    Not ideal, but need the filename(s).
    """

    artist_url = vargs['artist_url']
    num_tracks = vargs['num_tracks']

    # Restricting SoundScrape usage
    keep_previews = False
    folders = False
    downloadable = False

    id3_extras = {}
    one_track = False
    likes = False
    client = get_client()

    if 'soundcloud' not in artist_url.lower():
        puts(colored.red("Could not find soundcloud in the URL"))
        return None

    try:
        resolved = client.get('/resolve', url=artist_url, limit=200)
    except Exception as e:  # HTTPError?

        # SoundScrape is trying to prevent us from downloading this.
        # We're going to have to stop trusting the API/client and
        # do all our own scraping. Boo.

        if '404 Client Error' in str(e):
            puts(colored.red("Problem downloading [404]: ") + colored.white("Item Not Found"))
            return None

        message = str(e)
        item_id = message.rsplit('/', 1)[-1].split('.json')[0].split('?client_id')[0]
        hard_track_url = get_hard_track_url(item_id)

        track_data = get_soundcloud_data(artist_url)
        puts_safe(colored.green("Scraping") + colored.white(": " + track_data['title']))

        filenames = []
        filename = sanitize_filename(track_data['artist'] + ' - ' + track_data['title'] + '.mp3')

        if exists(filename):
            puts_safe(colored.yellow("Track already downloaded: ") + colored.white(track_data['title']))
            return None

        filename = download_file(hard_track_url, filename)
        tagged = tag_file(filename,
                 artist=track_data['artist'],
                 title=track_data['title'],
                 year='2017',
                 genre='',
                 album='',
                 artwork_url='')

        if not tagged:
            wav_filename = filename[:-3] + 'wav'
            os.rename(filename, wav_filename)
            filename = wav_filename

        filenames.append(filename)

    else:
        aggressive = False

        # This is is likely a 'likes' page.
        if not hasattr(resolved, 'kind'):
            tracks = resolved
        else:
            if resolved.kind == 'artist':
                artist = resolved
                artist_id = str(artist.id)
                tracks = client.get('/users/' + artist_id + '/tracks', limit=200)
            elif resolved.kind == 'playlist':
                id3_extras['album'] = resolved.title
                if resolved.tracks != []:
                    tracks = resolved.tracks
                else:
                    tracks = get_soundcloud_api_playlist_data(resolved.id)['tracks']
                    tracks = tracks[:num_tracks]
                    aggressive = True
                    for track in tracks:
                        download_track(track, resolved.title, keep_previews, folders, custom_path=vargs['path'])

            elif resolved.kind == 'track':
                tracks = [resolved]
            elif resolved.kind == 'group':
                group = resolved
                group_id = str(group.id)
                tracks = client.get('/groups/' + group_id + '/tracks', limit=200)
            else:
                artist = resolved
                artist_id = str(artist.id)
                tracks = client.get('/users/' + artist_id + '/tracks', limit=200)
                if tracks == [] and artist.track_count > 0:
                    aggressive = True
                    filenames = []

                    data = get_soundcloud_api2_data(artist_id)

                    for track in data['collection']:

                        if len(filenames) >= num_tracks:
                            break

                        if track['type'] == 'playlist':
                            track['playlist']['tracks'] = track['playlist']['tracks'][:num_tracks]
                            for playlist_track in track['playlist']['tracks']:
                                album_name = track['playlist']['title']
                                filename = download_track(playlist_track, album_name, keep_previews, folders, filenames, custom_path=vargs['path'])
                                if filename:
                                    filenames.append(filename)
                        else:
                            d_track = track['track']
                            filename = download_track(d_track, custom_path=vargs['path'])
                            if filename:
                                filenames.append(filename)

        if not aggressive:
            filenames = download_tracks(client, tracks, num_tracks, downloadable, folders, vargs['path'],
                                        id3_extras=id3_extras)

    return filenames
