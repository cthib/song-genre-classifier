class Genres(object):

    def __init__(self):
        self.genres = {
            "Rock": 0,
            "Experimental": 1,
            "Electronic": 2,
            "Hip-Hop": 3,
            "Folk": 4,
            "Pop": 5,
            "Instrumental": 6,
            "International": 7,
            "Classical": 8,
            "Jazz": 9,
            "Country": 10,
            "Soul-RnB": 11,
            "Blues": 12,
        }

    def get_genres(self):
        return self.genres

    def get_empty_genre13(self):
        return dict.fromkeys(self.genres, 0)

    def get_empty_genre10(self):
        gen_dict = dict.fromkeys(self.genres, 0)
        for i in range(3):
            gen_dict.popitem()
        return gen_dict

    def get_empty_genre8(self):
        gen_dict = dict.fromkeys(self.genres, 0)
        for i in range(5):
            gen_dict.popitem()
        return gen_dict
