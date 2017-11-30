# song-genre-classifier
A simple song genre classifier using basic music information retrieval techniques, such as pitch detection, texture, and tempo, in combination with Tensorflow. This was done as a project for the University of Victoria's CSC475 "Music Information Retrieval" course. 

## Setup

### 1. virtualenv and virtualenvwrapper
You will need to install virtualenvwrapper for this project. We will have some python dependencies for the project and it makes it super helpful when we have 3rd party packages.
  1. Download virtualenv (https://pypi.python.org/pypi/virtualenv)
  2. `pip install virtualenvwrapper` (https://virtualenvwrapper.readthedocs.io/en/latest/)
  3. Initialize (https://virtualenvwrapper.readthedocs.io/en/latest/#introduction) 
     ```
     export WORKON_HOME=~/Envs
     mkdir -p $WORKON_HOME
     source /usr/local/bin/virtualenvwrapper.sh
     ```
### 2. Creating a virtualenv for the project
To help with conflicts in project dependencies we'll create a virtualenv for the project. The command reference: 
  1. `mkvirtualenv -p python3 sgc` *NOTE: You will need python3 for this. Also the -p python3 command may vary with your OS*
  2. `workon sgc` this switches to your song-genre-classifier env. Ensure that you have `(sgc)` at the very most left point of your command line
  3. While in your env: `pip install -r requirements.txt`
  4. `pip freeze` will show the project dependencies

### 3. Running song-genre-classifier
For a demo of downloading a song from SoundCloud, running feature extraction using Librosa, training a tensor flow model on 13 genres, and then classifying the genre...
`python song-genre-classifier.py -d`

## Main Dependent Resources

### FMA: A Dataset For Music Analysis (https://github.com/mdeff/fma)
The FMA (Free Music Archive) dataset is a large set of audio files and metadata .csv files for 106,574 tracks. For this project the top-level genres from the `genres.csv` were used as well as the extracted `features.csv`. After joining the sets and removing empty genres and general case genres, resulting set was just over 48,000 tracks.

### TensorFlow (https://www.tensorflow.org/)
In this project an TensorFlow based SoftMax regression model was used. The training of the model was done over the 3 different genre feature sets obtained from the FMA dataset. The accuracy result for the balanced 13 genres averages at 85.6%.  

### Librosa (https://librosa.github.io/librosa/index.html)
Librosa is a python package for audio feature extraction. It was the main tool used in extracing the spectral centroid, spectral bandwidth, spectral rolloff, RMSE, and ZCR in the FMA dataset. For classification accuracy, the same Librosa library was used to extract features from audio files to create a feature vector.

### SoundScrape (https://github.com/Miserlou/SoundScrape)
To obtain a SoundCloud API key to download the audio files from SoundCloud, a modified version of SoundScrape was used. The main functionality was missing one small component, which was the return of the filename string which only needed small altercations.

## Future work
Future work is needed in saving the trained models as well as running validation on the features extracted from Librosa.