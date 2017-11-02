# song-genre-classifier
A simple song genre classifier using basic music information retrieval techniques, such as pitch detection, texture, and tempo, in combination with Tensorflow.

## Setup

### 1. virtualenv and virtualenvwrapper
You will need to install virtualenvwrapper for this project. We will have some python dependencies for the project and it makes it super helpful when we have 3rd party packages.
  1. Download virtualenv (https://pypi.python.org/pypi/virtualenv)
  2. `pip install virtualenvwrapper` (https://virtualenvwrapper.readthedocs.io/en/latest/)
### 2. Creating a virtualenv for the project
To help with conflicts in project dependencies we'll create a virtualenv for the project. The command reference: 
  1. `mkvirtualenv -p python3 sgc` *NOTE: You will need python3 for this. Also the -p python3 command may vary with your OS*
  2. `workon sgc` this switches to your song-genre-classifier env. Ensure that you have `(sgc)` at the very most left point of your command line
  3. While in your env: `pip install -r requirements.txt`
  4. `pip freeze` will show the project dependencies

## Dependent Resources

### TensorFlow (https://www.tensorflow.org/)
pass

### FMA: A Dataset For Music Analysis (https://github.com/mdeff/fma)
pass

### Librosa (https://librosa.github.io/librosa/index.html)
pass

### SoundScrape (https://github.com/Miserlou/SoundScrape)
pass
