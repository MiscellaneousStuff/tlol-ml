# TLoL - Jinx Early Game Behavioural Cloning

## Datasets

A brief description of datasets used for this project are listed below:

 - `jinx_833_ml_db.zip`
   - total files: 833
   - approx size: 7.7GB
   - map: Summoner's Rift
   - gamemode: Ranked Solo/Duo
   - source: [tlol](https://github.com/MiscellaneousStuff/tlol)

### Structure of .pkl files (Observations and action labels)

Each file contains an ordered sequence of roughly 1,428 frames
(the first 5 to 300 seconds of gameplay). Each file is a Python
pickle file where the pickled data is a large NumPy array
containing both observations and actions combined into a single
large array. These NumPy arrays contain 3,189 columns, with the last
20 columns being dedicated to action labels and parameters, while
all of the columns before that are for observations.