import numpy as np
import itertools
import pandas as pd
import re
import json
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import librosa
import librosa.display
import IPython.display as ipd
from random import randint
import os
from numpy import random as rd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from collections import defaultdict
from functools import reduce
from sklearn import preprocessing 
from scipy.cluster.hierarchy import dendrogram, linkage
from numpy import linalg
from sklearn.cluster import KMeans
from pathlib import Path, PurePath   
from tqdm.notebook import tqdm
from sklearn.metrics import silhouette_score
import scipy.spatial.distance as metric
from yellowbrick.cluster import KElbowVisualizer
import subprocess
import math
from sklearn.decomposition import PCA
import sys
import plotly.graph_objects as go
import traceback
import itertools
from sklearn import metrics

path = '/content/drive/MyDrive/HomeworkIV'

############### QUESTION 1 ###############

def convert_mp3_to_wav(audio:str) -> str:  
    """Convert an input MP3 audio track into a WAV file.

    Args:
        audio (str): An input audio track.

    Returns:
        [str]: WAV filename.
    """
    if audio[-3:] == "mp3":
        wav_audio = audio[:-3] + "wav"
        if not Path(wav_audio).exists():
                subprocess.check_output(f"ffmpeg -i {audio} {wav_audio}", shell=True)
        return wav_audio
    
    return audio

def plot_spectrogram_and_picks(track:np.ndarray, sr:int, peaks:np.ndarray, onset_env:np.ndarray) -> None:
    """[summary]

    Args:
        track (np.ndarray): A track.
        sr (int): Aampling rate.
        peaks (np.ndarray): Indices of peaks in the track.
        onset_env (np.ndarray): Vector containing the onset strength envelope.
    """
    HOP_SIZE = 512

    times = librosa.frames_to_time(np.arange(len(onset_env)),
                            sr=sr, hop_length=HOP_SIZE)

    plt.figure()
    ax = plt.subplot(2, 1, 2)
    D = librosa.stft(track)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time')
    plt.subplot(2, 1, 1, sharex=ax)
    plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    plt.vlines(times[peaks], 0,
            onset_env.max(), color='r', alpha=0.8,
            label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.8)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

def load_audio_picks(audio, duration, hop_size):
    """[summary]

    Args:
        audio (string, int, pathlib.Path or file-like object): [description]
        duration (int): [description]
        hop_size (int): 

    Returns:
        tuple: Returns the audio time series (track) and sampling rate (sr), a vector containing the onset strength envelope
        (onset_env), and the indices of peaks in track (peaks).
    """
    try:
        track, sr = librosa.load(audio, duration=duration)
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length=hop_size)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
    except Exception as e:
        print('An error occurred processing ', str(audio))
        print(e)

    return track, sr, onset_env, peaks

def preprocessing_converter(N_TRACKS, mp3_tracks):
    for track in tqdm(mp3_tracks, total=N_TRACKS):
        convert_mp3_to_wav(str(track))

def audio_signals(HOP_SIZE, DURATION, tracks):
    for idx, audio in enumerate(tracks):
        if idx >= 2:
            break
        track, sr, onset_env, peaks = load_audio_picks(audio, DURATION, HOP_SIZE)
        plot_spectrogram_and_picks(track, sr, peaks, onset_env)

def collect_query(query_tracks):
    query_list= []
    for q in query_tracks:
        query_list.append(str(q))
    return query_list

def get_tracks_informations(tracks, DURATION, HOP_SIZE):

  ''' This function return title, peaks frequencies of all songs in our dataset'''
  
  titles_tracks  = []
  peaks_freq = []
  for audio in tqdm(tracks):
    # find the titles of the songs
    titles_tracks.append(re.search('\d+-(.+?).wav', audio).group(1).replace('_',' '))
    # use the (given) function to get the peacks and their frequencies
    _, _, onset_env, peaks = load_audio_picks(audio, DURATION, HOP_SIZE) 
    f = [ '%.1f' % elem for elem in onset_env[peaks]]
    peaks_freq.append(list(map(float,f)))
  return titles_tracks, peaks_freq

def load_json_list(path, file_output_name):

  ''' This function allows to load and read correctly a .json file which 
  contains a list of lists from a specific path. '''

  f = open(path)
  file_output_name = json.load(f)
  return file_output_name

def load_json_dict(path, file_output_name):

  ''' This function allows to load and read correctly a .json file which 
  contains a dictionary from a specific path. '''

  with open(path) as json_file:
    file_output_name = json.load(json_file)

  return file_output_name

def create_shingles(peaks_freqencies):

  ''' This function create a list of all unique values of 'peaks frequencies', 
  called shingles. The input peaks_freqencies must be a list of lists. '''

  # define an empty array to append shingles 
  shingles = set()
  # collect all the 'peak frequence' that appear in the input array
  for sublist in peaks_freqencies:
      for item in sublist:
        shingles.add(round(item,1))

  return np.array(list(shingles))

def create_characteristic_matrix(peaks_freqencies, shingles):
  
  '''This function creates a matrix C that has all shingles values as rows and 
  all different songs as columns. Its generic value C_ij is equal to one if the j-th
  song has the i-th 'peak frequency' among its 'peaks frequencies' and 0 otherwise.
  The input peaks_freqencies must be a list of lists'''
  

  # set the shape of the output matrix
  n = len(shingles)   # number of rows
  m = len(peaks_freqencies)   # number of columns

  # define a matrix of zeros with the correct shape
  characteristic_matrix =  [ [ 0 for j in range(m) ] for i in range(n) ]
  for i in range(n):
    for j in range(m):
      # set '1' if the j-th song has the i-th 'peak frequency' among its 'peaks frequencies' 
      if shingles[i] in np.array(peaks_freqencies[j]).round(1):
        characteristic_matrix[i][j] = 1
  
  return np.array(characteristic_matrix)

def update_signature_matrix(characteristic_matrix):

  ''' This function returns a list that corresponds to a row of the signature matrix 
  which has all hash_songs as columns.
  Each row of signature matrix contains the value of row-index of the first value 
  equal to '1' in characteristic matrix that is given in input. '''

  # create an empty array 
  perm_row = []
  for j in range(len(characteristic_matrix[0])):
    for i in range(len(characteristic_matrix)):
      # serch the first value that is not equal to 0
      if characteristic_matrix[i][j] == 1:
        perm_row.append(i)
        # for loop must stop when the first 1 has been found
        break
  return perm_row

def get_perm(M):

  ''' This function returns a permutation of a list with values from 0 to M'''
  
  result = np.arange(M)
  return np.random.permutation(result)

def signature_matrix(peaks_freqencies, shingles):
  
  ''' This function returns the signature matrix taking in input only the peaks' frequencies.
  It calls the function 'create_characteristic_matrix' to compute characteristic matrix'''

  characteristic_matrix = create_characteristic_matrix(peaks_freqencies, shingles)

  signature_matrix = []
  list_permutation = [] # list to memorize index-permutations 

  # set 20 as the number of purmutation --> number of rows of the signature matrix
  for i in range(20):
    # get permuted rows' indexes
    permutation = get_perm(len(characteristic_matrix))
    list_permutation.append(permutation)
    
    # create the permuted characteristic matrix
    perm_mat = characteristic_matrix[permutation,:]
    
    # create the row of the signature matrix
    perm_row = update_signature_matrix(perm_mat)
    # fill the matrix
    signature_matrix.append(perm_row)
  
  return signature_matrix, list_permutation 

def define_buckets(signature_matrix, r = 4):
  ''' This function takes in input a signature matrix of shape (n_permutations, n_songs)
  and returns a dictionary which has as key a 'hash' value and as values all the songs that contain that 'hash'
  Specifically we set (default) r = 2 that means that we are going to split the matrix in 5 
  bands that contain 4 rows each (b = 5, r = 4). '''
  
  # create an empty dictionary
  buckets = dict()
  for col_idx, col in enumerate(np.transpose(signature_matrix)):
    # each bucket has 4 row (r = 4). Thus 
    for i in range(0, len(col), r):
      hash = tuple(col[i:i+r])
      # fill the dictionary
      if hash in buckets:
        buckets[hash].add(str(col_idx))
      else:
        buckets[hash] = {str(col_idx)}
  return(buckets)

def get_matches(DURATION, HOP_SIZE, titles_tracks, peaks_freq, queries, r = 4):

  ''' This function takes in input the queries (array), an array of songs' titles and a list of songs' peak-frequencies. 
  It computes the signature matrix for the tracks and creates a dictionary which map for each bucket a list of song 
  (that are mapped in that bucket) - from the signature matrix. 
  Then each query is compared only with the songs that are mapped in the same buckets of the query. The 'best match' of a query is
  the song that compares  with the major frequency among them.'''

  # call previous functions to get all objects to compute matching
  
  #titles_tracks, peaks_freq = get_tracks_informations(tracks)
  
  shingles = create_shingles(peaks_freq)
  signature_mat, permutations = signature_matrix(peaks_freq, shingles)
  bucket_dict = define_buckets(signature_mat, r)
  
  matching_dict = dict()
  titles_query = []

  for query in queries:

    # find the 'title' of the query
    title_query = str(query).split('/')[-1].split('.')[0]
    _, _, onset_env_query, peaks_query = load_audio_picks(query, DURATION, HOP_SIZE) 
    f = [ '%.1f' % elem for elem in onset_env_query[peaks_query]]
    peaks_freq_query = [list(map(float,f))] 

    bucket_query = []
    characteristic_matrix_query = create_characteristic_matrix(peaks_freq_query,shingles)
    
    # Get the signature matrix (vector) for a single query with the 
    # same permutations of the songs' signature matrix
    for perm in permutations:
      car_q = characteristic_matrix_query[perm]
      bucket_query.append(update_signature_matrix(car_q))
    
    # covert bucket_query (list of lists) into array
    b = np.matrix(bucket_query)
    bucket_query = list(np.array(b).reshape(-1,))
    
    # create a set of buckets to which that query has been mapped
    buckets = set()
    for i in range(0, len(bucket_query), r):
      buckets.add(tuple(bucket_query[i:i+r]))
    
    # serch the indexes of the songs' tracks that have been mapped in the same buckets of thet query
    comparing_tracks_index = []
    for b in buckets:
      if b in bucket_dict:
        comparing_buckets = bucket_dict[b]
        for track in comparing_buckets:
          comparing_tracks_index.append(track)

    # compute the frequencies of the songs that are mapped in the same buckets of the query
    matching_tracks = []
    scores = []
    unique, counts = np.unique(comparing_tracks_index, return_counts=True)
    frequecy_song = dict(zip(unique, counts))
    
    matching = max(frequecy_song, key=frequecy_song.get)
    matching_dict[title_query] = titles_tracks[int(matching)]

  return matching_dict

def get_tracks_list(tracks):
    tracks_list = []
    for i in tracks:
        tracks_list.append(str(i))
    return tracks_list

def save_peaks_freq(peaks_freq):
    with open(path+'/peaks_freq.json', 'w') as f:
        json.dump(peaks_freq, f)  

def save_titles_tracks(titles_tracks):
    with open(path+'/titles_tracks.json', 'w') as f:
        json.dump(titles_tracks, f)  

def save_matching_dict(matching_dict):
    with open(path+'/matching_dict.json', 'w') as f:
        json.dump(matching_dict, f)

def print_matching(input_track, matches_dict):

  track = list(matches_dict.keys())[int(input_track)-1]

  fig = go.Figure(data=[go.Table(
      columnwidth = 80,
      header=dict(values=['MATCHING SONG'],
                  line_color='silver',
                  fill_color='silver',
                  align='center',
                  font=dict(color='snow', size=12),
                  height=30),
      cells=dict(values=['<b>{}<b>'.format(matches_dict[track])],
                line_color='white',
                fill =  dict(color='dodgerblue'),
                font=dict(color='snow', size=24),
                height=50)

                      )])
  fig.update_layout(width=1000, height=300)
  fig.show()

def handle_q_1():
    N_TRACKS = 1413
    HOP_SIZE = 512
    DURATION = 30 # TODO: to be tuned!
    THRESHOLD = 0 # TODO: to be tuned!
    data_folder = Path(path + "/mp3s-32k/")
    mp3_tracks = data_folder.glob("*/*/*.mp3")
    tracks = data_folder.glob("*/*/*.wav")
    preprocessing_converter(N_TRACKS, mp3_tracks)
    tracks_list = get_tracks_list(tracks)
    data_folder_query = Path(path + "/query/")
    query_tracks = data_folder_query.glob("*.wav")
    query_list = collect_query(query_tracks)
    titles_tracks, peaks_freq = get_tracks_informations(tracks_list, DURATION, HOP_SIZE)
    save_peaks_freq(peaks_freq)
    save_titles_tracks(titles_tracks)
    matching_dict = get_matches(DURATION, HOP_SIZE, titles_tracks,peaks_freq, query_list, r = 4)
    save_matching_dict(matching_dict)
    matches = dict()    
    matches = load_json_dict(path+'/matching_dict.json', matches)
    return matches


############### QUESTION 2 ###############

def remove_NaN_values(dataset):
  ''' Given a dataset, this function fills NaN values with 0 '''
  for col in dataset.columns:
        if dataset[col].isnull().any() == True:
            if is_numeric_dtype(dataset[col]) == True: #case of numeric column
                dataset[col] = dataset[col].fillna(dataset[col].mean())
            elif is_string_dtype(dataset[col]) == True: # case of string column
                dataset[col] = dataset[col] .fillna("")  

def remove_object_col(dataset):
    ''' This function removes object type columns from the given dataset and returns a cleaned one '''
    dataset_cleaned = dataset.select_dtypes(exclude = 'object') 
    return dataset_cleaned

def scaling_features(dataset):
  ''' This function returna a dataset with all scaled features. 
  Each feature is obtained by subtracting the mean and dividing by the standard deviation. '''
  scaler = preprocessing.StandardScaler()
  # We are not going to consider for the scale the track_id column which is the first one 
  dataset_features = pd.DataFrame(scaler.fit_transform(dataset[dataset.columns[1:]].values), columns = dataset.columns[1:])
  # Add again track_id
  final_df = pd.concat([dataset[dataset.columns[:1]], dataset_features], axis = 1)
  
  return final_df

def number_of_components(pca, ratio = 0.755):
  ''' This function returns the number of components that explain about 75% (default) of the total variance'''
  n_components = len(pca.explained_variance_ratio_)
  n = len([np.cumsum(pca.explained_variance_ratio_)[i]  for i in range(n_components) if np.cumsum(pca.explained_variance_ratio_)[i] <= ratio])
  return(n)

def kmeans_scratch(k, ds):
    '''We pass the k number of clusters and a dataset ds, the function will return the k-means from scratch'''
    if isinstance(ds, pd.DataFrame):
      values = ds.values # Creating a values array with the values inside ds
    else:
      values = ds
    m = values.shape[1] # Columns
    n = values.shape[0] # Rows
    
    centroids = values[rd.choice(n, size = k, replace=False)] # Generate random the vector of centroids (initialization step)
    
    prior_cr = np.zeros((n,k)) # Array of zeros to store during the i-th iteration the centroids of the (i-1)-th iteration
    
    it = 0 #number of iterations

    # Loop that will stop if the centroids won't change or the iterations will be at maximum equal to 10 
    while it != 10 or np.array_equal(centroids, prior_cr) == False:
    
        prior_cr = centroids # Centroids for the next iteration
        d = np.zeros((n,k)) # Euclidean distance matrix
        cs = defaultdict(list) # Clusters collected into a dict    
        clusters = [] # Identified cluster 
        
        # Euclidean distances from each point to each centroid
        for i in range(n):
            for j in range(k):
                d[i][j] += linalg.norm(values[i]-centroids[j])

            ''' Assign to each element of the dataset a cluster such that its distance from 
            that cluster is minimized. '''
            clusters.append(np.where(d[i] == min(d[i]))[0][0]+1)
            
            # Dictionary that maps each cluster to the observations that belong to it
            cs[clusters[i]].append(i)
        
        for a in range(k):
            for j in range(m):
                my_values = []
                for i in cs[a+1]: # Clusters' labels are 1, 2, ..., K
                    # Take the values of the observation belonging to the i-th cluster
                    my_values.append(values[i][j])
                # New centroids for each cluster 
                centroids[a][j] = np.mean(my_values)
        it += 1
    return clusters, d, cs

def Elbow_scratch(data, K):

  ''' This function returns a plot with a curve that describes how the 
  withness-variance vary with different number of cluster. 
  It is needed to decide the better number of cluster with the elbow method. '''

  distortions = []
  for n_cluster in tqdm(K):

      # Take the distance matrix from kmeans function
      clusters, distance_matrix, cs = kmeans_scratch(n_cluster, data)

      # compute the squeared-sum of distances of each data-point from the correspondent centroid 
      distortions.append((sum(distance_matrix.min(axis=1)))**2)
  
  # plot the curve
  plt.figure(figsize=(12,5))
  plt.xlabel("Numbers of clusters")
  plt.ylabel("Distortions")
  plt.title('Elbow Curve')
  plt.plot(K,distortions,  linestyle='-', marker='o', color='tomato', lw = 3, alpha = .7, markersize = 8 )
  plt.show()  

def gap_stat(data, k):

    ''' This function returns a plot with a curve that describes how the 
    gap-statistic vary with different number of cluster. 
    It can be used to decide the better number of cluster. '''


    # Compute a matrix of Uniform data samples of the same shape of our input data
    randomReference = np.random.random_sample(size=data.shape)
    cost_r = []
    gaps = []
    s_k = [] 
    for n_cluster in tqdm(range(2,k)):
        # compute k-means and get only the distance metrix and the disctionary 
        _, distance_matrix_random, _ = kmeans_scratch(n_cluster, randomReference) # for the random data-matrix
        _, distance_matrix, _ = kmeans_scratch(n_cluster, data)                          # for our data
        
        # compute the squeared-sum of distances of each data-point from the correspondent centroid 
        costo = sum(distance_matrix.min(axis=1)**2)         # for the random data-matrix
        costr = sum(distance_matrix_random.min(axis=1)**2)  # for our data
        
        cost_r.append(costr) 

        # compute the gap statistic
        gap = np.log(np.mean(cost_r)) - np.log(costo)

        # keep the value of gap statistic for each value of k
        gaps.append(gap)

        # compute the standard deviation (of the random part)
        s_k.append(np.std(np.log(np.mean(cost_r)))*np.sqrt(1+(1/len(cost_r))))
    
    # plot the curve
    plt.figure(figsize=(12,5))
    plt.plot(list(range(2,k)), gaps, linestyle='-', marker='o', color='lightseagreen', lw = 3, alpha = .7, markersize = 6)
    plt.xlabel("Numbers of clusters")
    plt.ylabel("Gap Stistics")
    plt.title('Gap Stistic Curve')
    
    # Get the better value of k according to gap statistics
    # By definition k_star is the min {k | G[k] >= G[k+1] - s_k[k+1]
    k_ = []
    for z in range(0, k-3):
      if (gaps[z] >= gaps[z+1] - s_k[z+1]):
        k_.append(z)
    k_star = min(k_)
    plt.plot(k_star+2, gaps[k_star],  marker='o',markerfacecolor="darkcyan", markersize = 10)
    plt.show()
    return(k_star+2)

def Pivot(echonest, tracks, merged_df, clusters):
  pivot = pd.DataFrame() #create a new dataFrame
  a = []
  #
  for i in range(1,5):
      pivot.insert(i-1, echonest.columns[i], pd.cut(echonest[echonest.columns[i]], bins = 4,labels=["1", "2", "3","4"])) # pd.cut() function is used to separate the array elements into different bins
      #pd.cut will choose the bins to be evenly spaced according to the values themselves
  pivot["track_genre_top"] = tracks.track_genre_top	
  pivot.insert(0,"track_duration", pd.qcut(merged_df[merged_df.columns[7]], q = 4,labels=["1", "2", "3","4"])) #pd.qcut() tries to divide up the underlying data into equal sized bins.
  
  # aggregate categories of the feature track_language_code: we are going to have only four classes (english, french, spanish anh other)
  for lang in range(len(tracks.track_language_code)):
    if tracks.track_language_code[lang] not in ['en','es','fr','']:
      tracks.track_language_code[lang] = 'oth'
  
  pivot["track_language"] = tracks.track_language_code
  pivot.insert(0, "Clusters", clusters)
  return pivot

def PivotDuration(pivot):
    #duration
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
              # I subdivide the values of the pivot table by the number in the track_duration column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.track_duration == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2) #I transform values into percentages
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    track_duration_pivot = pd.DataFrame(t)
    track_duration_pivot.columns = l
    track_duration_pivot.loc[4] = sum(t)
    track_duration_pivot = track_duration_pivot.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})
  

    return track_duration_pivot 

def pivotAcousticness(pivot):
   #acousticness
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
               # I subdivide the values of the pivot table by the number in the audio_features_acousticness column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.audio_features_acousticness == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    audio_features_acousticness_pivot = pd.DataFrame(t)
    audio_features_acousticness_pivot.columns = l
    audio_features_acousticness_pivot.loc[4] = sum(t)
    audio_features_acousticness_pivot = audio_features_acousticness_pivot.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})


    return audio_features_acousticness_pivot 

def pivotDanceability(pivot):
   #danceability
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
              # I subdivide the values of the pivot table by the number in the audio_features_danceability column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.audio_features_danceability == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    audio_features_danceability_pivot = pd.DataFrame(t)
    audio_features_danceability_pivot.columns = l
    audio_features_danceability_pivot.loc[4] = sum(t)
    audio_features_danceability_pivot = audio_features_danceability_pivot.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})
 

    return audio_features_danceability_pivot  

def pivotEnergy(pivot):
   #energy 
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
              # I subdivide the values of the pivot table by the number in the audio_features_energy column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.audio_features_energy == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    audio_features_energy_pivot = pd.DataFrame(t)
    audio_features_energy_pivot.columns = l
    audio_features_energy_pivot.loc[4] = sum(t)
    audio_features_energy_pivot = audio_features_energy_pivot.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})


    return audio_features_energy_pivot  

def pivotInstrumentalness (pivot):
   #instrumentalness
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
              # I subdivide the values of the pivot table by the number in the audio_features_instrumentalness column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.audio_features_instrumentalness == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    audio_features_instrumentalness_pivot = pd.DataFrame(t)
    audio_features_instrumentalness_pivot.columns = l
    audio_features_instrumentalness_pivot.loc[4] = sum(t)
    audio_features_instrumentalness_pivot = audio_features_instrumentalness_pivot.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})

    return audio_features_instrumentalness_pivot

def pivotLanguage(pivot):
    #language
    t = np.zeros((len(pivot.track_language.unique()),6))
    l = []
    c = 0 

    for j in pivot.track_language.unique():
        for i in range(0,6):
          t[c][i] = ((pivot[(pivot.track_language == j) & (pivot.Clusters == i+1)].count()[0]))
        c += 1
    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    track_language_pivot = pd.DataFrame(t[:4])
    track_language_pivot.columns = l
    
    track_language_pivot.loc[len(pivot.track_language.unique())] = sum(t)
 
    track_language_pivot = track_language_pivot.rename(index={0: 'English', 1: 'Spanish', 2: 'Other', 3: 'French', 5: 'Total'})

    return track_language_pivot

def pivotGenre(pivot):
    #genre
    t = np.zeros((len(pivot.track_genre_top.unique()),6))
    l = []
    c = 0 
    for j in pivot.track_genre_top.unique():
        for i in range(0,6):
            t[c][i] = ((pivot[(pivot.track_genre_top == j) & (pivot.Clusters == i+1)].count()[0]))
        c += 1
    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))

    track_genre_top_pivot = pd.DataFrame(t)
    track_genre_top_pivot = track_genre_top_pivot.drop(2)
    track_genre_top_pivot.columns = l
    track_genre_top_pivot.loc[len(pivot.track_genre_top.unique())+1] = sum(t)
    track_genre_top_pivot = track_genre_top_pivot.rename(index={0: 'Hip-Hop', 1: 'Pop', 3:'Rock', 4: 'Experimental', 5: 'Folk', 
                                                                6:'Jazz',7: 'Electronic', 8:'Spoken',9:'International', 10:'Soul-RnB',
                                                               11: 'BluesCountry', 12:'Classical', 13: 'Old-Time / Historic', 
                                                                14:'Instrumental', 15:'Easy', 16: 'Listening', 18: 'Tot'})


    return track_genre_top_pivot

def SpecialPivot(dataTracksFeatures, cluster):
  pivot = pd.DataFrame()
  a = []
  pivot["track_genre_top_x"] = dataTracksFeatures.track_genre_top_x
  pivot.insert(0,"track_duration_x", pd.qcut(dataTracksFeatures[dataTracksFeatures.columns[5]], q = 3,labels=["1", "2", "3"]))
  pivot.insert(0, 'track_bit_rate', pd.qcut(dataTracksFeatures[dataTracksFeatures.columns[559]], q = 3,labels=["1", "2", "3"]))
  pivot["track_language_code_y"] = dataTracksFeatures.track_language_code_y
  pivot["track_location_x"] =  dataTracksFeatures.artist_location_x
  pivot.insert(0, "Clusters", cluster)
  return pivot

def SPecialPivotBitrate(pivot):
   #instrumentalness
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
              # I subdivide the values of the pivot table by the number in the audio_features_instrumentalness column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.track_bit_rate == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    special_bitrate = pd.DataFrame(t)
    special_bitrate.columns = l
    special_bitrate.loc[4] = sum(t)
    special_bitrate = special_bitrate.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})

    return special_bitrate

def SPecialPivotDuration(pivot):
   #instrumentalness
    t = np.zeros((4,6))
    l = []
    for j in range(0,4):
        for i in range(0,6):
              # I subdivide the values of the pivot table by the number in the audio_features_instrumentalness column and by the number of the cluster witch rappresent that value
              t[j][i] = ((pivot[(pivot.track_duration_x == str(j+1)) & (pivot.Clusters == i+1)].count()[0]))

    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    special_duration = pd.DataFrame(t)
    special_duration.columns = l
    special_duration.loc[4] = sum(t)
    special_duration = special_duration.rename(index={0: 'Low',1: 'Medium-Low',2: 'Medium-High', 3: 'High',4: 'Tot'})

    return special_duration

def SpecialPivotGenre(pivot):
    #genre
    t = np.zeros((len(pivot.track_genre_top_x.unique()),6))
    l = []
    c = 0 
    for j in pivot.track_genre_top.unique():
        for i in range(0,6):
            t[c][i] = ((pivot[(pivot.track_genre_top_x	 == j) & (pivot.Clusters == i+1)].count()[0]))
        c += 1
    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))

    track_genre_top_pivot = pd.DataFrame(t)
    track_genre_top_pivot = track_genre_top_pivot.drop(2)
    track_genre_top_pivot.columns = l
    track_genre_top_pivot.loc[len(pivot.track_genre_top.unique())+1] = sum(t)
    track_genre_top_pivot = track_genre_top_pivot.rename(index={0: 'Hip-Hop', 1: 'Pop', 3:'Rock', 4: 'Experimental', 5: 'Folk', 
                                                                6:'Jazz',7: 'Electronic', 8:'Spoken',9:'International', 10:'Soul-RnB',
                                                               11: 'BluesCountry', 12:'Classical', 13: 'Old-Time / Historic', 
                                                                14:'Instrumental', 15:'Easy', 16: 'Listening', 18: 'Tot'})


    return track_genre_top_pivot

def SpecialPivotLanguage(pivot):
    #language
    t = np.zeros((len(pivot.track_language_code_y.unique()),6))
    l = []
    c = 0 

    for j in pivot.track_language_code_y.unique():
        for i in range(0,6):
          t[c][i] = ((pivot[(pivot.track_language_code_y == j) & (pivot.Clusters == i+1)].count()[0]))
        c += 1
    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    track_language_pivot = pd.DataFrame(t[:4])
    track_language_pivot.columns = l
    
    track_language_pivot.loc[len(pivot.track_language_code_y.unique())] = sum(t)
 
    track_language_pivot = track_language_pivot.rename(index={0: 'English', 1: 'Spanish', 2: 'Other', 3: 'French', 5: 'Total'})

    return track_language_pivot

def SpecialPivotLocation(pivot):
    #language
    t = np.zeros((len(pivot.track_location_x.unique()),6))
    l = []
    c = 0 

    for j in pivot.track_location_x.unique():
        for i in range(0,6):
          t[c][i] = ((pivot[(pivot.track_location_x == j) & (pivot.Clusters == i+1)].count()[0]))
        c += 1
    for i in range(t.shape[1]):
        t[:,i] = np.around((t[:,i]/sum(t)[i])*100,2)
    l = []
    for i in range(1,7):
        l.append("Cluster "+str(i))
    location_special = pd.DataFrame(t[:4])
    location_special.columns = l
    
    location_special.loc[len(pivot.track_location_x.unique())] = sum(t)
 
    location_special = location_special.rename(index={0: 'Brooklyn, NY', 1: 'France', 2: 'Other', 3: 'New York, NY', 5: 'Total'})

    return location_special

############### QUESTION 3 ###############

def find_values_equal_s(A, s):
  myPairs = []
  for pair in itertools.combinations(A,2): # make all possibile combination
    if (not pair in myPairs) and (not tuple(reversed(pair)) in myPairs) and (s == pair[0] + pair[1]): #check if (x,y) and (y,x) are not on my list
      myPairs.append(pair)
  if len(myPairs) > 0:
      print(myPairs)
  else:
      print("There isn't any pairs whose sum is equal to", s)