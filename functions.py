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
import traceback

path = ''

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

def save_json_utils(peaks_freq, peaks_time, titles_tracks):
    with open(path+'/peaks_freq.json', 'w') as f:
        json.dump(peaks_freq, f)
    with open(path+'/peaks_time.json', 'w') as f:
        json.dump(peaks_time, f)
    with open(path+'/titles_tracks.json', 'w') as f:
        json.dump(titles_tracks, f)

def create_datasets(tracks, HOP_SIZE, DURATION):
    tracks_list = []
    for i in tracks:
        tracks_list.append(str(i))
    titles_tracks  = []
    peaks_time = []
    peaks_freq = []
    for audio in tqdm(tracks_list):
        # find the titles of the songs
        titles_tracks.append(re.search('\d+-(.+?).wav', audio).group(1).replace('_',' '))
        # use the (given) function to get the peacks and their frequencies
        _, _, onset_env, peaks = load_audio_picks(audio, DURATION, HOP_SIZE) 
        peaks_time.append(list(map(int,peaks)))
        f = [ '%.3f' % elem for elem in onset_env[peaks]]
        peaks_freq.append(list(map(float,f)))
    save_json_utils(peaks_time, peaks_freq, titles_tracks)

    return tracks_list

def load_json_list(path, file_output_name):

  ''' This function allows to load and read correctly a .json file which 
  containa a list of lists from a specific path. '''

  f = open(path)
  load = json.load(f)
  for i in load:
    file_output_name.append(i)
  return(file_output_name)

def load_utils():
    peaks_freq = []
    peaks_time = []
    titles_tracks = []

    peaks_freq = load_json_list(path+'/peaks_freq.json', peaks_freq)
    peaks_time = load_json_list(path+'/peaks_time.json', peaks_time)
    titles_tracks = load_json_list(path+'/titles_tracks.json',titles_tracks)

    return peaks_freq, peaks_time, titles_tracks

def create_shingles(peaks_freqencies):

  ''' This function create a list of all unique values of 'peaks frequencies', 
  called shingles. The input peaks_freqencies must be a list of lists. '''

  # define an empty array to append shingles 
  shingles = []
  # collect all the 'peak frequence' that appear in the input array
  for sublist in peaks_freqencies:
      for item in sublist:
        shingles.append(item)
  # drop duplicates
  shingles = list(dict.fromkeys(shingles))
  shingles = np.round(shingles,3)
  return(shingles)

def create_characteristic_matrix(peaks_freqencies):
  
  '''This function creates a matrix C that has all shingles values as rows and 
  all different songs as columns. Its generic value C_ij is equal to one if the j-th
  song has the i-th 'peak frequency' among its 'peaks frequencies' and 0 otherwise.
  The input peaks_freqencies must be a list of lists'''
  
  shingles = create_shingles(peaks_freqencies)

  # set the shape of the output matrix
  n = len(shingles)   # number of rows
  m = len(peaks_freqencies)   # number of columns

  # define a matrix of zeros with the correct shape
  characteristic_matrix =  [ [ 0 for j in range(m) ] for i in range(n) ]
  for i in range(n):
    for j in range(m):
      # set '1' if the j-th song has the i-th 'peak frequency' among its 'peaks frequencies' 
      if shingles[i] in peaks_freqencies[j]:
        characteristic_matrix[i][j] = 1
  return(characteristic_matrix)


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


def signature_matrix(peaks_freqencies):
  
  ''' This function returns the signature matrix taking in input only the peaks' frequencies.
  It calls the function 'create_characteristic_matrix' to compute characteristic matrix'''

  characteristic_matrix = create_characteristic_matrix(peaks_freqencies)

  signature_matrix = []
  # set 100 as the number of purmutation --> number of rows of the signature matrix
  for i in range(100):
    # permute rows of the characteristic_matrix
    perm_mat = np.random.permutation(characteristic_matrix)
    # create the row of the signature matrix
    perm_row = update_signature_matrix(perm_mat)
    # fill the matrix
    signature_matrix.append(perm_row)
  
  return signature_matrix

def save_signature_matrix(signature):
    with open(path+'/signature_matrix.json', 'w') as f:
        json.dump(signature, f)

def load_signature_matrix():
    signature_mat = []
    signature_mat = load_json_list(path+'/signature_matrix.json', signature_mat)
    return signature_mat

def define_buckets(signature_matrix, r = 2):
  ''' This function takes in input a signature matrix of shape (n_permutations, n_songs)
  and returns a dictionary which has as key a 'hash' value and as values all the songs that contain that 'hash'
  Specifically we set (default) r = 2 that means that we are going to split the matrix in 25 
  bands that contain 2 rows each (b = 50, r = 2). '''
  
  # create an empty dictionary
  buckets = dict()
  for col_idx, col in enumerate(np.transpose(signature_matrix)):
    # each bucket has 2 row (r = 2). Thus 
    for i in range(0, len(col), r):
      hash = tuple(col[i:i+2])
      # fill the dictionary
      if hash in buckets:
        buckets[hash].add(str(col_idx))
      else:
        buckets[hash] = {str(col_idx)}
  return(buckets)

def collect_query(query_tracks):
    queries = []
    for id, q in enumerate(query_tracks):
        queries.append(q)
    return queries

def extract_info(queries, DURATION, HOP_SIZE):
    # Get some useful informations from the query
    titles_query  = []
    peaks_time_query = []
    peaks_freq_query = []
    for q in tqdm(queries):
        # find the 'title' of the query
        titles_query.append(str(q).split('/')[-1].split('.')[0])
        
        # use the (given) function to get the peacks and their frequencies
        _, _, onset_env_query, peaks_query = load_audio_picks(q, DURATION, HOP_SIZE) 
            
        peaks_time_query.append(list(map(int,peaks_query)))
        f = [ '%.3f' % elem for elem in onset_env_query[peaks_query]]
        peaks_freq_query.append(list(map(float,f)))
    
    return titles_query, peaks_time_query, peaks_freq_query

def jaccard_similarity(list1, list2):

  ''' This function computes the jaccard similarity between two lists (which are given as parameters)'''
  s1 = set(list1)
  s2 = set(list2)
  return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def fingerprint_hashing(N_TRACKS, mp3_tracks, HOP_SIZE, DURATION, tracks):
    # Start preprocessing & analysis of mp3 audio tracks
    preprocessing_converter(N_TRACKS, mp3_tracks)
    audio_signals(HOP_SIZE, DURATION, tracks)
    # Create our workng data
    tracks_list = create_datasets(tracks, HOP_SIZE, DURATION)
    peaks_freq, peaks_time, titles_tracks = load_utils()
    # Implement minhash (LSH)
    sign_mat = signature_matrix(peaks_freq)
    save_signature_matrix(sign_mat)
    # sign_mat = load_signature_matrix() -> use this if you don't want to create it from scratch
    bucket_dict = define_buckets(sign_mat)

    return bucket_dict, tracks_list, peaks_freq, peaks_time, titles_tracks

def matching(signature_query, bucket_dict, tracks_list, peaks_freq_query, peaks_freq, titles_tracks, titles_query):
    dic = dict()
    r = 2
    all_matches = dict()
    threshold = 0.6

    for q in range(len(np.transpose(signature_query))):
        #for q in range(9):
        buckets = []
        hash_query = np.transpose(signature_query)[q]
        
        # create a list of buckets to which that query has been mapped
        for i in range(0, len(hash_query), r):
            buckets.append(tuple(hash_query[i:i+r]))
        
        # serch the indexes of the songs' tracks that have been mapped in the same buckets of thet query
        comparing_tracks_index = []
        for b in buckets:
            if b in bucket_dict.keys():
                comparing_buckets = bucket_dict[b]
                for track in comparing_buckets:
                    comparing_tracks_index.append(track)
    
    # 
    matching_tracks = []
    scores = []
    for t in range(len(tracks_list)):
        if str(t) in comparing_tracks_index:
            jaccard = jaccard_similarity(peaks_freq_query[q], peaks_freq[t])
            if(jaccard >= 0):
                scores.append(jaccard)
                matching_tracks.append((jaccard, titles_tracks[t]))
    
    # complete our dictionary
    dic[titles_query[q]] = max(matching_tracks)

def handle_q_1():
    N_TRACKS = 1413
    HOP_SIZE = 512
    DURATION = 30 # TODO: to be tuned!
    THRESHOLD = 0 # TODO: to be tuned!
    data_folder = Path(path + "/mp3s-32k/")
    mp3_tracks = data_folder.glob("*/*/*.mp3")
    tracks = data_folder.glob("*/*/*.wav")
    data_folder_query = Path(path + "/query/")
    query_tracks = data_folder_query.glob("*.wav")
    # We start loading our queries
    queries = collect_query(query_tracks)
    titles_query, peaks_time_query, peaks_freq_query = extract_info(queries, DURATION, HOP_SIZE)
    bucket_dict, tracks_list, peaks_freq, peaks_time, titles_tracks = fingerprint_hashing(N_TRACKS, mp3_tracks, HOP_SIZE, DURATION, tracks)
    signature_query = signature_matrix(peaks_freq_query)
    matching(signature_query, bucket_dict, tracks_list, peaks_freq_query, peaks_freq, titles_tracks, titles_query)

############### QUESTION 2 ###############

def load_ds_get_prev():
    echonest = pd.DataFrame(pd.read_csv(path+"/dataset_csv/echonest.csv", sep = ','))
    echonest.head()
    echonest.info()

    features = pd.DataFrame(pd.read_csv(path+"/dataset_csv/features.csv", sep = ','))
    features.head()
    features.info()

    tracks =  pd.DataFrame(pd.read_csv(path + "/dataset_csv/tracks.csv", sep = ','))
    tracks.head()
    tracks.info()

    return echonest, features, tracks

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

def provide_data():
    # Obtain datasets
    echonest, features, tracks = load_ds_get_prev()
    # Remove NaN values
    remove_NaN_values(echonest)
    remove_NaN_values(features)
    remove_NaN_values(tracks)
    # Clean columns
    echonest_cleaned = remove_object_col(echonest)
    features_cleaned = remove_object_col(features)
    tracks_cleaned = remove_object_col(tracks)
    # Show new infos
    echonest_cleaned.info()
    features_cleaned.info()
    tracks_cleaned.info()
    # Finalize our data to start working on
    echonest_final = scaling_features(echonest_cleaned)
    features_final = scaling_features(features_cleaned)
    tracks_final = scaling_features(tracks_cleaned)

    return echonest, features, tracks, echonest_final, features_final, tracks_final

def PCA_echo(echonest, echonest_final):
    # Set number of components equal to the min(n_sample, n_features)
    pca_echonest = PCA(n_components = echonest_final.shape[1]-1)
    # we need to exclude as always the trak_id column (the first one)
    echonest_fit = pca_echonest.fit_transform(echonest_final[echonest_final.columns[1:]])
    # let's choose the number of components
    n_components_echonest = number_of_components(pca_echonest)
    print('We select {} components.'.format(n_components_echonest))
    ratio = round(np.cumsum(pca_echonest.explained_variance_ratio_)[n_components_echonest],3)
    print('The ratio of variance explained by {} components is equal to {}.'.format(n_components_echonest, ratio))
    
    plt.figure(figsize = (25,8))
    plt.subplot(121) 

    plt.plot(pca_echonest.explained_variance_ratio_, lw = 4, color = 'cornflowerblue')
    plt.ylabel('Explained Variance')
    plt.xlabel('Components')

    plt.subplot(122) 
    plt.plot(np.cumsum(pca_echonest.explained_variance_ratio_), lw = 4, color = 'cornflowerblue')
    plt.plot( [n_components_echonest, n_components_echonest],[0.1, np.cumsum(pca_echonest.explained_variance_ratio_)[n_components_echonest] ], 'k--', lw=3, alpha = .4)
    plt.plot( [0, n_components_echonest],[np.cumsum(pca_echonest.explained_variance_ratio_)[n_components_echonest], np.cumsum(pca_echonest.explained_variance_ratio_)[n_components_echonest]  ], 'k--',lw=3, alpha = .4)
    plt.plot(n_components_echonest, np.cumsum(pca_echonest.explained_variance_ratio_)[n_components_echonest], marker="o", markersize=14, markerfacecolor="firebrick")
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('Components')
    plt.show()

    # Get the reduced dataframe of echonest
    echonest_pca = pd.DataFrame(echonest_fit).iloc[: , :n_components_echonest+1]
    echonest_pca.head()

    # Add the track_id column 
    echonest_pca = pd.concat([echonest['track_id'], echonest_pca], axis = 1)
    echonest_pca.head()

    return echonest_pca

def PCA_feat(features, features_final):
    # Set number of components equal to the min(n_sample, n_features)
    pca_features = PCA(features_final.shape[1]-1)
    # we need to exclude as always the trak_id column (the first one)
    features_fit = pca_features.fit_transform(features_final[features_final.columns[1:]])
    # let's choose the number of components
    n_components_features = number_of_components(pca_features)
    print('We select {} components.'.format(n_components_features))
    ratio = round(np.cumsum(pca_features.explained_variance_ratio_)[n_components_features],3)
    print('The ratio of variance explained by {} components is equal to {}.'.format(n_components_features, ratio))
    
    plt.figure(figsize = (25,8))
    plt.subplot(121) 

    plt.plot(pca_features.explained_variance_ratio_, lw = 4, color = 'cornflowerblue')
    plt.ylabel('Explained Variance')
    plt.xlabel('Components')

    plt.subplot(122)
    plt.plot(np.cumsum(pca_features.explained_variance_ratio_), lw = 4 ,color = 'cornflowerblue')
    plt.plot( [n_components_features, n_components_features],[0.1, np.cumsum(pca_features.explained_variance_ratio_)[n_components_features] ], 'k--', lw=3, alpha = .4)
    plt.plot( [0, n_components_features],[np.cumsum(pca_features.explained_variance_ratio_)[n_components_features], np.cumsum(pca_features.explained_variance_ratio_)[n_components_features]  ], 'k--', lw=3 , alpha = .4)
    plt.plot(n_components_features, np.cumsum(pca_features.explained_variance_ratio_)[n_components_features], marker="o", markersize=14, markerfacecolor="firebrick")
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('Components')
    plt.show()

    # Get the reduced dataframe of features
    features_pca = pd.DataFrame(features_fit).iloc[: , :n_components_features+1]
    features_pca.head()

    # Add the track_id column 
    features_pca = pd.concat([features['track_id'], features_pca], axis = 1)
    features_pca.head()

    return features_pca

def PCA_tracks(tracks, tracks_final):
    # Set number of components equal to the min(n_sample, n_features)
    pca_tracks = PCA(n_components=tracks_final.shape[1]-1)
    # we need to exclude as always the trak_id column (the first one)
    tracks_fit = pca_tracks.fit_transform(tracks_final[tracks_final.columns[1:]])
    # let's choose the number of components
    n_components_tracks = number_of_components(pca_tracks)
    print('We select {} components.'.format(n_components_tracks))
    ratio = round(np.cumsum(pca_tracks.explained_variance_ratio_)[n_components_tracks],3)
    print('The ratio of variance explained by {} components is equal to {}.'.format(n_components_tracks, ratio))

    plt.figure(figsize = (25,8))
    plt.subplot(121) 

    plt.plot(pca_tracks.explained_variance_ratio_, lw = 4, color = 'cornflowerblue')
    plt.ylabel('Explained Variance')
    plt.xlabel('Components')

    plt.subplot(122)
    plt.plot(np.cumsum(pca_tracks.explained_variance_ratio_), lw = 4, color = 'cornflowerblue' )
    plt.plot( [n_components_tracks, n_components_tracks],[0.22, np.cumsum(pca_tracks.explained_variance_ratio_)[n_components_tracks] ], 'k--', lw=3, alpha = .4)
    plt.plot( [0, n_components_tracks],[np.cumsum(pca_tracks.explained_variance_ratio_)[n_components_tracks], np.cumsum(pca_tracks.explained_variance_ratio_)[n_components_tracks]  ], 'k--', lw=3, alpha = .4)
    plt.plot(n_components_tracks, np.cumsum(pca_tracks.explained_variance_ratio_)[n_components_tracks],marker="o", markersize=14, markerfacecolor="firebrick")
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('Components')
    plt.show()

    # Get the reduced dataframe of tracks
    tracks_pca = pd.DataFrame(tracks_fit).iloc[: , :n_components_tracks+1]
    tracks_pca.head()

    # Add the track_id column 
    tracks_pca = pd.concat([tracks['track_id'], tracks_pca], axis = 1)
    tracks_pca.head()

    return tracks_pca

def pca_all():
    '''
    In order to get a more complete final dataset, we need to select some meaningful variables from the tracks dataframe.

    We chose the following variables to have a more detailed description of the tracks:

    track_title
    album_title
    artist_name
    track_duration
    track_language_code
    track_genre_top
    track_license
    '''
    echonest, features, tracks, echonest_final, features_final, tracks_final = provide_data()
    selected_variables_df = tracks[['track_id', 'track_title', 'album_title','artist_location', 'artist_name','track_duration', 'track_language_code', 'track_genre_top', 'track_license']]
    selected_variables_df.head()
    echonest_pca = PCA_echo(echonest, echonest_final)
    features_pca = PCA_feat(features, features_final)
    tracks_pca = PCA_tracks(tracks, tracks_final)

    '''
    Now we can merge all together the three reduced dataframe obtained from the PCA and the dataframe with the selected variables.

    The new dataframe contains only 125 features!
    '''
    merge_1 = selected_variables_df.merge(echonest_pca, on = "track_id")
    merge_2 = merge_1.merge(features_pca, on = "track_id")
    merged_df = merge_2.merge(tracks_pca, on = "track_id")

    print(merged_df.shape)
    merged_df.head()
    
    # Save our new dataset
    merged_df.to_csv(path+'/dataset_final.csv')

    return merged_df

def kmeans_scratch(k, ds):
    '''We pass the k number of clusters and a dataset ds, the function will return the k-means from scratch'''
    
    values = np.array(ds).reshape(ds.shape[0], ds.shape[1]) # Creating a values array with the values inside ds
    m = values.shape[1] # Columns
    n = values.shape[0] # Rows
    
    centroids = values[rd.choice(n, size = k, replace=False)] # Generate random the vector of centroids (initialization step)
    
    prior_cr = np.zeros((n,k)) # Array of zeros to store during the i-th iteration the centroids of the (i-1)-th iteration
    
    it = 0 #number of iterations

    # Loop that will stop if the centroids won't change or the iterations will be at maximum equal to n (number of observation)
    while it != 2 or np.array_equal(centroids, prior_cr) == False:
        
        
        prior_cr = centroids # Centroids for the next iteration
        d = np.zeros((n,k)) # Cuclidean distance matrix
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
        
        for k in range(k):
            for j in range(m):
                my_values = []
                for i in cs[k+1]: # Clusters' labels are 1, 2, ..., K
                    # Take the values of the observation belonging to the i-th cluster
                    my_values.append(values[i][j])
                # New centroids for each cluster 
                centroids[k][j] = np.mean(my_values)
        
        it += 1
        
    return clusters, d, cs

def Elbow_scratch(data, K):
  distortions = []
  for k in tqdm(K):
      _, distance_matrix, _ = kmeans_scratch(k, data)
      distortions.append((sum(distance_matrix.min(axis=1)))**2)
  plt.figure(figsize=(16,8))
  plt.xlabel("Numbers of clusters")
  plt.ylabel("Distortions")
  plt.plot(K,distortions, 'bx-')
  plt.show()  

# Create new random reference set
def gap_stat(data, k):
    randomReference = np.random.random_sample(size=data.shape)
    cost_r = []
    gaps = []
    sd_k = [] 
    for i in tqdm(range(2,k)):
        _,distance_matrix_random, clus_random = kmeans_scratch(i, randomReference)
        _,distance_matrix, clus = kmeans_scratch(i, data)
        costo = sum(distance_matrix.min(axis=1)**2)
        costr = sum(distance_matrix_random.min(axis=1)**2)
        cost_r.append(costr)
        gap = np.log(np.mean(cost_r)) - np.log(costo)
        gaps.append(gap)
        sd_k.append(np.std(np.log(np.mean(cost_r))))
    plt.figure(figsize=(16,8))
    plt.plot(list(range(2,k)), gaps, linestyle='-', marker='x', color='blue')
    plt.title("Gap statistics")
    plt.xlabel("K")
    plt.ylabel("Gap Stistics")
    plt.show()
    k_star = []
    for j in range(0, k-3):
      if (gaps[j] >= gaps[j+1] - sd_k[j+1]):
        k_star.append(j)
    return(min(k_star))

def cluster_handle_scratch():
    merged_df = pca_all()
    data = remove_object_col(merged_df)

    clusters, distance_matrix, clus =  kmeans_scratch(12, data)
    print(distance_matrix,clusters)
    len(distance_matrix[0])
    clus.keys()
    Elbow_scratch(data, range(2,15))
    k_star = gap_stat(data, 15)
    print(k_star)

def gap_stat_normal(data, k):
    randomReference = np.random.random_sample(size=data.shape)
    cost_r = []
    gaps = []
    sd_k = [] 
    for i in tqdm(range(2,k)):
        distance_matrix_random =  KMeans(n_clusters=k, init='k-means++').fit_transform(randomReference)
        distance_matrix = KMeans(n_clusters=k, init='k-means++').fit_transform(data)
        costo = sum(distance_matrix.min(axis=1)**2)
        costr = sum(distance_matrix_random.min(axis=1)**2)
        cost_r.append(costr)
        gap = np.log(np.mean(cost_r)) - np.log(costo)
        gaps.append(gap)
        sd_k.append(np.std(np.log(np.mean(cost_r))))
    plt.figure(figsize=(16,8))
    plt.plot(list(range(2,k)), gaps, linestyle='-', marker='x', color='blue')
    plt.title("Gap statistics")
    plt.xlabel("K")
    plt.ylabel("Gap Stistics")
    plt.show()
    k_star = []
    for j in range(0, k-3):
      if (gaps[j] >= gaps[j+1] - sd_k[j+1]):
        k_star.append(j)
    return(min(k_star))

def cluster_handle_lib():
    merged_df = pca_all()
    data = remove_object_col(merged_df)

    kmeans = KMeans(n_clusters=12, init='k-means++').fit(data)
    kmeans.cluster_centers_

    # Instantiate the clustering model and visualizer
    visualizer = KElbowVisualizer(KMeans(n_clusters=12, init='k-means++').fit(data), k=(2,15))

    visualizer.fit(data)        # Fit the data to the visualizer
    visualizer.show()

    gap_stat_normal(data,15)
