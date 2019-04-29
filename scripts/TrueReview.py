#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import seaborn as sb
import numpy as np
import time
import csv
import math as m
from matplotlib import pyplot as plt
import os
import json

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# CSV file parsing

dataset_name = "ground_truth_1"
dataset_folder_path = f"../data/{dataset_name}/"
info_filename = f"{dataset_folder_path}info.csv"
ratings_filename = f"{dataset_folder_path}ratings.csv"
authors_filename = f"{dataset_folder_path}authors.csv"

info = pd.read_csv(info_filename)
paper_authors = pd.read_csv(authors_filename)
paper_authors = paper_authors.values
paper_ratings = pd.read_csv(ratings_filename)
paper_ratings = paper_ratings.values

# Initial setup

dataset_name = info["Dataset"][0] 
papers_number = info["Paper"][0]
readers_number = info["Reader"][0] 
ratings_number = info["Rating"][0]
papers = np.arange(papers_number)
readers = np.arange(readers_number)
ratings = np.arange(ratings_number)
paper_score = np.zeros(papers_number)
rating_informativeness = np.zeros(ratings_number)
rating_accuracy_loss = np.zeros(ratings_number)
rating_bonus = np.zeros(ratings_number)
reader_bonus = np.zeros(readers_number)
reader_score = np.zeros(readers_number)


def quadratic_loss(a, b):
    return m.pow((a - b), 2)


def logistic_function(value):
    return 1 / 1 + (m.exp((-1 * (value - 0.5))))


start_time = time.time()

for current_paper in papers:

    print(f"---------- CURRENT PAPER: {current_paper} ----------")

    current_paper_ratings = []
    ratings_sum = 0
    
    # For each paper, consider only its ratings and throw away the other ones
    
    for index, entry in enumerate(paper_ratings):
    
        # Example: <1,1,2,0.8>
        # At Timestamp 1 Reader 1 gave to Paper 2 a Rating of 0.8
        timestamp = int(entry[0])
        reader = int(entry[1])
        paper = int(entry[2])
        rating = entry[3]
        
        if paper == current_paper:
            current_paper_ratings.append(entry)

    # For each rating of the paper under consideration, compute the required quantities
     
    for index, entry in enumerate(current_paper_ratings):

        # Example: <1,1,2,0.8>
        # At Timestamp 1 Reader 1 gave to Paper 2 a Rating of 0.8
        timestamp = int(entry[0])
        reader = int(entry[1])
        paper = int(entry[2])
        rating = entry[3]
        
        percentage = 100*index/ratings_number
        if percentage % 10 == 0:
            print(f"{int(index)}/{ratings_number} ({int(percentage)}/100%)")
        # print("---------- CURRENT ENTRY ----------")
        # print(f"TIMESTAMP {timestamp} - READER {reader} - PAPER {paper} - SCORE {rating}")
    
        # 0 < i < n
    
        if 0 < index < len(current_paper_ratings)-1:
            
            # COMPUTATION START: QI_PAST
            
            past_ratings = current_paper_ratings[:index] 
            past_scores = []
            for past_index, past_entry in enumerate(past_ratings):
                past_rating = past_entry[3]
                past_scores.append(past_rating)
                
            # COMPUTATION START: QI_FUTURE
                
            future_ratings = current_paper_ratings[(index+1):]
            future_scores = []
            for future_index, future_entry in enumerate(future_ratings):
                future_rating = future_entry[3]
                future_scores.append(future_rating)
            
            qi_past_ratings = sum(past_scores) / len(past_scores)
            qi_future_ratings = sum(future_scores) / len(future_scores)
                                
            # COMPUTATION START: INFORMATIVENESS and ACCURACY LOSS
            
            rating_informativeness[timestamp] = quadratic_loss(qi_past_ratings, qi_future_ratings)
            rating_accuracy_loss[timestamp] = quadratic_loss(rating, qi_future_ratings)
            
            # COMPUTATION START: RATING BONUS
            
            rating_bonus[timestamp] = rating_informativeness[timestamp] * logistic_function(rating_accuracy_loss[timestamp])
            
            # COMPUTATION START: READER BONUS - it is the sum of the bonus computed for each of its ratings
            
            reader_bonus[reader] = reader_bonus[reader] + rating_bonus[timestamp]
            
        # Sum the current rating to compute the mean at the end
            
        ratings_sum = ratings_sum + rating
            
    # COMPUTATION START: PAPER SCORE - scores can be aggregated with an index of your choice
    
    paper_score[current_paper] = ratings_sum / len(current_paper_ratings)
    
    print(f"{int(ratings_number)}/{ratings_number} (100/100%)")
    
elapsed_time = time.time() - start_time    
print("--------------------")
print("ELAPSED TIME: ", elapsed_time)
    


# In[12]:


# Summary

print("RATING INFORMATIVENESS: ", rating_informativeness)
print("RATING ACCURACY LOSS:   ", rating_accuracy_loss)
print("RATING BONUS:           ", rating_bonus)
print("READER BONUS:           ", reader_bonus)
print("PAPER  SCORE:           ", paper_score)

result_folder_path = f"../models/{dataset_name}/"

# Quantities output handling

columns = ['Quantity', 'Identifiers','Values']
dictionary = [
    {'Quantity': 'Rating Informativeness', 'Identifiers': ratings.tolist(), 'Values': rating_informativeness.tolist()},
    {'Quantity': 'Rating Accuracy Loss', 'Identifiers': ratings.tolist(), 'Values': rating_accuracy_loss.tolist()},
    {'Quantity': 'Rating Bonus', 'Identifiers': ratings.tolist(), 'Values': rating_bonus.tolist()},
    {'Quantity': 'Reader Bonus', 'Identifiers': readers.tolist(), 'Values': reader_bonus.tolist()},
    {'Quantity': 'Paper Score', 'Identifiers': papers.tolist(), 'Values': paper_score.tolist()},
]

quantities_filename = f"{result_folder_path}truereview/quantities.json"
os.makedirs(f"{result_folder_path}truereview/", exist_ok=True)

print(f"PRINTING QUANTITIES TO .JSON FILE AT PATH {quantities_filename}")

with open(quantities_filename, 'w') as outfile:  
    json.dump(dictionary, outfile)
    
# Rating matrix output handling

rating_matrix = np.zeros((readers_number, papers_number))

for index, entry in enumerate(paper_ratings) :

    # Example: <1,1,2,0.8,0>
    # At Timestamp 1 Reader 1 gave to Paper 2 a Rating of 0.8
    timestamp = int(entry[0])
    reader = int(entry[1])
    paper = int(entry[2])
    rating = entry[3]
    
    rating_matrix[reader][paper] = rating
    
ratings_filename = f"{result_folder_path}truereview/ratings.csv"
os.makedirs(f"{result_folder_path}truereview/", exist_ok=True)

print(f"PRINTING RATING MATRIX TO .CSV FILE AT PATH {ratings_filename}")

with open(ratings_filename, mode='w', newline='') as rating_file:
    rating_writer = csv.writer(rating_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for rating_entry in rating_matrix:
        rating_writer.writerow(rating_entry)
rating_file.close()

# Info output handling

elapsed_time = time.time() - start_time 
    
info_filename = "{}truereview/info.json".format(result_folder_path)
os.makedirs("{}truereview/".format(result_folder_path), exist_ok=True)
    
dictionary = [{'Time': elapsed_time}]
    
print("PRINTING INFO TO .JSON FILE AT PATH {}".format(info_filename))
        
with open(info_filename, 'w') as outfile:  
    json.dump(dictionary, outfile)
outfile.close()


# In[ ]:




