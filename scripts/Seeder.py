#!/usr/bin/env python
# coding: utf-8

# In[21]:



import pandas as pd
import seaborn as sb
import numpy as np
import math as m
import os
import csv
import random as rn
from matplotlib import pyplot as plt
import scipy as sp
from scipy.stats import truncnorm as tn

# Quantities to seed

papers_number = 10000
readers_number = 2500
authors_number = 25

papers = np.arange(papers_number)
readers = np.arange(readers_number)
authors = np.arange(authors_number)

# Seed folder path

dataset_name = "seed_2/p_4_lot_variance"
dataset_folder_path = "../data/{}/".format(dataset_name)
info_file_path = "{}info.csv".format(dataset_folder_path)
ratings_file_path = "{}ratings.csv".format(dataset_folder_path)
authors_file_path = "{}authors.csv".format(dataset_folder_path)

os.makedirs(dataset_folder_path, exist_ok=True)

print("DATASET NAME: ", dataset_name)
print("DATASET FOLDER PATH: ", dataset_folder_path)
print("INFO FILE PATH: ", info_file_path)
print("RATINGS FILE PATH: ", ratings_file_path)
print("AUTHORS FILE PATH: ", authors_file_path)


# In[22]:



# Papers distribution generation

print("---------- PAPER DISTRIBUTIONS GENERATION STARTED ----------")

paper_distributions = np.empty(papers_number)
for index in range(0, papers_number):
    percentage = 100*index/papers_number
    if percentage % 10 == 0:
        print("{}/{} ({}/100%)").format(int(index), papers_number, int(percentage))
    distribution = tn(0, 1, loc=rn.uniform(0, 1), scale=rn.uniform(0, 0.9)).rvs(1)
    paper_distributions[index] = distribution
print("{}/{} (100/100%)").format(papers_number, papers_number)
    
print("---------- PAPER DISTRIBUTIONS GENERATION COMPLETED ----------")


# In[23]:



# Ratings file generation

# N sets of readers, each one has X% of the total

readers_percent = 20
reader_sets_number = m.floor(100 / readers_percent)
readers_amount = m.floor((readers_number*readers_percent)/100)

readers_set = set(readers)
readers_sets = []

# Readers rate papers with a certain frequence

paper_frequencies = [2, 4, 8, 30, 90]

print("---------- READERS SETS GENERATION STARTED ----------")

ratings_number = sum(paper_frequencies) * readers_amount
for x in range(0, reader_sets_number):
    current_readers_set = rn.sample(readers_set, readers_amount)
    # Removing last index
    if readers_number in current_readers_set: current_readers_set.remove(readers_number)
    readers_sets.append(current_readers_set)
    for reader in current_readers_set:
        readers_set.remove(reader)
    print("SET {}: ", current_readers_set).format(x)
     
print("---------- READERS SETS GENERATION COMPLETED ----------")

print("---------- RATINGS GENERATION STARTED ----------")

generated_ratings = 0
rated_papers = []
rated_readers = []
with open(ratings_file_path, mode='w', newline='') as ratings_file:
    ratings_writer = csv.writer(ratings_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ratings_writer.writerow(['Timestamp', 'Reader', 'Paper', 'Score'])
    for current_set in range(0, reader_sets_number):
        paper_per_reader = paper_frequencies[current_set]
        readers_set = readers_sets[current_set]
        for reader in readers_set:
            for index, paper in enumerate(rn.sample(set(papers), paper_per_reader)):
                paper_distribution = paper_distributions[paper]
                percentage = 100*generated_ratings/ratings_number
                if percentage % 10 == 0:
                    print("{}/{} ({}/100%)").format(int(generated_ratings), ratings_number, int(percentage))
                current_tuple = {
                    "Reader": reader, 
                    "Paper": paper, 
                    "Score": round(paper_distribution, 2), 
                }
                ratings_writer.writerow([generated_ratings, current_tuple["Reader"], current_tuple["Paper"], current_tuple["Score"]])
                generated_ratings+=1
                rated_readers.append(reader)
                rated_papers.append(paper)
    
    # Filling gaps
    unrated_papers = set(papers) - set(rated_papers)    
    for paper in unrated_papers:
        for reader in rn.sample(set(readers), 3): 
            paper_distribution = paper_distributions[paper]
            current_tuple = {
                "Reader": reader, 
                "Paper": paper, 
                "Score": round(paper_distribution, 2), 
            }
            ratings_writer.writerow([generated_ratings, current_tuple["Reader"], current_tuple["Paper"], current_tuple["Score"]])    
            generated_ratings = generated_ratings + 1

    print("{}/{} (100/100%)").format(ratings_number, ratings_number)
    
ratings_file.close()

print("---------- RATINGS GENERATION ENDED ----------")


# In[24]:



# Authors file generation

print("---------- AUTHORS GENERATION STARTED ----------")

with open(authors_file_path, mode='w', newline='') as authors_file:
    authors_writer = csv.writer(authors_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    authors_writer.writerow(["Author", "Paper"])
    for index, author in enumerate(authors):
        percentage = 100*index/authors_number
        if percentage % 10 == 0:
            print("{}/{} ({}/100%)").format(int(index), authors_number, int(percentage))
        # An author writes a number of paper between 1 and paper_fraction
        author_papers_number = rn.randint(1, (papers_number-1))
        papers_written = np.random.choice(papers, author_papers_number).tolist()
        papers_written = set(papers_written)
        if len(papers_written) > 1:
            papers_written = map(str, list(papers_written))
            papers_written = ";".join(papers_written)
        authors_writer.writerow([author, papers_written])
    print("{}/{} (100/100%)").format(authors_number, authors_number)
authors_file.close()
        
print("---------- AUTHORS GENERATION ENDED ----------")


# In[25]:



# Info file generation

print("---------- INFO GENERATION STARTED ----------")

info_dataframe = pd.DataFrame(columns=["Dataset", "Paper", "Reader", "Rating", "Author"])
info_dataframe = info_dataframe.append(
    {
        "Dataset": dataset_name, 
        "Paper": papers_number, 
        "Reader": readers_number, 
        "Rating": ratings_number, 
        "Author": authors_number
    }, ignore_index=True)
info_dataframe.to_csv(info_file_path, index=False)

print("---------- INFO GENERATION ENDED ----------")


# In[ ]:




