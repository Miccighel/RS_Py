#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
import numpy as np
import math as m
import os
import collections
import csv
import random as rn
from scipy.stats import beta as beta

# ------------------------------
# ----- PARAMETERS & SETUP -----
# ------------------------------

# Parameter setting

dataset_name = "seed_shuffle_1"
papers_number = 300
readers_number = 1000
authors_number = 40
months_number = 1
paper_frequencies = [
    2 * months_number, 
    6 * months_number, 
    8 * months_number, 
    14 * months_number, 
    20 * months_number
]
shuffling = True
shuffle_number = 100

assert (papers_number > (sum(paper_frequencies)) and (papers_number % 10) == 0),     "ERROR: papers_number must be greater than (equal to) {} and it must be a multiple of 10.".format(sum(paper_frequencies)) 

# Seed folder path

dataset_folder_path = "../data/{}/".format(dataset_name)
dataset_shuffle_folder_path = "../data/{}/shuffle/".format(dataset_name)
info_file_path = "{}info.csv".format(dataset_folder_path)
ratings_file_path = "{}ratings.csv".format(dataset_folder_path)
authors_file_path = "{}authors.csv".format(dataset_folder_path)
stats_file_path = "{}stats.csv".format(dataset_folder_path)

# Setting up arrays

papers = np.arange(papers_number)
readers = np.arange(readers_number)
authors = np.arange(authors_number)

os.makedirs(dataset_folder_path, exist_ok=True)

print("DATASET NAME: ", dataset_name)
print("DATASET FOLDER PATH: ", dataset_folder_path)
print("INFO FILE PATH: ", info_file_path)
print("RATINGS FILE PATH: ", ratings_file_path)
print("AUTHORS FILE PATH: ", authors_file_path)


# In[147]:


# ------------------------------
# ---- CORE IMPLEMENTATION -----
# ------------------------------

# Papers distribution generation with beta distribution

print("---------- PAPER DISTRIBUTIONS GENERATION STARTED ----------")

generated_configurations = {"0":{},"1":{},"2":{},"3":{},"4":{}}

beta_distributions_frequencies = [(0, int(round((5*papers_number/100))))]
beta_distributions_frequencies.append((1, int(round(30*papers_number/100))))
beta_distributions_frequencies.append((2, int(round(20*papers_number/100))))
beta_distributions_frequencies.append((3, int(round(30*papers_number/100))))
beta_distributions_frequencies.append((4, int(round(15*papers_number/100))))

papers_set = set(papers)
paper_distributions = [None] * papers_number

generated_papers_distributions = 0
for (index, papers_amount) in beta_distributions_frequencies:
    current_paper_set = rn.sample(papers_set, papers_amount)
    generated_configurations["{}".format(index)]["papers_ids"] = current_paper_set
    generated_configurations["{}".format(index)]["papers_amount"] = papers_amount
    for paper in current_paper_set:
        a = 0
        b = 0
        if index==0:
            # CASE 1: a == b == 1, 5% of papers
            a = 1
            b = 1
        if index==1:
            # CASE 2: a == b > 1, 30% of papers
            a = rn.randint(2, 10)
            b = a
        if index == 2:
            # CASE 3: 0 < (a ^ b) < 1, 30% of papers
            a = rn.uniform(0.001, 1)
            b = rn.uniform(0.001, 1)
        if index == 3:
            # CASE 4: (a V b) == 1, (a > b V b > a), 20% of papers
            a = 1
            b = rn.randint(1, 10)
            if rn.randint(0,1) > 0.5:
                a, b = b, a
        if index == 4:
            # CASE 5: (a ^ b) > 1, (a > b V b > a), 15% of papers
            a = rn.randint(2, 10)
            b = rn.randint(2 + a, 10 + a)
            if rn.randint(0,1) > 0.5:
                a, b = b, a
        percentage = 100*generated_papers_distributions/papers_number
        if percentage % 10 == 0:
            print("{}/{} ({}/100%)".format(int(generated_papers_distributions), papers_number, int(percentage)))
        paper_distributions[paper] = [a, b]
        generated_papers_distributions = generated_papers_distributions + 1
        papers_set.remove(paper)
print("{}/{} (100/100%)".format(papers_number, papers_number))

print(generated_configurations)

print("---------- PAPER DISTRIBUTIONS GENERATION COMPLETED ----------")


# In[148]:


# Ratings file generation

# N sets of readers, each one has X% of the total

readers_percent = 20
reader_sets_number = m.floor(100 / readers_percent)
readers_amount = m.floor((readers_number*readers_percent)/100)

readers_sets = []

# Readers rate papers with a certain frequency

print("---------- READERS SETS GENERATION STARTED ----------")

ratings_number = sum(paper_frequencies) * readers_amount

for x in range(0, reader_sets_number):
    current_readers_set = np.random.choice(readers, readers_amount, False) 
    readers = np.setdiff1d(readers, current_readers_set)
    readers_sets.append(current_readers_set)
    print("SET {}: {}".format(x, current_readers_set))

print("---------- READERS SETS GENERATION COMPLETED ----------")

print("---------- RATINGS GENERATION STARTED ----------")

generated_ratings = 0
rated_papers = []
with open(ratings_file_path, mode='w', newline='') as ratings_file:
    ratings_writer = csv.writer(ratings_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    ratings_writer.writerow(['Timestamp', 'Reader', 'Paper', 'Score'])
    for current_set in range(0, reader_sets_number):
        frequency = paper_frequencies[current_set]
        readers_set = readers_sets[current_set]
        for reader in readers_set:
            sample = np.random.choice(papers, frequency, False)     
            for paper in sample:    
                paper_distribution = beta(paper_distributions[paper][0],paper_distributions[paper][1])
                percentage = 100*generated_ratings/ratings_number
                if percentage % 10 == 0:
                    print("{}/{} ({}/100%)".format(int(generated_ratings), ratings_number, int(percentage)))
                generated_rating = round(paper_distribution.rvs(1)[0], 2)
                if generated_rating == 0:
                    generated_rating = 0.01
                ratings_writer.writerow([
                    generated_ratings, 
                    reader, 
                    paper, 
                    generated_rating
                ])
                rated_papers.append(paper)
                generated_ratings+=1
    
    # Filling gaps
    readers = np.arange(readers_number)
    unrated_papers = set(papers) - set(rated_papers)    
    for paper in unrated_papers:
        for reader in np.random.choice(readers, 5, False):
            paper_distribution = paper_distributions[paper]
            generated_rating = round(paper_distribution.rvs(1)[0], 2)
            if generated_rating == 0:
                generated_rating = 0.01
                ratings_writer.writerow([
                    generated_ratings, 
                    reader, 
                    paper,
                    generated_rating
                ])
                generated_ratings+=1
        
    print("{}/{} (100/100%)".format(ratings_number, ratings_number))
    
ratings_file.close()

paper_ratings = pd.read_csv(ratings_file_path)
paper_ratings = paper_ratings.sample(frac=1)
paper_ratings["Timestamp"] = range(len(paper_ratings))
paper_ratings.reset_index(drop=True, inplace=True)

paper_ratings.to_csv(ratings_file_path, index=False, header=True, sep=",")

print("---------- RATINGS GENERATION ENDED ----------")


# In[149]:


# Authors file generation

print("---------- AUTHORS GENERATION STARTED ----------")

with open(authors_file_path, mode='w', newline='') as authors_file:
    authors_writer = csv.writer(authors_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    authors_writer.writerow(["Author", "Paper"])
    for index, author in enumerate(authors):
        percentage = 100*index/authors_number
        if percentage % 10 == 0:
            print("{}/{} ({}/100%)".format(int(index), authors_number, int(percentage)))
        # An author writes a number of paper between 1 and paper_fraction
        author_papers_number = rn.randint(1, (papers_number-1))
        papers_written = np.random.choice(papers, author_papers_number).tolist()
        papers_written = set(papers_written)
        if len(papers_written) > 1:
            papers_written = map(str, list(papers_written))
            papers_written = ";".join(papers_written)
        else:
            papers_written = list(papers_written)[0]
        authors_writer.writerow([author, papers_written])
    print("{}/{} (100/100%)".format(authors_number, authors_number))
authors_file.close()
        
print("---------- AUTHORS GENERATION ENDED ----------")


# In[150]:


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


# In[151]:


# Stats file generation

print("---------- STATS GENERATION STARTED ----------")

temp_ratings_dataframe = pd.read_csv(ratings_file_path)
temp_ratings_dataframe[temp_ratings_dataframe.columns] = temp_ratings_dataframe[temp_ratings_dataframe.columns].apply(pd.to_numeric)

stats_dataframe = temp_ratings_dataframe.copy()
stats_dataframe[stats_dataframe > 0.0000001] = 1

print("---------- COMPUTING STATS FOR PAPERS ----------")

sums_paper = stats_dataframe.copy().sum(axis=0)
sums_paper_dataframe = pd.DataFrame(sums_paper)

max_ratings_paper = sums_paper_dataframe.max()
min_ratings_paper = sums_paper_dataframe.min()
mean_ratings_paper = sums_paper_dataframe.mean()

temp_ratings_dataframe = temp_ratings_dataframe.T
paper_counter = 0
for index, row in temp_ratings_dataframe.iterrows():
    if len(np.unique(row)) == 1:
        paper_counter+=1
        
print("---------- COMPUTING STATS FOR READERS ----------")

sums_reader = stats_dataframe.copy().sum(axis=1)
counter=collections.Counter(sums_reader)
sums_reader_dataframe = pd.DataFrame(sums_reader)

max_ratings_reader = sums_reader_dataframe.max()
min_ratings_reader = sums_reader_dataframe.min()
mean_ratings_reader = sums_reader_dataframe.mean()

temp_ratings_dataframe = temp_ratings_dataframe
reader_counter = 0
for index, row in temp_ratings_dataframe.iterrows():
    if len(np.unique(row)) == 1:
        reader_counter+=1
        
# Writing stats to file

stats_dataframe = pd.DataFrame(columns=[
    "Dataset",
    "Max Number Rating Paper", 
    "Min Number Rating Paper", 
    "Mean Number Rating Paper",
    "Number Papers Unique Ratings",
    "Max Number Rating Reader", 
    "Min Number Rating Reader", 
    "Mean Number Rating Reader"
    "Number Readers Unique Rating"
])
stats_dataframe = stats_dataframe.append(
    {
        "Dataset": dataset_name, 
        "Max Number Rating Paper": int(max_ratings_paper.values[0]), 
        "Min Number Rating Paper": int(min_ratings_paper.values[0]), 
        "Number Papers Unique Ratings": paper_counter, 
        "Mean Number Rating Paper": int(mean_ratings_paper.values[0]), 
        "Max Number Rating Reader": int(max_ratings_reader.values[0]), 
        "Min Number Rating Reader": int(min_ratings_reader.values[0]), 
        "Mean Number Rating Reader": int(mean_ratings_reader.values[0]), 
        "Number Readers Unique Rating": reader_counter, 
    }, ignore_index=True)
stats_dataframe.to_csv(stats_file_path, index=False)

print("---------- STATS GENERATION COMPLETED ----------")


# In[145]:


# Data generation for experiments

# ------------------------------
# -- EXP 1-A: DATA GENERATION --
# ------------------------------

print("---------- SPECIAL RATINGS STARTED ----------")

gaussian_beta_distributions = generated_configurations["2"]
papers_identifiers = gaussian_beta_distributions["papers_ids"]
for paper in papers_identifiers:
    mean = (paper_distributions[paper][0]/(paper_distributions[paper][0] + paper_distributions[paper][1]))
    SR1_rating_id = generated_ratings
    SR1_reader = "SR#{}".format(readers_number)
    SR1_paper = paper
    SR1_rating_score = mean
    SR2_rating_id = generated_ratings+1
    SR2_reader = "SR#{}".format(readers_number+1)
    SR2_paper = paper
    SR3_rating_id = generated_ratings+2
    SR3_reader = "SR#{}".format(readers_number+2)
    SR3_paper = paper
    if mean <= 0.5:
        SR2_rating_score = 0
        SR3_rating_score = (1-mean)/2
    else:
        SR2_rating_score = 0
        SR3_rating_score = mean/2
    #print([SR1_rating_id, SR1_reader, SR1_paper, SR1_rating_score])
    #print([SR2_rating_id, SR2_reader, SR2_paper, SR2_rating_score])
    #print([SR3_rating_id, SR3_reader, SR3_paper, SR3_rating_score])
    generated_ratings = generated_ratings + 3

print("---------- SPECIAL RATINGS COMPLETED  ----------")


# In[ ]:


# ------------------------------
# -- EXP 1-B: DATA GENERATION --
# ------------------------------

print("---------- RATINGS SHUFFLING STARTED ----------")

if shuffling:
    os.makedirs(dataset_shuffle_folder_path, exist_ok=True)
    for s in range(shuffle_number):
        c = 0
        if s % 10 == 0:
            print("{}/{} ({}/100%)".format(s, shuffle_number, s))
        current_shuffle_file_path = "{}/shuffle_{}.csv".format(dataset_shuffle_folder_path, s)
        shuffled_papers_ratings = paper_ratings.sample(frac=1)
        for i, row in shuffled_papers_ratings.iterrows():
            shuffled_papers_ratings.at[i,'Timestamp'] = c
            c  = c + 1
        shuffled_papers_ratings.to_csv(current_shuffle_file_path, index=False, header=True, sep=",")
    print("{}/{} (100/100%)".format(shuffle_number, shuffle_number))
    
print("---------- RATINGS SHUFFLING COMPLETED ----------")


# In[ ]:




