#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import numpy as np
import math as m
import os
import csv
import random as rn
from random import randint
from scipy.stats import beta


papers_number = 500
readers_number = 250
authors_number = 25

papers = np.arange(papers_number)
readers = np.arange(readers_number)
authors = np.arange(authors_number)

# Seed folder path

dataset_name = "seed_2/small"
dataset_folder_path = f"../data/{dataset_name}/"
info_file_path = f"{dataset_folder_path}info.csv"
ratings_file_path = f"{dataset_folder_path}ratings.csv"
authors_file_path = f"{dataset_folder_path}authors.csv"

os.makedirs(dataset_folder_path, exist_ok=True)

print("DATASET NAME: ", dataset_name)
print("DATASET FOLDER PATH: ", dataset_folder_path)
print("INFO FILE PATH: ", info_file_path)
print("RATINGS FILE PATH: ", ratings_file_path)
print("AUTHORS FILE PATH: ", authors_file_path)

# Papers distribution generation with beta distribution

print("---------- PAPER DISTRIBUTIONS GENERATION STARTED ----------")

# CASE 1: a == b == 1, 5% of papers
beta_distributions_frequencies = [(m.floor((5*papers_number)/100), (1, 1))]
# CASE 2: a == b > 1, 30% of papers
a = randint(2, 10)
b = a
beta_distributions_frequencies.append((m.floor((30*papers_number)/100), (a, b)))
# CASE 3: 0 < (a ^ b) < 1, 30% of papers
a = rn.uniform(0.001, 1)
b = rn.uniform(0.001, 1)
beta_distributions_frequencies.append((m.floor((20*papers_number)/100), (a, b)))
# CASE 4: (a V b) == 1, (a > b V b > a), 20% of papers
a = 1
b = randint(1, 10)
if rn.randint(0,1) > 0.5:
    a, b = b, a
beta_distributions_frequencies.append((m.floor((30*papers_number)/100), (a, b)))
# CASE 5: (a ^ b) > 1, (a > b V b > a), 15% of papers
a = randint(2, 10)
b = randint(2 + a, 10 + a)
if rn.randint(0,1) > 0.5:
    a, b = b, a
beta_distributions_frequencies.append((m.floor((15*papers_number)/100), (a, b)))

papers_set = set(papers)
paper_distributions = [None] * papers_number

generated_papers_distributions = 0
for (papers_amount, (a, b)) in beta_distributions_frequencies:
    current_paper_set = rn.sample(papers_set, papers_amount)
    for paper in current_paper_set:
        percentage = 100*generated_papers_distributions/papers_number
        if percentage % 10 == 0:
            print(f"{int(generated_papers_distributions)}/{papers_number} ({int(percentage)}/100%)")
        paper_distributions[paper] = beta(a=a, b=b)
        generated_papers_distributions = generated_papers_distributions + 1
        papers_set.remove(paper)
print(f"{papers_number}/{papers_number} (100/100%)")

print("---------- PAPER DISTRIBUTIONS GENERATION COMPLETED ----------")


# In[3]:



# Ratings file generation

# N sets of readers, each one has X% of the total

readers_percent = 20
reader_sets_number = m.floor(100 / readers_percent)
readers_amount = m.floor((readers_number*readers_percent)/100)

readers_sets = []

# Readers rate papers with a certain frequence

paper_frequencies = [2, 4, 8, 30, 90]

print("---------- READERS SETS GENERATION STARTED ----------")

ratings_number = sum(paper_frequencies) * readers_amount

for x in range(0, reader_sets_number):
    current_readers_set = np.random.choice(readers, readers_amount, False) 
    readers = np.setdiff1d(readers, current_readers_set)
    readers_sets.append(current_readers_set)
    print(f"SET {x}: ", current_readers_set)

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
                paper_distribution = paper_distributions[paper]
                percentage = 100*generated_ratings/ratings_number
                if percentage % 10 == 0:
                    print(f"{int(generated_ratings)}/{ratings_number} ({int(percentage)}/100%)")
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
        
    print(f"{ratings_number}/{ratings_number} (100/100%)")
    
ratings_file.close()

paper_ratings = pd.read_csv(ratings_file_path)
paper_ratings = paper_ratings.sample(frac=1)
paper_ratings["Timestamp"] = range(len(paper_ratings))
paper_ratings.reset_index(drop=True, inplace=True)

paper_ratings.to_csv(ratings_file_path, index=False, header=True, sep=",")

print("---------- RATINGS GENERATION ENDED ----------")


# In[4]:



# Authors file generation

print("---------- AUTHORS GENERATION STARTED ----------")

with open(authors_file_path, mode='w', newline='') as authors_file:
    authors_writer = csv.writer(authors_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    authors_writer.writerow(["Author", "Paper"])
    for index, author in enumerate(authors):
        percentage = 100*index/authors_number
        if percentage % 10 == 0:
            print(f"{int(index)}/{authors_number} ({int(percentage)}/100%)")
        # An author writes a number of paper between 1 and paper_fraction
        author_papers_number = rn.randint(1, (papers_number-1))
        papers_written = np.random.choice(papers, author_papers_number).tolist()
        papers_written = set(papers_written)
        if len(papers_written) > 1:
            papers_written = map(str, list(papers_written))
            papers_written = ";".join(papers_written)
        authors_writer.writerow([author, papers_written])
    print(f"{authors_number}/{authors_number} (100/100%)")
authors_file.close()
        
print("---------- AUTHORS GENERATION ENDED ----------")


# In[5]:



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


# In[1]:




