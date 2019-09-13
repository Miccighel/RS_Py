#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import math as m
import linecache
from collections import deque
import csv
import zipfile
import threading
import numpy as np
import os
import json
import time
from ReadersourcingToolkit import ReadersourcingToolkit

# Scroll to the bottom for the samples section

def readersourcing(parameters : ReadersourcingToolkit):
    
    # Checking parameters for weird things
    
    if parameters.days_serialization:
        if parameters.days_number % parameters.days_serialization_threshold != 0:
            raise ValueError('days_serialization_threshold must be a divider of days_number')
        if  parameters.days_serialization_threshold < 0 or parameters.days_serialization_threshold > parameters.days_number:
            raise ValueError('this must be correct: 0 < days_serialization_threshold <= days_number')
        if parameters.current_day < 0 or parameters.current_day > parameters.days_number:
            raise ValueError('this must be correct: 0 <= current_day <= days_number')
        if parameters.days_serialization_cleaning:
            if parameters.days_cleaning_threshold < 0 or parameters.days_cleaning_threshold > parameters.days_number:
                raise ValueError('this must be correct: 0 < days_cleaning_threshold <= days_number')
    if parameters.data_shuffled:
         if parameters.current_shuffle < 0:
             raise ValueError('this must be correct: current_shuffle >= 0')
         if parameters.shuffle_amount < 0:
             raise ValueError('this must be correct: shuffle_amount >= 0')

    # Reader score must be set to a very small value otherwise there will be a division by 0
    
    epsilon = 0.000001
    
    # CSV file parsing
    
    info_filename = "{}info.csv".format(parameters.dataset_entries_path)
    ratings_filename = "{}ratings.csv".format(parameters.dataset_entries_path)
    authors_filename = "{}authors.csv".format(parameters.dataset_entries_path)
    
    info = pd.read_csv(info_filename)
    paper_authors = pd.read_csv(authors_filename)
    paper_authors = paper_authors.values
    paper_ratings = pd.read_csv(info_filename)
    paper_ratings = paper_ratings.values
        
    csv_offset = 2
        
    # Initial Readersourcing setup
    
    dataset_name = info["Dataset"][0]
    papers_number = info["Paper"][0]
    readers_number = info["Reader"][0]
    ratings_number = info["Rating"][0]
    authors_number = info["Author"][0]
    papers = np.arange(papers_number)
    readers = np.arange(readers_number)
    ratings = np.arange(ratings_number)
    authors = np.arange(authors_number)
    paper_steadiness = np.zeros(papers_number)
    paper_score = np.zeros(papers_number)
    rating_goodness = np.zeros(ratings_number)
    reader_steadiness = np.zeros(readers_number)
    reader_score = np.zeros(readers_number)
    reader_score.fill(epsilon)
    author_steadiness = np.zeros(authors_number)
    author_score = np.zeros(authors_number)
    
    # Day serialization handling
    
    if parameters.days_serialization:
        ratings_number_per_day = m.floor(int(ratings_number / parameters.days_number))
        computed_days = 1
        written = False
        # cleaned = False
        
    # Data shuffling handling
    
    if parameters.data_shuffled:
        ratings_filename = "{}shuffle_{}.csv".format(parameters.dataset_shuffle_path, parameters.current_shuffle)
    
    # Output handling
    
    result_file_paths = []
    
    # Function to retrieve the authors of a given paper
    
    def get_author(current_paper) :
        
        found_authors = []
        
        for author_index, author_entry in enumerate(paper_authors) :
            current_author = int(author_entry[0])
            written_papers = author_entry[1].split(";")
            written_papers = [int(x) for x in written_papers]
            if current_paper in written_papers :
                found_authors.append(current_author)
                
        return np.asarray(found_authors)
    
    # Function to output result to file
    
    def serialize_result(current_index, verbose, parameters):
                         
        if parameters.data_shuffled:
            result_folder_path = "{}shuffle_{}/".format(parameters.result_shuffle_base_path, parameters.current_shuffle)
        else:
            if parameters.days_serialization:
                result_folder_path = "{}day_{}/".format(parameters.result_days_path, parameters.current_day)
            else:
                result_folder_path = parameters.result_folder_base_path
        
        os.makedirs(result_folder_path, exist_ok=True)
    
        # Quantities output handling
    
        dictionary = [
            {'Quantity': 'Paper Steadiness', 'Identifiers': papers.tolist(), 'Values': paper_steadiness.tolist()},
            {'Quantity': 'Paper Score', 'Identifiers': papers.tolist(), 'Values': paper_score.tolist()},
            {'Quantity': 'Reader Steadiness', 'Identifiers': readers.tolist(), 'Values': reader_steadiness.tolist()},
            {'Quantity': 'Reader Score', 'Identifiers': readers.tolist(), 'Values': reader_score.tolist()},
            {'Quantity': 'Author Steadiness', 'Identifiers': authors.tolist(), 'Values': author_steadiness.tolist()},
            {'Quantity': 'Author Score', 'Identifiers': authors.tolist(), 'Values': author_score.tolist()},
        ]
        
        result_quantities_filename = "{}quantities.json".format(result_folder_path, parameters.current_day)
        
        if verbose:
            print("------------------------------")
            print("PRINTING QUANTITIES TO .JSON FILE AT PATH {}".format(result_quantities_filename))
        
        with open(result_quantities_filename, 'w') as result_quantities_file:  
            json.dump(dictionary, result_quantities_file)
        result_quantities_file.close()
            
        # Rating and goodness matrix output handling
        
        rating_matrix = np.zeros((readers_number, papers_number))
        goodness_matrix = np.zeros((readers_number, papers_number))
                
        for rating_index in range(csv_offset, csv_offset + current_index):
                    
            current_entry = linecache.getline(ratings_filename, rating_index).split(",")
                                
            # Example: <1,1,2,0.8,0>
            # At Timestamp 1 Reader 1 gave to Paper 2 a Rating of 0.8
            current_timestamp = int(current_entry[0])
            current_reader = int(current_entry[1])
            current_paper = int(current_entry[2])
            current_rating = float(current_entry[3])
                
            rating_matrix[current_reader][current_paper] = current_rating
            goodness_matrix[current_reader][current_paper] = rating_goodness[current_timestamp]
        
        result_ratings_filename = "{}ratings.csv".format(result_folder_path, parameters.current_day)
        result_goodness_filename = "{}goodness.csv".format(result_folder_path, parameters.current_day)
        
        if verbose:
            print("PRINTING RATING MATRIX TO .CSV FILE AT PATH {}".format(result_ratings_filename))
                
        paper_ratings_dataframe = pd.read_csv(ratings_filename)
        ratings_matrix = paper_ratings_dataframe.pivot_table(index="Reader", columns="Paper", values="Score")
        ratings_matrix.fillna(0, inplace=True)
        ratings_matrix.to_csv(result_ratings_filename, sep=",", header=False, index=False)
        
        with open(result_ratings_filename, mode='w', newline='') as result_ratings_file:
            ratings_writer = csv.writer(result_ratings_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for rating_entry in rating_matrix:
                ratings_writer.writerow(rating_entry)
        result_ratings_file.close()
            
        if verbose:
            print("PRINTING RATING GOODNESS MATRIX TO .CSV FILE AT PATH {}".format(result_goodness_filename))
        
        with open(result_goodness_filename, mode='w', newline='') as result_goodness_file:
            goodness_writer = csv.writer(result_goodness_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for goodness_entry in goodness_matrix:
                goodness_writer.writerow(goodness_entry)
        result_goodness_file.close()
        
        # Info output handling
        
        result_elapsed_time = time.time() - start_time 
        
        dictionary = [{'Time': result_elapsed_time}]
        
        result_info_filename = "{}info.json".format(result_folder_path, parameters.current_day)
        
        if verbose:
            print("PRINTING INFO TO .JSON FILE AT PATH {}".format(result_info_filename))
            print("------------------------------")
            
        with open(result_info_filename, 'w') as result_info_file:  
            json.dump(dictionary, result_info_file)
        result_info_file.close()
        
        if parameters.result_compression:
            print("--------------------------------------")
            print("---------- COMPRESSING RESULTS ----------")
            archive_filename = "{}{}.zip".format(result_folder_path, parameters.archive_name)
            archive_file = zipfile.ZipFile(archive_filename, 'w')
            for folder, subfolders, files in os.walk(result_folder_path):
                for file in files:
                    if file.endswith('.csv') or file.endswith('.json'):
                        archive_file.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), result_folder_path), compress_type = zipfile.ZIP_DEFLATED)
                        thread = threading.Thread(
                            target=os.remove,
                            args=[os.path.relpath(os.path.join(folder,file))]
                        )
                        thread.daemon = True
                        thread.start()
            archive_file.close()           
            print("RESULT COMPRESSED INTO A .ZIP ARCHIVE AT PATH {}".format(archive_filename))
            print("--------------------------------------")
            
        file_paths = [result_ratings_filename, result_goodness_filename, result_quantities_filename, result_info_filename]
        
        return result_elapsed_time, file_paths
    
    # Function to clean unwanted results
    
    def clean_results(results_to_clean):
        print("--------------------------------------")
        print("---------- CLEANING RESULTS ----------")
        for file_path in results_to_clean:
            if os.path.isfile(file_path):
                print("Deleting file at path: ", file_path)
                thread = threading.Thread(
                    target=os.remove,
                    args=[file_path],
                )
                thread.daemon = True
                thread.start()
        print("--------------------------------------")
    
    # There are many "print" that you can uncomment if you have to do some debugging
    # print("##########")
        
    start_time = time.time()
    
    for index in range(csv_offset, (ratings_number + csv_offset)):
        
        if parameters.days_serialization:
            if index % (ratings_number_per_day * computed_days) == 0:
                parameters.current_day = computed_days
                written = False
                cleaned = False
                if parameters.days_serialization_cleaning and computed_days % parameters.days_cleaning_threshold == 0 and not cleaned:
                     clean_results(result_file_paths)
                     result_file_paths = []
                     cleaned = True
                if computed_days % parameters.days_serialization_threshold == 0 and not written:
                    print("---------- DAY {}/{} ----------".format(parameters.current_day, parameters.days_number))
                    elapsed_time, paths = serialize_result(index, verbose=False, parameters=parameters)
                    result_file_paths = result_file_paths + paths
                    written = True
                computed_days += 1
                
        percentage = 100*index/ratings_number
        if percentage % 5 == 0:
            print("{}/{} ({}/100%)".format(int(index), ratings_number, int(percentage)))
                    
        entry = linecache.getline(ratings_filename, index).split(",")
        
        # Example: <1,1,2,0.8,0>
        # At Timestamp 1 Reader 1 gave to Paper 2 a Rating of 0.8
        timestamp = int(entry[0])
        reader = int(entry[1])
        paper = int(entry[2])
        rating = float(entry[3])
        authors_of_paper = get_author(paper)
        
        # print("---------- CURRENT ENTRY ----------")
        # print(f"TIMESTAMP {timestamp} - READER {reader} - PAPER {paper} - SCORE {rating}")
    
        # COMPUTATION START: PAPER AND READER SCORE
    
        # Saving values at time t(i)
    
        old_paper_steadiness = paper_steadiness[paper]
        old_paper_score = paper_score[paper]
        old_reader_steadiness = reader_steadiness[reader]
        old_rating_goodness = rating_goodness[timestamp]
        old_reader_score = reader_score[reader]
        
        # print("---------- PRINTING VALUES AT TIME T(I) ----------")
        # print("PAPER STEADINESS T(I) ", old_paper_steadiness)
        # print("PAPER SCORE T(I) ", old_paper_score)
        # print("READER STEADINESS T(I) ", old_paper_score)
        # print("RATING GOODNESS T(I) ", rating_goodness[timestamp])
        # print("READER SCORE T(I) ", old_reader_score)
    
        # Updating values at time t(i+1)
    
        paper_steadiness[paper] = old_paper_steadiness + old_reader_score
        paper_score[paper] = ((old_paper_steadiness * old_paper_score) + (old_reader_score * rating)) / paper_steadiness[paper]
        rating_goodness[timestamp] = (1 - (m.sqrt(abs(rating - paper_score[paper]))))
        reader_steadiness[reader] = (old_reader_steadiness + paper_steadiness[paper])
        reader_score[reader] = (((old_reader_steadiness * old_reader_score) + (paper_steadiness[paper] * rating_goodness[timestamp])) / reader_steadiness[reader])
    
        # print("---------- PRINTING VALUES AT TIME T(I+1) ----------")
        # print("PAPER STEADINESS T(I+1) ", paper_steadiness[paper])
        # print("PAPER SCORE T(I+1) ", paper_score[paper])
        # print("READER STEADINESS T(I+1) ", reader_steadiness[reader])
        # print("RATING GOODNESS T(I+1) ", rating_goodness[timestamp])
        # print("READER SCORE T(I+1) ", reader_score[reader])
    
        # COMPUTATION START: AUTHOR SCORE
    
        for author in authors_of_paper :
            # Saving values at time t(i)
    
            old_author_steadiness = author_steadiness[author]
            old_author_score = author_score[author]
    
            # Updating values at time t(i+1)7
    
            author_steadiness[author] = old_author_steadiness + old_reader_score
            author_score[author] = ((old_author_steadiness * old_author_score) + (old_reader_score * rating)) / author_steadiness[author]
    
        # COMPUTATION START: PROPAGATING CHANGES TO PREVIOUS READERS
            
        previous_ratings = []        
        with open(ratings_filename) as rating_file:
            raw_previous_ratings = deque([next(rating_file) for x in range(csv_offset, (index + csv_offset))])
            raw_previous_ratings.popleft()
        rating_file.close()
        for raw_previous_rating in raw_previous_ratings:
            splitted_raw_previous_rating = raw_previous_rating.split(",")
            if len(splitted_raw_previous_rating) > 4:
                previous_rating = splitted_raw_previous_rating[:-1]
            else: 
                previous_rating = splitted_raw_previous_rating
            previous_ratings.append(previous_rating)
        previous_ratings = np.array(previous_ratings, dtype=float)
        previous_ratings = previous_ratings[
             (previous_ratings[:,1]!=float(reader)) &
             (previous_ratings[:,2]==float(paper))
        ]            
                       
        # print(" ----- PREVIOUS PAPER RATINGS -----")
    
        for previous_index, previous_entry in enumerate(previous_ratings):
                        
            # Example: <1,1,2,0.8,0>
            # At Timestamp 1 Reader 1 gave to Paper 2 a Rating of 0.8 written by Author 0
            previous_timestamp = int(previous_entry[0])
            previous_reader = int(previous_entry[1])
            previous_paper = int(previous_entry[2])
            previous_rating = previous_entry[3]
    
            # print(f"PREVIOUS TIMESTAMP {previous_timestamp} - PREVIOUS READER {previous_reader} - PREVIOUS PAPER {previous_paper} - PREVIOUS RATING {previous_rating}")
    
            # Saving previous values at time t(i)
    
            old_previous_reader_steadiness = reader_steadiness[previous_reader]
            old_previous_reader_score = reader_score[previous_reader]
            old_previous_rating = previous_rating
            old_previous_rating_goodness = rating_goodness[previous_timestamp]
    
            # Updating previous values at time t(i+1)
    
            rating_goodness[previous_timestamp] = 1 - (m.sqrt(abs(old_previous_rating - paper_score[paper])))
            reader_steadiness[previous_reader] = (old_previous_reader_steadiness + old_reader_score)
            reader_score[previous_reader] = (
                                                (old_previous_reader_steadiness * old_previous_reader_score) -
                                                (old_paper_steadiness * old_previous_rating_goodness) +
                                                (paper_steadiness[paper] * rating_goodness[previous_timestamp])
                                            ) / reader_steadiness[previous_reader]
               
        # print(" ----- PREVIOUS PAPER RATINGS END -----")
            
        # print("---------- PRINTING FINAL VALUES AT TIME T(I+1) ----------")
        # print("PAPER STEADINESS: ", paper_steadiness)
        # print("PAPER SCORE: ", paper_score)
        # print("READER STEADINESS: ", reader_steadiness)
        # print("READER SCORE: ", reader_score)
        # print("##########")
    
    elapsed_time, result_file_paths = serialize_result(ratings_number, verbose=True, parameters=parameters)
    print("ELAPSED TIME: ", elapsed_time)
    
    # ----- ALGORITHM ENDS HERE ----- #
    
    # Summary
    
    #print("PAPER STEADINESS:  ", paper_steadiness)
    #print("PAPER SCORE:       ", paper_score)
    #print("READER STEADINESS: ", reader_steadiness)
    #print("READER SCORE:      ", reader_score)
    #print("AUTHOR STEADINESS: ", author_steadiness)
    #print("AUTHOR SCORE:      ", author_score)


# In[8]:


# Samples

# ------------------------------
# ---------- SAMPLE 1 ----------
# ------------------------------

seed = ReadersourcingToolkit(
   dataset_name="ground_truth_2", 
   dataset_folder_path="../data/{}/",
)
try:
   readersourcing(seed)
except ValueError as error:
     print(repr(error))


# In[6]:


# ------------------------------
# ---------- SAMPLE 2 ----------
# ------------------------------

seed = ReadersourcingToolkit(
    dataset_name="seed_1/p_1_beta", 
    dataset_folder_path="../data/{}/", 
    days_serialization=True,
    days_number=30,
    days_serialization_threshold=5,
)
   
try:
   readersourcing(seed)
except ValueError as error:
    print(repr(error))


# In[ ]:


# ------------------------------
# ---------- SAMPLE 3 ----------
# ------------------------------

seed = ReadersourcingToolkit(
    dataset_name="seed_1/p_1_beta", 
    dataset_folder_path="../data/{}/", 
)
try:
   readersourcing(seed)
except ValueError as error:
    print(repr(error))


# In[ ]:



# ------------------------------
# ---------- SAMPLE 4 ----------
# ------------------------------

seed = ReadersourcingToolkit(
    dataset_name="seed", 
    dataset_folder_path="../data/{}/", 
    data_shuffled=True, 
    current_shuffle = 0,
    shuffle_amount=100
)
 
try:
   for index_shuffle in range(seed.shuffle_amount):
       print("---------------------------------")
       print("----------- SHUFFLE {} -----------".format(index_shuffle))
       seed.current_shuffle = index_shuffle
       readersourcing(seed)
except ValueError as error:
   print(repr(error))
   


# In[5]:


# ------------------------------
# ---------- SAMPLE 5 ----------
# ------------------------------

seed = ReadersourcingToolkit(
     dataset_name="seed_shuffle_1_special", 
     dataset_folder_path="../data/{}/", 
     data_shuffled=True, 
     current_shuffle = 0,
     shuffle_amount=100
)
  
try:
    for index_shuffle in range(seed.shuffle_amount):
        print("---------------------------------")
        print("----------- SHUFFLE {} -----------".format(index_shuffle))
        seed.current_shuffle = index_shuffle
        readersourcing(seed)
except ValueError as error:
     print(repr(error))


# In[ ]:


# ------------------------------
# ---------- SAMPLE 6 ----------
# ------------------------------

seed = ReadersourcingToolkit(
    dataset_name="seed_power_law_1", 
    dataset_folder_path="../data/{}/", 
)
try:
   readersourcing(seed)
except ValueError as error:
    print(repr(error))


# In[ ]:




