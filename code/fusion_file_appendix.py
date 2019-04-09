# -*- encoding: utf-8 -*-
#---------------------------------------------AUTHOR---------------------------------------------------------------------------------------
# Written by: Maximilian Hartmann
# Date: 15.10.2018
# Email: hartmanm@student.ethz.ch
# Cause: Master Thesis
# ENTIRE PROJECT CODE IS ALSO AVAILABLE ON GITHUB
# GitHub: https://github.com/Bellador
#-------------------------------------INFORMATION TO SCRIP USAGE----------------------------------------------------------------------------
#This scrip was not primarily intendet to be reused by other users. Therefore the legibility, comments and code adaptations are limited.
#This means that path names, credentials, file names etc. have to be changed to ones own machine.
# If questions arise please feel free to contact me under the above given E-Mail address.
#-------------------------------------------------------------------------------------------------------------------------------------------
#-------------API's----------------------------------------
import credentialsInstaAPI as credAPI
from configparser import ConfigParser
import GeocodingAPIcredentials as credGeocode
from google.cloud import vision
from google.cloud.vision import types
from google.cloud import storage
from gcloud import pubsub
#-----------Natural-Language--------------------------------
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.cistem import Cistem
import nltk
#---------Language-detection-and-spell-checking ------------
import enchant #needs 32bit python to work!!!
#-----------Machine-Learning--------------------------------
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import export_graphviz
import mglearn
#---------------Data-Management-----------------------------
import pandas as pd
#----------statistics-and-plotting---------------------------
import matplotlib.pyplot as plt
import numpy as np
#---------------URL-Requests--------------------------------
import bs4 as bs
import urllib.request
import requests
#----------postgreSQL----------------------------------------
import psycopg2
import psycopg2.errorcodes
from psycopg2 import sql
#----------miscellaneous-------------------------------------
from collections import Counter
from collections import defaultdict
# import graphviz
import datetime
import joblib
import csv
import re
import json
import time
import os.path
import ast
###########--------------------------------Notice-------------------------------------------------###################
#Create out of the original, single py.files:
# 1. instagramAPI.py --> only adds location data to the netlytic instagram media objects
# 2. postgreSQL_setup.py
# 3. FLICKR HAS TO BE INCLUDED AS WELL
#
# This fusion file does not include the Flickr or Instagram media object acquisition
###########---------------------------------------------------------------------------------------###################

'''
should coordinate all other classes to achieve a clean execution
'''

class OperationHandler:

    def __init__(self):
        self.operation()
    '''
    Asks the user for raw input what operation to execute next
    '''
    def operation(self):
        print("Welcome to postgreSQL DB interface")
        print("Creating PostgresHandler")
        postgresql_inst = PostgresqlHandler()
        while True:
            print("#########################################################################################################")
            print("1 - modify table                                                                                        #")
            print("2 - Merge Netlytic / Instagram csv files to one                                                         #")
            print("3 - DataAddOn to Instagram CSV-File (location info)                                                     #")
            print("4 - DataAddOn to Flickr TXT-File (location info)                                                        #")
            print("5 - add Instagram or Flickr data to existing table (includes in boundary check, data & text processing) #")
            print("6 - add all Foursquare files from a directory to existing table                                         #")
            print("7 - add Google Vision labels to all tables                                                              #")
            print("8 - drop table(s)                                                                                       #")
            print("9 - Merge BIG training/test dataset over all 10 locations                                               #")
            print("10 - Train ML model on media_objects_trainingdata_instagram and save model                              #")
            print("11 - Predict research area media objects accroding to ML model                                          #")
            print("12 - Compare with and without vision label classification                                               #")
            print("0 - EXIT                                                                                                #")
            print("#########################################################################################################")
            choice = int(input())
            if not isinstance(int(choice), int):
                print("Invalid input")
                continue
            if choice == 1:
                print('Which of the following tables needs modification?')
                print(PostgresqlHandler.selectable_table_names_gen())
                table_to_modify = input()

            elif choice == 2:
                print('Please provide a path to a file which contains the filepaths of all csv files to merge, each in a seperate row')
                filepaths_file = input()
                DataProcessHandler.csv_merger(filepaths_file.strip('"').strip("'"))

            elif choice == 3:
                print(OperationHandler.print_error_format("Make sure you created a merged CSV file over all subregions (see STEP 2)!"))
                input_filepath = input("Input instagram csv-filePATH to which location data should be appended\n")
                print("Output file will be in the same directory and with the '_locationAddOn' extension.")
                LocAddOnInst = LocDataAddOnHandler(input_filepath.strip('"').strip("'"), choice)

            elif choice == 4:
                input_filepath = input("Input flickr txt-filePATH to which location data should be appended\n")
                print("Output file will be in the same directory and with the '_locationAddOn' extension.")
                LocAddOnInst = LocDataAddOnHandler(input_filepath.strip('"'), choice)

            elif choice == 5:
                while True:
                    OperationHandler.print_error_format("Make sure the location data has been previously added")
                    print('Add data to which of the media_objects tables?')
                    print("Remark: It is only possible to add data to media_objects tables, corresponding location data will be added simultaneously")
                    table_dict = PostgresqlHandler.selectable_table_names_gen()
                    print(json.dumps(table_dict, indent=2))
                    response = input()
                    try:
                        response = int(response)
                        break
                    except Exception:
                        print("Invalid input")
                table_to_modify = table_dict[str(response)]
                print('-'*30)
                file = input("name of source file \n")
                transmit = {'operation': 5, 'file': file.strip('"').strip("'"), 'table': table_to_modify}
                postgresql_inst.db_operation(transmit)

            #add all Foursquare files from a directory to existing table
            elif choice == 6:
                while True:
                    print("Adding data to the fusion db table: 'media_objects_cantonzug_foursquare'")
                    response = input("Please input the directory-PATH to the Foursquare inputfiles:\n").strip('"')
                    if os.path.isdir(response):
                        break
                    else:
                        print("Invalid input")
                        continue
                path = response
                print('-'*30)
                transmit = {'operation': 6, 'table': 'media_objects_cantonzug_foursquare', 'directory': path}
                postgresql_inst.db_operation(transmit)

            #add vision labels to all tables
            elif choice == 7:
                transmit = {'operation': 7}
                postgresql_inst.db_operation(transmit)

            #drop table
            elif choice == 8:
                reassure = input("Do you really want to drop a db table? (Y/N)")
                if reassure == 'Y':
                    all_single = input("Drop all or a single table (all/single)?\n")
                    if all_single == "all":
                        reassure2 = input("Last warning: Do you really want to drop all tables? (Y/N)")
                        if reassure2 == 'Y':
                            transmit = {'operation': 7, 'table': "all"}
                            postgresql_inst.db_operation(transmit)
                    elif all_single == "single":
                        table_to_drop = input("Which table ( + CASCADING) should be dropped?\n")
                        reassure3 = input("Last warning: Do you really want to drop {}? (Y/N)".format(table_to_drop))
                        if reassure3 == 'Y':
                            transmit = {'operation': 7, 'table': table_to_drop}
                            postgresql_inst.db_operation(transmit)
                        else:
                            continue
                else:
                    continue

            elif choice == 9:
                DataProcessHandler.csv_merger_traindata()

            #train ML model
            elif choice == 10:
                postgresql_inst.db_operation(command_dic={'operation': 10})

            #predict with ML model
            elif choice == 11:
                postgresql_inst.db_operation(command_dic={'operation': 11})

            # compare with and without vision label classification
            elif choice == 12:
                postgresql_inst.db_operation(command_dic={'operation': 12})

            elif choice == 0:
                print("Exiting.")
                break

            another_operation = input("Another operation? (Y/N)")
            if str(another_operation) == 'Y':
                continue
            elif str(another_operation) == 'N':
                print("Bye")
                break

    @staticmethod
    def print_error_format(msg):
        if str(msg) == "":
            msg = "Empty error string!"
        print("____" * 20)
        print("!!" * 30)
        print(str(msg).strip(r'\n'))
        print("!!" * 30)
        print("____" * 20)


class MachineLearningHandler:
    dump_path = "INSERT_PATH_HERE"
    dump_filename = "INSERT_PATH_HERE"
    logfile_path = "INSERT_PATH_HERE"
    crossval_log_path = "INSERT_PATH_HERE"
    d_tree_output_path = "INSERT_PATH_HERE"

    @staticmethod
    def train_model(connection):
        columns = ['text', 'label']
        best_none_class_objects = 0
        best_parameters_output = []
        best_feature_count = 0
        best_accuracy_train_score = 0
        best_accuracy_test_score = 0
        best_class_acc_list_train = 0
        best_class_acc_list_test = 0
        best_cross_val_score = 0
        best_precision_score_weighted = 0
        best_precision_score_all = 0
        best_recall_score_weighted = 0
        best_recall_score_all = 0
        best_f1_score_weighted = 0
        best_f1_score_all = 0
        best_test_f1_score_no_none = 0
        best_confusion_matrix = 0

        #write CSV-file header and clear log from earlier session
        with open(MachineLearningHandler.logfile_path, 'wt', encoding='utf-8') as log_f:
            log_f.write("""index, time, none_objects, precision_score_weighted, recall_score_weighted, f1_score_weighted, 
            precision_score_all, recall_score_all, f1_score_all, f1_score_no_none, train_accuracy_score, test_accuracy_score, 
            class_acc_list_train, class_acc_list_test, class_acc_mean_no_none_train, class_acc_mean_no_none_test, 
            best_crossval_score, best_params, feature_count\n""")

        # tune over multiple amounts of media objects of the 'None' class
        for iterator, none_class_objects in enumerate([700]):
            t = datetime.datetime.now()
            dump_filename = "linearSCV_model_" + str(t.strftime('%d_%m_%Y'))
            print("___________________________________________________________________________")
            print("Fetching trainingsdata from the db...")
            with connection.cursor() as cursor:
                # 1. get data from the db table media_objects_trainingdata_instagram where model_data is TRUE
                cursor.execute("""
                               (SELECT processed_text, vision_labels, classification
                               FROM media_objects_trainingdata_instagram
                               WHERE is_model_data IS true
                                    AND processed_text != ''
                                    AND classification != 'None'
                                    AND classification != 'Borderline'
                                )
                                UNION ALL
                               (SELECT processed_text, vision_labels, classification
                                FROM media_objects_trainingdata_instagram
                                WHERE is_model_data IS true
                                    AND processed_text != ''
                                    AND classification = 'None'
                                    AND vision_labels IS NOT NULL
                                    AND detected_language != 'None'
                                ORDER  BY random()
                                limit %(amount)s
                                );                                
                               """, {'amount': none_class_objects})

                '''
                borderline cases were excluded on purpose to achieve a better generalization of the model
                '''

                content = cursor.fetchall()
            connection.commit()

            #2. read into pandas dataframe
            df = pd.DataFrame(columns=columns)
            print("Populating dataframe...")
            for index, element in enumerate(content):
                if index % 100 == 0:
                    print("Processed {} of {}".format(index, len(content)))
                '''
                Merge processed_text and vision_labels
                '''
                error_count = 0
                processed_text = element[0]
                v_labels = element[1]
                classification = element[2]
                if v_labels:
                    # 1. create list of vision labels, extracted from the list of tuples with their accuracy score
                    try:
                        if v_labels == 'NO_DATA':
                            vision_labels = []
                        else:
                            vision_labels = [label[0] for label in ast.literal_eval(v_labels)]
                    except Exception as e:
                        error_count += 1
                        vision_labels = []
                    # 2. text process the vision labels
                    vision_labels_processed = [DataProcessHandler.text_processing_core(label,
                                                                                       DataProcessHandler.read_topography_names(),
                                                                                       SnowballStemmer("english"),
                                                                                       stopwords.words("english"),
                                                                                       enchant.Dict("en_GB")
                                                                                       ) for label in vision_labels]

                    # 3. create a fusion text string out of the processed text and vision labels
                    labels_merge = [' '.join(l_element) for l_element in vision_labels_processed]
                    text = str(processed_text) + " " + str(' '.join(labels_merge))

                # if the vision labels are excluded, then the model prediction is soley based on the user generated text
                elif not v_labels:
                    text = str(processed_text)

                df.loc[index, 'text'] = text
                df.loc[index, 'label'] = classification.replace("Borderline", "None")


            '''
            Model 1 best parameters with 3500 non-class media objects!
            '''
            best_parameters_m1 = {
                    'vect__sublinear_tf': [True],
                    'vect__min_df': [12],
                    'vect__ngram_range': [(1, 1)],
                    'chi__k': ['all'],
                    'clf__C': [1],
                    'clf__penalty': ['l2'],
                    'clf__max_iter': [1000000]
                }

            '''
            Model 2 best parameters with 700 non-class media objects!
            '''
            best_parameters_m2 = {
                'vect__sublinear_tf': [True],
                'vect__min_df': [7],
                'vect__ngram_range': [(1, 1)],
                'chi__k': ['all'],
                'clf__C': [1],
                'clf__penalty': ['l2'],
                'clf__max_iter': [1000000]
            }

            ngram_C = {
                'vect__sublinear_tf': [True],
                'vect__min_df': [7],
                'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
                'chi__k': ['all'],
                'clf__C': [0.01, 0.1, 1, 10, 100],
                'clf__penalty': ['l2'],
                'clf__max_iter': [1000000]
            }

            linear_svc_parameters = {
                    'vect__min_df': [7, 8, 9, 10, 11, 12, 13],
                    'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
                    'chi__k': ['all'],
                    'clf__C': [0.001, 0.1, 1, 10, 100]
                }

            randomforest_parameters = {
                    'vect__min_df': [9, 10, 11, 12, 13, 14],
                    'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
                    'chi__k': ['all'],
                    'clf__max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, None]
                }

            X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'])

            pipeline = Pipeline([
                ('vect', TfidfVectorizer(stop_words=None, sublinear_tf=True)),
                ('chi', SelectKBest(chi2)),
                ('clf', LinearSVC(random_state=42, max_iter=1000000, penalty='l2'))
            ])
            # cv parameter stands for the amount of performed cross-validations
            print("Tuning model...")
            grid = GridSearchCV(pipeline, best_parameters_m2, n_jobs=-1, cv=10, scoring='f1_weighted')
            # #used for tuning and validating the model
            # model = grid.fit(X_train, y_train)

            #for final fit -> trained on the entire trainings-set!
            model = grid.fit(df['text'], df['label'])

            vectorizer = model.best_estimator_.named_steps["vect"]
            print("###" * 30)

            print("Amount of features: {}".format(len(vectorizer.get_feature_names())))
            print("First 10 features: {}".format(vectorizer.get_feature_names()[:10]))
            print("Last 10 features: {}".format(vectorizer.get_feature_names()[-10:]))
            print("random features: {}".format(vectorizer.get_feature_names()[::100]))
            cv_results = pd.DataFrame(grid.cv_results_)

            accuracy_train_score = model.score(X_train, y_train)
            accuracy_test_score = model.score(X_test, y_test)

            print("_______________________________________________________________________________________")
            print("Currently used amount of objects from the 'None' class: {}".format(none_class_objects))
            print("Best cross-validation score: {}".format(model.best_score_))
            print("Model mean train accuracy: {}".format(accuracy_train_score))
            print("Model mean test accuracy: {}".format(accuracy_test_score))
            print("Best parameters of the grid: {}".format(model.best_params_))
            print("_______________________________________________________________________________________")
            # confusion-matrix for evaluating classification specific accuracies on TRAIN
            pred_values_train = model.predict(X_train)
            labels_cm = ['None', 'walking', 'hiking', 'jogging', 'biking', 'picnic', 'dog_walking', 'horse_riding']
            confusion_train = confusion_matrix(y_train, pred_values_train, labels=labels_cm)
            print("TRAIN Confusion matrix:\n{}".format(confusion_train))
            # Now the normalize the diagonal entries
            confusion_train = confusion_train.astype('float') / confusion_train.sum(axis=1)[:, np.newaxis]
            # The diagonal entries are the accuracies of each class
            print("Diagonal entries show accuracy for corresponding class:\n{}\n{}".format(labels_cm, confusion_train.diagonal()))
            class_acc_mean_no_none_train = sum(confusion_train.diagonal()[1:]) / float(len(confusion_train.diagonal()[1:]))
            class_acc_list_train = confusion_train.diagonal()
            print("mean of all class accuracies except None: {}"
                  .format(class_acc_mean_no_none_train))
            print("_______________________________________________________________________________________")
            # confusion-matrix for evaluating classification specific accuracies on TEST
            pred_values_test = model.predict(X_test)
            c = Counter(pred_values_test)
            print("predicted values on X_test: {}".format(json.dumps(c, indent=2)))
            confusion_test_original = confusion_matrix(y_test, pred_values_test, labels=labels_cm)
            print("TEST Confusion matrix:\n{}".format(confusion_test_original))
            # Now the normalize the diagonal entries
            confusion_test = confusion_test_original.astype('float') / confusion_test_original.sum(axis=1)[:, np.newaxis]
            # The diagonal entries are the accuracies of each class
            print("Diagonal entries show accuracy for corresponding class:\n{}\n{}".format(labels_cm, confusion_test.diagonal()))
            class_acc_mean_no_none_test = sum(confusion_test.diagonal()[1:]) / float(len(confusion_test.diagonal()[1:]))
            class_acc_list_test = confusion_test.diagonal()
            print("mean of all class accuracies except None: {}".format(class_acc_mean_no_none_test))
            print("_______________________________________________________________________________________")
            #some other important model metrices:
            precision_score_weighted = precision_score(y_test, pred_values_test, average='weighted')
            precision_score_all = precision_score(y_test, pred_values_test, average=None)
            print("Precision score: {}".format(precision_score_weighted))
            recall_score_weighted = recall_score(y_test, pred_values_test, average='weighted')
            recall_score_all = recall_score(y_test, pred_values_test, average=None)
            print("Recall score: {}".format(recall_score_weighted))
            f1_score_weighted = f1_score(y_test, pred_values_test, average='weighted')
            f1_score_all = f1_score(y_test, pred_values_test, average=None)
            print("f1 score: {}".format(f1_score_weighted))

            f1_score_no_none = sum(f1_score_all[1:]) / float(len(f1_score_all[1:]))
            print("mean of all class f1 scores except None: {}".format(f1_score_no_none))

            print("###"*30)

            if f1_score_no_none > best_test_f1_score_no_none:
                best_test_f1_score_no_none = f1_score_no_none
                best_none_class_objects = none_class_objects
                best_class_acc_list_test = confusion_test.diagonal()
                best_parameters_output = model.best_params_
                best_cross_val_score = model.best_score_
                best_feature_count = len(vectorizer.get_feature_names())
                best_precision_score_weighted = precision_score_weighted
                best_precision_score_all = precision_score_all
                best_recall_score_weighted = recall_score_weighted
                best_recall_score_all = recall_score_all
                best_f1_score_weighted = f1_score_weighted
                best_f1_score_all = f1_score_all
                best_accuracy_train_score = accuracy_train_score
                best_accuracy_test_score = accuracy_test_score
                best_confusion_matrix = confusion_test_original

            class_acc_list_train = np.array2string(class_acc_list_train)
            class_acc_list_test = np.array2string(class_acc_list_test)

            with open(MachineLearningHandler.logfile_path, 'at', encoding='utf-8', newline='') as log_f:
                result_list = [(iterator+1),
                        str(time.strftime('%H %M %S - %d %m %Y')),
                        none_class_objects,
                        precision_score_weighted,
                        recall_score_weighted,
                        f1_score_weighted,
                        precision_score_all,
                        recall_score_all,
                        f1_score_all,
                        f1_score_no_none,
                        accuracy_train_score,
                        accuracy_test_score,
                        class_acc_list_train,
                        class_acc_list_test,
                        class_acc_mean_no_none_train,
                        class_acc_mean_no_none_test,
                        model.best_score_,
                        model.best_params_,
                        len(vectorizer.get_feature_names())]
                writer = csv.writer(log_f, delimiter=',')
                writer.writerow(result_list)

            '''
            write to cross-validation log file
            '''
            cv_results.to_csv(MachineLearningHandler.crossval_log_path, header=True, index=True, sep=',', mode='a')

        with open(MachineLearningHandler.logfile_path, 'at', encoding='utf-8') as log_f:
            result_list = ['FINAL',
                str(time.strftime('%H %M %S - %d %m %Y')),
                best_none_class_objects,
                best_precision_score_weighted,
                best_recall_score_weighted,
                best_f1_score_weighted,
                best_precision_score_all,
                best_recall_score_all,
                best_f1_score_all,
                best_test_f1_score_no_none,
                best_accuracy_train_score,
                best_accuracy_test_score,
                '-',
                '-',
                best_class_acc_list_train,
                best_class_acc_list_test,
                best_cross_val_score,
                best_parameters_output,
                best_feature_count]
            writer = csv.writer(log_f, delimiter=',')
            writer.writerow(result_list)
            for row in best_confusion_matrix:
                writer.writerow(row)

        save = input("Do you want to save the model? (Y/N)\n")
        if save == 'Y':
            full_dump_path = os.path.join(MachineLearningHandler.dump_path, MachineLearningHandler.dump_filename)
            joblib.dump(model, full_dump_path)

        label_sequence = ['None', 'Biking', 'Dog Walking', 'Hiking', 'Horse riding', 'Jogging', 'Picnic', 'Walking']
        print('creating data visualization graphs...')

        # the second parameter of reshape has to correspond to the different n_gram settings
        scores = grid.cv_results_['mean_test_score'].reshape(-1, len(ngram_C['vect__ngram_range'])).T
        # visualize heat map
        heatmap = mglearn.tools.heatmap(scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
                                        xticklabels=ngram_C['clf__C'],
                                        yticklabels=ngram_C['vect__ngram_range'])
        plt.colorbar(heatmap)
        plt.show()

        while True:
            to_predict = input("Predict label of following text:\n")
            print("Model predicts: {}".format(model.predict([str(to_predict)])))
            answer = input("Do you want to predict something else? (Y/N)")
            if answer == 'N':
                break
            else:
                continue

    @staticmethod
    def predict_media_objects(connection, v_labels):
        #keep track of the found classes with following dict:
        counter_dict = {
            'walking': 0,
            'hiking': 0,
            'jogging': 0,
            'dog_walking': 0,
            'horse_riding': 0,
            'biking': 0,
            'picnic': 0,
            'None': 0
        }
        error_count = 0
        #create result dictionary that links ids to their predicted classification
        prediction_dict = {}
        #import model
        model_path = input("Enter path to desired model:\n").strip('"')
        model = joblib.load(model_path)
        #enter table name
        target = input("Which table should be accessed by the model?\nPlease insert the corresponding target table name nr.:\n {}\n"
                       .format(json.dumps(PostgresqlHandler.selectable_table_names_gen(), indent=1)))
        target_table = str(PostgresqlHandler.selectable_table_names_gen()['{}'.format(target)])
        #request media objects to predict from db
        if v_labels:
            print("Classification is done considering google vision image labels")
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("""SELECT media_object_id, processed_text, vision_labels
                                   FROM {}
                                   WHERE included_after_dataprocessing IS TRUE
                                        AND in_boundary IS TRUE
                                        AND classification_m2_w_labels IS NULL
                                """).format(sql.Identifier(target_table)))
                content = cursor.fetchall()
        elif not v_labels:
            print("Classification is done WITHOUT considering google vision image labels!")
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("""SELECT media_object_id, processed_text
                                   FROM {}
                                   WHERE included_after_dataprocessing IS TRUE
                                        AND in_boundary IS TRUE
                                        AND classification_m2_without_labels IS NULL
                                """).format(sql.Identifier(target_table)))
                content = cursor.fetchall()
        #doing the actual classification prediction with the model based on the processed text content and the stemmed vision_labels
        for indexer, element in enumerate(content, 1):
            md_id = element[0]
            processed_text = element[1]

            #check if vision labels should be included for the model classification prediction of the media objects
            if v_labels:
                #1. create list of vision labels, extracted from the list of tuples with their accuracy score
                try:
                    if element[2] == 'NO_DATA':
                        vision_labels = []
                    else:
                        vision_labels = [label[0] for label in ast.literal_eval(element[2])]
                except Exception as e:
                    print("Error occurred: {}".format(e))
                    error_count += 1
                    vision_labels = []
                #2. text process the vision labels
                vision_labels_processed = [DataProcessHandler.text_processing_core(label,
                                                                                   DataProcessHandler.read_topography_names(),
                                                                                   SnowballStemmer("english"),
                                                                                   stopwords.words("english"),
                                                                                   enchant.Dict("en_GB")
                                                                                   ) for label in vision_labels]

                #3. create a fusion text string out of the processed text and vision labels
                labels_merge = [' '.join(l_element) for l_element in vision_labels_processed]
                to_predict = str(processed_text) + " " + str(' '.join(labels_merge))

            #if the vision labels are excluded, then the model prediction is soley based on the user generated text
            elif not v_labels:
                to_predict = str(processed_text)

            #4. let model make a prediction
            if to_predict == '':
                prediction = ['None']
            else:
                prediction = model.predict([to_predict])
            #5. Adjust counter_dict
            counter_dict[prediction[0]] += 1
            #6. assign the prediction to the corresponding media object ID
            prediction_dict[md_id] = prediction[0]
            if indexer % 100 == 0:
                print("_"*25 + str(datetime.datetime.now()) + "_"*25)
                print("Processed {} of {}".format(indexer, len(content)))
                print("Current classification status:\n{}\n".format(json.dumps(counter_dict, indent=1)))
                print("Errors encountered: {}".format(error_count))

        #update db according to the prediction_dict entries
        if v_labels:
            for md_id in prediction_dict:
                with connection.cursor() as cursor:
                    cursor.execute(sql.SQL("""UPDATE {}
                                    SET classification_m2_w_labels = %(class)s
                                    WHERE media_object_id = %(id)s""").format(sql.Identifier(target_table)),
                                                                        {'class': prediction_dict[md_id],
                                                                        'id': md_id})
        elif not v_labels:
            for md_id in prediction_dict:
                with connection.cursor() as cursor:
                    cursor.execute(sql.SQL("""UPDATE {}
                                    SET classification_m2_without_labels = %(class)s
                                    WHERE media_object_id = %(id)s""").format(sql.Identifier(target_table)),
                                                                        {'class': prediction_dict[md_id],
                                                                        'id': md_id})
        print("Successfully updated {} classifications of the table: {}".format(len(prediction_dict.keys()), target_table))
        print("Result:\n{}\n".format(json.dumps(counter_dict, indent=1)))
        print("Total Errors encountered: {}".format(error_count))

    @staticmethod
    def compare_w_without_labels_classification(connection):
        match_counter = 0
        mismatch_counter = 0


        class_match = {
            'walking': 0,
            'hiking': 0,
            'jogging': 0,
            'dog_walking': 0,
            'horse_riding': 0,
            'biking': 0,
            'picnic': 0,
            'None': 0
        }
        class_occurance = {
            'walking': 0,
            'hiking': 0,
            'jogging': 0,
            'dog_walking': 0,
            'horse_riding': 0,
            'biking': 0,
            'picnic': 0,
            'None': 0
        }
        # enter table name
        target = input(
            "Which table should be accessed by the model?\nPlease insert the corresponding target table name nr.:\n {}\n"
            .format(json.dumps(PostgresqlHandler.selectable_table_names_gen(), indent=1)))
        target_table = str(PostgresqlHandler.selectable_table_names_gen()['{}'.format(target)])
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("""SELECT classification_w_labels, classification_without_labels
                               FROM {}
                               WHERE included_after_dataprocessing IS TRUE
                                    AND in_boundary IS TRUE""").format(sql.Identifier(target_table)))
            content = cursor.fetchall()

        for element in content:
            if element[0] == element[1]:
                match_counter += 1
                class_match[element[0]] += 2

            else:
                mismatch_counter += 1
            class_occurance[element[0]] += 1
            class_occurance[element[1]] += 1
        print("_"*60)
        print("Identical classifications: {}\n Mismatches: {}\n Ratio: {}"\
              .format(match_counter, mismatch_counter, (match_counter/(match_counter+mismatch_counter))))
        print("*"*30)
        print("Class specific accordance - amount of occurances that appeared in pairs compared to the total class appearance:")
        for item in class_match:
            print("{}: {}".format(item, (class_match[item]/class_occurance[item])*100))

        print("_" * 60)

'''
Order of the Data Add on to the original CSV has to follow a specific order.
So that the new columns can just be added to the end to achieve consistency 
over all input files
Oder: 
1. adding of location infos - columns: location_id, location name, latitude, longitude
2. adding of Google Vision labels - colums: lables (takes a list of tuples with the label and its corresponding
certainty value
'''
class LocDataAddOnHandler:
    insta_access_token = credAPI.access_token
    insta_client_id = credAPI.client_id
    insta_base_url = "https://api.instagram.com/v1/"

    def __init__(self, input_filepath, choice):
        self.in_filepath = input_filepath
        if choice == 3:
            self.instagram_location_info_to_file()
        elif choice == 4:
            self.flickr_location_info_to_file()

    '''
    Important question to remember:
    For the instagram data - the instagram locations throught the API were chosen over the Geocoding API of Google
    which was/will be used for the flickr media objects coordinates.
    The reason for this inconsistency is the fact, that the geo position of the instagram data does not fully correspond
    to the given lat/lng but rather to a user-gererated, instagram specific location that the user chose as reference
    while creating the post. That is why two different location data sources were chosen.
    '''
    def instagram_location_info_to_file(self):
        enpoint_deactivated = True
        # 1. extract coordinates from netlytic exported CSV. file
        self.out_filepath = self.in_filepath.strip('"').strip(".csv") + "_locationAddOn" + ".csv"
        coordinates = []
        coord_info_dic = {}

        with open(self.in_filepath, "rt", encoding='utf-8') as f, open(self.out_filepath, "wt", encoding='utf-8') as new_file:
            content = csv.reader(f, delimiter=',')
            print("-" * 40)
            print("Getting instagram API location data for the netlytic coordinates pair")
            for linecounter, line in enumerate(content, 1):
                if linecounter == 1:
                    continue
                else:
                    #grabs the coordinate string
                    coordinates.append(line[2])
            print("-" * 40)

            # 2. create a (unique) set out of the coordinates list
            print("All coordinates: {}".format(len(coordinates)))
            unique_coordinates = set(coordinates)
            print("Unique coordinates: {}".format(len(unique_coordinates)))
            search_match_counter = 0
            # 3. make API requests for every element in unique_coordinates --> create dicitonary where coordinate pair has a name, etc.
            for coord_counter, coord in enumerate(unique_coordinates):
                lat, lng = coord.split(",")

                if enpoint_deactivated:
                    coord_info_dic['{location_id}'.format(location_id=coord_counter)] = {
                        'name': 'API_endpoint_expired',
                        'latitude': lat,
                        'longitude': lng}

                else:
                    url_request = "{base}locations/search?lat={latitude}&lng={longitude}&distance={dist}&access_token={token}"\
                        .format(base=self.insta_base_url, latitude=lat, longitude=lng, dist=1, token=self.insta_access_token)
                    data = requests.get(url_request).json()

                    #check if the API returned more then one unique location for the given coordinates
                    try:
                        len_request_dict = len(data['data'])
                        for index, result in enumerate(data['data'], 1):
                            result_lat = result['latitude']
                            result_lng = result['longitude']
                            print("-" * 40)
                            if np.isclose(float(result_lat), float(lat), rtol=1e-08, atol=1e-06, equal_nan=False) and \
                                    np.isclose(float(result_lng), float(lng), rtol=1e-06, atol=1e-06, equal_nan=False):
                                print("Unique coord {} of {} - Found correct location from request return.".format((coord_counter+1), len(unique_coordinates)))
                                coord_info_dic['{location_id}'.format(location_id=result['id'])] = {
                                    'name': result['name'],
                                    'latitude': result['latitude'],
                                    'longitude': result['longitude']}
                                search_match_counter += 1
                                break
                            else:
                                if index >= len_request_dict:
                                    coord_info_dic['{location_id}'.format(location_id=coord_counter)] = {
                                        'name': 'NoMatch',
                                        'latitude': lat,
                                        'longitude': lng}
                                print("Unique coord {} of {} - NoMatch found from request return.".format(
                                    (coord_counter + 1), len(unique_coordinates)))
                            print("-" * 40)
                            #check if input coordinates correspond with the once given from the API return

                    except KeyError:
                        print("Error: {}".format(data['meta']['error_type']))
                        print("Error Message: {}".format(data['meta']['error_message']))
                        if data['meta']['code'] == 400 and data['meta']['error_type'] == 'APINotAllowedError':
                            coord_info_dic['{location_id}'.format(location_id=coord_counter)] = {
                                'name': 'API_endpoint_expired',
                                'latitude': lat,
                                'longitude': lng}
                        continue
                    print("FINISH STEP 1 - Search request match Counter: {} of {}".format(search_match_counter, len_request_dict))


            '''
            Here: (a) either I create a seperate csv file with the location data - and leave the netlytic csv file how it is without appending anything.
            Then the location data would be added to the database via matching the long,lat tuple with the seperate loaciton file -> extracting the location-ID.
            (b) Or I append the location data directly into the netlytic csv-file, altering it (could also copy it but what ever).

            decided to append to a copy of the original netlytic csv. output file the location data from the Instagram API:
            create a copy of the original file for later altering by adding additional location data
            needed because it already iterated through the entire file once, so the pointer is at the very end.
            '''

            f.seek(0)
            match_counter = 0
            for line_counter, line_ in enumerate(content, 1):
                if line_counter == 1:
                    match_counter += 1
                    print("Line before pop: " + str(line_))
                    line_.pop()
                    print("Line after pop: " + str(line_))
                    line_.append('location_id')
                    line_.append('location_name')
                    line_.append('latitude')
                    line_.append('longitude')
                    for counter_, element in enumerate(line_, 1):
                        if counter_ < len(line_):
                            new_file.write(element)
                            new_file.write(',')
                        else:
                            new_file.write(element)
                    new_file.write('\n')
                else:
                    '''
                    process of finding the correct location_ID for a given coordinate pair from the coord_info_dic [Dictionary]
                    '''
                    coord_string = line_[2]
                    add_on = []
                    lat, lng = coord_string.split(",")
                    len_coord_info_dic = len(coord_info_dic.keys())
                    for index2, element in enumerate(coord_info_dic, 1):
                        # you CANNOT refer to the sub dictionary by iterating over it, you always have to give the entire 'path' with the mother dictionary
                        if np.isclose(float(coord_info_dic[element]["latitude"]), float(lat), rtol=1e-08, atol=1e-06, equal_nan=False) and \
                                np.isclose(float(coord_info_dic[element]["longitude"]), float(lng), rtol=1e-06, atol=1e-06, equal_nan=False):
                            print("line {} - location MATCH".format(line_counter))
                            match_counter += 1

                            add_on.append(element)
                            add_on.append(coord_info_dic[element]["name"])
                            add_on.append(coord_info_dic[element]["latitude"])
                            add_on.append(coord_info_dic[element]["longitude"])

                            new_line = line_ + add_on
                            for counter_, element2 in enumerate(new_line, 1):
                                '''
                                removes the trailing empty elements of the list, but not if they are in the middle 
                                (could occur that know data exists for certain columns - they would otherwise be removed too.
                                '''
                                if counter_ >= 12 and element2 == "":
                                    continue
                                elif counter_ < len(new_line):
                                    new_file.write('"' + str(element2) + '"')
                                    new_file.write(',')
                                else:
                                    new_file.write('"' + str(element2) + '"')
                            new_file.write('\n')
                            break
                        else:
                            if index2 >= len_coord_info_dic:
                                continue
            print("search matches {}, line matches {}".format(search_match_counter, match_counter))
        print("Done.")

    '''
    Currently copied from flickr_location_info_to_file() - removed after adaption complete
    '''
    def flickr_location_info_to_file(self):
        #https://maps.googleapis.com/maps/api/geocode/json?latlng=40.714224,-73.961452&key=YOUR_API_KEY
        self.out_filepath = self.in_filepath.strip(".txt") + "_locationAddOn" + ".txt"
        google_api_key = credGeocode.google_api_key
        '''
        #########################################################################
        the location_type is evidence of the accuracy of the geocoding api return.
        Everything besides 'ROOFTOP' return an address that is close 
        --> therefore an approximation!
        #########################################################################
        '''

        '''
        1. extract coordinates from netlytic exported CSV. file
        '''
        coordinates = []
        unique_coordinates_indexed = []
        '''
        contains lat/lng tuple related to the media object id
        '''
        coord_info_dic = {}
        with open(self.in_filepath, "rt", encoding='utf-8') as f, open(self.out_filepath, "wt", encoding='utf-8') as new_file:
            content = csv.reader(f, delimiter=',')
            print("-" * 60)
            print("Getting Google Geocoding API location data for the netlytic coordinates pair")
            for linecounter, line in enumerate(content, 1):
                if linecounter == 1:
                    continue
                else:
                    #grabs lat / lng
                    coordinates.append((line[1],line[2]))
                    print("Appended coord: {}, {}".format(line[1],line[2]))
            print("-" * 60)
            print("len of coordinates list: {}".format(len(coordinates)))
            '''
            2. create a (unique) set out of the coordinates list
            '''
            unique_coordinates = set(coordinates)
            '''
            3. make API requests for every element in unique_coordinates 
            --> create dictionary where coordinate pair has a unique index used as reference later on
            (because the location ID of the Geocoding API cannot be used, because multiple coordinate pairs,
            can be potentially be snapped or matched to the same address and therefore the location ID cannot be used as
            a unique identifier or key in a dictionary!
            '''
            for index, element in enumerate(unique_coordinates ,1):
                unique_coordinates_indexed.append((element[0],element[1],index))

            # google API key=136bfab2eeb2a79b83e9ba758d10c2d57bfd8cc2 add to external file!
            len_unique_coordinates = len(unique_coordinates)
            for coord_counter, coord_tuple in enumerate(unique_coordinates_indexed, 1):
                try_counter = 0
                while True:
                    if try_counter > 3:
                        OperationHandler.print_error_format("Too many flickr location retries - sleeping 5s")
                        time.sleep(5)
                        break
                    try:
                        try_counter += 1
                        print("Geocoding API request {} of {}".format(coord_counter, len_unique_coordinates))
                        lat = coord_tuple[0]
                        lng = coord_tuple[1]
                        location_id = coord_tuple[2]
                        url_request = "https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={key}"\
                            .format(lat=lat, lng=lng, key=google_api_key)
                        data = requests.get(url_request).json()
                    except Exception:
                        time.sleep(3)
                        OperationHandler.print_error_format("Flickr location error - sleeping 3s")
                    '''
                    Excepting most of all IndexErrors 
                    '''
                    try:
                        name = data['results'][0]['formatted_address'].replace(",", "")
                    except Exception:
                        name = ''
                        print("'name' not found in results...")
                    try:
                        location_type = data['results'][0]['geometry']['location_type'].replace(",", "")
                    except Exception:
                        location_type = ''
                        print("'location_type' not found in results...")

                    coord_info_dic['{location_id}'.format(location_id=location_id)] = {
                        'name': name,
                        'location_type': location_type,
                        'lat': lat,
                        'lng': lng}
                    break

            print(json.dumps(coord_info_dic, indent=3))
            f.seek(0)
            match_counter = 0
            for line_counter, line_ in enumerate(content, 1):
                if line_counter == 1:
                    line_.append('location_id')
                    line_.append('location_name')
                    line_.append('location_type')
                    for counter_, element in enumerate(line_, 1):
                        if counter_ < len(line_):
                            new_file.write(element)
                            new_file.write(',')
                        else:
                            new_file.write(element)
                    new_file.write('\n')
                elif linecounter != 1:
                    '''
                    process of finding the correct location_ID for a given coordinate pair 
                    from the coord_info_dic [Dictionary]
                    '''
                    latitude = line_[1]
                    longitude = line_[2]
                    add_on = []

                    len_dict = len(coord_info_dic)
                    for index, element in enumerate(coord_info_dic.keys(), 1):
                        # you CANNOT refer to the sub dictionary by iterating over it,
                        # you always have to give the entire 'path' with the mother dictionary
                        if np.isclose(float(coord_info_dic[element]['lat']), float(latitude), rtol=1e-06, atol=1e-06,
                                      equal_nan=False) and np.isclose(float(coord_info_dic[element]['lng']), float(longitude),
                                                                      rtol=1e-06, atol=1e-06, equal_nan=False):

                            '''
                            element (the dictionary key) represents the location_id
                            lat, lng do not have to get added again, since they are already in the CSV file
                            '''
                            add_on.append(element)
                            add_on.append(coord_info_dic[element]['name'])
                            add_on.append(coord_info_dic[element]['location_type'])

                            new_line = line_ + add_on
                            for counter_, element2 in enumerate(new_line, 1):
                                '''
                                removes the trailing empty elements of the list, but not if they are in the middle 
                                (could occur that know data exists for certain columns 
                                --> they would otherwise be removed too.
                                '''
                                if re.match(r'(\s{3,})', element2):
                                    print("WHITESPACE JUNK DETECTED")
                                    element2 = ''
                                if counter_ >= 15 and element2 == "":
                                    continue
                                elif counter_ < len(new_line):
                                    new_file.write(str(element2).strip())
                                    new_file.write(',')
                                else:
                                    new_file.write(str(element2).strip())
                            new_file.write('\n')
                            match_counter += 1
                            break
                        else:
                            if index < len_dict:
                                continue
                            else:
                                print("no match")
                        print("Handled {} of {} dictionary keys/entries".format(index, len_dict))
            print("Match_counter: {}".format(match_counter))

        print("Done")

class StatisticHandler:

    @staticmethod
    def posts_per_author_graph(counter_object, target_table):
        while True:
            while True:
                top_x = input("Include top X authors in graph:")
                try:
                    top_x = int(top_x)
                    break
                except Exception:
                    print("Invalid input")
            authors = []
            count = []
            x_label = list(range(1,(top_x+1)))

            for element in counter_object.most_common(top_x):
                #affected posts
                count.append(element[1])
            authors.sort(reverse=True)
            count.sort(reverse=True)
            plt.bar(x_label, count)
            xtick_labels = [1]
            for ten in range((int(round(top_x, -1)/10))):
                xtick_labels.append(((ten+1)*10))

            plt.xticks(xtick_labels)
            axes = plt.gca()
            axes.set_xlim([1, top_x])
            plt.xlabel("top {} authors".format(top_x))
            plt.ylabel("effected media objects")
            plt.title("{}".format(target_table))
            plt.show()
            while True:
                response = input("Create another graph? (Y/N)")
                if response == 'Y' or response == 'N':
                    break
                else:
                    print("Invalid input")
            if response == 'Y':
                    continue
            elif response == 'N':

                while True:
                    nr_authors_to_remove = input("""According to the presented graphs, how many top 
                    authors should be excluded from further dataprocessing?""")
                    sure = input("""the top {} authors will be removed, correct? (Y/N)""".format(nr_authors_to_remove))
                    if sure == 'Y':
                        try:
                            nr_authors_to_remove = int(nr_authors_to_remove)
                            return nr_authors_to_remove

                        except Exception:
                            print("Invalid input")
                            continue
                    elif sure == 'N':
                        continue

                    else:
                        print("Invalid input")
                        continue

    @staticmethod
    def bulk_upload_graph(data_dict, target_table):
        # the graph will only contain data that has not yet been dismissed
        # by the dominant author and in_boundary data processing step!
        bulk_thresholds = []
        affected_media_objects = []

        for element in data_dict:
            bulk_thresholds.append(element[5:-8])
            affected_media_objects.append(data_dict[element])

        caption_text = "shows data that has not been dismissed by previous data processing"

        plt.bar(bulk_thresholds, affected_media_objects)
        plt.title("{}".format(target_table))
        plt.xlabel("bulk-upload thresholds")
        plt.ylabel("effected media objects")
        plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()

        while True:
            bulk_threshold = input("According to the presented graph, please input a suitable bulk threshold:\n")
            try:
                bulk_threshold = int(bulk_threshold)
                return bulk_threshold
            except Exception:
                print("Invalid input")

class DataProcessHandler:
    '''
    Summarizes all methods which are linked to data or text processing

    Important regex patterns used for text processing
    '''
    #removes all no-char elements
    onlychars_pattern = re.compile(r"[^a-züöäßâàç]")
    # removes instagram mentions and email addresses
    profile_pattern = re.compile(r'[^\s]*@[^\s]*')
    # removes url's
    uri_pattern = re.compile(r'(https://|http://|www.)[^\.]*[\S]*')
    # removes html tags e.g. <a>...</a> and <b>...</b>
    html_pattern = re.compile(r'<.+>[^<>]*</.>')
    # removes letters with number combos e.g. IMG3245
    char_num_pattern = re.compile(r"[^\s0-9]+[0-9]+")
    # removes often occuring geographic references
    geo_pattern = re.compile(r'[\S]*zug[\S]*')
    # umlaute patterns
    sharp_s_pattern = re.compile(r'(ß)')
    u_umlaute_pattern = re.compile(r'(ü)')
    o_umlaute_pattern = re.compile(r'(ö)')
    a_umlaute_pattern = re.compile(r'(ä)')
    # spell check
    spellcheck_pattern = re.compile(r'\s+')
    @staticmethod
    def csv_merger_traindata():
        wdirs = {"final_insta": "INSERT_PATH_HERE",
                 "final_insta_6_10": "INSERT_PATH_HERE",
                 "final_insta_22_11": "INSERT_PATH_HERE",
                 "instagram_1": "INSERT_PATH_HERE",
                 "isntagram_2": "INSERT_PATH_HERE",
                 "instagram_3": "INSERT_PATH_HERE"
                 }
        output_filepath = ""
        location_names = ["Aarau", "Arosa", "Ebmatingen", "LocarnoNord", "Neuchatel",
                          "Ovronnaz", "SChanf", "Scuol", "Uetliberg", "ZurichDolder"]
        '''
        Step 1: Iterating over each working directory and finding the files of the same location
        From those files a (unique) set of media_object_ids is created so that the final merge file 
        for that location does NOT contain duplicates        
        '''
        total_lines = 0
        regex_error_counter = 0
        pattern_1 = re.compile(r'[\d]+,([\d,\_,"]+),')
        pattern_2 = re.compile(r'([\d]+,([\d,\_,"]+),"[^"]*","[^"]*","[^"]*","[^"]*","(.+?(?=","))",(.+?(?=","))",(.+?(?=",))",[^,]*,"[^"]*",)')
        for location in location_names:
            print("-"*30 + "handling all files from the location {}".format(location) + "-"*30)
            file_content_dict = {}
            file_match_counter = 0
            for wd in wdirs:
                for index, file in enumerate(os.listdir(wdirs[wd]), 1):
                    if re.search(r'{}'.format(location), file):
                        file_match_counter += 1
                        print("File match: {}".format(file_match_counter))
                        with open(os.path.join(wdirs[wd], file), 'rt', encoding='utf-8') as f_read:
                            content = f_read.readlines()
                            previously_matched_id = 0
                            for index_2, line in enumerate(content):
                                if index_2 == 0:
                                    continue
                                else:
                                    '''
                                    check if last element of the line is a comma, if not it is likely to be a multi line object..
                                    '''
                                    total_lines += 1
                                    try:
                                        matched_id = re.match(pattern_1, line).group(1)
                                        previously_matched_id = matched_id
                                        file_content_dict[matched_id] = line
                                    #can occur when one line extends to the next
                                    except Exception as e:
                                        print("regex error at file {} index {} \nline: {}".format(wdirs[wd], (index_2+1), line))
                                        regex_error_counter += 1
                                        '''
                                        Handles cases where media objects are splitted over multiple lines which then don't start with an id
                                        '''
                                        file_content_dict[previously_matched_id] = file_content_dict[previously_matched_id]\
                                                                                       .strip('\n') + str(line).strip('\n') + '\n'
                                        continue
            print("Total lines processed: {}".format(total_lines))
            print("Keys in Dict (amount of unique ids: {}".format(len(file_content_dict.keys())))
            print("-"*30 + "finding matches" + "-"*30)
            '''
            Step 2: Write new file with only unique entries
            '''
            def determine(compare, element):
                if re.match(pattern_1, element).group(1) == compare:
                    # print("returning true in determine function...")
                    return True
            def dict_element_len(dict):
                total_len = 0
                for key in dict:
                    total_len += len(dict[key])

                return total_len
            output_filename = "Switzerland_" + str(location) + ".csv"

            with open(os.path.join(output_filepath, output_filename), 'wt', encoding='utf-8') as f_write:
                f_write.write("id,guid,coords,link,medialink,pubdate,author,title,description,like_count,filter,location")
                f_write.write('\n')
                for key, line in file_content_dict.items():
                    line_n_striped = line.rstrip('\n')
                    index = -1
                    while True:
                        if line_n_striped[index] == '"' or line_n_striped[index] == ',':
                            index -= 1
                        else:
                            index += 1
                            break

                    line_mod = line_n_striped[0:index] + '","' + str(location) + '"' + "\n"
                    f_write.write(line_mod)
        print("Regex errors: {}".format(regex_error_counter))
        print("-"*30 + "Done" + "-"*30)

    @staticmethod
    def csv_merger(filepaths_file):
        #output file will be in the parent directory of all included files
        #this method has been rewritten from a csv_reader approach (which did not work) to a regex / readlines method.
        filepaths = []
        media_object_ids_dict = {}
        total_media_objects = 0
        '''
        reading filepaths from input file
        '''
        with open(filepaths_file, 'rt', encoding='utf-8') as f:
            for path in f:
                mod_path = path.strip('\n').strip('"')
                filepaths.append(mod_path)
        output_filepath = re.match(r'(C:/Users/Max/Documents/Master_Thesis/Code_hub/Output/Netlytic/[^/]+/)',
                                   filepaths[0]).group(1) + "/merge_file.csv"
        '''
        Step 1: Creating a set of media_object_ids so the final merge file does NOT contain dublicates        
        '''
        for file in filepaths:
            with open(file, 'rt', encoding='ISO-8859-1') as f_read:
                content = f_read.readlines()
                for index, line in enumerate(content):
                    if index == 0:
                        continue
                    total_media_objects += 1
                    matched_id = re.match(r'[^,]+,([^,]+),', line).group(1)
                    media_object_ids_dict[matched_id] = line

        #iterate somehow over the set!
        #-----------------------------------------------------------------------------
        '''
        Step 2: Write new file with only unique entries
        '''
        print("Total media objects: {}\n Unique media objects: {}\n Dublicates: {}"
              .format(total_media_objects, len(media_object_ids_dict.keys()), (total_media_objects-len(media_object_ids_dict.keys()))))
        with open(output_filepath, 'wt', encoding='utf-8') as f_write:
            f_write.write("id,guid,coords,link,medialink,pubdate,author,title,description,like_count,filter")
            f_write.write('\n')
            for counter, unique_id in enumerate(media_object_ids_dict):
                f_write.write(media_object_ids_dict[unique_id])

    @staticmethod
    def nltk_wordlist_adaption(word_list):
        sharp_s_pattern = r'(ß)'
        u_umlaute_pattern = r'(ü)'
        o_umlaute_pattern = r'(ö)'
        a_umlaute_pattern = r'(ä)'

        mod_word_list = []
        for element in word_list:
            mod_word = re.sub(sharp_s_pattern, 'ss',
                            re.sub(u_umlaute_pattern, 'ue',
                                re.sub(o_umlaute_pattern, 'oe',
                                    re.sub(a_umlaute_pattern, 'ae', element)))).lower()
            mod_word_list.append(mod_word)

        return mod_word_list

    @staticmethod
    def remove_double_quotes_inside_string_flickr(input_filepath, target_table):
        print("doing some data prep... hold on")
        return input_filepath

    #gets skipped at the moment
    @staticmethod
    def remove_double_quotes_inside_string_instagram(input_filepath, target_table):
        """
        CSV-files provided by Netlytic (Instagram) as well as Flickr Data can contain user generated text (strings)
        which themselfs contain double quotes. These quotes screw with the entire format and must be removed/replaced.
        """
        print("doing some data prep... hold on")
        output_filepath = input_filepath.split(".")[0] + "_quotes_gone.csv"
        pattern_unionzug = re.compile(r'("[\d]+","[^"]+","[^"]+","[^"]+","[^"]+","[^"]+",")(.+?(?=","))","(.*?(?=","))","(.*?(?=","))","(.+)')
        pattern_trainingdata = re.compile(r'([\d]+,)([^,]+)(,"[^"]*","[^"]*","[^"]*","[^"]*",")(.+?(?=","))","(.*?(?=","))","(.*)')
        pattern_end = re.compile(r'[\d]*,[^,]*,[^,]+$')
        if re.search(r'unionzug', target_table):
            pattern = pattern_unionzug
            data_type = 'unionzug'
        elif re.search(r'trainingdata', target_table):
            pattern = pattern_trainingdata
            data_type = 'trainingdata'
        attribute_errors = 0
        with open(input_filepath, 'rt', encoding='ISO-8859-1') as f, open(output_filepath, 'wt', encoding='utf-8') as out_f:
            content = f.readlines()
            for index, line in enumerate(content):
                if index == 0:
                    if data_type == 'unionzug':
                        out_f.write(line)
                    continue
                else:

                    if data_type == 'unionzug':
                        matches = re.match(pattern, line)
                        quotes_removed = str(matches.group(1)) + \
                                         str(matches.group(2).replace('"', '').replace(',', ' ')) + '","' + \
                                         str(matches.group(3).replace('"', '').replace(',', ' ')) + '","' + \
                                         str(matches.group(4).replace('"', '').replace(',', ' ')) + '","' + \
                                         str(matches.group(5))
                        out_f.write(quotes_removed)
                        out_f.write('\n')
                    elif data_type == 'trainingdata':
                        """
                        with re.search and $ match then end of the line and cut of the like_count, filter and location
                        REASON: THEN THE POSITIVE LOOKAHEAD OF "," WORKS IN ALL CASES (HOPEFULLY)!
                        """
                        try:
                            line_end = re.search(pattern_end, line).group(0)
                        except AttributeError:
                            print("ENDSTRING Attribute Error: {}".format(line))
                            attribute_errors += 1
                            continue
                        end_len = len(line_end) * -1
                        line_cutoff = line[0:(end_len - 1)]
                        matches = re.match(pattern, line_cutoff)
                        try:
                            quotes_removed = str(matches.group(1)) + \
                                             '"' + str(matches.group(2).strip('"')) + '"' + \
                                             str(matches.group(3)) + \
                                             str(matches.group(4).replace('"', '').replace(',', ' ')) + '","' + \
                                             str(matches.group(5).replace('"', '').replace(',', ' ')) + '","' + \
                                             str(matches.group(6).replace('"', '').replace(',', ' ')) + '",'
                        except AttributeError as e:
                            #if this error occures, title or description contain only digits which causes an error
                            #it was decided, that these posts are neglected due to the missing textural information!
                            print("Line: {} - Attribute Error: {}".format((index+1), line))
                            attribute_errors += 1
                            continue

                        out_f.write(quotes_removed+line_end)

        print("File: {} - created".format(output_filepath))
        print("Attribute errors: {}".format(attribute_errors))
        return output_filepath

    @staticmethod
    def in_boundary(connection, table_name):
        '''
        !!!!Filename format: 'media_objects_RESEARCHAREA_DATASOURCE_inboundary.csv' !!!
        --> Important to assure that data gets appended to the correct db table
        Reads from a QGIS exported csv-file which was created through an intersection of media_objects and the
        shapefile of a research area. All media_objects in this in_boundy_media_objects - file will have the
        'in_boundry' db column changed to 'TRUE'
        -> the export settings in QGIS: csv format, everything else on default

        :param connection:
        :param cursor:
        :param in_boundary_media_objects:
        :return:
        '''
        print("-" * 30 + "in_boundary" + "-" * 30)
        print("Checking which media objects lie within the research area boundaries...")
        filepath = input("--->INPUT in_boundary filePATH from GIS in the format 'media_objects_RESEARCHAREA_DATASOURCE_inboundary.csv:\n").strip('"')
        '''
        extract corresponding table name from input filepath
        '''
        in_boundary_ids = []
        with open(filepath, 'rt', encoding='utf-8') as f:
            content = csv.reader(f, delimiter=',')
            for index, line in enumerate(content):
                if index == 0:
                    continue
                else:
                    #IMPORTANT - SOMETIMES AN ADDITIONAL COLUMN 'FID' IS ADDED BY QGIS CLIP
                    #the input file with the extension _inboundary will have an additional column 'fid'
                    #therefore, the column indexes for the media_object_id's do not correspond with the original files!
                    if re.search(r'instagram', table_name):
                        media_object_id = line[0] ####1!!!!
                    elif re.search(r'flickr', table_name):
                        media_object_id = line[6]
                    in_boundary_ids.append(media_object_id)
        '''
        write 'in_boundary_ids' to fusion_db and turn boolean column 'in_boundary' to TRUE
        '''
        print("in_boundary fields to update: {}".format(len(in_boundary_ids)))
        print("Executing in_boundary SQL update commands...")
        for id in in_boundary_ids:
            with connection.cursor() as cursor:
                cursor.execute("UPDATE {table_name} SET in_boundary = true WHERE media_object_id = %(mo_id)s;"\
                .format(table_name=table_name), {'mo_id': id})
            connection.commit()

        print("Step: in_boundary - done.")

    @staticmethod
    def data_processing(connection, table):
        """
        Handels bulk uploads, dominant users etc.
        The result of the processing of each row will be visible in the 'included_after_dataprocessing' boolean column
        :return:
        """
        location_dict = {
            '1': "Aarau",
            '2': "Arosa",
            '3': "Ebmatingen",
            '4': "LocarnoNord",
            '5': "Neuchatel",
            '6': "Ovronnaz",
            '7': "Schanf",
            '8': "Scuol",
            '9': "Uetliberg",
            '10': "ZurichDolder"
        }
        print("-" * 30 + "data_processing" + "-" * 30)
        '''
        first handling dominant users MAKE SURE TO ALWAYS CHECK FOR FILTERING DONE BEFORE.
        IF ITS FALSE THEN SKIP!
        '''
        print("removing dominant users in the table: {}".format(table))
        # get all author names and create a counter object to detect the author with the most uploaded posts
        # if its instagram take column 'author', if it is a flickr table take 'author_id'!
        with connection.cursor() as cursor:
            if re.search(r"instagram", table):
                if re.search(r"trainingdata", table):
                    ######
                    ###### Insert here adapted sql request with trainingdata location
                    ######
                    print(json.dumps(location_dict, indent=1))
                    location_name = input("Please choose the location index this trainingdata belongs to: \n")
                    cursor.execute("""SELECT author FROM media_objects_trainingdata_instagram 
                                             WHERE in_boundary IS TRUE
                                             AND location_name = %(location_name)s;
                                             """, {"location_name": location_dict[location_name]})
                else:
                    cursor.execute("SELECT author FROM media_objects_unionzug_instagram WHERE in_boundary IS TRUE;")

                authors = cursor.fetchall()
                connection.commit()
                total_nr_authors = len(authors)
                if re.search(r"trainingdata", table):
                    print("{} number of lines selected with the location: {}".format(total_nr_authors, location_dict[location_name]))
                c = Counter(authors)
                '''
                creating a posts per user graph for better evaluate the threshold for the dominant user data processing
                '''
                nr_authors_to_remove = StatisticHandler.posts_per_author_graph(c, table)

                print("{} authors will be removed of a total of {}".format(nr_authors_to_remove, total_nr_authors))
                removed_posts = 0
                for element in c.most_common(int(nr_authors_to_remove)):
                    removed_posts += element[1]
                    author = element[0][0]
                    if re.search(r"trainingdata", table):
                        cursor.execute("""UPDATE media_objects_trainingdata_instagram SET included_after_dataprocessing = false 
                                          WHERE author = %(author)s
                                          AND location_name = %(location)s;
                                          """, {'author': author, 'location': location_dict[location_name]})

                    else:
                        cursor.execute("""UPDATE media_objects_unionzug_instagram SET included_after_dataprocessing = false 
                                          WHERE author = %(author)s;
                                       """, {'author': author})
                    connection.commit()

                print("Total of {} posts from {} authors marked.".format(removed_posts, int(nr_authors_to_remove)))

            elif re.search(r"flickr", table):
                cursor.execute("SELECT author_id FROM media_objects_cantonzug_flickr WHERE in_boundary IS TRUE;")
                authors = cursor.fetchall()
                connection.commit()
                total_nr_authors = len(authors)
                c = Counter(authors)
                '''
                creating a posts per user graph for better evaluate the threshold for the dominant user data processing
                '''
                nr_authors_to_remove = StatisticHandler.posts_per_author_graph(c, table)

                print("The following {} authors will be removed of a total of {}".format(nr_authors_to_remove, total_nr_authors))
                print(str(c.most_common(int(nr_authors_to_remove))))
                removed_posts = 0
                for element in c.most_common(int(nr_authors_to_remove)):
                    removed_posts += element[1]
                    author_id = element[0][0]
                    with connection.cursor() as cursor:
                        cursor.execute("""UPDATE media_objects_cantonzug_flickr SET included_after_dataprocessing = false 
                                 WHERE author_id = %(author_id)s;""", {'author_id': author_id})
                    connection.commit()
                print("Total of {} posts from {} authors marked.".format(removed_posts, int(nr_authors_to_remove)))
            print("the top {} of users in table {} are taken care of.".format(nr_authors_to_remove, table))
        # -----------------------------------Bulk_upload handling-------------------------------------------------------
        print("Addressing bulk uploads. Hold on...")
        #defaul value which can get overwritten if the user choses to do so according to the bulk graph
        bulk_threshold = 3

        #dict for bulk graph creation
        bulk_counter_dict = {"bulk_2_counter": 0,
                             "bulk_3_counter": 0,
                             "bulk_4_counter": 0,
                             "bulk_5_counter": 0,
                             "bulk_6_counter": 0,
                             "bulk_7_counter": 0,
                             "bulk_8_counter": 0,
                             "bulk_9_counter": 0,
                             "bulk_10_counter": 0,
                             }
        '''
        create a list of unique authors for the following steps
        because the current list authors inludes the authors of every ROW and therefor many dublicates!!!!
        '''
        temp_author_container = []
        for author in authors:
            temp_author_container.append(author[0])
        unique_authors = set(temp_author_container)
        #######-------------------
        '''
        optimize db-connection by sending a bulk of requests as bulk, process them, and send the answer back
        Till now, for every author a connection was established, and torn down again.. really inefficent.
        Connecting to the db takes proportionally the longest time of all steps including sending, parsing the query etc.
        '''
        #######-------------------
        '''
        def bulk-upload: more than {bulk_threshold} uploads per user and hour
        get all the 'pubdate' timestamps of a specific author and count posts with the same 'modified' timestampt
        (to make them similar down to the hour  
        --> mark all of them as bulk-uploads (set included_after_processing = false)
        '''
        with connection.cursor() as cursor:
            if re.search(r"flickr", table):
                cursor.execute("""SELECT author_id, pubdate FROM media_objects_cantonzug_flickr
                                        WHERE in_boundary IS TRUE 
                                        AND included_after_dataprocessing IS TRUE;""")

            elif re.search(r"instagram", table):
                if re.search(r"trainingdata", table):
                    cursor.execute("""SELECT author, pubdate 
                                        FROM media_objects_trainingdata_instagram
                                        WHERE in_boundary IS TRUE 
                                        AND included_after_dataprocessing IS TRUE
                                        AND location_name = %(location)s;""", {'location': location_dict[location_name]})
                else:
                    cursor.execute("""SELECT author, pubdate 
                                        FROM media_objects_unionzug_instagram
                                        WHERE in_boundary IS TRUE 
                                        AND included_after_dataprocessing IS TRUE;""")

            result = cursor.fetchall()
        connection.commit()
        #creates a dictionary of which every key value is by default already a list
        result_dict = defaultdict(list)
        #create a dictionary where the keys are the author(_ids) and the value a list of modified pubdates
        for element in result:
            author = element[0]
            pubdate = element[1]
            #cuts off minutes and seconds
            mod_pubdate = pubdate[0:-6]
            result_dict[author].append(mod_pubdate)
        #create Counter items out of each list:
        for author in result_dict:
            result_dict[author] = Counter(result_dict[author])
            for pubdate in result_dict[author]:
                counter = result_dict[author][pubdate]
                '''
                Following code serves the bulk_threshold graph creation
                '''
                if int(counter) >= 2:
                    bulk_counter_dict["bulk_2_counter"] += counter
                if int(counter) >= 3:
                    bulk_counter_dict["bulk_3_counter"] += counter
                if int(counter) >= 4:
                    bulk_counter_dict["bulk_4_counter"] += counter
                if int(counter) >= 5:
                    bulk_counter_dict["bulk_5_counter"] += counter
                if int(counter) >= 6:
                    bulk_counter_dict["bulk_6_counter"] += counter
                if int(counter) >= 7:
                    bulk_counter_dict["bulk_7_counter"] += counter
                if int(counter) >= 8:
                    bulk_counter_dict["bulk_8_counter"] += counter
                if int(counter) >= 9:
                    bulk_counter_dict["bulk_9_counter"] += counter
                if int(counter) >= 10:
                    bulk_counter_dict["bulk_10_counter"] += counter

        bulk_threshold = StatisticHandler.bulk_upload_graph(bulk_counter_dict, table)
        print("Updating db accordingly. Hold on..")
        start = time.time()
        for author in result_dict:
            '''
            def bulk-upload: more than {bulk_threshold} uploads per user and hour
            get all the 'pubdate' timestamps of a specific author and count posts with the same 'modified' timestampt
            (to make them similar down to the hour
            --> mark all of them as bulk-uploads (set included_after_processing = false)
            '''
            for pubdate in result_dict[author]:
                counter = result_dict[author][pubdate]
                if int(counter) >= bulk_threshold:
                    with connection.cursor() as cursor:
                        if re.search(r'instagram', table):
                            # % represents a wildecard in SQL
                            if re.search(r"trainingdata", table):
                                cursor.execute("""UPDATE media_objects_trainingdata_instagram
                                                SET included_after_dataprocessing = false
                                                WHERE pubdate ~* %(pubdate)s 
                                                AND author = %(author)s
                                                AND location_name = %(location)s;
                                """, {'author': author, 'pubdate': pubdate, 'location': location_dict[location_name]})
                            else:
                                cursor.execute("""UPDATE media_objects_unionzug_instagram
                                                SET included_after_dataprocessing = false
                                                WHERE pubdate ~* %(pubdate)s 
                                                AND author = %(author)s;
                                """, {'author': author, 'pubdate': pubdate})
                        elif re.search(r'flickr', table):
                            cursor.execute("""UPDATE media_objects_cantonzug_flickr
                                            SET included_after_dataprocessing = false
                                            WHERE pubdate ~* %(pubdate)s 
                                            AND author_id = %(author)s;
                            """, {'author': author, 'pubdate': pubdate})
                    connection.commit()
        end = time.time()
        print("-*"*60)
        print("Time taken for bulk sql updates: {} s".format((end-start)))
        print("cleaned out bulk uploads.")
        print("Step: data processing - done.")
        print("-*" * 60)

    @staticmethod
    def lang_detect(test_str, topo_name_list, words_en, words_de, words_swiss, words_french, words_italian):
        # uses the library langdetect for detecting which language a string has
        # additionally the library pyenchant is used for spellchecking
        print("-" * 30 + "language-detection" + "-" * 30)
        counter_en = 0
        counter_de = 0
        counter_swiss = 0
        counter_french = 0
        counter_italian = 0
        det_en_words = []
        det_de_words = []
        det_swiss_words = []
        det_french_words = []
        det_italian_words = []
        list_of_test_strings_mod = []
        min_word_length = 3

        '''
        Doing some small text processing like in text_processing_core
        vowel mutations are being replaced, because the language dictionaries of german and swiss german have been 
        modified to have the same form to increase the overall matchrate across different ways of writing
        '''
        [list_of_test_strings_mod.append(x.strip())
         for x in re.sub(DataProcessHandler.onlychars_pattern, ' ',
            re.sub(DataProcessHandler.sharp_s_pattern, 'ss',
                re.sub(DataProcessHandler.u_umlaute_pattern, 'ue',
                    re.sub(DataProcessHandler.o_umlaute_pattern, 'oe',
                        re.sub(DataProcessHandler.a_umlaute_pattern, 'ae',
                            re.sub(DataProcessHandler.geo_pattern, ' ',
                                re.sub(DataProcessHandler.char_num_pattern, ' ',
                                    re.sub(DataProcessHandler.profile_pattern, ' ',
                                        re.sub(DataProcessHandler.uri_pattern, ' ',
                                            re.sub(DataProcessHandler.html_pattern, ' ', test_str.lower())))))))))).split()
                                   if len(x) >= min_word_length if x not in topo_name_list]

        for element in list_of_test_strings_mod:
            for word in words_en:
                if element == word:
                    det_en_words.append(element)
                    counter_en += 1
                    break
            for word in words_de:
                if element == word:
                    det_de_words.append(element)
                    counter_de += 1
                    break
            for word in words_swiss:
                if element == word:
                    det_swiss_words.append(element)
                    counter_swiss += 1
                    break
            for word in words_french:
                if element == word:
                    det_french_words.append(element)
                    counter_french += 1
                    break
            for word in words_italian:
                if element == word:
                    det_italian_words.append(element)
                    counter_italian += 1
                    break

        print("Counter_en: {} - Words: {}".format(counter_en, det_en_words))
        print("Counter_de: {} - Words: {}".format(counter_de, det_de_words))
        print("Counter_swiss: {} - Words: {}".format(counter_swiss, det_swiss_words))
        print("Counter_french: {} - Words: {}".format(counter_french, det_french_words))
        print("Counter_italian: {} - Words: {}".format(counter_italian, det_italian_words))
        # return the language to which most words of the test string were assigned to
        #english has priority, followed by german and swissgerman if the counters are equal
        if counter_en == 0 and counter_de == 0 and counter_swiss == 0:
            return None
        elif counter_en >= counter_de and counter_en >= counter_swiss and counter_en >= counter_french and counter_en >= counter_italian:
            return 'en'
        elif counter_de >= counter_en and counter_de >= counter_swiss and counter_de >= counter_french and counter_de >= counter_italian:
            return 'de'
        elif counter_swiss >= counter_en and counter_swiss >= counter_de and counter_swiss >= counter_french and counter_swiss >= counter_italian:
            return 'swiss'
        elif counter_french >= counter_en and counter_french >= counter_de and counter_french >= counter_swiss and counter_french >= counter_italian:
            return 'fr'
        elif counter_italian >= counter_en and counter_italian >= counter_de and counter_italian >= counter_swiss and counter_italian >= counter_french:
            return 'it'
        else:
            return None

    @staticmethod
    def spell_check(lan_dict, to_test):
        if lan_dict is not None and not re.search(DataProcessHandler.spellcheck_pattern, to_test):
            if lan_dict.check(to_test) is True:
                return to_test
            else:
                suggestions = lan_dict.suggest(to_test)
                try:
                    corrected = suggestions[0]
                    return corrected
                except IndexError:
                    return to_test
        else:
            return to_test

    @staticmethod
    def read_topography_names():
        wd = "C:/Users/Max/Documents/Master_Thesis/Code_hub/Output/GIS/topography_names_swiss.txt"
        topo_names_list = []
        with open(wd, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                topo_names_list.append(line)
        return topo_names_list

    @staticmethod
    def text_processing_core(test_string, topo_name_list, stemmer=None, words=None, lan_dict=None):

        list_of_strings_mod = []
        list_of_strings_mod_split = []
        list_of_strings_mod_split_stemmed = []
        min_word_length = 3
        if None not in (stemmer, words, lan_dict):
            [list_of_strings_mod.append(DataProcessHandler.spell_check(lan_dict, x).lower().strip())
             for x in re.sub(DataProcessHandler.onlychars_pattern, ' ',
                 re.sub(DataProcessHandler.geo_pattern, ' ',
                      re.sub(DataProcessHandler.char_num_pattern, ' ',
                         re.sub(DataProcessHandler.profile_pattern, ' ',
                             re.sub(DataProcessHandler.uri_pattern, ' ',
                                re.sub(DataProcessHandler.html_pattern, ' ', test_string.lower())))))).split()
                                    if len(x) >= min_word_length
                                        if x not in words
                                            if x not in topo_name_list]

            [list_of_strings_mod_split.extend(re.sub(DataProcessHandler.onlychars_pattern, '',
                                                 re.sub(DataProcessHandler.sharp_s_pattern, 'ss',
                                                    re.sub(DataProcessHandler.u_umlaute_pattern, 'ue',
                                                        re.sub(DataProcessHandler.o_umlaute_pattern, 'oe',
                                                            re.sub(DataProcessHandler.a_umlaute_pattern, 'ae', x.lower()))))).split())
             for x in list_of_strings_mod if len(x) >= min_word_length]

            [list_of_strings_mod_split_stemmed.append(stemmer.stem(x)) for x in list_of_strings_mod_split
                                                                            if x not in words
                                                                                if x not in topo_name_list]
            return list_of_strings_mod_split_stemmed
        #italian language case
        elif None not in (stemmer, words) and lan_dict == None:
            [list_of_strings_mod.append(x.lower().strip())
             for x in re.sub(DataProcessHandler.onlychars_pattern, ' ',
                             re.sub(DataProcessHandler.geo_pattern, ' ',
                                re.sub(DataProcessHandler.char_num_pattern, ' ',
                                    re.sub(DataProcessHandler.profile_pattern, ' ',
                                       re.sub(DataProcessHandler.uri_pattern, ' ',
                                              re.sub(DataProcessHandler.html_pattern, ' ', test_string.lower())))))).split()
                         if len(x) >= min_word_length
                            if x not in words
                                if x not in topo_name_list]

            [list_of_strings_mod_split.extend(re.sub(DataProcessHandler.sharp_s_pattern, 'ss',
                                                     re.sub(DataProcessHandler.u_umlaute_pattern, 'ue',
                                                            re.sub(DataProcessHandler.o_umlaute_pattern, 'oe',
                                                                   re.sub(DataProcessHandler.a_umlaute_pattern, 'ae',
                                                                          x.lower())))).split())
             for x in list_of_strings_mod if len(x) >= min_word_length]

            [list_of_strings_mod_split_stemmed.append(stemmer.stem(x)) for x in list_of_strings_mod_split
                                                                            if x not in words
                                                                                if x not in topo_name_list]
            return list_of_strings_mod_split_stemmed
        else:
            [list_of_strings_mod.append(x.strip())
             for x in re.sub(DataProcessHandler.onlychars_pattern, ' ',
               re.sub(DataProcessHandler.sharp_s_pattern, 'ss',
                 re.sub(DataProcessHandler.u_umlaute_pattern, 'ue',
                    re.sub(DataProcessHandler.o_umlaute_pattern, 'oe',
                       re.sub(DataProcessHandler.a_umlaute_pattern, 'ae',
                         re.sub(DataProcessHandler.geo_pattern, ' ',
                            re.sub(DataProcessHandler.char_num_pattern, ' ',
                               re.sub(DataProcessHandler.profile_pattern, ' ',
                                  re.sub(DataProcessHandler.uri_pattern, ' ',
                                     re.sub(DataProcessHandler.html_pattern, ' ', test_string.lower())))))))))).split()
                                        if len(x) >= min_word_length if x not in topo_name_list]

            return list_of_strings_mod

    @staticmethod
    def text_processing(connection, table):
        """
        processing includes: sentence splitting, stemming, removing of stop-words and numbers, spelling-correction etc.
        Instagram - Text includes the columns 'title' and 'description' --> CHECK IF THEY ARE IDENTICAL!!!!
        Flickr - Text includes the columns 'title', 'description' and 'tags'
        :return:
        """
        print("-" * 30 + "text_processing" + "-" * 30)
        # ---------------creating-all-necessary-objects-and-dictionaries as deutsch:
        nltk.download('stopwords')
        nltk.download('words')
        stemmer_en = SnowballStemmer("english")
        stemmer_fr = SnowballStemmer("french")
        stemmer_it = SnowballStemmer("italian")
        stemmer_de = Cistem()
        stopwords_en = stopwords.words("english")
        stopwords_de = DataProcessHandler.nltk_wordlist_adaption(stopwords.words("german"))
        stopwords_fr = stopwords.words("french")
        stopwords_it = stopwords.words("italian")
        dict_en = enchant.Dict("en_GB")
        dict_de = enchant.Dict("de")
        dict_fr = enchant.Dict("fr")
        #no spell-correction for italian
        path_german_words_file = "INSERT_PATH_HERE"
        path_swiss_words_file = "INSERT_PATH_HERE"
        path_french_words_file = "INSERT_PATH_HERE"
        path_italian_words_file = "INSERT_PATH_HERE"
        words_en = DataProcessHandler.nltk_wordlist_adaption(nltk.corpus.words.words())
        topo_name_list = DataProcessHandler.read_topography_names()
        words_de = []
        words_swiss = []
        words_french = []
        words_italian = []

        with open(path_german_words_file, 'rt') as german:
            content = german.readlines()
            for word in content:
                words_de.append(word.strip())
        with open(path_swiss_words_file, 'rt') as swiss:
            content = swiss.readlines()
            for word in content:
                words_swiss.append(word.strip())
        with open(path_french_words_file, 'rt') as french:
            content = french.readlines()
            for word in content:
                words_french.append(word.strip())
        with open(path_italian_words_file, 'rt') as italian:
            content = italian.readlines()
            for word in content:
                words_italian.append(word.strip())

        with connection.cursor() as cursor:
            if re.search(r'instagram', table):
                '''
                Handling an instagram table with text columns 'title' and 'description' which often are identical!!!
                '''
                if re.search(r'trainingdata', table):
                    cursor.execute("""SELECT media_object_id, title, description FROM media_objects_trainingdata_instagram 
                                        WHERE processed_text IS NULL 
                                        AND included_after_dataprocessing IS TRUE
                                        ORDER BY media_object_id ASC;""")

                else:
                    cursor.execute("""SELECT media_object_id, title, description FROM media_objects_unionzug_instagram 
                                                        WHERE processed_text IS NULL 
                                                        AND included_after_dataprocessing IS TRUE
                                                        AND in_boundary IS TRUE;""")
                content = cursor.fetchall()
                connection.commit()
                for index, row in enumerate(content):
                    '''
                    Execute and commit commands regularly
                    '''
                    if index % 5 == 0 and index != 0:
                        print("Commiting processed text string...")
                        connection.commit()

                    print("Handling row {}: {}".format((index + 1), row))
                    media_object_id = row[0]
                    title = row[1]
                    description = row[2]
                    '''
                    language detection and spelling correction                
                    '''
                    det_language = DataProcessHandler.lang_detect(str(title + " " + description), topo_name_list,
                                                                  words_en, words_de, words_swiss, words_french, words_italian)
                    print("Row {}: Detected language: {}".format((index + 1), det_language))
                    '''
                    actual text processing
                    '''
                    title_mod = []
                    description_mod = []
                    final_text_container = []
                    print("-" * 30 + "spell-checking and stemming" + "-" * 30)
                    if det_language == 'en':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_en, stopwords_en, dict_en)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_en, stopwords_en, dict_en)

                    elif det_language == 'de' or det_language == 'swiss':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_de, stopwords_de, dict_de)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_de, stopwords_de, dict_de)

                    elif det_language == 'fr':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_fr, stopwords_fr, dict_fr)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_fr, stopwords_fr, dict_fr)

                    elif det_language == 'it':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_it, stopwords_it)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_it, stopwords_it)

                    elif det_language is None:
                        det_language = 'None'
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list)
                    '''
                    check if title and description are identical. If that is the case they should not be both added to
                    avoid overvalueing each word
                    '''
                    match_counter = 0
                    len_title_mod = len(title_mod)
                    to_similar = False
                    if len(title_mod) >= 1 and len(description_mod) >= 1:
                        for element in title_mod:
                            if element in description_mod:
                                match_counter += 1
                        # checks if 75% or more words in title_mod are also in description_mod included
                        try:
                            if (match_counter / len_title_mod) >= 0.75:
                                to_similar = True
                        except ZeroDivisionError:
                            to_similar = False

                        if not to_similar:
                            final_text_container = title_mod + description_mod

                        elif to_similar:
                            final_text_container = set(title_mod + description_mod)
                    else:
                        final_text_container = title_mod + description_mod

                    '''
                    convert container elements to a final string in which the elements / words are sepereated by a space!
                    double and single quotes have to be removed again due to stemming which could lead to single quotes again!
                    '''
                    final_string = str(' '.join(final_text_container)).replace('"', '').replace("'", "")
                    '''
                    write sql command which updates row and pastes 'final_text_container' into 'processed_text' cell
                    '''
                    if re.search(r'trainingdata', table):
                        cursor.execute("""UPDATE media_objects_trainingdata_instagram 
                                        SET processed_text = %(container)s, detected_language = %(language)s
                                        WHERE media_object_id = %(id)s;""", {'container': final_string,
                                                                             'language': det_language,
                                                                             'id': media_object_id
                                                                             })
                    else:
                        cursor.execute("""UPDATE media_objects_unionzug_instagram
                                    SET processed_text = %(container)s, detected_language = %(language)s
                                    WHERE media_object_id = %(id)s;""", {'container': final_string,
                                                                         'language': det_language,
                                                                         'id': media_object_id
                                                                         })

            elif re.search(r'flickr', table):
                '''
                Handling an flickr table with text columns 'title', 'description' and 'tags' --> check if identical!!!
    
    
                IMPORTANT NOTE: The Flickr generated image tags were neglected for 
                the text processing and further analysis – due to possible overlapping with the tags / labels that 
                were generated with the google cloud vision image recognition algorithm!
                '''
                cursor.execute("""SELECT media_object_id, title, description, tags
                                FROM media_objects_cantonzug_flickr 
                                WHERE processed_text IS NULL
                                AND included_after_dataprocessing IS TRUE
                                AND in_boundary IS TRUE
                                ORDER BY media_object_id ASC;""")

                content = cursor.fetchall()
                connection.commit()
                for index, row in enumerate(content):
                    '''
                    Execute and commit commands regularly
                    '''
                    if index % 5 == 0 and index != 0:
                        print("Commiting processed text string...")
                        connection.commit()

                    print("Handling row {}: {}".format((index + 1), row))
                    media_object_id = row[0]
                    title = row[1]
                    description = row[2]
                    tags = row[3]
                    '''
                    language detection and spelling correction                
                    '''
                    det_language = DataProcessHandler.lang_detect(str(title + " " + description), topo_name_list, words_en,
                                                                  words_de, words_swiss, words_french, words_italian)
                    print("Row {}: Detected language: {}".format(index, det_language))
                    '''
                    actual text processing
                    '''
                    title_mod = []
                    description_mod = []
                    final_text_container = []
                    print("-" * 30 + "spell-checking and stemming" + "-" * 30)
                    if det_language == 'en':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_en, stopwords_en, dict_en)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_en, stopwords_en, dict_en)

                    elif det_language == 'de' or det_language == 'swiss':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_de, stopwords_de, dict_de)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_de, stopwords_de, dict_de)

                    elif det_language == 'fr':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_fr, stopwords_fr, dict_fr)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_fr, stopwords_fr, dict_fr)

                    elif det_language == 'it':
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list, stemmer_it, stopwords_it)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list, stemmer_it, stopwords_it)

                    elif det_language is None:
                        det_language = 'None'
                        title_mod = DataProcessHandler.text_processing_core(title, topo_name_list)
                        description_mod = DataProcessHandler.text_processing_core(description, topo_name_list)

                    tags_mod = DataProcessHandler.text_processing_core(tags, topo_name_list, stemmer_en, stopwords_en, dict_en)
                    '''
                    check if title and description are identical. If that is the case they should not be both added to
                    avoid overvalueing each word
                    '''
                    match_counter = 0
                    len_title_mod = len(title_mod)
                    to_similar = False
                    if len(title_mod) >= 1 and len(description_mod) >= 1:
                        for element in title_mod:
                            if element in description_mod:
                                match_counter += 1
                        # checks if 75% or more words in title_mod are also in description_mod included
                        try:
                            if (match_counter / len_title_mod) >= 0.75:
                                to_similar = True
                        except ZeroDivisionError:
                            to_similar = False

                        if not to_similar:
                            final_text_container = title_mod + description_mod + tags_mod

                        elif to_similar:
                            final_text_container = list(set(title_mod + description_mod)) + tags_mod
                    else:
                        final_text_container = title_mod + description_mod + tags_mod

                    '''
                    convert container elements to a final string in which the elements / words are sepereated by a space!
                    double and single quotes have to be removed again due to stemming which could lead to single quotes again!
                    '''
                    final_string = str(' '.join(final_text_container)).replace('"', '').replace("'", "")
                    '''
                    write sql command which updates row and pastes 'final_text_container' into 'processed_text' cell
                    '''
                    cursor.execute("""UPDATE media_objects_cantonzug_flickr 
                                    SET processed_text = %(container)s, detected_language = %(language)s 
                                    WHERE media_object_id = %(id)s;""", {'container': final_string,
                                                                         'language': det_language,
                                                                         'id': media_object_id
                                                                        })
        connection.commit()
        print("Step: text processing - done.")

'''
Google Vision labels will not be added to a CSV file but rather the db itself (unlike the location data).    
'''
class VisionDataAddOnHandler:
    """
    Here I am trying to query the db directly (not the csv files anylonger) to acquire the link to the
    picture of each media object to feed it into the google cloud vision API. The labels with their corresponding
    score will be saved under the 'vision_labels' column as a list of tuples.
    """
    def __init__(self):
        pass

    @staticmethod
    def implicit():
        # If you don't specify credentials when constructing the client, the
        # client library will look for credentials in the environment.
        storage_client = storage.Client()

        # Make an authenticated API request
        buckets = list(storage_client.list_buckets())
        print(buckets)

    @staticmethod
    def detect_labels_uri(url):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "INSERT_PATH_HERE"
        VisionDataAddOnHandler.implicit()
        '''
        Creating necessary objects
        '''
        client = vision.ImageAnnotatorClient()
        image = vision.types.Image()
        '''
        object attribute / parameter URI will be given a new value
        '''
        image.source.image_uri = url
        '''
        Detects labels in the file located in Google Cloud Storage or on the
        Web.
        '''
        response = client.label_detection(image=image)
        labels = response.label_annotations

        # print('Labels:')
        label_score_tuples = []
        for label in labels:
            print(label.description, ", " + str((round(label.score, 2)*100)) + " %")
            label_score_tuples.append((label.description, (round(label.score, 2)*100)))
        return label_score_tuples

class PostgresqlHandler:
    '''
    add extra boolean column to specify if a media_object or location is inside the research area!
    needs to be done with arcgis which returns a list of id's which are inside or python gis library maybe
    '''

    def __init__(self):
        self.db_operation({'operation': 0})

    @staticmethod
    def config(filename='database.ini', section='postgresql'):
        # create a parser
        parser = ConfigParser()
        # read config file
        parser.read(filename)
        # get section, default to postgresql
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))

        return db

    '''
    command_dic['operation'] - operation type [INT], command_dic['file'] - source file, command_dic['table'] - target table
    '''

    def db_operation(self, command_dic):
        conn = None
        '''
        read the connection parameters
        '''
        print("reading connection parameters...")
        params = PostgresqlHandler.config()
        '''
        connect to the PostgreSQL server
        '''
        print("establishing connection...")
        conn = psycopg2.connect(**params)
        print("connected")

        #### THIS BELOW WILL ALL BE TAKEN OUT SO THAT IT DOES NOT GET EXECUTED AUTOMATICALLY!
        #### CONECTION HANDLER WILL THEN ONLY SET UP AND MANAGE THE CONNECTION

        '''
        ------------------------------------------------------------
        execute commands:
        ------------------------------------------------------------
        
        creating tables if not already existing -> default operation 0
        '''
        if command_dic['operation'] == 0:
            PostgresqlHandler.create_tables(conn)
        #add data to instagram/flickr table
        elif command_dic['operation'] == 5:
            source_file = command_dic['file']
            target_table = command_dic['table']
            PostgresqlHandler.populate_table(conn, target_table, source_file)
        #Foursquare data handling
        elif command_dic['operation'] == 6:
            target_table = command_dic['table']
            source_directory = command_dic['directory']
            PostgresqlHandler.populate_table_foursquare(conn, target_table, source_directory)
        #acquire google cloud vision labels for given media url's
        elif command_dic['operation'] == 7:
            #maybe pass the connection as well to make regular commits due to huge data and time for labeling
            #in case of API error the entire process might be lost.
            PostgresqlHandler.write_vision_labels_to_db(conn)
        #drop table
        elif command_dic['operation'] == 8:
            target_table = command_dic['table']
            if target_table == 'all':
                PostgresqlHandler.drop_all_tables(conn)
            else:
                PostgresqlHandler.drop_table(conn, target_table)
        #train ML model
        elif command_dic['operation'] == 10:
            MachineLearningHandler.train_model(conn)

        #predict media objects with ML model
        elif command_dic['operation'] == 11:
            while True:
                w_labels = input("Shall vision labels of the media objects be included for the model class prediction? [Y/N]\n")
                if w_labels == 'Y':
                    w_labels = True
                    break
                elif w_labels == 'N':
                    w_labels = False
                    break
                else:
                    print("Invalid input")
            MachineLearningHandler.predict_media_objects(conn, w_labels)

        #compare with and without vision label classification
        elif command_dic['operation'] == 12:
            MachineLearningHandler.compare_w_without_labels_classification(conn)
        '''
        close communication with the PostgreSQL database server
        only cursor not the direct connection itself!
        '''
        # cur.close()
        '''
        close communication with the PostgreSQL database server            
        only cursor not the direct connection itself!
        '''
        conn.commit()

    @staticmethod
    def drop_table(connection, table_name):
        with connection.cursor() as cursor:
            sql_drop = "DROP TABLE IF EXISTS {} CASCADE;".format(table_name)
            cursor.execute(sql_drop)
            print("Table {} has been successfully dropped.".format(table_name))
        connection.commit()

    @staticmethod
    def drop_all_tables(connection):
        objects = ['media_objects', 'locations']
        research_areas = ['seepromenade', 'zugerberg']
        datasources = ['instagram']
        table_names_dict = {}

        # creating possible mutations
        table_index = 1
        for item in objects:
            for area in research_areas:
                for datasource in datasources:
                    gen_name = str(item + '_' + area + '_' + datasource)
                    table_names_dict[str(table_index)] = gen_name
                    table_index += 1
        table_names_dict[str(table_index)] = "media_objects_cantonzug_flickr"
        table_names_dict[str(table_index+1)] = "locations_cantonzug_flickr"
        for table in table_names_dict:
            with connection.cursor() as cursor:
                sql_drop = "DROP TABLE IF EXISTS {};".format(table_names_dict[table])
                cursor.execute(sql_drop)
            connection.commit()
            print("Table {} has been successfully droped.".format(table_names_dict[table]))
    '''
    For instagram and flickr dataobjects
    --> Foursquare data will most likely require a new table creation method
    '''
    @staticmethod
    def selectable_table_names_gen():
        table_name_dict = {}
        table_name_dict['1'] = "media_objects_cantonzug_flickr"
        table_name_dict['2'] = "media_objects_unionzug_instagram"
        table_name_dict['3'] = "media_objects_trainingdata_instagram"
        return table_name_dict

    @staticmethod
    def create_tables(connection):
        """ create tables in the PostgreSQL database
        locations must be first in list so they get created first.
        because media_objects foreign key needs reference!
        """
        '''
        following code only compiles Instagram table names after the research_area adaption of 07.11.2018 where was decided
        that Flickr data will be acquired for the entire Canton of Zug!
        '''
        objects = ['locations', 'media_objects']
        research_areas = ['unionzug', 'trainingdata']
        '''
        creating possible table mutations
        '''
        # in_boundy -> is the media_object actually located inside the research_area
        # processed_text -> after processing, including sentence splitting, stemming, removing of
        # stop-words and numbers, spelling-correction etc.
        # classification -> result of the machine learning text classification
        for object_ in objects:
            for area in research_areas:
                if area == 'unionzug':
                    with connection.cursor() as cursor:
                        if object_ == 'media_objects':
                                cursor.execute("""
                                        CREATE TABLE IF NOT EXISTS media_objects_unionzug_instagram (
                                                media_object_id TEXT PRIMARY KEY,
                                                location_id BIGINT,
                                                link TEXT,
                                                media_link TEXT,
                                                pubdate TEXT,
                                                author TEXT,
                                                title TEXT,
                                                description TEXT,
                                                processed_text TEXT,
                                                like_count BIGINT,
                                                filter TEXT,
                                                detected_language TEXT,
                                                vision_labels TEXT,
                                                classification_m1_w_labels TEXT,
                                                classification_m1_without_labels TEXT,
                                                classification_m2_w_labels TEXT,
                                                classification_m2_without_labels TEXT,
                                                in_boundary BOOLEAN DEFAULT FALSE,
                                                included_after_dataprocessing BOOLEAN DEFAULT TRUE,
                                                is_model_data BOOLEAN DEFAULT TRUE,
                                                FOREIGN KEY (location_id) REFERENCES locations_unionzug_instagram(id)
                                                );""")

                        elif object_ == 'locations':
                            cursor.execute("""
                                            CREATE TABLE IF NOT EXISTS locations_unionzug_instagram (
                                                    id BIGINT PRIMARY KEY,
                                                    name TEXT NOT NULL,
                                                    latitude DOUBLE PRECISION,
                                                    longitude DOUBLE PRECISION
                                                    );""")
                        cursor.execute("""
                                    CREATE UNIQUE INDEX IF NOT EXISTS index_id_union_instagram
                                        ON public.media_objects_unionzug_instagram USING btree
                                        (media_object_id COLLATE pg_catalog."default" varchar_ops)
                                        TABLESPACE pg_default;        
                                        """)

                elif area == 'trainingdata':
                    if object_ == 'media_objects':
                        with connection.cursor() as cursor:
                            cursor.execute("""
                                    CREATE TABLE IF NOT EXISTS media_objects_trainingdata_instagram (
                                            media_object_id TEXT PRIMARY KEY,
                                            latitude DOUBLE PRECISION,
                                            longitude DOUBLE PRECISION,
                                            link TEXT,
                                            media_link TEXT,
                                            pubdate TEXT,
                                            author TEXT,
                                            title TEXT,
                                            description TEXT,
                                            processed_text TEXT,
                                            like_count BIGINT,
                                            filter TEXT,
                                            detected_language TEXT,
                                            vision_labels TEXT,
                                            classification TEXT DEFAULT 'None',
                                            in_boundary BOOLEAN DEFAULT FALSE,            
                                            included_after_dataprocessing BOOLEAN DEFAULT TRUE,
                                            is_model_data BOOLEAN DEFAULT TRUE,
                                            location_name TEXT                                      
                                            );""")
                            cursor.execute("""
                                        CREATE UNIQUE INDEX IF NOT EXISTS index_media_object_id
                                        ON public.media_objects_trainingdata_instagram USING btree
                                        (media_object_id COLLATE pg_catalog."default" varchar_ops)
                                        TABLESPACE pg_default;""")
                            cursor.execute("""
                                        CREATE INDEX IF NOT EXISTS index_author
                                        ON public.media_objects_trainingdata_instagram USING btree
                                        (author COLLATE pg_catalog."default" varchar_ops)
                                        TABLESPACE pg_default;""")
                            cursor.execute("""
                                        CREATE INDEX IF NOT EXISTS index_pubdate
                                        ON public.media_objects_trainingdata_instagram USING btree
                                        (pubdate COLLATE pg_catalog."default" varchar_ops)
                                        TABLESPACE pg_default;""")
                            cursor.execute("""
                                        CREATE INDEX IF NOT EXISTS index_processed_text
                                        ON public.media_objects_trainingdata_instagram USING btree
                                        (processed_text COLLATE pg_catalog."default" varchar_ops)
                                        TABLESPACE pg_default;""")
                            cursor.execute("""
                                        CREATE INDEX IF NOT EXISTS index_language
                                        ON public.media_objects_trainingdata_instagram USING btree
                                        (detected_language COLLATE pg_catalog."default" varchar_ops)
                                        TABLESPACE pg_default;""")
        connection.commit()
        # reference changes to original flickr csv. file (to keep similar shema as instagram tables):
        # NAME changed to title
        # Owner changed to author
        # UserID changed to author_id
        # PhotoID changed to media_object_id
        # URL changed to link
        # UploadDate changed to pubdate
        # views replaced like_count from Instagram table

        #in_boundy -> is the media_object actually located inside the research_area
        #processed_text -> after processing, including sentence splitting, stemming, removing of
        # stop-words and numbers, spelling-correction etc.
        #classification -> result of the machine learning text classification
        with connection.cursor() as cursor:
            cursor.execute("""
                        CREATE TABLE IF NOT EXISTS locations_cantonzug_flickr (
                                id BIGINT PRIMARY KEY,
                                name TEXT NOT NULL,
                                location_type TEXT,
                                latitude DOUBLE PRECISION NOT NULL,
                                longitude DOUBLE PRECISION NOT NULL
                                );
                        """)

            cursor.execute("""
                      CREATE TABLE IF NOT EXISTS media_objects_cantonzug_flickr (
                              media_object_id BIGINT PRIMARY KEY,
                              location_id BIGINT,
                              link TEXT,
                              datetaken TEXT,
                              pubdate TEXT,
                              author_id TEXT,
                              author TEXT,
                              author_origin TEXT,
                              title TEXT,
                              description TEXT,
                              processed_text TEXT,
                              views BIGINT,
                              faves BIGINT,
                              tags TEXT,
                              detected_language TEXT,                                             
                              vision_labels TEXT,                                              
                              classification_m1_w_labels TEXT,
                              classification_m2_without_labels TEXT,
                              classification_m2_w_labels TEXT,
                              classification_m2_without_labels TEXT,
                              in_boundary BOOLEAN DEFAULT FALSE,
                              included_after_dataprocessing BOOLEAN DEFAULT TRUE,
                              is_model_data BOOLEAN DEFAULT FALSE,
                              FOREIGN KEY (location_id) REFERENCES locations_cantonzug_flickr(id)
                              );""")

            cursor.execute("""
                        CREATE UNIQUE INDEX IF NOT EXISTS index_id_flickr
                            ON public.media_objects_cantonzug_flickr USING btree
                            (media_object_id)
                            TABLESPACE pg_default;        
                            """)

            cursor.execute("""
                      CREATE TABLE IF NOT EXISTS media_objects_cantonzug_foursquare (
                              media_object_id TEXT PRIMARY KEY,
                              venue_name TEXT,
                              latitude DOUBLE PRECISION, 
                              longitude DOUBLE PRECISION,
                              formatted_address TEXT,
                              category_id TEXT,
                              category_name TEXT,
                              rating FLOAT(2),
                              verified BOOLEAN 
                              );""")

        print("Created or checked existence of needed db tables")
        connection.commit()

    @staticmethod
    def populate_table(connection, target_table, source_filepath):
        #!!make file_name correspond directly to the created from the corresponding method!!
        #when adding to a media_objects table the location data from the same csv file will be added simultaneously!!
        print("-" * 30 + "populate_table {}".format(target_table) + "-" * 30)
        filepath = source_filepath
        target_table = target_table
        if re.search(r'instagram', target_table):
            file_quotes_removed = DataProcessHandler.remove_double_quotes_inside_string_instagram(filepath, target_table)
            print("handling instagram data...")
        elif re.search(r'flickr', target_table):
            file_quotes_removed = DataProcessHandler.remove_double_quotes_inside_string_flickr(filepath, target_table)
            print("handling flickr data...")

        with open(file_quotes_removed, "rt", encoding='ISO-8859-1') as f:
            content = csv.reader(f, delimiter=',')
            '''
            finding the length of the content object for print purposes
            '''
            len_content = sum(1 for line in content)
            print("Total amount of data rows: {}".format((len_content-1)))
            f.seek(0)
            for linecounter, line in enumerate(content, 1):
                '''
                Here we need to differentiate between Flickr and Instagram source files due to their
                different structure!
                '''
                if linecounter == 1:
                    continue

                if linecounter % 100 == 0:
                    connection.commit()

                with connection.cursor() as cursor:
                    if re.search('instagram', target_table):
                        if re.search('unionzug', target_table):
                            '''
                            Inputs for media_objects table

                            to add a string with a single quote, replace the single quote with a double single quote
                            !!!! SQL demands single quotes around the inputs after values, even with format etc.!!!!
                            '''
                            media_object_id = line[1]
                            link = line[3]
                            media_link = line[4]
                            pubdate = line[5]
                            author = line[6].replace('"', "").replace("'", "''")
                            title = line[7].replace('"', "").replace("'", "''")
                            description = line[8].replace('"', "").replace("'", "''")
                            like_count = line[9]
                            filter = line[-5]
                            print("Media_object_id of line {}: {}".format(linecounter, media_object_id))
                            '''
                            location data
                            '''
                            location_id = line[-4]
                            location_name = line[-3].replace('"', "").replace("'", "''")
                            latitude = line[-2]
                            longitude = line[-1]

                            cursor.execute("""INSERT INTO locations_unionzug_instagram (id, name, latitude, longitude)
                                       VALUES (%(location_id)s,%(location_name)s,%(latitude)s,%(longitude)s)
                                       ON CONFLICT DO NOTHING;""", {'location_id': location_id,
                                                                    'location_name': location_name,
                                                                    'latitude': latitude,
                                                                    'longitude': longitude
                                                                    })

                            cursor.execute("""INSERT INTO media_objects_unionzug_instagram
                            (media_object_id, location_id, link, media_link, pubdate, author, title, description, like_count, filter)
                            VALUES (%(media_object_id)s,%(location_id)s,%(link)s,%(media_link)s,%(pubdate)s,%(author)s,%(title)s,
                            %(description)s,%(like_count)s,%(filter)s) ON CONFLICT DO NOTHING;""",
                                           {'media_object_id': media_object_id,
                                            'location_id': location_id,
                                            'link': link,
                                            'media_link': media_link,
                                            'pubdate': pubdate,
                                            'author': author,
                                            'title': title,
                                            'description': description,
                                            'like_count': like_count,
                                            'filter': filter
                                            })

                        elif re.search('trainingdata', target_table):
                            media_object_id = line[1]
                            coordinates = line[2]
                            if len(line[2]) == 0:
                                latitude = 9999
                                longitude = 9999
                            else:
                                latitude = line[2].split(",")[0]
                                longitude = line[2].split(",")[1]
                            link = line[3]
                            media_link = line[4]
                            pubdate = line[5]
                            author = line[6].replace('"', "").replace("'", "''")
                            title = line[7].replace('"', "").replace("'", "''")
                            description = line[8].replace('"', "").replace("'", "''")
                            like_count = line[9]
                            if like_count == '':
                                like_count = 0
                            if media_object_id.find('_') != -1:
                                filter = line[10]
                                try:
                                    location_name = line[11]
                                except IndexError as e:
                                    print("Index Error: {}".format(line))

                            elif media_object_id.find('_') == -1:
                                filter = line[-2]
                                try:
                                    location_name = line[-1]
                                except IndexError as e:
                                    print("Index Error: {}".format(line))

                            in_boundary = True

                            cursor.execute("""INSERT INTO media_objects_trainingdata_instagram
                            (media_object_id, latitude, longitude, link, media_link, pubdate, author, title,
                            description, like_count, filter, location_name, in_boundary)
                            VALUES (%(id)s, %(lat)s, %(lng)s, %(link)s, %(media_link)s, %(pubdate)s, %(author)s, %(title)s,
                                    %(desc)s, %(like)s, %(filter)s, %(loc)s, %(bound)s) ON CONFLICT DO NOTHING;""",
                            {'id': media_object_id,
                             'lat': latitude,
                             'lng': longitude,
                             'link': link,
                             'media_link': media_link,
                             'pubdate': pubdate,
                             'author': author,
                             'title': title,
                             'desc': description,
                             'like': like_count,
                             'filter': filter,
                             'loc': location_name,
                             'bound': in_boundary})

                    elif re.search('flickr', target_table):
                        #following still needs to be adapted to flickr source file!
                        media_object_id = line[6]
                        link = line[5]
                        datetaken = datetime.datetime.strptime(line[-8], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                        pubdate = datetime.datetime.strptime(line[-7], '%m/%d/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                        author_id = line[7]
                        author = line[8].replace('"', "").replace("'", "''")
                        author_origin = line[9].replace('"', "").replace("'", "''")
                        title = line[3].replace('"', "").replace("'", "''")
                        description = line[4].replace('"', "").replace("'", "''")
                        views = line[-6]
                        faves = line[-5]
                        tags = line[-4].replace('"', "").replace("'", "''")
                        '''
                        Inputs for locations table
                        '''
                        location_id = line[-3]
                        location_name = line[-2].replace('"', "").replace("'", "''")
                        location_type = line[-1].replace('"', "").replace("'", "''")
                        latitude = line[1]
                        longitude = line[2]

                        cursor.execute("""INSERT INTO locations_cantonzug_flickr (id, name, location_type, latitude, longitude)
                                   VALUES (%(location_id)s,%(location_name)s,%(location_type)s,%(latitude)s,%(longitude)s)
                                   ON CONFLICT DO NOTHING;""", {'location_id': location_id,
                                                                'location_name': location_name,
                                                                'location_type': location_type,
                                                                'latitude': latitude,
                                                                'longitude': longitude
                                                                })

                        cursor.execute("""INSERT INTO media_objects_cantonzug_flickr (media_object_id, location_id, link, datetaken,
                                    pubdate, author_id, author, author_origin, title, description, views, faves, tags)
                                    VALUES (%(media_object_id)s,%(location_id)s,%(link)s,%(datetaken)s,%(pubdate)s,
                                    %(author_id)s,%(author)s,%(author_origin)s,%(title)s,%(description)s,%(views)s,
                                    %(faves)s,%(tags)s) ON CONFLICT DO NOTHING;""", {'media_object_id': media_object_id,
                                                                                    'location_id': location_id,
                                                                                    'link': link,
                                                                                    'datetaken': datetaken,
                                                                                    'pubdate': pubdate,
                                                                                    'author_id': author_id,
                                                                                    'author': author,
                                                                                    'author_origin': author_origin,
                                                                                    'title': title,
                                                                                    'description': description,
                                                                                    'views': views,
                                                                                    'faves': faves,
                                                                                    'tags': tags
                                                                                     })

        connection.commit()
        print("Step: populating db table {} with raw data - done.".format(target_table))
        '''
        Add in_boundary information
        '''
        if not re.search('trainingdata', target_table):
            DataProcessHandler.in_boundary(connection, target_table)
        '''
        Initiating the data processing; takes care of dominant users and bulk uploads
        '''
        DataProcessHandler.data_processing(connection, target_table)
        '''
        Initiating the text processing; takes care of stemming, spell correction, stop words
        '''
        DataProcessHandler.text_processing(connection, target_table)
        print("Population Process done.")

    @staticmethod
    def populate_table_foursquare(connection, target_table, source_directory):
        #iterate over each file in the given directory
        for file in os.listdir(source_directory):
            with open(os.path.join(source_directory, file), 'rt', encoding='utf-8') as f:
                data = json.loads(f.read())
                #iterate over every venue_id in the dict
                with connection.cursor() as cursor:
                    for venue in data:
                        #make sure the venue_ids match
                        if str(venue) == str(data[venue]['response']['venue']['id']):
                            media_object_id = data[venue]['response']['venue']['id']
                            venue_name = data[venue]['response']['venue']['name'].replace("'", '')
                            latitude = data[venue]['response']['venue']['location']['lat']
                            longitude = data[venue]['response']['venue']['location']['lng']
                            formatted_address = data[venue]['response']['venue']['location']['formattedAddress'][0].replace("'", '')
                            category_id = data[venue]['response']['venue']['categories'][0]['id']
                            category_name = data[venue]['response']['venue']['categories'][0]['name'].replace("'", '')
                            try:
                                description = data[venue]['response']['venue']['listed']['groups'][0]['items'][0]['description'].replace("'", '')
                            except (KeyError, IndexError):
                                description = ''
                            try:
                                rating = data[venue]['response']['venue']['rating']
                            except KeyError:
                                rating = 9999
                            verified = data[venue]['response']['venue']['verified']

                            cursor.execute("""INSERT INTO media_objects_cantonzug_foursquare (media_object_id, venue_name, latitude, longitude, 
                            formatted_address, category_id, category_name, rating, verified)
                            VALUES (%(media_object_id)s,%(venue_name)s,%(latitude)s,%(longitude)s,%(formatted_address)s,
                            %(category_id)s,%(category_name)s,%(rating)s,%(verified)s) 
                            ON CONFLICT DO NOTHING;""", {'media_object_id': media_object_id,
                                                          'venue_name': venue_name,
                                                          'latitude': latitude,
                                                          'longitude': longitude,
                                                          'formatted_address': formatted_address,
                                                          'category_id': category_id,
                                                          'category_name': category_name,
                                                          'rating': rating,
                                                          'verified': verified
                                                         })

                        elif str(venue) != str(data[venue]['response']['venue']['id']):
                            OperationHandler.print_error_format("""venue id: '{}' does not match with response id: '{}'
                                                                """.format(venue, data[venue]['response']['venue']['id']))
                            print("Exiting...")
                            exit()
            connection.commit()
            print("Processing Foursquare data from file '{}' done.".format(file))

    @staticmethod
    def write_vision_labels_to_db(connection):
        #check if the labels have already been added to a given row - skip if necessary
        #will add labels to all existing tables simultaniously
        '''
        Hint:
        A problem 'no results fetched' occurred here. It had something to do with looping over the cursor
        after a first sql command execution and in that loop trying to execute a second sql command.
        Problem could be solved by appending all needed sql commands to a list (like done before) and
        executing all of them after leaving the stated loop! BUT OTHER POSSIBILITIES ARE:
        - reading all the content of the cursor into a variable with fetchall() and freeing up the cursor for
          another command execution
        - potentially, the creation of an additional cursor object could do the trick as well.
        -> this was important because the vision label acquirement for an entire table takes ages, therefore only one
        execution and commit at the end of every table is very risky!
        '''
        #Returns dict of table names
        table_dict = PostgresqlHandler.selectable_table_names_gen()
        total_processed_rows = 0
        none_objects_vision = 10
        with connection.cursor() as cursor:
            for table in table_dict:
                unfinished = True
                table_name = table_dict[table]
                print("adding google cloud vision labels to table {}".format(table_name))
                '''
                differentiate between Instagram and Flickr tables
                '''
                while unfinished:
                    if re.search(r'instagram', table_name):
                        if re.search(r'trainingdata', table_name):
                            '''
                            select LINK not MEDIA_LINK because latter is expired! Image will be extracted from the browser
                            as can be seen in the following steps. Due to time constraints only the media objects which 
                            were already indentified as legit trainingsdata will be queried for image /vision labels + 1000!
                            '''
                            cursor.execute("""
                                           (SELECT media_object_id, link
                                           FROM media_objects_trainingdata_instagram
                                           WHERE vision_labels IS NULL                                                                 
                                              AND included_after_dataprocessing IS TRUE
                                              AND classification != 'None'
                                              AND classification != 'Borderline'
                                            )
                                            UNION ALL
                                           (SELECT media_object_id, link
                                            FROM media_objects_trainingdata_instagram
                                            WHERE vision_labels IS NULL                                                                 
                                              AND included_after_dataprocessing IS TRUE
                                              AND classification = 'None'
                                            ORDER  BY random()
                                            limit %(amount)s
                                            );                                
                                           """, {'amount': none_objects_vision})
                        else:
                            cursor.execute("""SELECT  media_object_id, media_link FROM media_objects_unionzug_instagram
                                          WHERE vision_labels IS NULL
                                          AND in_boundary IS TRUE
                                          AND included_after_dataprocessing IS TRUE
                                          ORDER  BY random();""")

                        row = cursor.fetchone()
                        if row is None:
                            unfinished = False
                            break
                        total_processed_rows += 1
                        '''
                        retrieving the total line count for better progress monitoring
                        '''
                        print("_"*30 + "ID: {}".format(row[0]) + "_"*10 + str(total_processed_rows) + "_"*10)
                        image_url = row[1]
                        media_object_id = row[0]
                        '''
                        if instagram_trainingsdata extract the image scr from the LINK (not MEDIA LINK) via
                        the following function:
                        '''
                        print("URL:\n {}".format(image_url))
                        if re.search(r'trainingdata', table_name):
                            try:
                                sauce = urllib.request.urlopen(image_url).read()
                                soup = bs.BeautifulSoup(sauce, "lxml")
                                table_soup = soup.find("meta", {"property": "og:image"}) #{"class": "FFVAD"}
                                print('Table src: ' + str(table_soup['content']))
                                image_url = table_soup['content']

                            except (PermissionError, TypeError, TimeoutError, urllib.error.HTTPError, ConnectionResetError) as e:
                                print('Error:')
                                print('{}'.format(e))
                                print('Media object id: {}'.format(media_object_id))
                                print('Image URL: {}'.format(image_url))
                                print('process continues...')

                        print("Requesting Vision labels...")
                        vision_labels = 'NO_DATA'
                        while True:
                            try:
                                vision_labels = VisionDataAddOnHandler.detect_labels_uri(image_url)
                                break
                            except Exception as e:
                                print("Vision API exception: {} :occurred. Retry in 10s...".format(e))
                                time.sleep(10)
                        '''
                        empty lists return false -> make sure to not include an emtpy list as vision labels into a row.
                        If the API does not return anything (due to error etc.) the same row will be queried again 
                        when the script runs again due to the vision_label field still being 'Null'
                        '''
                        if not vision_labels:
                            if re.search(r'instagram', table_name):
                                if re.search(r'trainingdata', table_name):
                                    cursor.execute("""UPDATE media_objects_trainingdata_instagram SET vision_labels = 'NO_DATA'
                                                    WHERE media_object_id = %(id)s AND vision_labels IS NULL;""", {'id': media_object_id})

                                else:
                                    cursor.execute("""UPDATE media_objects_unionzug_instagram SET vision_labels = 'NO_DATA'
                                                    WHERE media_object_id = %(id)s AND vision_labels IS NULL;""", {'id': media_object_id})
                            continue
                        '''
                        SQL: strings must always be surrounded by single quotes. Therefore all quotes inside the argument 
                        have to be double-quotes!
                        Instagram media_object_id has to be surrounded by single quotes as well -> it's a string due to '_'
                        '''
                        formatted_vision_labels = str(vision_labels).replace("'", '"')
                        if re.search(r'trainingdata', table_name):
                            cursor.execute("""UPDATE media_objects_trainingdata_instagram 
                                            SET vision_labels = %(labels)s
                                            WHERE media_object_id = %(id)s;""",
                                                           {'labels': formatted_vision_labels,
                                                            'id': media_object_id
                                                            })
                        else:
                            cursor.execute("""UPDATE media_objects_unionzug_instagram 
                                            SET vision_labels = %(labels)s
                                            WHERE media_object_id = %(id)s;""",
                                                            {'labels': formatted_vision_labels,
                                                             'id': media_object_id
                                                             })
                        connection.commit()

                    elif re.search(r'flickr', table_dict[table]):
                        cursor.execute("""SELECT  media_object_id, link FROM media_objects_cantonzug_flickr 
                                        WHERE vision_labels IS NULL
                                        AND in_boundary = true
                                        AND included_after_dataprocessing = true
                                        ORDER  BY random();""")
                        row = cursor.fetchone()
                        if row is None:
                            unfinished = False
                            break
                        total_processed_rows += 1
                        '''
                        retrieving the total line count for better progress monitoring, rouwcount is not a function but a 
                        read-only cursor attribute!
                        '''
                        # total_lines = cursor.rowcount
                        '''
                        returns a list of tuples with (id, link)
                        '''
                        # for index, row in enumerate(content, 1):
                        media_object_id = row[0]
                        # print("_" * 30 + "REQUEST {} OF {}".format(index, total_lines) + "_" * 30)
                        print("_"*30 + "ID: {}".format(media_object_id) + "_"*10 + str(total_processed_rows) + "_"*10)
                        '''
                        creation of the media_link to the actual image from the link
                        '''
                        tries = 0
                        while True:
                            try:
                                tries += 1
                                sauce = urllib.request.urlopen(row[1]).read()
                                break

                            except urllib.error.HTTPError as e:
                                if tries >= 3:
                                    break
                                print("HTTP error code {} occurred. Retry in 5s...".format(e.code))
                                time.sleep(5)
                        if tries >= 3:
                            continue

                        soup = bs.BeautifulSoup(sauce, "lxml")
                        table = soup.find("img", class_="main-photo")
                        if table is None:
                            OperationHandler.print_error_format("No class 'main-photo' found from image_url {}!".format(row[1]))
                            cursor.execute("""UPDATE media_objects_cantonzug_flickr SET vision_labels = 'NO_DATA'
                                                WHERE media_object_id = %(id)s AND vision_labels IS NULL;""", {'id': media_object_id})

                        else:
                            image_url = str('https:' + table['src'])
                            print('image_url acquired: {}'.format(image_url))
                            print("Requesting Vision labels...")
                            while True:
                                try:
                                    vision_labels = VisionDataAddOnHandler.detect_labels_uri(image_url)
                                    break
                                except Exception:
                                    print("Vision API exception occurred. Retry in 10s...")
                                    time.sleep(10)
                            '''
                            empty lists return false -> make sure to not include an emtpy list as vision lables into a row.
                            If the API does not return anything (due to error etc.) the same row will be queried again 
                            when the script runs again due to the vision_label field still being 'Null'
                            '''
                            if not vision_labels:
                                cursor.execute("""UPDATE media_objects_cantonzug_flickr SET vision_labels = 'NO_DATA'
                                                        WHERE media_object_id = %(id)s AND vision_labels IS NULL;""",
                                               {'id': media_object_id})
                                continue
                            '''
                            SQL: strings must always be surrounded by single quotes. Therefore all quotes inside the argument 
                            have to be double-quotes!
                            '''
                            formatted_vision_labels = str(vision_labels).replace("'", '"')
                            cursor.execute("""UPDATE media_objects_cantonzug_flickr SET vision_labels = %(labels)s
                                                    WHERE media_object_id = %(id)s AND vision_labels IS NULL;""",
                                           {'labels': formatted_vision_labels,
                                            'id': media_object_id
                                            })
                        connection.commit()
            print("-" * 60)
            print("-" * 60)
            print("Final Executing and Commit...")
            connection.commit()
            print("Done.")
            print("-" * 60)

    @staticmethod
    def exe_commit_commands(connection, commands):
        print("-"*60)
        print("Executing SQL command(s)...")
        start = time.time()
        with connection.cursor() as cursor:
            try:
                #check if commands is actually a list consisting of multiple command strings
                if isinstance(commands, (list,)):
                    print("Received: List of SQL commands")
                    for index, command in enumerate(commands):
                        cursor.execute(command)
                        # print("Executed command: {}".format(index))
                        connection.commit()
                #check if commands is actually just one single command as a string
                elif isinstance(commands, (str,)):
                    print("Received: single SQL string command")
                    cursor.execute(commands)
                    connection.commit()
            except Exception as e:
                print("Error occurred while handling {}: {} / {}".format(command, psycopg2.errorcodes.lookup(e.pgcode), psycopg2.errorcodes.lookup(e.pgcode[:2])))
                cursor.close()
        end = time.time()
        print("Time taken: {0:.2f} s".format((end-start)))
        print("-" * 60)

if __name__ == '__main__':
    '''
    Normal operation with user command input
    '''
    instance = OperationHandler()