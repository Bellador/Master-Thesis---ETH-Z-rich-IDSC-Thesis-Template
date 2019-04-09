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

#add the variable names from bb_dict directly to this files namespace; therefore something like bb_dict.xy is not needed
#but each variable can directly be accessed via its original name. Not really recommended, because namespaces can overlap!
#from bb_dict import *

import bb_dict_cantonzug
import flickrapi
import flickrAPI_credentials
import json
from collections import Counter
from prettytable import PrettyTable
import re
import requests
import sys
import bs4 as bs
import os
import datetime
import urllib.request
import math
import copy
import time

#---------------------INFO--BBOX--ARGUMENT-------------------------
#'section1': [min_long, min_lat, max_long, max_lat]
# HAS TO BE A COMMA DELIMITED LIST OF ONE SINGLE STRING!!!!
#------------------------------------------------------------------
#Search is delimited by the accuracy level, the geo_context
#------------------------------------------------------------------

'''
divided into smaller areas to get more results!!
See for reverence: https://www.flickr.com/groups/51035612836@N01/discuss/72157680701360093/!!!
'''

# spatial_search_data_cantonzug = {
#     '1': ['8.381881713867188,47.19951070111655,8.45672607421875,47.250805241535076'],
#     '2': ['8.44573974609375,47.19297874437906,8.521957397460938,47.23915174981453'],
#     '3': ['8.507537841796875,47.184112659842015,8.589935302734375,47.22982711058905'],
#     '4': ['8.570709228515625,47.16590921476792,8.633880615234375,47.21676985912017'],
#     '5': ['8.624267578125,47.135556272359196,8.707351684570312,47.18084583431804'],
#     '6': ['8.612594604492188,47.089760603235646,8.693618774414062,47.141628247621185'],
#     '7': ['8.563156127929688,47.12808212013255,8.628387451171875,47.170110563639014'],
#     '8': ['8.514404296875,47.154703993097705,8.57551574707031,47.186912636007655'],
#     '9': ['8.45672607421875,47.164041841932914,8.517837524414062,47.19717795172789'],
#     '9_1': ['8.455352783203125,47.18364598278901,8.482818603515625,47.19764450981031'],
#     '9_2': ['8.4814453125,47.184112659842015,8.518524169921875,47.19764450981031'],
#     '9_3': ['8.453292846679688,47.16777652195966,8.517837524414062,47.18737928434322'],
#     '10': ['8.390121459960938,47.16590921476792,8.460845947265625,47.20557536955536'],
#     '11': ['8.403167724609375,47.11873795272715,8.459472656249998,47.170110563639014'],
#     '12': ['8.45123291015625,47.12434465012191,8.570022583007812,47.17151093941472'],
#     '12_1': ['8.512344360351562,47.150501425705635,8.568649291992186,47.1705773596669'],
#     '12_2': ['8.507537841796875,47.12200859801305,8.572769165039062,47.153303174225954'],
#     '12_3': ['8.45123291015625,47.12107414844729,8.517837524414062,47.17104415159213'],
#     '12_4': ['8.50839614868164,47.15149372853215,8.574743270874023,47.16701793632996'],
#     '13': ['8.545989990234375,47.0747983802725,8.620147705078125,47.13835880864309'],
#     '14': ['8.445053100585938,47.086020441438556,8.550109863281248,47.13275358836622']
# }

class FlickrMining:
    #Key and Secret credentials were removed. Personal credentials can be entered in unicode format below.
    def __init__(self, kwargs):
        self.now = datetime.datetime.now()
        self.flickr = flickrapi.FlickrAPI(flickrAPI_credentials.api_key, flickrAPI_credentials.api_secret, format='parsed-json')
        self.arguments = kwargs
        self.result_dic = {}
        if kwargs:
            #kvars requires the passing of a dictionary in the object creation
            #Not all of these functions have to be executed. Uncomment everything that shall not be executed
            # (just some functions rely on the output of others!)
            print('search in working...')
            self.search_process(self.arguments)
            print('1. eliminating duplicates due to search area overlap...')
            print('processing data...')
            # creates unique set of all ID's (no duplicates). Still needed for get_user_info
            self.unique_set = self.create_unique_set()
            self.creating_DB_input()
            print('done.')

        else:
            print('No arguments received')

    def search_process(self, kwargs):
        # apparently is that the max and not 500 like mentioned in the manual
        photos_per_page = 250
        for sector in kwargs['bbox']:
            print("-"*60)
            print("currently handling sector: {}".format(sector))
            print("-"*60)
            try:
                index_name = str(sector) + "_Page1"
                # dynamic creation in next step that includes other parameters!
                self.result_dic[index_name] = self.flickr.photos.search(bbox=kwargs['bbox'][sector],
                                                                        accuracy=kwargs['accuracy'], per_page=photos_per_page)
            except Exception as e:
                print("-"*60)
                print("")
                print("Error occurred: {}".format(e))
                print("")
                print("sleeping 5s...")
                print("-" * 60)
                time.sleep(5)
            if self.result_dic[index_name]['photos']['pages'] != 1 and self.result_dic[index_name]['photos']['pages'] != 0:
                print('Amount of result pages in sector' + str(sector) + ' : ' + str((self.result_dic[index_name]['photos']['pages']-1)))
                for page in range(self.result_dic[index_name]['photos']['pages'] - 1):
                    while True:
                        try:
                            index_name_2 = str(sector)+'_Page'+str(page+1)
                            print(str(index_name_2))
                            self.result_dic[index_name_2] = self.flickr.photos.search(bbox=kwargs['bbox'][sector],
                                                                                      accuracy=kwargs['accuracy'],
                                                                                      per_page=photos_per_page,
                                                                                      page=(page+2)
                                                                                      )
                            break
                        except Exception as e:
                            print("-" * 60)
                            print("")
                            print("Error occured: {}".format(e))
                            print("")
                            print("retry same page in 5s...")
                            print("-" * 60)
                            time.sleep(5)

            else:
                print("Amount of pages is only at: " + str(self.result_dic[index_name]['photos']['pages']))
                continue

    def create_unique_set(self):
        print('_____________________________________________')
        print('Number of search results per section')
        # displays the amount of search results per section
        [print(str(element) + ' : ' + str(len(self.result_dic[element]['photos']['photo']))) for element in self.result_dic]
        # creating a set out of all searches over all sectors to eliminate dublicates
        # creating a set of only the ID's
        liste_ = []
        [liste_.append(ID['id']) for section in self.result_dic
                                    for ID in self.result_dic[section]['photos']['photo']]
        print("Len liste original: {}".format(len(liste_)))
        set_ = set(liste_)
        print("Len set: {}".format(len(set_)))
        #set_ is a list with all the unique ID's across all sections. It is not a slimmed result_dic!
        return set_

    #Data processing: upper 5% of high and low frequency user exclusion (exclusion of lower end has been uncommented -> is possible)
    #'REMOVE' FUNCTION INSTEAD OF 'DEL' DUE TO CHANGE OF INDEX WHEN ELIMINATING AN ELEMENT FROM THE LIST WHILE ITERATING OVER IT!
    def data_processing(self, user_ids):
        #threshold * 100 in percent which will be excluded from further analysis according to the user total post count in the region
        # upper_threshold = 0.05
        lower_threshold = 0.08
        users_to_exclude = []
        #if just = is used, then they still REFER to one another!! They are only individual copies "copy.deepcopy(self.result_dic)"!
        processed_result_dic = copy.deepcopy(self.result_dic)
        c = Counter(user_ids)
        #transform dic into list of tuple
        postcount_list_tuple = []

        for id_ in c:
            postcount_list_tuple.append((id_, c[id_]))

        #sort user_postcount_tuple after their count (not sure if even necessary)
        sorted_user_postcount = sorted(postcount_list_tuple, key = lambda tuple: tuple[1], reverse=True)
        len_user_postcount = len(sorted_user_postcount)
        print('User Length: ' +str(len_user_postcount))
        cut_off_top = math.ceil(upper_threshold * len_user_postcount)
        # might not be necessary because most users only posted once (would be a random cut off)
        # cut_off_bottom = math.ceil(lower_threshold * len_user_postcount)
        # print('Cut off top and bottom: ' + str(cut_off_top) + ' ¦ ' + str(cut_off_bottom))
        print('Cut off bottom: ' + str(cut_off_top))

        if cut_off_top != 1:
            for element in sorted_user_postcount[:cut_off_top]:
                users_to_exclude.append(element[0])
        else:
            users_to_exclude.append(sorted_user_postcount[0][0])
        # if cut_off_bottom != 1:
        #     for element in sorted_user_postcount[-(cut_off_bottom):]:
        #         users_to_exclude.append(element[0])        #
        # else:
        #     users_to_exclude.append(sorted_user_postcount[-1:][0])
        #following: extract these users from the result dic or unique_set list for further analysis

        photo_ids = []
        actual_eliminated_users = []
        duplicates_found = []
        processed_photos_counter = 0
        for item in self.result_dic:
            for count, photo in enumerate(self.result_dic[item]['photos']['photo']):
                if photo['owner'] in users_to_exclude:
                    if photo['id'] in photo_ids:
                        duplicates_found.append(photo['id'])
                    actual_eliminated_users.append(photo['owner'])
                    processed_result_dic[item]['photos']['photo'].remove(photo)
                    print('Removed: {}'.format(photo))
                elif photo['id'] in photo_ids:
                    duplicates_found.append(photo['id'])
                    processed_result_dic[item]['photos']['photo'].remove(photo)
                photo_ids.append(photo['id'])
                processed_photos_counter += 1

        print('processed photos: {}'.format(processed_photos_counter))
        #create unique pic ID list from processed result_dic
        unique_ID_processed_result_dic = []
        backup_photo_counter = 0
        for item in processed_result_dic:
            for photo in processed_result_dic[item]['photos']['photo']:
                unique_ID_processed_result_dic.append(photo['id'])
                backup_photo_counter += 1
        print('counter check processed photos: {}'.format(backup_photo_counter))

        if processed_photos_counter != backup_photo_counter:
            processed_photos_counter = backup_photo_counter

        ID_controll_result_dic = []

        for item in self.result_dic:
            for photo in self.result_dic[item]['photos']['photo']:
                ID_controll_result_dic.append(photo['id'])

        actual_eliminated_users = set(actual_eliminated_users)
        print('actual excluded users: ' + str(actual_eliminated_users))
        print('processed photos: ' + str(processed_photos_counter))

        return processed_result_dic, processed_photos_counter, unique_ID_processed_result_dic

    def word_frequency_analysis(self):
        all_words_in_title = [title['title'].split() for section in self.processed_result_dic
                                                         for title in self.processed_result_dic[section]['photos']['photo']]

        flattened_list = [word for element2 in all_words_in_title
                                    for word in element2]
        c = Counter(flattened_list)
        #console output with PrettyTable
        pt = PrettyTable(field_names=['Word', 'Count'])
        for count, element3 in enumerate(c.most_common(20)):
            pt.add_row(element3)
            pt.align['Count'], pt.align['Word'] = 'l', 'r'
        print(pt)

    def retrieve_pictures(self, filename):
        filename = str(filename)
        #building URL's to the pictures with the model: http://flickr.com/photo.gne?id= + ID from the location list which includes the ID's
        id_url_tuple_list = []
        lines = 0
        with open(filename, 'rt', encoding='utf-8') as file:
            for line in file:
                line_striped = str(line.strip())
                # print(line_striped)
                url = re.search(r'(https?://www\.flickr\.com/photos/[^,]*[/]{1}[^,]*[/]{1})(?=,)', line_striped)
                # url = re.search(r'(https://www.flickr[.]{1}com/photos/)', line_striped)
                id_1 = re.match(r'[^,]*[,][^,]*[,][^,]*[,][^,]*[,][^,]*[,]([\d]+)(?=,)', line_striped)
                lines += 1
                try:
                    id_url_tuple_list.append((id_1.group(1), url.group(1)))
                except AttributeError:
                    continue

        for counter, tuple_ in enumerate(id_url_tuple_list):
            try:
                file_path = 'C:/Users/Max/Documents/ETH/Master_Thesis/Code_hub/Output/retrieved_flickr_pics/'
                file_name = str(file_path) + str(tuple_[0]) + '.jpg'
                sauce = urllib.request.urlopen(tuple_[1]).read()
                soup = bs.BeautifulSoup(sauce, "lxml")
                table = soup.find("img", class_="main-photo is-hidden")

                print('Table src: ' + str(table['src']))
                create_pic_url = str('https:' + table['src'])
                print('pic_url: ' + str(create_pic_url))
                urllib.request.urlretrieve(create_pic_url, filename=str(file_name))
                print('{} pictures of {} fetched'.format((counter + 1), lines))
            except (PermissionError, TypeError, TimeoutError):
                print('Error occurred, process continues...')


    #CSV file need to contain: ID,Latitude,Longitude,NAME,URL,PhotoID,Owner Name,UserID,DateTaken,
    # UploadDate,Views(flickr.stats.getPhotoStats),Tags,MTags
    def creating_Tag_Map_input(self):
        dic_for_all_IDs_with_CSV_data = {}
        #photo ids are from post processing
        mismatch_counter = 0
        filepath = str('INPUT_PATH_HERE' + str(self.now.strftime('%M %H %d %m %Y')) + '.txt')
        with open(filepath, 'w') as file:
            file.write('ID,Latitude,Longitude,NAME,URL,PhotoID,Owner,UserID,DateTaken,UploadDate,Views,Tags,MTags\n')
        #sort unique_ID_processed so that the resulting dictionary keys are in order
        sorted_unique_ID_processed = sorted(self.unique_set)

        for counter, ID in enumerate(sorted_unique_ID_processed): #self.unique_ID_processed !!!!BIGCHANGE!!!
            if counter <= len(self.unique_set):
                #1. create temporary list / bucket to hold info for current ID and append counter for numberation.
                # Fetch all necessary data via with .getInfo
                bucket = [(counter+1)]
                temp = self.flickr.photos.getInfo(photo_id =ID)
                getInfo = temp['photo']
                #2. Append Lat / Long to bucket
                bucket.append(getInfo['location']['latitude'])
                bucket.append(getInfo['location']['longitude'])
                #3.1 Append Title and description (maybe concatenate them to one string, so everything gets analysed)
                # and append to bucket
                title = getInfo['title']['_content']
                #3.2 strip all the ',' commas out of the text to prevent the csv file from being falsely read!
                title_stripped = title.replace(',','')
                bucket.append(title_stripped)
                #4. Append URL
                url = getInfo['urls']['url'][0]['_content']
                bucket.append(url)
                #5. get PhotoID, Username, UserID
                photo_ID_getInfo = getInfo['id']
                if ID == photo_ID_getInfo:
                    bucket.append(photo_ID_getInfo)
                else:
                    mismatch_counter += 1

                username = getInfo['owner']['username']
                userID = getInfo['owner']['nsid']
                bucket.append(username)
                bucket.append(userID)
                #6. Append date taken and upload date
                date_taken = getInfo['dates']['taken']
                date_uploaded = getInfo['dates']['posted']
                date_taken = datetime.datetime.strptime(date_taken, '%Y-%m-%d %H:%M:%S').strftime('%m/%d/%Y %H:%M:%S')
                date_uploaded = date_taken
                bucket.append(date_taken)
                bucket.append((date_uploaded))
                #7. Append views
                views = getInfo['views']
                bucket.append(views)
                #8. Append Tags and MTags
                tags = []
                tag_string = ';'
                for tag in getInfo['tags']['tag']:
                    tags.append(tag['raw'])
                #create ; separated string out of all tags in list tags
                for element in tags:
                    tag_string += str(element)
                    tag_string += ';'
                bucket.append(tag_string)
                #append emtpy element for MTags
                dic_for_all_IDs_with_CSV_data[ID] = bucket

                print('{} of {} done.'.format((counter+1), len(sorted_unique_ID_processed))) #CHANGE TO PHOTO_IDS

            else:
                break

        #write total output into a .txt file as ouput
        with open(filepath, 'a', encoding='utf-8') as file:
            for key in dic_for_all_IDs_with_CSV_data:
                for element in dic_for_all_IDs_with_CSV_data[key]:
                    try:
                        file.writelines(str(element)+str(','))
                    except UnicodeEncodeError:
                        print('UnicodeEncodeError')
                        file.write(str(','))
                file.write('\n')

    #creating a CSV file with all the necessary information about the media object, user
    #location information will be acquired seperatly via Google Geocoding API
    #Important: strip all text input if commas!
    def creating_DB_input(self):
        accuracy = self.arguments['accuracy']
        dic_for_all_IDs_with_CSV_data = {}
        mismatch_counter = 0
        filepath = str('''INSERT_PATH_HERE'''.format(accuracy) + str(self.now.strftime('%M %H %d %m %Y')) + '.txt')
        with open(filepath, 'w') as file:
            file.write('''ID,latitude,longitude,title,description,url,media_object_id,author_id,
            author,author_origin,dateTaken,uploadDate,views,faves,tags\n''')
        #sort unique_set so that the resulting dictionary keys are in order
        '''
        Here the unprocessed media objects are intentionally used - which means no adjustments to high frequency users 
        has yet been made. This is because the netlytic data is the same, filtering is done later via SQL
        '''
        sorted_unique_ID = sorted(self.unique_set)
        for counter, ID in enumerate(sorted_unique_ID, 1):
            tries_tracker = 0
            while True:
                tries_tracker += 1
                if tries_tracker > 3:
                    time.sleep(10)
                    print("-" * 60)
                    print("")
                    print("Over 3 tries.. sleeping for 10s")
                    print("")
                    print("-" * 60)
                    break
                '''
                1. create temporary list / bucket to hold info for current ID and append counter for pagination.
                Fetch all necessary data via with .getInfo
                '''
                bucket = [counter]
                try:
                    temp = self.flickr.photos.getInfo(photo_id =ID)
                except Exception as e:
                    print("-" * 60)
                    print("")
                    print("Error occured: {}".format(e))
                    print("sleeping for 5s")
                    print("")
                    print("-" * 60)
                    time.sleep(5)
                    continue
                getInfo = temp['photo']
                '''
                2. Append Lat / Long to bucket
                '''
                bucket.append(getInfo['location']['latitude'])
                bucket.append(getInfo['location']['longitude'])
                '''
                3.1 Append Title and description (maybe concatenate them to one string, so everything gets analysed)
                and append to bucket
                '''
                title = getInfo['title']['_content']
                description = getInfo['description']['_content']
                '''
                3.2 strip all the ',' commas out of the text to prevent the csv file from being falsely read!
                '''
                title_stripped = title.replace(',','').replace('\n', '')
                description_stripped = description.replace(',', '').replace('\n', '')
                bucket.append(title_stripped)
                bucket.append(description_stripped)
                '''
                4. Append URL
                '''
                url = getInfo['urls']['url'][0]['_content']
                bucket.append(url)
                '''
                5. get PhotoID, Username, UserID, user origin
                '''
                photo_ID_getInfo = getInfo['id']
                if ID == photo_ID_getInfo:
                    bucket.append(photo_ID_getInfo)
                else:
                    mismatch_counter += 1

                author_id = getInfo['owner']['nsid']
                author = getInfo['owner']['username']
                author_stripped = author.replace(',', '').replace('\n', '')
                author_origin = getInfo['owner']['location']
                author_origin_stripped = author_origin.replace(',', '').replace('\n', '')
                bucket.append(author_id)
                bucket.append(author_stripped)
                bucket.append(author_origin_stripped)
                '''
                6. Append date taken and upload date
                '''
                date_taken = getInfo['dates']['taken']
                date_uploaded = getInfo['dates']['posted']
                date_taken = datetime.datetime.strptime(date_taken, '%Y-%m-%d %H:%M:%S').strftime('%m/%d/%Y %H:%M:%S')
                date_uploaded = date_taken
                bucket.append(date_taken)
                bucket.append((date_uploaded))
                '''
                7. Append views and favorites (similar to likes by Instagram)
                '''
                views = getInfo['views']
                faves = getInfo['isfavorite']
                bucket.append(views)
                bucket.append(faves)
                '''
                8. Append Tags and MTags
                '''
                tags = []
                tag_string = ';'
                for tag in getInfo['tags']['tag']:
                    tags.append(tag['raw'].replace(',', ''))
                #create ; separated string out of all tags in list tags
                for element in tags:
                    tag_string += str(element)
                    tag_string += ';'
                bucket.append(tag_string)
                #append emtpy element for MTags
                dic_for_all_IDs_with_CSV_data[ID] = bucket
                if mismatch_counter != 0:
                    print("-" * 60)
                    print("")
                    print("Mismatch_counter : {}".format(mismatch_counter))
                    print("sleeping for 5s")
                    print("")
                    print("-" * 60)

                print('{} of {} photos.getInfo requests processed'.format(counter, len(sorted_unique_ID)))
                break
            '''
            write total output into a .txt file as ouput
            '''
        with open(filepath, 'a', encoding='utf-8') as file:
            for key in dic_for_all_IDs_with_CSV_data:
                for element in dic_for_all_IDs_with_CSV_data[key]:
                    try:
                        file.writelines(str(element)+str(','))
                    except UnicodeEncodeError:
                        print('UnicodeEncodeError')
                        file.write(str(','))
                file.write('\n')

search = FlickrMining({'bbox' : bb_dict_cantonzug.spatial_search_data_cantonzug, 'accuracy' : 12})


