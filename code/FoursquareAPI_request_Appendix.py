# -*- encoding: utf-8 -*-
#---------------------------------------------AUTHOR---------------------------------------------------------------------------------------
# Written by: Maximilian Hartmann
# Date: 17.11.2018
# Email: hartmanm@student.ethz.ch
# Cause: Master Thesis
# ENTIRE PROJECT CODE IS ALSO AVAILABLE ON GITHUB
# GitHub: https://github.com/Bellador
#-------------------------------------INFORMATION TO SCRIP USAGE----------------------------------------------------------------------------
#This scrip was not primarily intendet to be reused by other users. Therefore the legibility, comments and code adaptations are limited.
#This means that path names, credentials, file names etc. have to be changed to ones own machine.
# If questions arise please feel free to contact me under the above given E-Mail address.
#-------------------------------------------------------------------------------------------------------------------------------------------
#Used Foursquare Enpoints to query the Canton of Zug, Switzerland:
#(1) Regular venue endpoint
#(2) Premium venue enpoint
#----------------------------------------
import json, requests
import foursquareAPI_credentials as creds
import os
import time
import importlib.util
'''
INFORMATION:

venue search (regulard API endpoint) already returns information about all the venues in the 
research area, but key elements such as 'verified' are missing in this regular Foursquare API request.
To get check-in's, verified data etc. one has to use the premium endpoint 'venues details' which only allows 50 request
per day!
Therefore the whole data acquiry process has to be smart, to always request a new set of 50 venues each day.
'''
#Outdoors & Recreation
#categoryId="4d4b7105d754a06377d81259"

# Canton Zug
# sw = "47.070589497943686,8.382568359375"
# ne = "47.25686404408872,8.718338012695312"
# acts as a container for the json results from all sectors
class FoursquareHandler:
    '''
    Import the bounding boxes from the 'bb_dict_cantonzug.py' file which was also used in the Flickr API mining
    '''
    spec = importlib.util.spec_from_file_location("bb_dict_cantonzug", "bounding_boxes_dictionary_cantonzug.py")
    bbox = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bbox)
    spatial_search_data_cantonzug = bbox.spatial_search_data_cantonzug

    #regular endpoint
    venue_search_url = "https://api.foursquare.com/v2/venues/search"
    venue_search_unique_ids_file = "INSERT_PATH_HERE"

    @staticmethod
    def get_unique_venue_ids():
        search_result_dict = {}
    # same as flickr API, the request has to be divided into smaller sub-requests due to a maximum return of 50 results!
        for sector in FoursquareHandler.spatial_search_data_cantonzug:
            #extract sw and ne parameters
            sw_1, sw_2, ne_1, ne_2 = FoursquareHandler.spatial_search_data_cantonzug[sector][0].split(',')
            sw = str(sw_2) + "," + str(sw_1)
            ne = str(ne_2) + "," + str(ne_1)
            params = dict(
                client_id=creds.client_ID,
                client_secret=creds.client_secret,
                sw="{}".format(sw),
                ne="{}".format(ne),
                intent="browse",
                categoryId="4d4b7105d754a06377d81259",
                v=20181116,
                limit=50
            )

            resp = requests.get(url=FoursquareHandler.venue_search_url, params=params)
            data = json.loads(resp.text)
            # print(json.dumps(data, indent=2))
            search_result_dict[sector] = data
            #print number of found venues
            print("_"*30 + "SEKTOR: {}".format(sector) + "_"*30)
            print(len(data['response']['venues']))
            print("_" * 70)
        '''
        get all venue id's and create a set with unique entries - meaning no dublicates
        '''
        ids = []
        for sector in search_result_dict:
            for counter_, venue in enumerate(search_result_dict[sector]['response']['venues']):
                ids.append(venue['id'])
        unique_ids = set(ids)
        print("Length of unique ids: {}".format(len(unique_ids)))

        with open(FoursquareHandler.venue_search_unique_ids_file, 'wt', encoding='utf-8') as f:
            for id in unique_ids:
                f.write(id)
                f.write('\n')

        print("Output file created. - Done.")

    @staticmethod
    def premium_search_venue_details():
        '''
        1. retrieve the filename that is in the directory /FoursquareAPI/unique_ids_to_request/
        2. read all lines - request the premium venue details endpoint for the first 50 lines/ids, write the
        remaining ids to a new file, with a +1 higher integer suffix e.g venue_search_unique_ids_1.txt
        3. safe the output in a final venue_details file.
        --> repeat everyday till all 379 unique ids have been requested.
        :return:
        '''
        venue_details_dict = {}
        wd = "INSERT_PATH_HERE"
        '''
        search the directory and find the file with the highest integer extension at the end, symbolising the latest file
        '''
        suffix_list = []
        for file in os.listdir(wd):
            suffix_list.append(file[24:-4])
        try:
            work_file_suffix = max(suffix_list)
        except ValueError:
            work_file_suffix = ''
        work_filepath = str(wd + "venue_search_unique_ids_" + str(work_file_suffix) + ".txt")
        if work_file_suffix == '':
            output_filepath = str(wd + "venue_search_unique_ids_" + str(1) + ".txt")
        elif work_file_suffix != '':
            output_filepath = str(wd + "venue_search_unique_ids_" + str(int(work_file_suffix) + 1) + ".txt")
        '''
        retrieving the first 50 entries
        '''
        id_batch = []
        retained_ids = []
        with open(work_filepath, 'rt') as f:
            for index, line in enumerate(f, 1):
                if index <= 50:
                    id_batch.append(line.strip('\n'))
                else:
                    retained_ids.append(line)
        '''
        re-writing the file - removing the 50 batched ids and only writing the retained ids
        '''
        with open(output_filepath, 'wt') as f_new:
            for id in retained_ids:
                f_new.write(id)

        print("Created new file {}".format(output_filepath[86:-4]))

        '''
        making the GET request for the batched 50 ids to the Foursquare API
        '''
        for index, id in enumerate(id_batch, 1):
            while True:
                resp = requests.get(url='''https://api.foursquare.com/v2/venues/{venue_id}
                ?client_id={client_id}&client_secret={client_secret}&v={version}'''
                                    .format(venue_id=id, client_id=creds.client_ID, client_secret=creds.client_secret, version=20181117))
                data = json.loads(resp.text)
                print("_"*30 + "NR: {} of 50 - id: {}".format(index, id) + "_"*30)
                print(json.dumps(data, indent=2))
                print("_" * 70)
                '''
                checking for quota exceeded alert message
                '''
                try:
                    print(data['meta']['errorDetail'])
                    if data['meta']['code'] == 500:
                        print("experienced server error 500 - retry in 5s...")
                        time.sleep(5)
                        continue
                    #else it is either a quota exceeded error or something else - exit the script either way
                    else:
                        exit()
                #key error only occures if the error message is NOT returned
                except KeyError:
                    '''
                    adding the response to the dictionary container
                    '''
                    venue_details_dict[id] = data
                    break

        '''
        writing the entire venue_details_dict to a new file
        '''
        response_outputfilepath = "INSERT_PATH_HERE"
        if work_file_suffix == '':
            response_outputfilename = "batchNr_1.txt"
        elif work_file_suffix != '':
            response_outputfilename = "batchNr_{}.txt".format(str(int(work_file_suffix)+1))

        with open(str(response_outputfilepath+response_outputfilename), 'wt', encoding='utf-8') as response_file:
            response_file.write(json.dumps(venue_details_dict, indent=1))

        print("Batch file: {} - created.".format(response_outputfilename))

    @staticmethod
    def read_batch_files():
        with open("INSERT_PATH_HERE", 'rt', encoding='utf-8') as f_batch:
            data = json.loads(f_batch.read())
        print(json.dumps(data, indent=1))


'''
Aquire all unique venue ids from all input bounding boxes 
'''
# FoursquareHandler().get_unique_venue_ids()
'''
get special venue data from each unique venue ID from the premium Foursquare API endpoint which allows 50 calls / day
'''
FoursquareHandler().premium_search_venue_details()
# FoursquareHandler().read_batch_files()
