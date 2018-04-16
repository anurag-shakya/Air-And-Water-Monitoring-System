
import csv
import os
from requests import get
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from sklearn.ensemble import RandomForestRegressor




def water_quality():
	
	# I am converting the whole address to latitude and longitude (geo-coordinates) using google geocoding
	# Hence once Refining Actual Dataset "IndiaAffectedWaterQualityAreas.csv" I created "IndiaAffectedWaterQualityAreas_refined_dataset.csv"
	# That's why the following code is made as comment as now we have refined dataset

    '''
    def req_address_formacxt_for_url(req_add):
        req_add = re.sub(r'[^\w]', ' ', req_add)
        req_add = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", req_add)

        req_add = req_add.replace(" ", "+")
        req_add = req_add.replace("_", "")
        return req_add

    try:
        os.remove('delete_me.csv')
    except:
        pass

    try:
        os.remove('api_usable_addresses.csv')
    except:
        pass

    temp_address = [0, 0, 0]
    with open('IndiaAffectedWaterQualityAreas.csv', "r") as csv_data, open('delete_me.csv', "a",
                                                                            newline='') as delete_me_data:
        read_object = csv.reader(csv_data)
        write_object = csv.writer(delete_me_data)
        i = 0
        for x in read_object:

            if (i == 0):
                i += 1
                write_object.writerow(["Year", "Address", "Quality Parameter"])

            else:

                temp_address[0] = x[3]
                temp_address[1] = x[2]
                temp_address[2] = x[1]

                address = temp_address[0] + ", " + temp_address[1] + ", " + temp_address[2]
                address = req_address_format_for_url(address)

                write_object.writerow([x[0], address, x[7]])

    with open('delete_me.csv', "r") as csv_data, open('api_usable_addresses.csv', "a", newline='') as delete_me_data:
        read_object = csv.reader(csv_data)
        write_object = csv.writer(delete_me_data)

        for x in read_object:

            if x[1].find("++") == -1:
                write_object.writerow([x[0], x[1], x[2]])

            else:
                if x[1].find("+++") == -1:
                    x[1] = x[1].replace("++", ",+")
                    write_object.writerow([x[0], x[1], x[2]])


                else:
                    x[1] = x[1].replace("+++", ",+")
                    write_object.writerow([x[0], x[1], x[2]])
    os.remove('delete_me.csv')

    base_url = "https://maps.googleapis.com/maps/api/geocode/json?address="
    keys = ["AIzaSyC2RF19jRuODujmLOcDnhvGrdbEUIWJn4A", "AIzaSyCdx51nxvTIdyFAMeh_sbZbo5V0aOgjZQ4",
            "AIzaSyCI2dZSOWI8pwqOgUcJwV_BLaQFqbs9alU", "AIzaSyDruQx7-KiH2cy0uww51LAyJGiiB-vya0I",
            "AIzaSyBQVviYMPeIfzI4p3q4unvsXtxjTPYaxx0",
            "AIzaSyAPXCBFTZBXngvgS16O6cStcZP9q81Nvmo""AIzaSyBQppfFm5yX5Fp3Xjkl4XCSfTWlK_1MhIk",
            "AIzaSyD7lNNyDcGlEFd23AbhfH17bUIMN7IPHc8", "AIzaSyC_cVlg7XQpChPYaC-RjEC8QZreMeHfrTg",
            "AIzaSyDmoTl7ZeQSGqlV5Fc_MI4JrCxrl_K9wFs", "AIzaSyArzDXyAoYwCJNa-fXN4-EBBKIlIQ7c0kQ",
            "AIzaSyAtBABIDM8OFssqn7FoRYqVLaZAxgf07PU", "AIzaSyBXNqYxYnjCiaYwata30gsE6Bk256-86U4",
            "AIzaSyB4dhdu4kuk_r5uBfy4KugIFEowONXAW5A"]
    key = "AIzaSyARzIHeXrp8jC4IYP-Blygc1KilMrIwX9w"
    key = keys[0]
    with open('api_usable_addresses1.csv', "r") as csv_data, open('water_quality_dataset.csv', "a",
                                                                  newline='') as real_dataset:
        read_object = csv.reader(csv_data)
        write_object = csv.writer(real_dataset)

        i = 0
        fails = 0
        q = 1
        stop_flag = 0
        set_counter = 0
        for x in read_object:
            if i < 180000:
                i += 1
                continue

            if (i == 0):
                write_object.writerow(["Year", "Latitude", "Longitude", "Quality Parameter"])

            else:
                try:
                    if (i % 850) == 0 and q < 13:

                        key = keys[q]
                        q += 1
                        if (q == 13):
                            set_counter += 1

                    request_url = base_url + x[1] + "&key=" + key
                    response = get(request_url).json()

                    lat = float(response["results"][0]["geometry"]["location"]["lat"])
                    long = float(response["results"][0]["geometry"]["location"]["lng"])
                    write_object.writerow([int(x[0]), lat, long, x[2]])
                    print(i)

                except:
                    fails += 1
                    pass
            i += 1
            if set_counter == 1:
                stop_flag += 1
                if stop_flag == 850:
                    break

    '''





    menu1 = True
    while (menu1):
        print("""\n\n  Water Quality Monitor

        1) Ground Water			
        2) River Water
        3) Go to Main Menu
        4) exit
       """)
        option = str(input())

        if option == '1':
			
			# Actual Dataset name : IndiaAffectedWaterQualityAreas.csv
			# Dataset after preprocessing(Refining) : IndiaAffectedWaterQualityAreas_refined_dataset.csv
			
            # loading dataset
            dataset = pd.read_csv('IndiaAffectedWaterQualityAreas_refined_dataset.csv')

            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, 3].values

            # Splitting the dataset into the Training set and Test set

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=86)

            # feature scaling

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)

            

                    # Model 1: Multiclass Logistic Classification
            logisticregression = LogisticRegression()

                    # Fitting Multiclass Logistic Classification to the Training set
            logisticregression.fit(X_train, y_train)

                    # Predicting the Test set results
            y_pred = logisticregression.predict(X_test)

            print("\nModel trained successfully \n ")
            print("with accuracy : %.4f" %(logisticregression.score(X_test, y_test))*100)
            # plot between actual labels(y) -vs- predicted labels(y)
            plt.title('Predictions for Ground Water')
            plt.plot(y_test[:50], color='blue', label='True labels')
            plt.plot(y_pred[:50], 'r--', label='Predicted labels')
            plt.xlabel('No. of Test Examples ---->')
            plt.ylabel('    Classes ---->')
            plt.grid(True)
            plt.show()

               



        elif option == "2":
		
			# In "river_quality_dataset.csv", a column "WQI" : river water Quality is evaluted, which would be the labelled output 
			# river water Quality is evaluted with given pollutant in each row in the dataset using Standard formula used be Govt. of India
			# after preprocessing "river_quality__refined_dataset.csv" will be created as the usable dataset
			
            wn = [0.3723, 0.2190, 0.3723, 0.0412, 0.371];
            wn_sum = sum(wn)
            sn = [5, 7.5, 5, 45, 300]
            ideal = [14.6, 7, 0, 0, 0]

            try:
                os.remove('river_quality__refined_dataset.csv')
            except:
                pass

            with open('river_quality_dataset.csv', "r") as csv_data, open('river_quality__refined_dataset.csv', "a", newline='') as delete_me_data:
                read_object = csv.reader(csv_data)
                write_object = csv.writer(delete_me_data)
                line = 0
                for x in read_object:

                    if (line == 0):
                        line += 1
                        write_object.writerow(
                            ["Year", "State", "DO", "PH", "BOD", "Nitrate", "Conductivity", "Water Quality"])
                    elif (line < 3):
                        line += 1
                        continue

                    else:
                        '''
                        print(line)
                        line += 1
                        '''

                        do = float(x[9])
                        ph = float(x[12])
                        bod = float(x[18])
                        nitrate = float(x[21])
                        conductivity = float(x[15])

                        q = [0, 0, 0, 0, 0]
                        q[0] = (((do - ideal[0]) / (sn[0] - ideal[0])) * 100)
                        q[1] = (((ph - ideal[1]) / (sn[1] - ideal[1])) * 100)
                        q[2] = (((bod - ideal[2]) / (sn[2] - ideal[2])) * 100)
                        q[3] = (((nitrate - ideal[3]) / (sn[3] - ideal[3])) * 100)
                        q[4] = (((conductivity - ideal[4]) / (sn[4] - ideal[4])) * 100)

                        wqi = 0
                        for i in range(0, 5):
                            wqi += (q[i] * wn[i])
                        wqi = wqi / (wn_sum)

                        write_object.writerow([x[0], x[3], do, ph, bod, nitrate, conductivity, wqi])

                        # ---
            dataset = pd.read_csv('river_quality__refined_dataset.csv')

            def handle_non_numerical_data(dataset):
                columns = dataset.columns.values

                for column in columns:
                    text_digit_vals = {}

                    def convert_to_int(val):
                        return text_digit_vals[val]

                    if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
                        column_contents = dataset[column].values.tolist()
                        unique_elements = set(column_contents)
                        x = 0
                        for unique in unique_elements:
                            if unique not in text_digit_vals:
                                text_digit_vals[unique] = x
                            x += 1

                        dataset[column] = list(map(convert_to_int, dataset[column]))

                return dataset

            dataset = handle_non_numerical_data(dataset)


            # ---

            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, 7].values
            from sklearn.preprocessing import OneHotEncoder
            onehotencoder = OneHotEncoder(categorical_features=[1])
            X = onehotencoder.fit_transform(X).toarray()

            # Splitting the dataset into the Training set and Test set

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

            # feature scaling

            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)

            logisticregression = LogisticRegression()
            logisticregression.fit(X_train, y_train.astype('int'))

            # Predicting the Test set results
            y_pred = logisticregression.predict(X_test)
            plt.title('Predictions for River Water')
            plt.plot(y_test[:50], color='blue', label='True labels')
            plt.plot(y_pred[:50], 'r--', label='Predicted labels')

            plt.xlabel('No. of Test Examples ---->')
            plt.ylabel('    Predicted Values ---->')
            plt.grid(True)
            plt.show()


        elif option == "3":
            print("\n Entering Main Menu....\n")
            return 0
        elif option == '4':
            print("quiting....\n Thank you")
            return 1

        else:
            print("\nPlease provide a valid input \n")


def air_quality() :
    
	# Used Dataset: "air_quality_delhi_dataset.csv" data provided by US Embassy in Delhi	
	# Parameters: Temperature, Wind speed, Relative Humidity, Traffic index, Air quality of previous day, Industrial parameters such as power plant emissions
    
    # loading dataset
    dataset = pd.read_csv('air_quality_delhi_dataset.csv')

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 8].values

    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

    # feature scaling

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    abc = RandomForestRegressor(n_estimators=10)
    abc.fit(X_test, y_test)
    y_pred = abc.predict(X_test)
    

    # plot between actual labels(y) -vs- predicted labels(y)
    plt.title('Predictions for Air Quality ')
    plt.plot(y_test[:50], color='blue', label='True labels')
    plt.plot(y_pred[:50], 'r--', label='Predicted labels')
    plt.xlabel('No. of Test Examples ---->')
    plt.ylabel('    Scaled Values ---->')
    plt.grid(True)
    plt.show()
   

    print("\nModel trained successfully \n with accuracy : %.4f" %(abc.score(X_test, y_test)))



showmenu = True
while (showmenu):
    print ("""\n\n \033[4;31;47m Air and Water Quality Monitor \033[0;0m
    
    
    1) Water Quality Prediction
    2) Air quality Prediction
    3) exit
          
   """)
    option = str(input())

   
    if option == "1":

        res_w = water_quality()
        if res_w == 0:
            pass
        elif res_w == 1:
            break
    
    elif option == "2":
        res_a = air_quality()
        if res_a == 0:
            pass
        elif res_a == 1:
            break
    elif option == '3':
        print("quiting....\n Thank you")
        showmenu = False

    else:
        print("\nPlease provide a valid input \n")
        
        