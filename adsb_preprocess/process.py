import argparse
import os
import csv
from glob import glob
from collections import defaultdict
import datetime
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

from utils import get_runway_transform, convert_frame
from getWindVelocity import wind_params_runway_frame

class Data:
    def __init__(self,datapath,weather_path):
        self.path = datapath + "/raw_data/"
        self.base_path = datapath
        self.filelist = [y for x in os.walk(self.path) for y in glob(os.path.join(x[0], '*.csv'))]
        self.data = defaultdict(lambda: defaultdict())
        self.window = 150
        self.R = get_runway_transform()    
        self.filtered_data = defaultdict(lambda: defaultdict())
        self.filtered_id = 0
        self.out = 1
        self.weather_path = weather_path
        self.last_wind_dir = 0
        self.last_wind_speed = 0

        self.read_weather()
        self.process_data()
        
    def read_weather(self):
        
        self.weather = pd.read_csv(self.weather_path)
        self.weather['datetime'] = pd.to_datetime(self.weather['valid'],format="%Y-%m-%d %H:%M")
#         print(self.weather)f

    def parse_date_time(self, date_str, time_str):
        try:
            # Remove the 'u' prefix and other unwanted characters
            date_str = date_str.replace("u'", "'").replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
            time_str = time_str.replace("u'", "'").replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
            
            # Split into components
            year, month, day = date_str.split(',')
            hour, minute, seconds_full = time_str.split(',')
            
            # Handle microseconds - truncate to 6 digits for datetime compatibility
            if '.' in seconds_full:
                seconds_parts = seconds_full.split('.')
                seconds = seconds_parts[0]
                microseconds = seconds_parts[1][:6]  # Truncate to 6 digits max
            else:
                seconds = seconds_full
                microseconds = '0'
            
            # Format as a standard datetime string
            formatted_datetime = f"{month}/{day}/{year} {hour}:{minute}:{seconds}.{microseconds}"
            return datetime.datetime.strptime(formatted_datetime, '%m/%d/%Y %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc)
        except Exception as e:
            print(f"Error parsing date/time: {e}")
            print(f"Date string: {date_str}")
            print(f"Time string: {time_str}")
            return None
                
    def process_data(self):
        ##main loop: Reads each file
        for i in tqdm(self.filelist):
            # print(i)
            with open(i, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                try:
                    for row in csv_reader:
                        if row["ID"] is not "" and  row["Range"] is not None and row["Bearing"] is not None :
                            ID = row["ID"]
                            Range = row["Range"]
                            Bearing = row["Bearing"]
                            Altitude = row["Altitude"]
                            Tail = row["Tail"]
                            k = Range + Bearing + ID
                            if k not in self.data and int(Altitude)<6000 and float(Range)<5:
                                frame_time = self.parse_date_time(row["Date"], row["Time"])
                                if frame_time is None:
                                    continue
                                
                                self.data[k]["Frame"] = frame_time
                                self.data[k]["ID"] = ID
                                self.data[k]["Range"] = Range
                                self.data[k]["Bearing"] = Bearing
                                self.data[k]["Altitude"] = Altitude
                                try:
                                    self.data[k]["Tail"] = int(Tail, 36) # base36 encode tail number
                                except:
                                    self.data[k]["Tail"] = 66728889815 # 'UNKNOWN' tail number
                except csv.Error as e:
                    print('file %s, line %d: %s' % (csv_file, csv_reader.line_num, e))
                            
                if not self.data:
                    print("Empty Dict")
                    continue
                df = self.convert_to_local_df()
                df_sorted = self.interp_data(df)
                # Wind information gets appended below
#                 print(df_sorted.head())
                utc_timestamps = pd.DataFrame()
                df_sorted["utc"] = df_sorted.index
#                 utc_timestamps["utc"] = df_sorted.index
#                 print(utc_timestamps.type)

#                 utc_timestamps.parallel_apply(self.get_wind, axis=0)
                df_sorted["wind"] = df_sorted.apply(lambda x : self.get_wind(x.utc),axis = 1)
                utc_timestamps[["x","y"]] = pd.DataFrame(df_sorted.wind.tolist(),index = df_sorted.index)
#                 print(utc_timestamps)
                df_sorted['Headwind'] = utc_timestamps.apply(lambda l: l.x,axis =1)
                df_sorted['Crosswind'] = utc_timestamps.apply(lambda l: l.y,axis =1)
#                 df_sorted.insert(5, "Headwind", headwind)
#                 df_sorted.insert(6, "Crosswind", crosswind)
                
                # Save the timestamp before dropping the utc column
                # df_sorted['Timestamp'] = df_sorted["utc"].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_sorted['Timestamp'] = df_sorted["utc"].astype('int64') / 1e9
                
                df_sorted = df_sorted.drop(["utc","wind"],axis=1)
#                 print(df_sorted)
                
                self.seg_and_save(df_sorted)                
                self.data = defaultdict(lambda: defaultdict())
                            
    def get_wind(self,utc):
#         print(utc)
        curr_utc = str(utc)
        utc_formatted = curr_utc[0:10] + "-" + curr_utc[11:-3]
        utc_time = datetime.datetime.strptime(utc_formatted, "%Y-%m-%d-%H:%M").replace(tzinfo=datetime.timezone.utc)
        
        result_index = self.weather['datetime'].sub(utc_time).abs().idxmin()
#         print(self.weather["sknt"].iloc[result_index])
        try:
            wind_speed = float(self.weather["sknt"].iloc[result_index])*0.51444 ##knots to m/s
            wind_angle = float(self.weather["drct"].iloc[result_index])*np.pi/180.0 ##knots to m/s
            self.last_wind_dir = wind_angle
            self.last_wind_speed = wind_speed
        except ValueError :
#             print(self.weather["sknt"].iloc[result_index],utc_time)
            wind_angle = self.last_wind_dir
            wind_speed = self.last_wind_speed
        
        h_i , c_i = wind_params_runway_frame(wind_speed,wind_angle)
        if (np.isnan(h_i)):
            print("nan",h_i,utc)
        return h_i,c_i
        
     
    def seg_and_save(self,df):
        ## segregates the data into scenes and saves
        filename = self.base_path + "/processed_data/" + str(self.out) + ".txt" 
        # print("Filename = ",filename)
        file = open(filename,'w')
        csv_writer = csv.DictWriter(file, fieldnames=["Frame","ID","x","y","z","Headwind","Crosswind","Timestamp","Tail"],delimiter = " ")
        first_time = int(df.iloc[0]["Frame"])
        for index , row in df.iterrows():
            last_time = int(row["Frame"])
            if not ((last_time-first_time) > 1):
                row_write = row.to_dict()
                csv_writer.writerow(row_write)
            else:
                file.close()
                self.out = self.out + 1
                filename = self.base_path + "/processed_data/" + str(self.out) + ".txt"
                # print(filename)
                file = open(filename,'w')
                csv_writer = csv.DictWriter(file,fieldnames=["Frame","ID","x","y","z","Headwind","Crosswind","Timestamp","Tail"],delimiter = " ")
            first_time = last_time
        self.out = self.out + 1    
        file.close()
    
    
    def interp_data(self,data):
        ##interpolates the data
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'],format="%m/%d/%Y,%H:%M:%S")
        df.index = df['datetime']
        del df['datetime']
        
        # Separate the position and tail number columns
        position_cols = ['z', 'x', 'y']
        tail_col = ['Tail']

        # Resample and apply mean to position columns
        df_numeric = df.groupby('ID').resample('S')[position_cols].mean()

        # For the tail number column, take the first value in each time window, and forward fill any NaNs
        df_non_numeric = df.groupby('ID').resample('S')[tail_col].first().ffill()
        
        # Combine the numeric and non-numeric DataFrames back together
        df_interpol = pd.concat([df_numeric, df_non_numeric], axis=1)
        
        # df_interpol = df.groupby('ID').resample('S').mean()
        df_interpol['x'] = df_interpol['x'].interpolate(limit=60)
        df_interpol['y'] = df_interpol['y'].interpolate(limit=60)
        df_interpol['z'] = df_interpol['z'].interpolate(limit=60)
#         del df_interpol['ID']
        df_interpol.reset_index(level=0, inplace=True)
        df_sorted = df_interpol.sort_values(by="datetime")
        df_sorted["time"] = df_sorted.index
        first = df_sorted["time"].iloc[0]
        df_sorted["Frame"] = (df_sorted["time"]-first).dt.total_seconds()
        df_sorted["Frame"] = df_sorted["Frame"].astype("int")
        del df_sorted["time"]
        df_sorted = df_sorted.dropna()
        return df_sorted

    def convert_to_local_df(self):
        ##converts data to local frame
        df = pd.DataFrame.from_dict(self.data,orient='index')
        data = pd.DataFrame()
        data["datetime"] = df["Frame"]
        data['z'] = df.apply(lambda x: float(x["Altitude"])*0.3048/1000.0, axis =1)
        df["pos"] = df.apply(lambda x : convert_frame(float(x["Range"]),float(x["Bearing"]),self.R),axis = 1)
        df[["x","y"]] = pd.DataFrame(df.pos.tolist(),index = df.index)
        data['x'] = df.apply(lambda l: l.x[0],axis =1)
        data['y'] = df.apply(lambda l: l.y[0],axis =1)
        data['ID'] = df["ID"]
        data['Tail'] = df["Tail"]
        
        return data

            
            
if __name__ == '__main__':
    
    ##Dataset params
    parser = argparse.ArgumentParser(description='Train TrajAirNet model')
    parser.add_argument('--dataset_folder',type=str,default='/dataset/')
    parser.add_argument('--dataset_name',type=str,default='7days1')
    parser.add_argument('--weather_folder',type=str,default='weather_data')

    args=parser.parse_args()

    data_path = os.getcwd() + args.dataset_folder + args.dataset_name 
    print("Processing data from ",data_path)
    weather_path = os.getcwd() + args.dataset_folder + args.weather_folder + "/weather.csv"
    data = Data(data_path,weather_path)