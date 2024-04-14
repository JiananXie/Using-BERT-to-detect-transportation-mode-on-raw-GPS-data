import pandas as pd
import numpy as np
from math import floor
import math
from pyproj import Transformer
from vincenty import vincenty

# Read data
df = pd.read_csv('geolife_processed_full_truncated32.csv',header=0,names=['lat','lon','ts','no','label'],dtype={'lat':float,'lon':float,'ts':str,'no':str,'label':str })

df['label'] = df['label'].replace('walk', 0)
df['label'] = df['label'].replace('bike', 1)
df['label'] = df['label'].replace('bus', 2)
df['label'] = df['label'].replace('driving', 3)
df['label'] = df['label'].replace('train_', 4)

df['ts'] = pd.to_datetime(df['ts'])

lat_min = df['lat'].min()
lat_max = df['lat'].max()
lon_min = df['lon'].min()
lon_max = df['lon'].max()

grid_num= 36,875,800 #revisable to set a proper gird_size, here we use 0.1m as grid_size

#Generate grid data    
df_grid = df.reset_index(drop=True)
df_grid_1 = df_grid


# calculate speed, unit m/s
def cal_speed(lat1, lon1, lat2, lon2, ts1, ts2):

    distance = vincenty((lat1, lon1), (lat2, lon2)) * 1000   

    time = (ts2 - ts1).seconds

    speed = round(distance / time, 2)

    return speed

for id in range(len(df_grid_1)-1):
    if df_grid_1.loc[id,'no'] == df_grid_1.loc[id+1,'no'] :
        lat1 = df_grid_1.loc[id, 'lat']
        lon1 = df_grid_1.loc[id, 'lon']
        lat2 = df_grid_1.loc[id+1, 'lat']
        lon2 = df_grid_1.loc[id+1, 'lon']   
        ts1 = df_grid_1.loc[id, 'ts']
        ts2 = df_grid_1.loc[id+1, 'ts']
        speed = cal_speed(lat1, lon1, lat2, lon2, ts1, ts2)
        df_grid_1.loc[id, 'speed'] = speed 

# calculate acceleration, unit m/s^2
def cal_acceleration(speed1, speed2, time1, time2):

    speed_diff = speed2 - speed1

    time_diff = (time2 - time1).seconds

    acceleration = round(speed_diff / time_diff,2)
    return acceleration

for id in range(len(df_grid_1)-1):
    if df_grid_1.loc[id,'no'] == df_grid_1.loc[id+1,'no'] :
        speed1 = df_grid_1.loc[id, 'speed']
        speed2 = df_grid_1.loc[id+1, 'speed']
        ts1 = df_grid_1.loc[id, 'ts']
        ts2 = df_grid_1.loc[id+1, 'ts']
        acceleration = cal_acceleration(speed1, speed2, ts1, ts2)
        df_grid_1.loc[id, 'acceleration'] = acceleration

# calculate jerk, unit m/s^3
def cal_jerk(acceleration1,acceleration2,time1,time2):

    acceleration_diff = acceleration2 - acceleration1

    time_diff = (time2 - time1).seconds

    jerk = round(acceleration_diff / time_diff,2)
    return jerk

for id in range(len(df_grid_1)-1):
    if df_grid_1.loc[id,'no'] == df_grid_1.loc[id+1,'no'] :
        acceleration1 = df_grid_1.loc[id, 'acceleration']  
        acceleration2 = df_grid_1.loc[id+1, 'acceleration']
        ts1 = df_grid_1.loc[id, 'ts']
        ts2 = df_grid_1.loc[id+1, 'ts']
        jerk = cal_jerk(acceleration1,acceleration2,ts1,ts2)
        df_grid_1.loc[id, 'jerk'] = jerk 


df_grid_1.fillna(0, inplace=True)


# # define a transform function
# def convert_coordinates(row):
#     x, y = transformer.transform(row['lon'], row['lat'])
#     return (x, y)

# mercator = df_grid.apply(convert_coordinates, axis=1)

lat_min = df_grid['lat'].min()  
lat_max = df_grid['lat'].max()
lon_min = df_grid['lon'].min()  
lon_max = df_grid['lon'].max()  

horizontal_distance = max(vincenty((lat_min, lon_min), (lat_min, lon_max)),vincenty((lat_max,lon_min),(lat_max,lon_max)) )* 1000   
vertical_distance = max(vincenty((lat_min, lon_min), (lat_max, lon_min)),vincenty((lat_max,lon_max),(lat_min,lon_max)) )* 1000

grid_size = min(horizontal_distance, vertical_distance) / grid_num #正方形格网坐标下的边长
df_grid_1['id'] = df_grid.apply(lambda x: min(floor(vincenty((x[0],x[1]),(x[0],lon_min))/grid_size),grid_num-1)*grid_num + min(floor(vincenty((x[0],x[1]),(lat_max,x[1]))/grid_size),grid_num-1),axis=1)


def resample_and_split_dataset(df_grid):

    # convert feature sequence to sentence
    final_df = df_grid.groupby(['no','label']).agg(lambda x: ' '.join(x.astype(str))).reset_index()
    final_df = final_df[['label','speed','acceleration','jerk','id']]
    
    count = final_df['id'].str.split().apply(len)
    final_df = final_df[count >= 10].reset_index(drop=True) #remove the trajectories with less than 10 points  
    
    #split train, valid, test in 8:1:1 ratio within each mode
    Classification_train = []
    Classification_valid = []
    Classification_test = []
    for l in final_df['label'].unique():
        temp = final_df[final_df['label'] == l]
        train, valid_test = np.split(temp.sample(frac=1,random_state=42), [int(0.8*len(temp))], axis=0)
        valid, test = np.split(valid_test.sample(frac=1,random_state=42), [int(0.5*len(valid_test))], axis=0)
        Classification_train.append(train)
        Classification_valid.append(valid)
        Classification_test.append(test)
    data_train = pd.concat(Classification_train, axis=0, join='outer', ignore_index=True)
    data_valid = pd.concat(Classification_valid, axis=0, join='outer', ignore_index=True)
    data_test = pd.concat(Classification_test, axis=0, join='outer', ignore_index=True)
    return data_train, data_valid, data_test

if __name__ == "__main__":
    data_train, data_valid, data_test = resample_and_split_dataset(df_grid_1)
    data_train.to_csv(f'data_train_full_{grid_size:.2f}_ws32.csv', index=False)
    data_valid.to_csv(f'data_valid_full_{grid_size:.2f}_ws32.csv', index=False)
    data_test.to_csv(f'data_test_full_{grid_size:.2f}_ws32.csv', index=False)