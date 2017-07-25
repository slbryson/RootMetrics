def get_average_distance(clean_bs_dict,df):
    from math import sin, cos, sqrt, atan2, radians, isnan
    import math

    count =0
    tot_dist =0
    avg_dist = 0.0

    for index, row in df.iterrows():

        try:
            lat = radians(clean_bs_dict[row['Base_Station_ID']][0])
            lon = radians(clean_bs_dict[row['Base_Station_ID']][1])
            lat2 = radians(row['Latitude'])
            lon2 = radians(row['Longitude'])
            distance = get_dist_to_cell(lat,lon,lat2,lon2)

            if not math.isnan(distance):
                count = count +1
                tot_dist = distance + tot_dist
                avg_dist = tot_dist/count
        except KeyError:
            null ={} 
        except:
            print "General Math Error for distance"

    return avg_dist

def create_base_station_dict(df):
    from math import isnan
    ################################
    # This Code creates a dictionary for base station locations from the sample data set.  We could create a dictionary
    # from the entire data set.

    # let's create a dictionary
    bs_coord = zip (df['Base_Station_Latitude'].values,df['Base_Station_Longitude'].values)
    bs_dict = dict(zip (df['Base_Station_ID'],bs_coord))
    #Base_Station_ID  Base_Station_Longitude
   
    #######################
    # Remove NaNs from dictionary
    

    clean_bs_dict = dict((k, v) for k, v in bs_dict.items() if not (type(k) == float \
        or isnan(k)))
    return clean_bs_dict
# Calculate Distance using lat/long
def get_dist_to_cell(lat1,lon1,lat2,lon2):
    from math import sin, cos, sqrt, atan2, radians, isnan
 
    # approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance

def get_box(df):
    import math
    import calc_distance
    import numpy as np
    
    #In 2H 2015 the field name changes to just Latitude and Longitude
    lat_total = df['Latitude'].dropna(subset=['Longitude'], how ='any', inplace =0).values
    long_total = df['Longitude'].dropna(subset=['Longitude'], how ='any', inplace =0).values
     
    left_top = (long_total.min(), lat_total.max())
    left_bottom = (long_total.min(), lat_total.min())
    right_top = (long_total.max(), lat_total.max())
    right_bottom = (long_total.max(), lat_total.max())

    right_top = (float(lat_total.max()), float(long_total.min()) )
    right_bottom = (float(lat_total.min()), float(long_total.min()))
    left_top = (float(lat_total.max()), float(long_total.max()))
    left_bottom = (float(lat_total.min()), float(long_total.max()))

     
    # Distance
    dimA = calc_distance.get_dist_to_cell(left_bottom[0],left_bottom[1],left_top[0],left_top[1])
    dimB = calc_distance.get_dist_to_cell(left_bottom[0],left_bottom[1],right_bottom[0],right_bottom[1])
    area = dimA*dimB
    area = np.sqrt(area)

    box = (left_bottom, left_top, right_bottom, right_top)
    if False:
        print dimA, dimB, dimA*dimB, area
        print type(box), len(box) , box[3][0]
        print box
    return box, area