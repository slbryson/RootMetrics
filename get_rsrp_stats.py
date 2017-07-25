def read_get_ss_stats(df2):
    import pandas as pd
    import numpy as np

    import itertools
    import time

    start = time.time()
       
    #Setup only for failed calls
    
    dm_fail = df2.loc[df2['Test_Summary']=='Access Ok / Retain Fail']
    failed_tests = []
    failed_tests = dm_fail['Test_Cycle_ID'].values
    is_failed = df2['Test_Summary']=='Access Ok / Retain Fail'
    del dm_fail
    
    is_fail_index = df2['Test_Cycle_ID'].isin(failed_tests)

    is_call = df2['Test'] == 'Outgoing Call Test End'
    is_call2 = df2['Test'] == 'Outgoing Call Test'
    is_call3 = df2['Test'] == 'Outgoing Call Test Start'
    ################################################################
    
   
    #df2[is_fail_index & is_call & is_failed].describe()
    # All rows with failed indexes]
    dm_fail = df2[is_fail_index&(is_call|is_call2|is_call3)]
    if dm_fail.empty:
        print ('DataFrame is empty')
        end = time.time()
        final = end-start
        return final, [], []
    ################################################################
    #Remove NA values on CDMA Signal Strength
    dm_fail.dropna(subset=['CDMA_Signal_Strength'], how ='any', inplace =1)
    dm_fail[['Test','CDMA_Ecio','CDMA_Signal_Strength','CDMA_Ecio' \
               ,'EVDO_RSSI','EVDO_Ecio', 'LTE_RSRP','LTE_RSRQ']][:5].sort('CDMA_Signal_Strength', ascending=0)
    #dm_sig = dm_fail[['CDMA_Signal_Strength','Test_Cycle_ID']].sort('CDMA_Signal_Strength', ascending=1)
    
    dm_sig = dm_fail[['CDMA_Signal_Strength','Test_Cycle_ID']]
    fail_call_ss =dm_sig['CDMA_Signal_Strength'].values
    fail_call_ss = fail_call_ss.astype(float)  
    ################################################################
     
    fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min()]
     
    ################################################################
    dm_acc = dm_fail[['CDMA_Signal_Strength','Device_Satellite_Count','Device_Location_Accuracy']].sort('CDMA_Signal_Strength', ascending=1)
    #Device_Satellite_Count Device_Location_Accuracy
    sat_acc = dm_acc['Device_Location_Accuracy'].values
    #Could use Satellite count
    #sat_acc = dm_acc['Device_Satellite_Count'].values

    sat_acc = sat_acc.astype(float)

    
    sat_stat =[sat_acc.mean(), sat_acc.max(), sat_acc.min()]
    del dm_fail
    dm_fail = df2[is_fail_index&(is_call|is_call2|is_call3)]
    #Only return the compute time and the stats for now
    end = time.time()
    final = end-start
     
    return final,fail_ss, dm_fail
def read_get_rsrp_stats(df2):
    import pandas as pd
    import numpy as np
    import re
    import itertools
    import time

    

    #inputFile = r'New Haven/NewHaven-CT_2015-1H_Sprint_Detail.csv'

    start = time.time() 
    #Not Used in this iteration
    #dm_call = df2[['Test','Test_Summary','Test_Cycle_ID']].groupby([df2['Test_Cycle_ID'],\
    #                                                       df2['Test_Summary']]).agg(len)
    #Downlink Throughput Test

    ################################################################
    #Setup only for failed calls
    
    dm_fail = df2.loc[df2['Test_Summary']=='Failure']
    failed_tests = []
    failed_tests = dm_fail['Test_Cycle_ID'].values
    is_failed = df2['Test_Summary']=='Failure'
    del dm_fail
    is_fail_index = df2['Test_Cycle_ID'].isin(failed_tests)

    is_call = df2['Test'] == 'Downlink Throughput Test'
  
    ################################################################
    
   
    #df2[is_fail_index & is_call & is_failed].describe()
    # All rows with failed indexes]
    dm_fail = df2[is_fail_index&(is_call)]
    if dm_fail.empty:
        print ('DataFrame is empty')
        end = time.time()
        final = end-start
        return final, [], []
    ################################################################
    #Remove NA values on CDMA Signal Strength
    dm_fail.dropna(subset=['LTE_RSRP'], how ='any', inplace =1)
    #dm_fail[['Test','CDMA_Ecio','CDMA_Signal_Strength','CDMA_Ecio' ,'EVDO_RSSI','EVDO_Ecio', 'LTE_RSRP','LTE_RSRQ']][:5].sort('LTE_RSRP', ascending=0)
    #dm_sig = dm_fail[['CDMA_Signal_Strength','Test_Cycle_ID']].sort('CDMA_Signal_Strength', ascending=1)
    

    dm_sig = dm_fail[['LTE_RSRP','Test_Cycle_ID']]
    fail_call_ss =dm_sig['LTE_RSRP'].values
    fail_call_ss = fail_call_ss.astype(float)  
    ################################################################
     
    fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min()]
     
    ################################################################
    dm_acc = dm_fail[['LTE_RSRP','Device_Satellite_Count','Device_Location_Accuracy']].sort('LTE_RSRP', ascending=1)
    #Device_Satellite_Count	Device_Location_Accuracy
    sat_acc = dm_acc['Device_Location_Accuracy'].values
    #Could use Satellite count
    #sat_acc = dm_acc['Device_Satellite_Count'].values

    sat_acc = sat_acc.astype(float)

    
    sat_stat =[sat_acc.mean(), sat_acc.max(), sat_acc.min()]
    #Only return the compute time and the stats for now
    #####################
    del dm_fail
    dm_fail = df2[is_fail_index&(is_call)]
    end = time.time()
    final = end-start
     
    return final, fail_ss, dm_fail