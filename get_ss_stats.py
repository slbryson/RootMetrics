def read_get_ss_stats(df2):
    # Note this is modified for 2H 2015 version
    import pandas as pd
    import numpy as np

    import itertools
    import time

    start = time.time()
       
    #Setup only for failed calls
    
    #dm_fail = df2.loc[df2['Test_Summary']=='Access Ok / Retain Fail']
    #need to make modifications as the 2H 2015 file adds two new fields replacing Test_Summary
    # The first is Access_Summary and the second is Task_Summary
    # We are really looking for call failures so we need Access to be success and Task_Summary failure
    dm_fail = df2.loc[df2['Task_Summary']=='Failure']
    failed_tests = []
    failed_tests = dm_fail['Test_Cycle_ID'].values
    is_failed = df2['Task_Summary']=='Failure'
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
    #dm_fail[['Test','CDMA_Ecio','CDMA_Signal_Strength','CDMA_Ecio' \
    #           ,'EVDO_RSSI','EVDO_Ecio', 'LTE_RSRP','LTE_RSRQ']][:5].sort('CDMA_Signal_Strength', ascending=0)
    #dm_sig = dm_fail[['CDMA_Signal_Strength','Test_Cycle_ID']].sort('CDMA_Signal_Strength', ascending=1)
    
    dm_sig = dm_fail[['CDMA_Signal_Strength','Test_Cycle_ID']]
    fail_call_ss =dm_sig['CDMA_Signal_Strength'].values
    fail_call_ss = fail_call_ss.astype(float)  
    ################################################################
     
    fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min()]
     
    ################################################################
    dm_acc = dm_fail[['CDMA_Signal_Strength','Device_Satellite_Count','Device_Location_Accuracy']].sort('CDMA_Signal_Strength', ascending=1)
    #Device_Satellite_Count	Device_Location_Accuracy
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