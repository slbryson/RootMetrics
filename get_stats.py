def get_ss_stats(df2,stat_name):
    #Modified for 2H 2015
    import pandas as pd
    import numpy as np

    import itertools
    import time

    start = time.time()
    if stat_name =="":
        print 'No valid stat passed'
        return [], []
       
    #Setup only for failed calls
    
    #dm_fail = df2.loc[df2['Test_Summary']=='Access Ok / Retain Fail']
    #need to make modifications as the 2H 2015 file adds two new fields replacing Test_Summary
    # The first is Access_Summary and the second is Task_Summary
    # We are really looking for call failures so we need Access to be success and Task_Summary failure
    dm_fail = df2.loc[df2['Task_Summary']=='Failure']
     
    failed_tests = []
    failed_tests = dm_fail['Test_Cycle_ID'].values
    #Not used but does return true/false vs the actual sub dataframe.
    #is_fail = df2['Test_Summary']=='Access Ok / Retain Fail'

    #dm_fail = df2.loc[(df2['Test_Summary']=='Access Ok / Retain Fail')| \
    #(df2['Test_Summary']=='Access Fail / Retain N/A')]

    
    del dm_fail
    
    is_fail_index = df2['Test_Cycle_ID'].isin(failed_tests)
    
     
   
    is_call  = df2['Test'] == 'Outgoing Call Test End'
    is_call2 = df2['Test'] == 'Outgoing Call Test'
    is_call3 = df2['Test'] == 'Outgoing Call Test Start'
    ################################################################
    
   
 
    # All rows with failed indexes]
    dm_fail = df2[is_fail_index&(is_call|is_call2|is_call3)]
     
    if dm_fail.empty:
        print ('DataFrame is empty')
        end = time.time()
        final = end-start
        return final, [], [], []
    ################################################################
    #Remove NA values on CDMA Signal Strength
    dm_fail.dropna(subset=['CDMA_Signal_Strength'], how ='any', inplace =1)
    
   
    fail_call_ss =dm_fail['CDMA_Signal_Strength'].values
    fail_call_ss = fail_call_ss.astype(float)  
    ################################################################
     
    fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min(),fail_call_ss.std()]
     
 
    ################################################################
    # Also calculate whether the failure might have been due to > 10 sec setup
    dm_sig = df2[is_fail_index & is_call]

        #Remove NA values on Call Setup Duration
    dm_sig.dropna(subset=['Outgoing_Call:Call_Setup_Duration'], how ='any', inplace =1)
     
   
    call_setup_dur =dm_sig['Outgoing_Call:Call_Setup_Duration'].values
    call_setup_dur = call_setup_dur.astype(float) 
    ################################################################
    del dm_fail
    del dm_sig
    # Reset dm_fail to failed calls.
    dm_fail = df2[is_fail_index&(is_call|is_call2|is_call3)]
    #Only return the compute time and the stats for now
    end = time.time()
    final = end-start
     
    return final,fail_ss, dm_fail, call_setup_dur

def get_gen_stat(df2,stat_name):
    import pandas as pd
    import numpy as np
    import time

    # Rather than Change working existing routines.  Create new mod that
    # will take a data frame and field and compute stats
    if stat_name =="":
        print 'No valid stat passed'
        return [], []

    start = time.time() 
    ################################################################
    ################################################################
    #Remove NA values on RSRP
    df2.dropna(subset=[stat_name], how ='any', inplace =1)
    ################################################################
    # Get Stats for Selected Calls

    call_ss =np.array(df2[stat_name].values)
    call_ss = call_ss.astype(float)  
    fail_ss = [call_ss.mean(), call_ss.max(), call_ss.min(), call_ss.std(), np.median(call_ss)]
    ################################################################ 
   

    end = time.time()
    final = end-start
     
    return final, fail_ss

def get_lte_fail_stats(df2, type_flag, stat_name):
    import pandas as pd
    import numpy as np
    import re
    import itertools
    import time

    # Valid values for type_flag = 'FAILED', 'ALL', 'CALL'
    start = time.time() 
    if stat_name =="":
        print 'No valid stat passed'
        return [], [], []
    ################################################################
    if type_flag == "":
        type_flag =='FAILED'
    ################################################################
    if type_flag =='FAILED':
        #Setup only for failed calls
        ################################################################
        dm_fail = df2.loc[df2['Task_Summary']!='100%']
        failed_tests = []
        failed_tests = dm_fail['Test_Cycle_ID'].values
        
        del dm_fail

        is_fail_index = df2['Test_Cycle_ID'].isin(failed_tests)
        is_call = df2['Test'] == 'Downlink Throughput Test'
        # Filter
        dm_fail = df2[is_fail_index&(is_call)]

        if dm_fail.empty:
            print ('DataFrame is empty for Failed Condition')
            end = time.time()
            final = end-start
            return final, [], []
        ################################################################
        #Remove NA values on stat_name
        dm_fail.dropna(subset=[stat_name], how ='any', inplace =1)

        fail_call_ss =np.array(dm_fail[stat_name].values)
        fail_call_ss = fail_call_ss.astype(float)  
        ################################################################
        fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min(),\
         fail_call_ss.std(), np.median(fail_call_ss)]
        ################################################################
        #Reset dataframe
        dm_fail = df2[is_fail_index&(is_call)]
        end = time.time()
        final = end-start

    ################################################################
    elif type_flag=='CALL':
         #Setup only for Success calls
        ################################################################
        dm_fail = df2.loc[df2['Task_Summary']=='Success']
        failed_tests = []
        failed_tests = dm_fail['Test_Cycle_ID'].values
        
        del dm_fail

        is_fail_index = df2['Test_Cycle_ID'].isin(failed_tests)
        is_call = df2['Test'] == 'Downlink Throughput Test'
        # Filter
        dm_fail = df2[is_fail_index&(is_call)]

        if dm_fail.empty:
            print ('DataFrame is empty for CALL Condition')
            end = time.time()
            final = end-start
            return final, [], []
        ################################################################
        #Remove NA values on CDMA Signal Strength
        dm_fail.dropna(subset=[stat_name], how ='any', inplace =1)

        fail_call_ss =np.array(dm_fail[stat_name].values)
        fail_call_ss = fail_call_ss.astype(float)  
        ################################################################
        fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min(),\
         fail_call_ss.std(), np.median(fail_call_ss)]
        ################################################################
        #Reset dataframe
        dm_fail = df2[is_fail_index&(is_call)]
        end = time.time()
        final = end-start


    ################################################################
    elif type_flag == 'ALL':
         #Setup only for All calls
        ################################################################
        # Filter
        dm_fail = df2 

        if dm_fail.empty:
            print ('DataFrame is empty')
            end = time.time()
            final = end-start
            return final, [], []
        ################################################################
        #Remove NA values on CDMA Signal Strength
        dm_fail.dropna(subset=[stat_name], how ='any', inplace =1)

        fail_call_ss =np.array(dm_fail[stat_name].values)
        fail_call_ss = fail_call_ss.astype(float)  
        ################################################################
        fail_ss = [fail_call_ss.mean(), fail_call_ss.max(),fail_call_ss.min(),\
         fail_call_ss.std(), np.median(fail_call_ss)]
        ################################################################
        #Reset dataframe
        dm_fail = df2
        end = time.time()
        final = end-start

    else:
        print 'Bad type_flag'
        return [], [],[]

    ################################################################
    return final, fail_ss, dm_fail
def get_tput_count(df2,stat_name,stat_name1):
    
    import numpy as np
    import pandas 
    
    if stat_name=="" or stat_name1 =="":
        print 'No valid stat passed'
        return [], []
     
    df = df2.dropna(subset=[stat_name], how ='any', inplace =0)
    
    ################################################################
    # df3 = df3.loc[(df3['Call Final Class qualifier'] !='109') & (df3['Call Final Class qualifier'] !='101')]
    lte_type  = df.loc[df2[stat_name] =='LTE']
     
    eHRPD_type  = df.loc[df[stat_name] =='eHRPD']
    revA_type  = df.loc[df[stat_name] =='EVDO revision A']
    mix_type  = df.loc[df[stat_name] =='LTE,eHRPD']
    RTT_type  = df.loc[df[stat_name] =='1xRTT']
    total_type = df2.dropna(subset=[stat_name1], how ='any', inplace =0)
    
 

    try:
        net_type= (lte_type[stat_name].count(), \
                  eHRPD_type[stat_name].count(), \
                  mix_type[stat_name].count(),\
                  revA_type[stat_name].count(), \
                  RTT_type[stat_name].count(), \
                  df2[stat_name1].count() 
                  #len(df)\
                  )
    except:
        print "Couldn't Package Net Types "
    # Get Throughputs from selected types
 
    #stat_name1= 'Uplink_Throughput:Final_Test_Speed'
    lte_type.dropna(subset=[stat_name1], how ='any', inplace =1)
    lte_type_value = np.array(lte_type[stat_name1].values, dtype =float)
     

    eHRPD_type.dropna(subset=[stat_name1], how ='any', inplace =1)
    eHRPD_type_value = np.array(eHRPD_type[stat_name1].values, dtype =float)
     
     

    mix_type.dropna(subset=[stat_name1], how ='any', inplace =1)
    mix_type_value = np.array(mix_type[stat_name1].values, dtype =float)
   
    # Calculate mean and median
    df = df2.dropna(subset=[stat_name1], how ='any', inplace =0)
    df = np.array(df[stat_name1].values, dtype =float)
    df_cumsum =df.cumsum()
    tenth, seventyfifth, ninetith = np.percentile(df, [10, 75, 90])
    # Calculate the Median Overall throughput as well
    
    
    # Adding the max
    #df_max = df.max()
    #  Get the median Access Duration per uplink or downlink
    if stat_name1[:2] == 'Do':
        access_duration = 'Downlink_Throughput:Access_Duration'
        ttfb = 'Downlink_Throughput:Time_To_First_Byte'
    elif stat_name1[:2] =='Up':
        access_duration = 'Uplink_Throughput:Access_Duration'
        ttfb = 'Uplink_Throughput:Median_TTC_All_Successful_Connections'

    try:
        data_acc_dur = df2.dropna(subset=[access_duration], how ='any', inplace =0)
        data_acc_dur = np.array(data_acc_dur[access_duration].values, dtype =float)

        data_ttfb = df2.dropna(subset=[ttfb], how ='any', inplace =0)
        data_ttfb = np.array(data_ttfb[ttfb].values, dtype =float)
    except:
        print 'Could not get access duration median and Time_To_First_Byte'
        print access_duration, '   ', ttfb, count

    # In addition to the break out of median speeds per technology, the 
    # percentiles are returned and the median speed over all technologies
    # I'm also adding max speeds to help sort out technologies
    net_tput = (lte_type_value.mean(), np.median(lte_type_value), \
     eHRPD_type_value.mean(), np.median(eHRPD_type_value), \
     mix_type_value.mean(), np.median(mix_type_value), \
     np.median(df), tenth,seventyfifth,ninetith,df.max(), \
     np.median(data_acc_dur), np.median(data_ttfb))
 
    return net_type, net_tput

def get_distribution(df,filelist):
    import numpy as np
    #Do somethimg
    #Attempt to extract Throughput vs RSRP from each data set
    # Often when the throughput value is recorded the RSRP is not!!
    # So we first group by Test Id, then find the Test Start, network type, end 
    # values and calculate the RSRP stats, RSRQ stats, RSSNR stats and throughput for those cases
    # later we can record the lat/long as well as the eNB Id
    ####################################################################
    stat_name= 'Downlink_Throughput:Final_Test_Speed'
    stat_name1 = 'LTE_RSRP'
    df2  = df.groupby([df['Test_Cycle_ID']])
    dlnet_list =[]
    ulnet_list =[]
    net_stat = {}
    rsrp_vec = []
    rsrq_vec = []
    rssnr_vec = []
    _stat = []
    for name, group in df2:
    ####################################################################
        count = 0
        for i, row in group.iterrows():
            # We ignore rows where LTE_RSRP is not valid, which may be too harsh.
            if not np.isnan(float(row['LTE_RSRP'])):

                #print(' New RSRP ', row['LTE_RSRP'])
                #print('New RSRQ ', row['LTE_RSRQ'], row['Test'])
                net_type  = row['Data_Network_Type']
                net_name = row['Network_Name']
                test_type = row['Test']
                #print row['Data_Network_Type']
                rsrp_vec.append(row['LTE_RSRP'])
                if np.isnan(float(row['LTE_RSRQ'])):
                    rsrq_vec.append('0')
                else:
                    rsrq_vec.append(row['LTE_RSRQ'])
                if np.isnan(float(row['LTE_RSSNR'])):
                    rssnr_vec.append('0')
                else:
                    rssnr_vec.append(row['LTE_RSSNR'])
                    # Get Base station information
                dl_lte_eci = row['LTE_eCI']
                dl_lte_enbid =row['LTE_eNB_ID']
                dl_pci_pcell = row['LTE_Physical_Cell_ID_Pcell']
                dl_pci_scell = row['LTE_Physical_Cell_ID_Scell']

                
                #Get device Lat/long right before test ends
                # Modified for 2H 2015
                dl_lat = row['Latitude']
                dl_long = row['Longitude']

            #Specific for Downlink
            if row['Test'] == 'Downlink Throughput Test End':
                market = row['Collection']
                market = filelist
                dl_tput = row['Downlink_Throughput:Final_Test_Speed']
                dl_acc_dur = float(row['Downlink_Throughput:Access_Duration'])
                dl_ttfb = float(row['Downlink_Throughput:Time_To_First_Byte'])

                # carrier Aggregation Data
                #LTE_DL_Bandwidth   LTE_UL_Bandwidth    LTE_Frequency_Band  
                # LTE_DL_EARFCN_Pcell LTE_UL_EARFCN_Pcell LTE_DL_Bandwidth_Index_Pcell    
                # LTE_DL_Bandwidth_Index_Scell    LTE_DL_EARFCN_Scell LTE_UL_EARFCN_Scell
                dl_bw = row['LTE_DL_Bandwidth']
                lte_freq = row['LTE_Frequency_Band']
                dl_earfcn_pcell = row['LTE_DL_EARFCN_Pcell']
                dl_earfcn_scell = row['LTE_DL_EARFCN_Scell']
                dl_bw_index_pcell = row['LTE_DL_Bandwidth_Index_Pcell']
                dl_bw_index_scell = row['LTE_DL_Bandwidth_Index_Scell']



        

                #Build Dictionary and calculate stats
                rsrp_vec = np.array(rsrp_vec, dtype=int)
                rsrq_vec = np.array(rsrq_vec, dtype=int)
                rssnr_vec = np.array(rssnr_vec, dtype=float)

                rsrp_mean = rsrp_vec.mean()
                rsrp_std = rsrp_vec.std()

                rssnr_mean = rssnr_vec.mean()
                rssnr_std = rssnr_vec.std()

                rsrq_mean = rsrq_vec.mean()
                rsrq_std = rsrq_vec.std()

                try:
                    net_stat[('test_id','test_iter')] = [(name, count), \
                        (net_type,dl_tput)]
                    net_stat['stat'] = [(\
                             rsrp_mean,rsrp_std, \
                             rsrq_mean, rsrq_std, \
                             rssnr_mean,rssnr_std)]
                 #create an alternate vector representation of the stats
                    _stat.append([market,dl_tput, rsrp_mean,rsrp_std, rsrq_mean,rsrq_std,\
                    rssnr_mean,rssnr_std, \
                    dl_lte_eci , dl_lte_enbid, dl_pci_pcell ,dl_pci_scell, \
                    dl_acc_dur, dl_ttfb,\
                    dl_lat, dl_long, dl_bw, \
                    lte_freq, dl_earfcn_pcell, dl_earfcn_scell, \
                    dl_bw_index_pcell, dl_bw_index_scell,  net_name])

                    dlnet_list.append(net_stat)
                except:
                    print ('DL something is wrong with setting net_stat'), type(net_stat),type(rsrp_mean)
                    _stat =[]
                    net_stat = {}
                    rsrp_vec = []
                    rsrq_vec = []
                    rssnr_vec = []
                    break
                #There are usually two tests after we have collected the previous RF
                # we increment the count and restart the vectors
                count = count+ 1
                net_stat = {}
                rsrp_vec = []
                rsrq_vec = []
                rssnr_vec = []
            #Specific for Uplink
            elif row['Test'] == 'Uplink Throughput Test End':
                ul_tput = row['Uplink_Throughput:Final_Test_Speed']
                market = row['Collection']
                market = filelist
                #Build Dictionary

                rsrp_vec = np.array(rsrp_vec, dtype=int)
                rsrq_vec = np.array(rsrq_vec, dtype=int)
                rssnr_vec = np.array(rssnr_vec, dtype=float)

                rsrp_mean = rsrp_vec.mean()
                rsrp_std = rsrp_vec.std()
                rssnr_mean = rssnr_vec.mean()
                rssnr_std = rssnr_vec.std()
                rsrq_mean = rsrq_vec.mean()
                rsrq_std = rsrq_vec.std()

                try:
                    net_stat[('test_id','test_iter')] = [(name, count), \
                        (net_type,dl_tput, \
                         rsrp_mean,rsrp_std, \
                         rsrq_mean, rsrq_std, \
                         rssnr_mean,rssnr_std,rsrq_vec)]
                    ulnet_list.append(net_stat)
                except:
                    print('Something wrong with UL netstat')
                count = count+ 1
                net_stat = {}
                rsrp_vec = []
                rsrq_vec = []
                rssnr_vec = []

        #break  for only one data point
    ####################################################################

    return  _stat, dlnet_list, ulnet_list

def get_plot(df,stat_name,buckets):
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np


    #Remove NA values on RSRP
    df2 = df.dropna(subset=[stat_name], how ='any', inplace =0)
    ################################################################
    # Get Stats for Selected Calls

    data=np.array(df2[stat_name].values)
    data = data.astype(float)/1000.0
    data_cumsum = data.cumsum()

    fig, ax = plt.subplots()
    end_range = data.max()
    counts, bins, patches = ax.hist(data, facecolor='yellow', \
        bins= buckets,edgecolor='gray',range=(0, end_range))

    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins)
    # Set the xaxis's tick labels to be formatted with 1 decimal place...
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    # Change the colors of bars at the edges...
    twentyfifth, seventyfifth = np.percentile(data, [10, 90])
    for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
        if rightside < twentyfifth:
            patch.set_facecolor('red')
        elif leftside > seventyfifth:
            patch.set_facecolor('green')


    med_annotate = 'Median = %0.2f'%np.median(data)   
    ax.annotate(med_annotate, xy=(35, 0), xycoords=('data', 'axes fraction'),
            xytext=(225, 200), textcoords='figure points', va='top', ha='center')

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -32), textcoords='offset points', va='top', ha='center')


    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    titles=df['Collection'].values
    my_title = titles[0] + '\n ' + stat_name + ' Median Throughput'
    plt.title(my_title)
    
    plt.grid(True)
       
    fname = titles[0]+'_'+stat_name[:10]+'.jpg'
 
    plt.savefig(fname, format='jpg') 
    plt.show()
    tenth, fiftith, ninetith = np.percentile(data, [10, 50, 90])
    print "Median ",stat_name,"throughput %2.2f"%np.median(data)
    print '10th %2.2f'%tenth, '50th %2.2f'%fiftith, '90th %2.2f'%ninetith
  
    return
def get_plmn_count(df):
    # Do I need any libraries

    # Grab Basestations seen in failed tests
    plmn_count = df[['Network_MCC','Network_MNC']].groupby([df['Network_MCC'],df['Network_MNC']]).agg(len)
    idx = plmn_count.index.tolist()
    _metro_map ={}
    _metro_map['RmMetro'] = df['Collection'][0] 
    _metro_map['mcc'] = idx
    # print _metro_map
    _stat = {}
    _stat = dict(zip(idx,plmn_count['Network_MCC'].values))
    x = _stat
    y = _stat
    _stat = { k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y) }
    #print '_stat  is the mapping of BS ID to number of times seen in the set', _stat
    print 'Different MCC values in market', len(_stat.keys())


def get_enbid_count(df):
    import numpy as np
    # Grab Basestations seen in failed tests
    fail_count = df[['LTE_eNB_ID']].groupby([df['LTE_eNB_ID']]).agg(len)
    idx = fail_count.index.tolist()
    #print df['Collection'][0]
    _metro_map ={}
    y = {}
    _metro_map['RmMetro'] = df['Collection'][0]
    _metro_map['eNB_ID'] = idx
    # print _metro_map
    _stat = {}
    _stat = dict(zip(idx,fail_count['LTE_eNB_ID'].values))
    x = _stat
    y = _stat
    _stat = { k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y) }
     
    y = _stat
    y = sorted(y.items(), key=lambda x: (x[1],x[0]), reverse=True) 
    #  print '_stat  is the mapping of BS ID to number of times seen in the set', y
    #print len(_stat.keys())
    enbid_count = int(len(_stat.keys()))

    return (y, enbid_count)