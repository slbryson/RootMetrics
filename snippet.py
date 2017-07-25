def get_plot_bak(df,stat_name):
    
    import pandas as pd
    import numpy as np
    import time
    import matplotlib as plt

    # Rather than Change working existing routines.  Create new mod that
    # will take a data frame and field and compute stats


    if stat_name =="":
        print 'No valid stat passed'


    start = time.time() 
    ################################################################
    ################################################################
    #Remove NA values on RSRP
    df2 = df.dropna(subset=[stat_name], how ='any', inplace =0)
    ################################################################
    # Get Stats for Selected Calls

    call_ss =np.array(df2[stat_name].values)
    call_ss = call_ss.astype(float)  
    fail_ss = [call_ss.mean(), call_ss.max(), call_ss.min(), call_ss.std(),np.median(call_ss)]
    end = time.time()

    # Histogram the number of times on technology
    #print np.median(call_ss), np.histogram(call_ss, bins = 25)

    

    print len(df)
    labels = call_ss.astype(str)
    plt.pyplot.figure()
  
    hist = plt.pyplot.hist(call_ss, bins=20, label =labels,cumulative=True, normed = True,log=False,edgecolor='b', facecolor='g', histtype='bar', range=(0,call_ss.max()))

    #hist = plt.pyplot.hist(call_ss, bins=25, label =labels,cumulative=False, log=False, histtype='bar', range=(call_ss.min(),call_ss.max() - np.median(call_ss)))
 
    titles=df['Collection'].values
    my_title = titles[0] + '\n ' + stat_name + ' Median Throughput'
    plt.pyplot.title(my_title)
    plt.pyplot.grid(True)
    amed = str(np.median(call_ss))
    plt.pyplot.annotate(amed,xy=(2, 1), xytext=(3, 1.5))
    plt.pyplot.xticks(range(call_ss.min(), call_ss.max()))
    
    fname = titles[0]+'_'+stat_name[:10]+'.jpg'
 
    plt.pyplot.savefig(fname, format='jpg')
    return