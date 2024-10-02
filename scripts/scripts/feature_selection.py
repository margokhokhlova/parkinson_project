import pandas as pd
import numpy as np

def get_feature(index_f, data_path = 'C:/Users/khokhlovam/Documents/kotelnikov/data/data_lstm_august24_PDET_left_right.csv', min_len_established = 600): 
    df =pd.read_csv(data_path, header=None, names=range(2150)) # 
    #df.head(15)
    freq_valid = [-np.inf, +np.inf] # was previosly used to remove some frequences, no longer used but shouldn't be changed
    X_PDL= []
    X_ETL= []
    min_len = float('inf')  # Using infinity to start
    patient_i = 1
    for i in range(index_f,len(df),14): # 14 per item 
        label = df.iloc[[i]][0].values[0]    
        feat_range = df.iloc[[i]].values.tolist()[0]        
        valid_indexes = [f for f in range(3,len(feat_range)) if str(feat_range[f]) !='nan' and  feat_range[f]>=freq_valid[0] and feat_range[f]<=freq_valid[1]]
        feat = [feat_range[x] for x in valid_indexes]   
        if len(feat)<min_len:
            min_len =  len(feat)
        if label == 'Left':
            X_PDL.append(feat[:min_len_established])            
        elif label == 'ETLeft':
            X_ETL.append(feat[:min_len_established]) 

        #print(f'Patient{i} verification {label},  {feat_range[2]}')          
        patient_i += 1
    return X_PDL, X_ETL