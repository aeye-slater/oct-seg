import os
import pandas as pd
import numpy as np
from pydicom import dcmread
from sklearn.model_selection import train_test_split


def findEye(input_string):
    eye = []
    parts = input_string.split("_")
    for text in parts:
        if text=="R":
            return "R"
        elif text == "OD":
            return "R"
        elif text == "L":
            return "L"
        elif text == "OS":
            return "L"
    return "NA"

def main():
    print("**************WARNING***********")
    print("This script assumes you have the Q")
    print("and Y drives mounted in /mnt/share")
    print("It was also designed to run on Nightmare")
    print("or Sweet Dreams.  It will fail on Windows")
    site=[]
    patient=[]
    file_nm=[]
    visit = []
    for root, dirs, files in os.walk("/mnt/share/Q/L drive cleanup - 2023/EPOCH"):
        for file in files:
            if file.lower().endswith("dcm"):
                parts = root.split("/")
                site.append(parts[6])
                patient.append(parts[7])
                visit.append(parts[-1])
                file_nm.append(file)

    data = pd.DataFrame({"Site":site,"Patient":patient,"Visit":visit,"File":file_nm})
    data.to_csv("/mnt/share/Y/OCT Segmentation/AllEpochFilesDicoms.csv")
    print(f"Found {data['Patient'].unique().shape} unique patients")
    data['Eye'] = data['File'].apply(findEye)
    data=data.sort_values(['Site','Patient','Eye','Visit'])
    data['key'] = data['Site']+data["Patient"]+data['Eye']
    x=data['key'].value_counts()
    data=data[data['key'].isin(x[x>2].index.to_list())]
    data=data[data['Eye']!="NA"]
    data=data.loc[data['Visit']=='Screen']
    data['Include']=False
    rng = np.random.default_rng(1776)
    for patient in data['Patient'].unique():
        pick = rng.choice(data[data['Patient']==patient].index)
        data.loc[pick,"Include"]=True
    files = data[data['Include']==True]
    mfg = []
    BASE = "/mnt/share/Q/L drive cleanup - 2023/EPOCH"
    print("Next step takes a few minutes, be patient")
    for i in range(files.shape[0]):
        tmp=os.path.join(BASE,files.iloc[i,0],files.iloc[i,1],files.iloc[i,2],files.iloc[i,3])
        try:
            tmp=dcmread(tmp)
            mfg.append(tmp.Manufacturer)
        except:
            mfg.append("badfile")
        if i%50==0:
            print(f"Done with {i} files")
    files['Mfg']=mfg
    print(files.shape)
    X_train,X_test = train_test_split(files,test_size=0.20,shuffle=True,random_state=1776, stratify=files["Mfg"])
    print("Writing File")
    data.to_csv("/mnt/share/Y/OCT Segmentation/file_list.csv",index=None)
    print("Write Training Files")
    X_train.to_csv("/mnt/share/Y/OCT Segmentation/training_file.csv",index=None)
    X_test.to_csv("/mnt/share/Y/OCT Segmentation/valiation_file.csv",index=None)
    files.to_csv("/mnt/share/Y/OCT Segmentation/trainval_combined_file.csv",index=None)
    print("Complete: Files are in Y/OCT Segmentation/")
    return 0

if __name__=="__main__":
    main()
