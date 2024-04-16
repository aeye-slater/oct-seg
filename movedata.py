import pandas as pd
import shutil
import os
import glob

def mostRecent(listofpaths):
    best = 'missing'
    best_time = 0
    for path_ in listofpaths:
        filetime = os.path.getmtime(path_)
        if "NFL" in path_:
            pass
        else:
            if filetime > best_time:
                best_time = filetime
                best = path_
    return best

def main():
    BASE = "/mnt/share/Q/L drive cleanup - 2023/EPOCH"
    LOCAL = "/home/shared/projects/oct_segmentation_rds/data/raw/train"
    train = pd.read_csv("/mnt/share/Y/OCT Segmentation/training_file.csv",dtype='object')
    inputs=[]
    targets = []
    sets = []
    for i in range(train.shape[0]):
        src = os.path.join(BASE, str(train.loc[i,"Site"]),str(train.loc[i,"Patient"]),"Screen",str(train.loc[i,"File"]))
        dest = os.path.join(LOCAL,train.loc[i,"File"])
        shutil.copy(src,dest)
        src=src.replace(".dcm","*.mat")
        files = glob.glob(src)
        seg_path = mostRecent(files)
        if seg_path != 'missing': 
            name = seg_path.split("/")[-1]
            dest = os.path.join(LOCAL,name)
            shutil.copy(seg_path,dest)
            inputs.append(train.loc[i,"File"])
            targets.append(name)
            sets.append('train')
    LOCAL = "/home/shared/projects/oct_segmentation_rds/data/raw/val"
    val = pd.read_csv("/mnt/share/Y/OCT Segmentation/validation_file.csv",dtype='object')
    for i in range(val.shape[0]):
        src = os.path.join(BASE, str(val.loc[i,"Site"]),str(val.loc[i,"Patient"]),"Screen",str(val.loc[i,"File"]))
        dest = os.path.join(LOCAL,val.loc[i,"File"])
        shutil.copy(src,dest)
        src=src.replace(".dcm","*.mat")
        files = glob.glob(src)
        seg_path = mostRecent(files)
        if seg_path != 'missing':
            name = seg_path.split("/")[-1]
            dest = os.path.join(LOCAL,name)
            shutil.copy(seg_path,dest)
            inputs.append(val.loc[i,"File"])
            targets.append(name)
            sets.append("val")
    df = pd.DataFrame({"Input":inputs,"Target":targets, "Set":sets})
    df.to_csv("/home/shared/projects/oct_segmentation_rds/code/data_list.csv")
    return 0

if __name__=="__main__":
    main()
