import os
import scipy.io
from pydicom import dcmread
import glob
import numpy as np
import nrrd
import pandas as pd

count=0
errors = []
err_type=[]
BASE = "/home/shared/projects/oct_segmentation_rds/data/raw/train"
files = os.listdir(BASE)
for file in files:
    if file.endswith("dcm"):
        count+=1
        try:
            tmp = dcmread(os.path.join(BASE,file))
            slices,X,Y = tmp.pixel_array.shape
        except:
            print("file won't open or corrupt {file}")
            errors.append(file)
            err_type.append("Bad Dicom")
        seg_search = file.replace(".dcm","*.mat")
        segs = glob.glob(f'/home/shared/projects/oct_segmentation_rds/data/raw/train/{seg_search}')
        if len(segs) < 1:
            errors.append(file)
            err_type.append("no segmentation")
        elif len(segs)==1:
            try:
                seg = scipy.io.loadmat(os.path.join(BASE,segs[0]))
                tmp=seg.get('OCTLayers')
                for i in range(tmp.shape[1]):
                    label = tmp[0][i][0]
                    if len(label) > 0:
                        if label[0].upper()=='ILM':
                            ilm=tmp[0][i][6]
                        if label[0].upper()=='RPE_ON':
                            rpe=tmp[0][i][6]
                area = np.zeros((slices,X,Y))
                for row in range(slices):
                    for col in range(Y):
                        tmp = np.zeros((X))
                        tmp[int(ilm[row,col]):int(rpe[row,col])]=1
                        area[row,:,col]=tmp

                nrrd_name = file.replace(".dcm",".seg.nrrd")
                header = {}
                header['space directions'] = np.array([
                                                  [0.023144  , 0.        , 0.        ],
                                                  [0.        , 0.003751  , 0.        ],
                                                  [0.        , 0.        , 0.06227703]])
                header['space origin'] = np.array([0., 0., 0.])
                header['space']= 'left-posterior-superior'
                nrrd.write(os.path.join(BASE,nrrd_name),area,header=header)
                errors.append(file)
                err_type.append("None")
            except:
                errors.append(file)
                err_type.append("nrrd creation failed")
    if count%10==0:
        print(count)
log = pd.DataFrame({"File Name":errors,"Error":err_type})
log.to_csv("segmentation_train_errors.csv")
