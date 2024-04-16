import os
from monai.transforms import LoadImaged, EnsureChannelFirstd,Orientationd,Resized,Compose
from multiprocessing import Pool
import nrrd
import numpy as np

def make_list(dir_):
    val_files = []
    for file in os.listdir(dir_):
        if file.endswith(".dcm"):
            mask = file.replace(".dcm",".seg.nrrd")
            if os.path.isfile(os.path.join(dir_,mask)):
                val_files.append({"image":os.path.join(dir_,file),"mask":os.path.join(dir_,mask)})
    return val_files

def process(i):
    print(i["image"],i["mask"])
    
    head=nrrd.read_header(i['mask'])
    out=train_transforms(i)
    head['sizes'] = np.array([512,496,97])
    image = i['image'].replace("raw","resize")
    image = image.replace(".dcm","RESIZED.nrrd")
    mask = i['mask'].replace("raw","resize")
    mask = mask.replace(".seg.nrrd","RESIZED.seg.nrrd")
    nrrd.write(image,out['image'].numpy(),header=head)
    nrrd.write(mask,out['mask'].numpy(),header=head)
    return 0


if __name__=="__main__":
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'mask']),
            EnsureChannelFirstd(keys=['image','mask']),
            Orientationd(keys=['mask'],axcodes='SPR'),
            Resized(keys=['image','mask'],spatial_size=(512, 496, 97))
        ])
    vals=make_list("/home/shared/projects/oct_segmentation_rds/data/raw/train")
    with Pool(16) as p:
        print(p.map(process, vals))
