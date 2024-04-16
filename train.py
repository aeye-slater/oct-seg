import os
import pandas as pd
from pathlib import Path
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from Models import Unetr
from monai.transforms import (
    Compose,
    LoadImaged,
    RandCropByLabelClassesD,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    EnsureChannelFirstd
)
from monai.data import (
    DataLoader,
    Dataset,
)

def make_list(dir_):
    val_files = []
    for file in os.listdir(dir_):
        if file.endswith("D.nrrd"):
            mask = file.replace(".nrrd",".seg.nrrd")
            if os.path.isfile(os.path.join(dir_,mask)):
                val_files.append({"image":os.path.join(dir_,file),"mask":os.path.join(dir_,mask)})
    return val_files

def main(
        model_desc,
        training_seed,
        project_dir,
        model_dims=(96, 96, 96),
        batch_size=4,
        num_workers=16) -> None:
    seed_everything(training_seed, workers=True)
    CKPT_PATH = '/home/shared/projects/oct_segmentation_rds/ckpts'

    mod = Unetr(model_dims=model_dims,
                num_masks=1,
                sw_batch_size=4)
    info_dir = project_dir / 'data' / 'tables' / 'train_tune' / 'info'
    working_data_dir = Path('/home/shared/data') / 'oct_segm_layers'
    model_dir = working_data_dir / 'models' / model_desc
    model_dir.mkdir(parents=True, exist_ok=True)
    train_ds_path = info_dir / 'train_resized_ds.csv'
    train_ds_files = make_list("/home/shared/projects/oct_segmentation_rds/data/resize/train")
    tune_ds_files = make_list("/home/shared/projects/oct_segmentation_rds/data/resize/val")
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'mask']),
            NormalizeIntensityd(keys=['image']),
            RandCropByLabelClassesD(
                    keys=["image", "mask"],
                    ratios=[1, 1],
                    spatial_size=(96, 96, 96),
                    label_key='mask',
                    num_samples=2,
                    num_classes=2
                ),
        ]
        )
    tune_transforms = Compose(
            [LoadImaged(keys=['image', 'mask']),
             NormalizeIntensityd(keys=['image'])]
            )
    
    train_ds = Dataset(data=train_ds_files,
                       transform=train_transforms)
    tune_ds = Dataset(data=tune_ds_files,
                      transform=tune_transforms)
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=True)
    tune_dl = DataLoader(tune_ds,
                         batch_size=batch_size,
                         num_workers=num_workers)
    chckpnt_dir = model_dir / 'chckpnt'
    chckpnt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
                dirpath=CKPT_PATH,
                filename='{epoch}-{validation_dice:2f}v2',
                monitor='validation_dice',
                mode='max',
                save_top_k=3
                                          )
    early_stop = EarlyStopping(monitor='validation_dice',
                               min_delta=0.0005,
                               patience=10,
                               mode='max')
    logger = TensorBoardLogger("tb_logs", name="loss_modelv2")

    trainer = Trainer(
        accelerator='gpu',
        devices=[0, 1],
        max_epochs=100,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop],
        strategy='ddp',
        logger=logger,
    )
    trainer.fit(mod,
                train_dataloaders=train_dl,
                val_dataloaders=tune_dl)
    print("Success")
    return 0


if __name__ == "__main__":
    main(model_desc='unetr_diceCE_volm_ilm_rpe',
         training_seed=42,
         project_dir=Path('/home') / 'shared' / 'projects' / 'oct_segm_layers',
         model_dims=(96, 96, 96),
         batch_size=8,
         num_workers=8
         )
