
import torch
import pytorch_lightning as pl
from pathlib import Path
from torchmetrics import Dice
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


def change_path(old_path: str, new_dir: str) -> str:
    new_path = new_dir / Path(old_path).name
    new_path = new_path.absolute().as_posix()
    return new_path


class Unetr(pl.LightningModule):
    def __init__(self, model_dims, num_masks, sw_batch_size):
        super().__init__()
        self._model = UNETR(
            in_channels=1,
            out_channels=num_masks,
            img_size=model_dims,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0
            )
        self.loss_function = DiceCELoss(sigmoid=True)
        #self.post_pred = AsDiscrete(argmax=True,
        #                            to_onehot=num_masks
        #                            )
        self.post_label = AsDiscrete()
        self.dice_metric = Dice()
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs = 1300
        self.check_val = 1
        self.warmup_epochs = 10
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []
        self.model_dims = model_dims
        self.num_labels = num_masks
        self.sw_batch_size = sw_batch_size
        return

    def forward(self, x):
        output = self._model(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(),
                                     lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, masks = (batch['image'], batch['mask'])
        batch_size = images.shape[0]
        outputs = self._model(images)
        loss = self.loss_function(outputs, masks)
        self.log('train_loss', loss,
                 sync_dist=True,
                 batch_size=batch_size,
                 on_epoch=True,
                 on_step=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        batch_size = images.shape[0]
        roi_size = self.model_dims
        outputs = sliding_window_inference(images, roi_size,
                                           self.sw_batch_size,
                                           self.forward)
        outputs = [i for i in decollate_batch(outputs)]
        outputs = torch.stack(outputs, dim=0)
        loss = self.loss_function(outputs, masks)
        targets = torch.tensor(masks, dtype=torch.int)
        dice = self.dice_metric(outputs, targets)
        self.log('validation_loss', loss,
                 sync_dist=True,
                 batch_size=batch_size,
                 on_epoch=True,
                 on_step=False,
                 logger=True,
                 prog_bar=True)
        self.log('validation_dice', dice,
                 sync_dist=True,
                 batch_size=batch_size,
                 on_epoch=True,
                 on_step=False,
                 logger=True,
                 prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        # might want to add some more logic here
        # see OCTPredict.ipynb
        images = batch["image"]
        roi_size = self.model_dims
        outputs = sliding_window_inference(images, roi_size,
                                           self.sw_batch_size,
                                           self.forward)
        outputs = [i for i in decollate_batch(outputs)]
        outputs = torch.stack(outputs, dim=0)
        return outputs
