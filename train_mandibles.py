import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

from jawfrac.datamodules import MandibleSegDataModule
from jawfrac.models import MandibleSegModule


def train():
    #with open('jawfrac/config/mandibles.yaml', 'r') as f:
    with open('./config/mandibles.yaml', 'r') as f:
        config = yaml.safe_load(f)


    pl.seed_everything(config['seed'])

    dm = MandibleSegDataModule(
        root=config['datamodule']['root'],
        batch_size=config['datamodule']['batch_size'],
        num_workers=config['datamodule']['num_workers'],
        patch_size=config['datamodule']['patch_size'],
        gamma_adjust=config['datamodule']['gamma_adjust'],
        max_patches_per_scan=config['datamodule']['max_patches_per_scan'],
        ignore_outside=config['datamodule']['ignore_outside'],
        regular_spacing=config['datamodule']['regular_spacing'],
        stride=config['datamodule']['stride'],
        regex_filter=config['datamodule']['regex_filter'],
        exclude=config['datamodule']['exclude'],  # Add this line
        val_size=config['datamodule']['val_size'],
        test_size=config['datamodule']['test_size'],
        pin_memory=config['datamodule']['pin_memory'],
        persistent_workers=config['datamodule']['persistent_workers'],
        seed=config['seed'],
    )

    model = MandibleSegModule(
        num_classes=dm.num_classes,
        **config['model'],
    )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=config['version'],
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)


    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='epoch',
        mode='max',
        filename='weights-{epoch:02d}',
    )
    loss_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='loss/val',
        filename='weights-{epoch:02d}',
    )
    metric_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='f1/val',
        mode='max',
        filename='weights-{epoch:02d}',
    )


    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=config['model']['epochs'],
        logger=logger,
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=2,
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            metric_checkpoint_callback,
            LearningRateMonitor(),
        ],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    train()
