# train.py
import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel

from datetime import datetime
import pytz


def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


if __name__ == '__main__':
    base_parser = ArgumentParser(add_help=False)
    base_parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
    base_parser.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="flowmatching")
    base_parser.add_argument("--no_wandb", action='store_true',
                             help="(保留选项供兼容) 不使用 W&B；我们统一使用 TensorBoard")
    temp_args, _ = base_parser.parse_known_args()

    parser = ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(),
                        default=temp_args.backbone)
    parser.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default=temp_args.ode)
    parser.add_argument("--no_wandb", action='store_true', default=temp_args.no_wandb)

    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    ode_class = ODERegistry.get_by_name(temp_args.ode)

    parser = pl.Trainer.add_argparse_args(parser)
    VFModel.add_argparse_args(parser.add_argument_group("VFModel", description=VFModel.__name__))
    ode_class.add_argparse_args(parser.add_argument_group("ODE", description=ode_class.__name__))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    SpecsDataModule.add_argparse_args(parser.add_argument_group("DataModule", description=SpecsDataModule.__name__))

    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser, args)

    dataset = os.path.basename(os.path.normpath(arg_groups['DataModule'].base_dir))
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    formatted_time_kst = now_kst.strftime("%Y%m%d%H%M%S")
    exp_name = f"dataset_{dataset}_{formatted_time_kst}"

    root_dir = getattr(args, "default_root_dir", None) or "lightning_logs"

    model = VFModel(
        backbone=args.backbone,
        ode=args.ode,
        data_module_cls=SpecsDataModule,
        **{
            **vars(arg_groups['VFModel']),
            **vars(arg_groups['ODE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule'])
        }
    )

    logger = TensorBoardLogger(save_dir=root_dir, name=exp_name)

    ckpt_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_last = ModelCheckpoint(
        dirpath=ckpt_dir, save_last=True, filename='{epoch}_last',
        save_on_train_epoch_end=False
    )
    checkpoint_pesq = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=20, monitor="pesq", mode="max",
        filename='{epoch}_{pesq:.2f}', save_on_train_epoch_end=False
    )
    checkpoint_si_sdr = ModelCheckpoint(
        dirpath=ckpt_dir, save_top_k=20, monitor="si_sdr", mode="max",
        filename='{epoch}_{si_sdr:.2f}', save_on_train_epoch_end=False
    )
    callbacks = [checkpoint_last, checkpoint_pesq, checkpoint_si_sdr]

    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=logger,
        default_root_dir=root_dir,
        log_every_n_steps=10,
        num_sanity_val_steps=1,
        callbacks=callbacks,
        gradient_clip_val=getattr(args, "gradient_clip_val", 1.0),
    )

    # 训练
    trainer.fit(model)
