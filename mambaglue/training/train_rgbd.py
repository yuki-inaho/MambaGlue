"""Hydra/CLI entry point for MambaGlue RGB-D training on top of glue-factory.

glue-factory hardcodes its optimizer table inside ``Trainer.construct_optimizer``
and never puts AMUSE into train mode, so this wrapper installs an
``AmuseTrainer`` subclass and then delegates to glue-factory's own CLI
(``gluefactory.train``). The glue-factory tree is left unmodified.

Example::

    uv run --no-sync python -m mambaglue.training.train_rgbd mambaglue_rgbd \
      --conf mambaglue/training/configs/superpoint_rgbd+mambaglue_tomato.yaml \
      data.root=/path/to/<staging-root> data.train_size=20 data.val_size=10 \
      train.epochs=1 train.log_every_iter=1
"""

from __future__ import annotations

import torch
from gluefactory import trainer as trainer_module
from gluefactory.trainer import Trainer

from mambaglue.training.optim.amuse import AMUSE
from mambaglue.training.optim.param_groups import build_amuse_param_groups

_AMUSE_OPTION_KEYS = (
    "muon_lr",
    "aux_lr",
    "warmup_steps",
    "beta1",
    "rho",
    "r",
    "weight_lr_power",
    "weight_decay",
    "weight_decay_at_y",
    "momentum",
    "aux_update_type",
)


class AmuseTrainer(Trainer):
    """Trainer that can build AMUSE and manages its train/eval mode."""

    @classmethod
    def construct_optimizer(cls, conf, model):
        if str(conf.optimizer) != "amuse":
            return super().construct_optimizer(conf, model)

        options = {
            str(key): value for key, value in dict(conf.optimizer_options).items()
        }
        unknown = sorted(set(options) - set(_AMUSE_OPTION_KEYS))
        if unknown:
            raise ValueError(f"unknown AMUSE optimizer_options: {unknown}")

        warmup_steps = int(options.pop("warmup_steps", 0))
        if warmup_steps <= 0:
            raise ValueError("AMUSE requires optimizer_options.warmup_steps > 0")

        groups = build_amuse_param_groups(
            model,
            muon_lr=float(options.pop("muon_lr", conf.lr)),
            aux_lr=float(options.pop("aux_lr", conf.lr)),
            weight_decay=float(options.pop("weight_decay", 0.0)),
            momentum=float(options.pop("momentum", 0.95)),
            aux_update_type=str(options.pop("aux_update_type", "adamw")),
        )
        return AMUSE(
            groups,
            warmup_steps=warmup_steps,
            beta1=float(options.pop("beta1", 0.9)),
            rho=float(options.pop("rho", 1.0)),
            r=float(options.pop("r", 0.0)),
            weight_lr_power=float(options.pop("weight_lr_power", 2.0)),
            weight_decay_at_y=float(options.pop("weight_decay_at_y", 0.0)),
        )

    def _toggle_amuse_mode(self, enter_train: bool) -> None:
        optimizer = getattr(self, "optimizer", None)
        if not isinstance(optimizer, AMUSE):
            return
        if enter_train:
            optimizer.train()
        else:
            optimizer.eval()

    def train_epoch(self, *args, **kwargs):
        self._toggle_amuse_mode(True)
        try:
            return super().train_epoch(*args, **kwargs)
        finally:
            self._toggle_amuse_mode(False)

    def train_loop(self, *args, **kwargs):
        self._toggle_amuse_mode(True)
        try:
            return super().train_loop(*args, **kwargs)
        finally:
            self._toggle_amuse_mode(False)


def _install_null_lr_scheduler() -> None:
    """Make ``lr_schedule.type=null`` a no-op (glue-factory crashes on None).

    AMUSE owns its own learning-rate schedule, so the surrounding trainer must
    not wrap it with an LR scheduler.
    """
    from gluefactory.utils import tools as tools_module

    original = tools_module.get_lr_scheduler

    def get_lr_scheduler(optimizer, conf):
        if conf.type is None:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        return original(optimizer, conf)

    tools_module.get_lr_scheduler = get_lr_scheduler


def install_amuse_trainer() -> None:
    """Make glue-factory's ``launch_training`` use :class:`AmuseTrainer`."""
    trainer_module.Trainer = AmuseTrainer
    _install_null_lr_scheduler()


def main() -> None:
    install_amuse_trainer()

    from gluefactory import train as glue_train

    args = glue_train.parse_args()
    output_dir = glue_train.create_training_dir(args.experiment, args)
    conf = glue_train.compose_cli_config(output_dir, args)
    conf.train.num_devices = conf.train.get("num_devices", 0)

    glue_train.save_code_snapshot(
        output_dir,
        conf.train.get("submodules", ()),
        compression=args.compress_snapshot,
    )

    if args.distributed and conf.train.num_devices < 1:
        conf.train.num_devices = torch.cuda.device_count()

    if conf.train.num_devices > 0:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(
            glue_train.main_worker,
            nprocs=conf.train.num_devices,
            args=(conf, output_dir, args),
        )
    else:
        glue_train.main_worker(0, conf, output_dir, args)


if __name__ == "__main__":
    main()
