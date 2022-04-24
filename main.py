from argparse import ArgumentParser
import importlib
from pathlib2 import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from loader import load_model_path_by_args


MODEL_NAME = "video_hw1_cnn"
DATA_MODULE_NAME = "video_hw1_data"


def load_model():
    camel_name = "".join([i.capitalize() for i in MODEL_NAME.split("_")])
    try:
        model = getattr(
            importlib.import_module(
                "model." + MODEL_NAME, package=__package__),
            camel_name,
        )
    except Exception:
        raise ValueError(
            f"Invalid Module File Name or Invalid Class Name {MODEL_NAME}.{camel_name}!"
        )
    return model


def load_dm():
    camel_name = "".join([i.capitalize() for i in DATA_MODULE_NAME.split("_")])
    try:
        dm = getattr(
            importlib.import_module(
                "data." + DATA_MODULE_NAME, package=__package__),
            camel_name,
        )
    except Exception:
        raise ValueError(
            f"Invalid Module File Name or Invalid Class Name {DATA_MODULE_NAME}.{camel_name}!"
        )
    return dm


def load_callbacks(args):
    callbacks = [
        plc.ModelCheckpoint(
            monitor=args.monitor,
            filename="-".join(["best", "{epoch:02d}", "{val_loss:.4f}", "{" + args.monitor + ":.4f}"]),
            save_top_k=args.save_top_k,
            mode="max",
            save_last=True,
            verbose=True,
        ),
        plc.LearningRateMonitor(logging_interval="step"),
    ]
    if not args.no_early_stop:
        callbacks.append(
            plc.EarlyStopping(
                monitor=args.monitor, mode="min", patience=5, min_delta=0.001, verbose=True
            ),
        )
    return callbacks


def main(parent_parser):
    # load model specific
    model = load_model()
    if hasattr(model, "add_model_specific_args"):
        parent_parser = model.add_model_specific_args(parent_parser)
    dm = load_dm()
    if hasattr(dm, "add_model_specific_args"):
        parent_parser = dm.add_model_specific_args(parent_parser)
    
    args = parent_parser.parse_args()
    
    # use nni to tune hyperparameters
    params = vars(args)
    if args.use_nni:
        import nni
        tuner_params = nni.get_next_parameter()
        params.update(tuner_params)
    
    pl.seed_everything(params['seed'])
    
    # initilize data module
    dm = dm(**params)
    # add some info from dataset
    ex_params = dm.add_data_info() if hasattr(dm, "add_data_info") else {}
    params.update(ex_params)
    
        
    # initilize model
    model = model(**params)
    
    # restart setting
    if args.load_dir is not None or args.load_ver is not None or args.load_v_num is not None:
        load_path = load_model_path_by_args(args)
        print(f'load model from {load_path}...')
    else:
        load_path = None
    
    # initilize logger
    if args.debug:
        logger = None
    else:
        if args.wandb:
            logger = WandbLogger(
                project=args.project_name, save_dir=args.log_dir, log_model=False,
                version=args.load_ver,
            )
            logger.watch(model)
        else:
            logger = TensorBoardLogger(
                save_dir=args.log_dir, name=args.project_name, version=args.load_ver,)
    
    # initilize callbacks
    args.callbacks = load_callbacks(args)
    
    # initilize trainer
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        strategy=args.multi_gpu_strategy,
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        min_epochs=None if args.no_early_stop else args.min_epochs,
        precision=args.precision,
        fast_dev_run=args.debug,
        callbacks=None if args.debug else args.callbacks,
    )
    
    # start training
    trainer.fit(model, dm, ckpt_path=load_path)
    # end of training
    if args.use_nni:
        metrics = trainer.validate(model, dm)
        nni.report_final_result(metrics[0][args.monitor])
        

def get_params():
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument("--log_dir", type=str, default="/data/contest/pytorch-lightning/lightning_logs")
    parser.add_argument("--project_name", type=str, default="default_run")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=30, type=int)
    parser.add_argument("--min_epochs", default=10, type=int)
    parser.add_argument("--gpus", default="-1", type=str)
    parser.add_argument("--multi_gpu_strategy", default=None)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_top_k", type=int, default=3)

    # Restart Control
    parser.add_argument("--load_best", action="store_true")
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--load_ver", default=None, type=str)
    parser.add_argument("--load_v_num", default=None, type=int)

    # Training Info
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--use_nni", action="store_true")
    parser.add_argument("--monitor", type=str, default="val_acc")

    # parser = pl.Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    # parser.set_defaults(max_epochs=40)
    # parser.set_defaults(gpus=1)
    
    return parser


if __name__ == "__main__":
    try:
        main(get_params())
        
    except Exception as exception:
        raise exception
