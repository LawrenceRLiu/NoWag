import torch
import os
import yaml
import tqdm

from ..data import get_loaders
from .ppl import ppl_eval_basic
from .zero_shot import zero_shot


@torch.no_grad()
def eval(compressed_model, cfg):

    compressed_model.eval()
    # ppl eval
    if hasattr(cfg.eval, "ppl_dataset"):
        if len(cfg.eval.ppl_dataset) == 0:
            print("No ppl datasets specified, skipping ppl eval")
        else:
            for dataset in cfg.eval.ppl_dataset:
                # seed(cfg.seed, seed_all = True)
                testloader = get_loaders(
                    dataset,
                    nsamples=0,
                    seqlen=compressed_model.seqlen,
                    model=cfg.base_model,
                    train_test="test",
                )

                ppl_eval_basic(
                    model=compressed_model,
                    testenc=testloader,
                    dataset_name=dataset,
                    results_log_yaml=os.path.join(cfg.save_path, "results.yaml"),
                    log_wandb=cfg.log_wandb,
                )

    # zero shot eval
    if hasattr(cfg.eval, "zero_shot_tasks"):
        if len(cfg.eval.zero_shot_tasks) == 0:
            print("No zero shot tasks specified, skipping zero shot eval")
            return
        # seed(cfg.seed,seed_all = True)
        zero_shot(
            cfg.base_model,
            compressed_model,
            tasks=cfg.eval.zero_shot_tasks,
            results_log_yaml=os.path.join(cfg.save_path, "results.yaml"),
            log_wandb=cfg.log_wandb,
        )
