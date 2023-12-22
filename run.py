import logging
import hydra
import submitit
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, open_dict
import time
from trainers import image_trainer, mixed_trainer
import trainers.trainer_utils as tut

LOG = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    ''' Important Note:  When defining a new model, trainer or dataset double check trainer_utils.get_trainer, model_utils.get_model and dataloaders.getDataloaders '''
    LOG.info(OmegaConf.to_yaml(cfg))
    trainer = tut.get_trainer(cfg.trainer.type, cfg)
    LOG.info(
        f"Output directory {cfg.trainer.output_dir}/{cfg.trainer.sync_key}")
    trainer()
    # trainer.eval() #added it in the  __call__ (better for submitit)
    LOG.info("Done with Run")
    # I/O overhead creates errors when SummaryWriter is called to recently
    time.sleep(1.5)
    return 0


if __name__ == "__main__":
    main()
