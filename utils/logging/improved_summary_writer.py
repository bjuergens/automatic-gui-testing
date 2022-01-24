import os

from torch.utils.tensorboard import SummaryWriter


class ImprovedSummaryWriter(SummaryWriter):

    def __init__(self, log_dir: str, name: str, **kwargs):

        root_save_dir = os.path.join(log_dir, name)

        if not os.path.exists(root_save_dir):
            os.makedirs(root_save_dir)

        self.version_number = len(os.listdir(root_save_dir))
        save_dir = os.path.join(root_save_dir, f"version_{self.version_number}")
        os.makedirs(save_dir)

        super().__init__(log_dir=save_dir, **kwargs)
