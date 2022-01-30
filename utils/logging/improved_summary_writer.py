import os
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class ImprovedSummaryWriter(SummaryWriter):

    def __init__(self, log_dir: str, name: Optional[str] = None, **kwargs):

        if name is not None:
            root_save_dir = os.path.join(log_dir, name)
        else:
            root_save_dir = log_dir

        if not os.path.exists(root_save_dir):
            os.makedirs(root_save_dir)

        root_dir_content = [sub_dir for sub_dir in os.listdir(root_save_dir) if "version" in sub_dir]
        root_dir_content.sort()

        if len(root_dir_content) > 0:
            self.version_number = int(root_dir_content[-1].split("version_")[-1]) + 1
        else:
            self.version_number = 0

        save_dir = os.path.join(root_save_dir, f"version_{self.version_number}")
        os.makedirs(save_dir)

        super().__init__(log_dir=save_dir, **kwargs)
