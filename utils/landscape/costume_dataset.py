import torch
import torchvision
from torchvision import transforms, datasets

from utils.attack.pgd import pgd_whitbox_image
from utils.dataset.build_datasetloader import build_dataloader
from utils.fileio.config import Config


def build_custom_dataset(cfg, model, log=None):
    cfg_atk = Config.fromfile("configs/Attack/pgd_whitbox_imagenette.py")
    data_loader = build_dataloader(cfg, log=log)
    flag = True
    for batch_idx, (data, target) in enumerate(data_loader["test"]):
        image_bat = data[0:100]
        target_bat = target[0:100]

        adv_image = pgd_whitbox_image(model, image_bat, target_bat, cfg.device, cfg_atk)
        # if flag:
        #     targets = target_bat
        #     adv_images = adv_image.cpu()
        #     flag = False
        # else:
        #     targets = torch.concat((targets, target_bat))
        #     adv_images = torch.concat((adv_images, adv_image.cpu()))

        
        # adv_images = image_bat
        dataset = Custom_Dataset(adv_image, target_bat)
        # dataset = Custom_Dataset(adv_images, targets)
        break
        # break
    test_loader = {"test":torch.utils.data.DataLoader(dataset, batch_size=300, shuffle=False)}
    return test_loader


class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, images, target) -> None:
        super().__init__()
        self._images = images
        self._target = target
    
    def __len__(self):
        return len(self._target)

    def __getitem__(self, idx):
        return self._images[idx], self._target[idx]

