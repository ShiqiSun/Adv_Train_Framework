import torch
from torchvision import transforms
from PIL import Image

def to_image(img, path='/home/shiqisun/train_framework/test_train_code/images/visual.jpg', img_size=224):
    img = torch.tensor(img.reshape(3, img_size, img_size))
    toPIL = transforms.ToPILImage()
    pic = toPIL(img)
    pic.save(path)


def load_image(path):
    img = Image.open(path)
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    return img

def inverse_transform(mean_f, std_f):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/std_f[0], 1/std_f[1], 1/std_f[2] ]),
                                transforms.Normalize(mean = [ -1*mean_f[0], -1*mean_f[1], -mean_f[2] ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    return invTrans