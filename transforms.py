# your_transforms.py
import random
import torchvision.transforms.functional as F

def get_transform(train: bool):
    def _trans(img, target):
        img = F.to_tensor(img)
        if train and random.random() < 0.5:
            img = F.hflip(img)
            w = img.shape[-1]
            boxes = target['boxes']
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            target['boxes'] = boxes
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-1)
        img = F.normalize(img,
                          mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225])
        return img, target
    return _trans
