import os
import numpy as np

import torch

class Logger:

    def __init__(self, ):
        self.log = {
            'epoch': None,
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'val_loss': None
        }

    def save(self, PATH):
        print('Saving log to {}'.format(PATH))
    
        torch.save(self.log, PATH)

    def load(self, PATH):
        print('Loading log from {}'.format(PATH))

        self.log = torch.load(PATH)

    def reset(self, **key_val_pairs):
        self.log = key_val_pairs

    def update(self, **key_val_pairs):
        '''
        Update an item in log. Add key-value pair to log if key does not exist.
        '''
        for key, value in key_val_pairs.items():
            self.log[key] = value


if __name__ == '__main__':
    import torchvision.models as models

    resnet18 = models.resnet18()
    resnet18.logger = Logger()

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    resnet18.logger.save('./checkpoints/log.t7')
    resnet18.logger.load('./checkpoints/log.t7')

    print(resnet18.logger.log)
    resnet18.logger.update(loss='bar', foo='bar')
    print(resnet18.logger.log)

    resnet18.logger.reset(foo='bar')
    print(resnet18.logger.log)

    