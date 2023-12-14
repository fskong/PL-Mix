import argparse
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
from config.advConfig import advParser
from config.common import common_args
from model.Adv import AdvModel
# from model.baseline import BasicModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def getarg():
    model_classes = {'ADV': AdvModel}
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(help='sub-command help.', dest='method')
    
    common_args(parser)
    ''' model '''
    # pctParser(subparsers)
    # mmdParser(subparsers)
    advParser(subparsers)
    # dannParser(subparsers)

    args = parser.parse_args()
    args.seed = args.seed or random.randint(0, 2**32-1)
    set_seed(args.seed)
    args.timestamp = args.timestamp or str(int(time.time())) + format(random.randint(0, 999), '03')
    args.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    args.model_class = model_classes[args.method] if args.method else BasicModel

    return args
