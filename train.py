import os,sys
import util
from data import get_split_dataset
from model import make_model, loss
from render import NeRFRenderer
import numpy as np
import torch.nn.functional as F 
import torch
from dotmap import DotMap
import os.path
import trainlib

args, conf = util.args.parse_args()
device = util.get_cuda(args.gpu_id[0])

#dataset, val_dataset, _ = get_split_dataset(args.dataset_format, args.datadir)