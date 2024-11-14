import numpy as np
import lightning as L
from lightning.pytorch import Trainer 
from huggingface_hub import hf_hub_download
from model import *
from dataset import *