import numpy as np
import lightning as L
from lightning.pytorch import Trainer 
from huggingface_hub import hf_hub_download
from model import *
from dataset import *
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch

gin.parse_config_file("config/gpt2-small.gin")
config = GPTConfig()

L.seed_everything(config.seed)
torch.manual_seed(config.seed)
np.random.seed(config.seed)


hf_hub_download(repo_id="pt-sk/chatgpt-dataset", filename="conversation_tokens.npy", repo_type="dataset", local_dir="/kaggle/working")
tokens = np.load("/kaggle/working/conversation_tokens.npy")



def trainer(rank, world_size):
    # 1. Initialize the Process Group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # 2. Set the Device for the Current Process
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # 3. Define Model, Loss, and Optimizer
    model = GPT.from_pretrained("gpt2")
    model.to(torch.bfloat16).to(device)
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)


    dataset = TokenDataset(config, tokens)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, drop_last=True)

    # 5. Training Loop
    model.train()
    for epoch in range(10):  # Example epochs
        sampler.set_epoch(epoch)  # Shuffle data per epoch for each process
        for batch, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            _, loss = model(inputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and batch % 10 == 0:  # Log only on rank 0
                print(f"Epoch [{epoch+1}/10], Batch [{batch+1}/{len(dataloader)}], Loss: {loss.item()}")

    # 6. Cleanup
    dist.destroy_process_group()


def run_ddp_training():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(trainer, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    import os
    import torch

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())  # Total number of GPUs on this node
    os.environ['RANK'] = '0'  # Rank 0 for a single-node setup

    run_ddp_training()