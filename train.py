import numpy as np
import lightning as L
from lightning.pytorch import Trainer 
from huggingface_hub import hf_hub_download
from model import *
from dataset import *


def main():
    gin.parse_config_file("config/gpt2-small.gin")
    config = GPTConfig()

    L.seed_everything(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    gpt2_small = GPT.from_pretrained("gpt2")
    gpt2_small.to(torch.bfloat16)

    hf_hub_download(repo_id="pt-sk/chatgpt-dataset", filename="conversation_tokens.npy", repo_type="dataset", local_dir="/kaggle/working")
    tokens = np.load("/kaggle/working/conversation_tokens.npy")

    dataset = TokenDataset(config, tokens)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    gpt2_wrapper = GPT2Wrapper(config, gpt2_small)

    trainer = Trainer(max_epochs=config.epochs, devices=2, accelerator="gpu", gradient_clip_val=1.0, strategy="ddp")
    trainer.fit(gpt2_wrapper, dataloader)

    np.save("/kaggle/working/train_loss.npy", gpt2_wrapper.train_loss, allow_pickle=True)


if __name__ == "__main__":
    main()