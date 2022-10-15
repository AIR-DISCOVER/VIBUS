import os
import torch
from tqdm import tqdm
import time
from logging import log, ERROR

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device, idx):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used

    log(ERROR, f"{total} - {used} - {block_mem}")
    x = torch.cuda.FloatTensor(256,1024,block_mem, device=f'cuda:{idx}')
    del x
    
if __name__ == '__main__':
    cuda_device=0
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    occumpy_mem(cuda_device)
    for _ in tqdm(range(60)):
        time.sleep(1)
    print('Done')