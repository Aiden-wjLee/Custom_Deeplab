import torch

num_gpus = torch.cuda.device_count()
print(f"Total available GPUs: {num_gpus}")

for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")