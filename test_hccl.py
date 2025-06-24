import torch, torch_npu
import torch.distributed as dist
import os

def main():
    """
    A minimal script to test torch.distributed.all_reduce on Ascend NPUs (HCCL backend).
    This script should be launched using `accelerate launch` or `torchrun`.
    """
    try:
        # Accelerate/torchrun will set these env vars
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # IMPORTANT: Use 'hccl' backend for Ascend NPUs
        print(f"[Rank {rank}] Initializing process group with 'hccl' backend...")
        dist.init_process_group(backend='hccl')
        print(f"[Rank {rank}] Process group initialized successfully.")

        # Each process gets a tensor on its corresponding device (NPU)
        device = f'npu:{local_rank}'
        torch.npu.set_device(device)
        
        # Create a tensor with value equal to its rank
        tensor = torch.tensor([rank + 1], dtype=torch.float32).to(device)
        
        print(f"[Rank {rank}] On device '{device}', before all_reduce, tensor is: {tensor.item()}")

        # Perform the all_reduce operation
        print(f"[Rank {rank}] Performing all_reduce...")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] all_reduce completed.")

        # Expected output: Sum of all ranks (e.g., for 8 ranks: 1+2+...+8 = 36)
        expected_sum = sum(range(1, world_size + 1))
        
        print(f"[Rank {rank}] After all_reduce, tensor is: {tensor.item()}")

        if tensor.item() == expected_sum:
            print(f"[Rank {rank}] Test PASSED!")
        else:
            print(f"[Rank {rank}] Test FAILED! Expected {expected_sum}, got {tensor.item()}")

    except KeyError:
        print(
            "\nERROR: Distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK) not found. "
            "Please run this script with a distributed launcher, for example: \n"
            "accelerate launch --num_processes=8 test_hccl.py\n"
        )
        raise
    except Exception as e:
        rank_str = os.environ.get('RANK', 'N/A')
        print(f"[Rank {rank_str}] An error occurred: {e}")
        # Re-raise the exception to make the launch script fail
        raise e

if __name__ == "__main__":
    main() 