import os
import torch
import torch.distributed as dist
from lm_eval.comm import apply_td_gather_object_patch, remove_td_gather_object_patch


def debug_rankwise(var):
    print(
        f"[RANK {dist.get_rank()}/{dist.get_world_size()}] - "
        f"{var = }"
    )

def init_distributed_hccl():
    """Initializes the distributed environment for Huawei HCCL."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("RANK and WORLD_SIZE environment variables are not set. Assuming single process.")
        # For local testing without torchrun, you might want to set them manually
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "29500" 
        # os.environ["RANK"] = "0"
        # os.environ["WORLD_SIZE"] = "1"
        # dist.init_process_group(backend='hccl', rank=0, world_size=1)
        return False # Indicate that distributed was not truly initialized

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # HCCL backend specific initialization
    # torch_npu.npu.set_device(rank % torch_npu.npu.device_count()) # if you have multiple NPUs per node
    torch.npu.set_device(rank % torch.npu.device_count())

    # MASTER_ADDR and MASTER_PORT should be set by the launch utility (e.g., torchrun)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    print(f"HCCL backend initialized for rank {rank} of {world_size}, npu {rank % torch.npu.device_count()} set")
    return True

def test_all_gather_object():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Each process creates a different tensor
    my_tensor = torch.tensor([rank * 10 + i for i in range(3)], dtype=torch.float32).to(rank) 
    debug_rankwise(my_tensor)

    # Prepare a list to store all gathered objects
    all_gathered_objects = [None] * world_size
    dist.all_gather_object(all_gathered_objects, my_tensor)
    debug_rankwise(all_gathered_objects)

    if rank == 0:
        print("\n--- Verifying torch.distributed.all_gather_object ---")
        correct = True
        for i in range(world_size):
            expected_tensor = torch.tensor([i * 10 + j for j in range(3)], dtype=torch.float32)
            # gathered tensors will be on CPU by default from all_gather_object
            if not torch.equal(all_gathered_objects[i].cpu(), expected_tensor):
                print(f"Rank 0: Error! all_gather_object for rank {i}: Expected {expected_tensor}, Got {all_gathered_objects[i]}")
                correct = False
        if correct:
            print("Rank 0: torch.distributed.all_gather_object verification PASSED.")
        else:
            print("Rank 0: torch.distributed.all_gather_object verification FAILED.")
        # print("Rank 0: All gathered objects (all_gather_object):")
        # for i, obj in enumerate(all_gathered_objects):
        # print(f"  Rank {i}'s tensor: {obj}")

def test_custom_gather_object():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dst_rank = 0

    # Each process creates a different tensor
    my_data = {"rank": rank, "data": [rank * 100 + i for i in range(2)], "device": f"npu:{rank}"}
    debug_rankwise(my_data)

    # Apply the patch to use the custom gather_object
    apply_td_gather_object_patch()

    gathered_objects_on_dst = None
    # if rank == dst_rank:
    #     gathered_objects_on_dst = [None] * world_size
    gathered_objects_on_dst = [None] * world_size

    # This now calls our custom_td_gather_object_with_all_gather
    dist.gather_object(
        my_data, 
        # gathered_objects_on_dst if rank == dst_rank else None, 
        gathered_objects_on_dst,
        dst=dst_rank
    )
    debug_rankwise(gathered_objects_on_dst)

    if rank == dst_rank:
        print("\n--- Verifying custom gather_object (via patched torch.distributed.gather_object) ---")
        correct = True
        for i in range(world_size):
            expected_data = {"rank": i, "data": [i * 100 + j for j in range(2)], "device": f"npu:{i}"}
            if gathered_objects_on_dst[i] != expected_data:
                print(f"Rank {dst_rank}: Error! Custom gather_object for rank {i}: Expected {expected_data}, Got {gathered_objects_on_dst[i]}")
                correct = False
        if correct:
            print(f"Rank {dst_rank}: Custom gather_object verification PASSED.")
        else:
            print(f"Rank {dst_rank}: Custom gather_object verification FAILED.")
        # print(f"Rank {dst_rank}: All gathered objects (custom gather_object):")
        # for i, obj in enumerate(gathered_objects_on_dst):
        #     print(f"  Rank {i}'s data: {obj}")

    # Clean up the patch
    remove_td_gather_object_patch()

def main():
    if not init_distributed_hccl():
        print("Failed to initialize distributed environment. Exiting.")
        # If not distributed, we can't run these tests meaningfully.
        # Optionally, run a single-process version of the logic for basic checks.
        # For now, we just exit if true distributed init fails.
        if "RANK" not in os.environ: # Only if it was a deliberate single process run
            print("Simulating single process behavior for custom_gather_object (no patch).")
            # Test the comm.py functions directly in a simulated way if needed
            # from lm_eval.comm import custom_td_gather_object_with_all_gather
            # my_obj = {"rank": 0, "data": [0,1], "device": "cpu"}
            # gather_list = [None]
            # custom_td_gather_object_with_all_gather(my_obj, gather_list, dst=0)
            # print(f"Single process custom_td_gather_object_with_all_gather result: {gather_list[0]}")
        return

    rank = dist.get_rank()
    print(f"Rank {rank}: Starting tests.")

    # Test 1: torch.distributed.all_gather_object
    test_all_gather_object()
    dist.barrier() # Ensure all processes complete test 1 before starting test 2

    # Test 2: lm_eval.comm.custom_td_gather_object_with_all_gather (via patch)
    test_custom_gather_object()
    dist.barrier()

    if rank == 0:
        print("\nAll tests completed.")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # luoyijie@job-174765237045708644842-luoyijie-master-0:~/src/lm-evaluation-harness$ 
    # torchrun --nproc_per_node=8 --nnodes=1 --master_addr=localhost --master_port=24444 test_gather.py
    main()
