from typing import Any, Optional, List
import torch
import torch_npu
import torch.distributed as dist

_original_td_gather_object = None

def custom_td_gather_object_with_all_gather(
    obj: Any,
    object_gather_list: Optional[List[Any]] = None,
    dst: int = 0,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Custom implementation of gather_object using torch.distributed.all_gather_object.
    Gathers objects from all ranks to all ranks, then populates
    object_gather_list on the destination (dst) rank.
    """
    if not dist.is_available() or not dist.is_initialized():
        # Fallback for non-distributed or uninitialized environment
        # This behavior might need adjustment based on how single-process runs are handled.
        if object_gather_list is not None and dst == 0: # Assuming rank 0 in single process
             if len(object_gather_list) > 0:
                object_gather_list[0] = obj
             else: # Or raise error if list is empty
                # Create list if it's meant to be an out param for single item
                object_gather_list.append(obj)

        return

    if group is None:
        # Ensure group is correctly resolved if not explicitly passed.
        # dist.all_gather_object handles default group if group is None.
        pass

    world_size = dist.get_world_size(group=group)
    current_rank = dist.get_rank(group=group)

    # all_gather_object requires a list on all ranks to store the gathered objects.
    # This list will be populated with objects from all ranks on every rank.
    all_gathered_objects_on_all_ranks = [None] * world_size
    dist.all_gather_object(all_gathered_objects_on_all_ranks, obj, group=group)

    if current_rank == dst:
        if object_gather_list is None:
            # The original torch.distributed.gather_object expects object_gather_list
            # to be a pre-sized list on the dst rank, and None otherwise.
            # We will adhere to this by requiring it.
            raise ValueError(
                "object_gather_list must be provided and correctly sized on the destination (dst) rank."
            )
        
        if not isinstance(object_gather_list, list):
            raise TypeError(
                f"object_gather_list must be a list on the destination (dst) rank, got {type(object_gather_list)}."
            )

        # Ensure the list is clear or correctly sized.
        # The original gather_object expects it to be pre-sized.
        # If it's not, we can't safely populate it without knowing the user's intent.
        # For safety, we'll assume it's pre-sized as per gather_object's contract.
        if len(object_gather_list) != world_size:
            # Resize or error? Original gather_object expects it to be correctly sized.
            # Let's error to match more closely.
            # Alternatively, we could do: object_gather_list.clear(); object_gather_list.extend([None] * world_size)
            # but that modifies the list structure more than just filling.
             raise ValueError(
                f"object_gather_list on dst rank must be of size {world_size}, but got {len(object_gather_list)}. "
                "Pre-allocate it as [None] * world_size on the dst rank."
            )

        for i in range(world_size):
            object_gather_list[i] = all_gathered_objects_on_all_ranks[i]
    # Non-dst ranks do nothing with object_gather_list, as per original gather_object behavior.

def apply_td_gather_object_patch():
    """
    Applies a monkey patch to torch.distributed.gather_object,
    replacing it with a custom implementation that uses all_gather_object.
    """
    global _original_td_gather_object
    if _original_td_gather_object is None:
        if dist.is_available(): # Only patch if distributed is available
            _original_td_gather_object = dist.gather_object
            dist.gather_object = custom_td_gather_object_with_all_gather
            print("Applied monkey patch to torch.distributed.gather_object to use all_gather_object.")
        else:
            print("torch.distributed is not available. Cannot apply gather_object patch.")


# Optional: A function to remove the patch if needed
def remove_td_gather_object_patch():
    """
    Removes the monkey patch from torch.distributed.gather_object if it was applied.
    """
    global _original_td_gather_object
    if _original_td_gather_object is not None and dist.is_available():
        dist.gather_object = _original_td_gather_object
        _original_td_gather_object = None
        print("Removed monkey patch from torch.distributed.gather_object.")

