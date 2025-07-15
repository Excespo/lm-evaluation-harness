import os      
import ray      
      
def apply(seed):  
    # 移除这行：seed = 42  
    print("applying ... ")      
    print(f"Process ID: {os.getpid()}")  
    print(f"Received seed: {seed}")  
      
    # 如果你想在每个进程中设置不同的seed，可以这样：  
    import random  
    random.seed(seed)  
    random_value = random.random()  
    print(f"Random value with seed {seed}: {random_value}")  

    return os.getpid()
      
@ray.remote(resources={"NPU": 1})      
def apply_across_all_resources(seed):  
    return apply(seed)  
      
if __name__ == "__main__":      
    ray.init()  
      
    cluster_resources = ray.cluster_resources()  
    num_npus = int(cluster_resources.get("NPU", 0))  
    print(f"\n可用NPU数量: {num_npus}")  
      
    if num_npus > 0:  
        # 为每个任务传递不同的seed值  
        tasks = [apply_across_all_resources.remote(i) for i in range(num_npus)]  
        pids = ray.get(tasks)  
        print(f"所有进程PID: {pids}")  
        print(f"唯一进程数量: {len(set(pids))}")  
    else:  
        print("没有可用的NPU资源")  
      
    ray.shutdown()