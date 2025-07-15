import logging  
import os  
  
# 创建文件日志处理器  
log_file = f"/tmp/vllm_debug_logger.log"  
file_handler = logging.FileHandler(log_file)  
file_handler.setLevel(logging.DEBUG)  
  
# 获取 vLLM 日志器并添加文件处理器  
from vllm.logger import init_logger  
logger = init_logger(f"vllm:{__name__}")  
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)  

logger.warning("这条WARNING消息应该出现在文件中")
logger.info("这条INFO消息应该出现在文件中")
logger.debug("这条DEBUG消息应该出现在文件中")