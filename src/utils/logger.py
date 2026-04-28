import logging
import json
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="eval/logs", exp_name="moe_lora_exp"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{exp_name}_{timestamp}.log")
        self.metric_file = os.path.join(self.log_dir, f"{exp_name}_{timestamp}.jsonl")
        
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("MoE-LoRA")
        logger.setLevel(logging.INFO)
        
        # 防止重复添加 handler
        if not logger.handlers:
            # 终端输出
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s', '%H:%M:%S')
            console_handler.setFormatter(console_formatter)
            
            # 文件输出
            file_handler = logging.FileHandler(self.log_file)
            file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s')
            file_handler.setFormatter(file_formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            
        return logger

    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)

    def log_metrics(self, step, metrics_dict):
        """
        将指标结构化保存为 JSONL 格式。
        方便后续跑完实验，直接写个小脚本读取它画折线图。
        """
        data = {"step": step}
        data.update(metrics_dict)
        
        with open(self.metric_file, "a") as f:
            f.write(json.dumps(data) + "\n")
            
        # 终端打印核心指标预览
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()])
        self.logger.info(f"[Step {step}] {metrics_str}")