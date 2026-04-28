import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .db_ch_evaluator import ClusterEvaluator

class TaskDiscoveryMonitor:
    def __init__(self, buffer_size=500, n_clusters_guess=2):
        self.buffer_size = buffer_size
        self.n_clusters_guess = n_clusters_guess
        self.feature_buffer = []
        self.evaluator = ClusterEvaluator()

    def update_buffer(self, hidden_state_features):
            """
            每次模型 forward 时，将提取的 hidden states 塞进来
            hidden_state_features: shape [batch_size, hidden_dim]
            """
            # 👇 --- 核心修复：添加 .float() 转换 --- 👇
            # NumPy 不认识 bfloat16，必须先强转为普通的 float32
            features_np = hidden_state_features.detach().cpu().float().numpy()
            # 👆 ----------------------------------- 👆
            
            for feat in features_np:
                self.feature_buffer.append(feat)
                
                # 如果缓冲区满了，触发一次“雷达扫描”
                if len(self.feature_buffer) >= self.buffer_size:
                    return self._analyze_and_flush()
            return False, None

    def _analyze_and_flush(self):
        """对缓冲区内的数据进行无监督聚类分析"""
        print(f"🔍 [Task Discovery] Buffer full ({self.buffer_size} samples). Running clustering analysis...")
        data = np.array(self.feature_buffer)
        
        # 1. 尝试使用轻量级 KMeans 将当前数据强行分为 n 簇
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters_guess, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(data)
        
        # 2. 调用我们之前写的 Evaluator 来评判这个聚类靠不靠谱
        # 如果 DB/CH 分数显示聚类非常清晰，说明确实出现了截然不同的新任务
        is_distinct = self.evaluator.should_split(data, labels)
        
        # 清空缓冲区，为下一轮监控做准备
        self.feature_buffer = []
        
        if is_distinct:
            print("🚨 [Task Discovery] Distinct new task cluster found!")
            # 返回 True，并告诉系统层：这批数据中哪一部分是“复杂的/新的”
            # （这里简单起见返回 True，实际系统可以返回具体的 cluster centroids 供 Router 微调）
            return True, kmeans.cluster_centers_
        else:
            print("⏳ [Task Discovery] Data distribution stable. No split needed.")
            return False, None