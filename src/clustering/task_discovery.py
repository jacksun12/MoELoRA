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
        向缓冲区添加一批前向过程中收集到的隐藏状态特征。
        Add one batch of hidden-state features collected during forward passes.

        hidden_state_features 的形状为 [batch_size, hidden_dim]。
        hidden_state_features has shape [batch_size, hidden_dim].
        """
        # NumPy 不支持 bfloat16，因此先转成 float32。
        # NumPy does not support bfloat16, so cast to float32 first.
        features_np = hidden_state_features.detach().cpu().float().numpy()

        for feat in features_np:
            self.feature_buffer.append(feat)

            # 缓冲区满时触发一次轻量级聚类扫描。
            # Trigger one lightweight clustering scan when the buffer is full.
            if len(self.feature_buffer) >= self.buffer_size:
                return self._analyze_and_flush()
        return False, None

    def _analyze_and_flush(self):
        """在缓冲特征上运行一次无监督聚类分析。 / Run an unsupervised clustering pass over the buffered features."""
        print(f"[Task Discovery] Buffer full ({self.buffer_size} samples). Running clustering analysis...")
        data = np.array(self.feature_buffer)

        # 探测最近缓冲区中是否出现新的兴趣区域。
        # Probe whether a new interest region has emerged in the recent buffer.
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters_guess, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(data)

        # 让 DB/CH 指标决定这次分裂是否足够显著。
        # Let DB/CH metrics decide whether this split is sufficiently distinct.
        is_distinct = self.evaluator.should_split(data, labels)

        # 重置缓冲区，为下一轮监控窗口做准备。
        # Reset the buffer for the next monitoring window.
        self.feature_buffer = []

        if is_distinct:
            print("[Task Discovery] Distinct new task cluster found.")
            return True, kmeans.cluster_centers_
        else:
            print("[Task Discovery] Data distribution stable. No split needed.")
            return False, None
