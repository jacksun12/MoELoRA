from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import numpy as np

class ClusterEvaluator:
    def __init__(self):
        self.history_features = []

    def add_feature(self, feat):
        self.history_features.append(feat.detach().cpu().numpy())

    def should_split(self, data, labels):
        """利用 DB/CH 指标评估当前聚类质量"""
        if data is None or len(data) == 0:
            return False
        if len(set(labels)) < 2:
            return False

        db_idx = davies_bouldin_score(data, labels)
        ch_idx = calinski_harabasz_score(data, labels)

        # DB 越小、CH 越大，聚类越明显，越应该分裂
        print(f"Metrics - DB Index: {db_idx:.4f}, CH Index: {ch_idx:.4f}")
        return db_idx < 1.2