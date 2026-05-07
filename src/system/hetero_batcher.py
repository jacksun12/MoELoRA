from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class InferenceRequest:
    request_id: str
    activated_loras: Tuple[int, ...]
    token_length: int


class HeteroBatchScheduler:
    """
    按激活的 LoRA 签名对请求分组。
    Group requests by activated LoRA signatures.

    并行度高的请求组送到 GPU，稀疏组送到 CPU。
    High-parallel groups go to GPU, while sparse groups go to CPU.
    """

    def __init__(self, gpu_min_batch: int = 4, gpu_min_tokens: int = 256):
        self.gpu_min_batch = gpu_min_batch
        self.gpu_min_tokens = gpu_min_tokens

    def group_by_signature(self, requests: List[InferenceRequest]) -> Dict[Tuple[int, ...], List[InferenceRequest]]:
        groups: Dict[Tuple[int, ...], List[InferenceRequest]] = {}
        for req in requests:
            sig = tuple(sorted(req.activated_loras))
            groups.setdefault(sig, []).append(req)
        return groups

    def schedule(self, requests: List[InferenceRequest]):
        groups = self.group_by_signature(requests)
        plan = {"gpu": [], "cpu": []}

        for signature, group in groups.items():
            total_tokens = sum(r.token_length for r in group)
            if len(group) >= self.gpu_min_batch and total_tokens >= self.gpu_min_tokens:
                plan["gpu"].append({"signature": signature, "requests": group, "total_tokens": total_tokens})
            else:
                plan["cpu"].append({"signature": signature, "requests": group, "total_tokens": total_tokens})

        return plan
