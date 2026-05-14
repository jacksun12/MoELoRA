from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class ServingRequest:
    """Synthetic serving request. / 合成推理请求。"""

    request_id: str
    arrival_time_ms: float
    activated_loras: Tuple[int, ...]
    prompt_tokens: int
    decode_tokens: int

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens + self.decode_tokens)

    @property
    def signature(self) -> Tuple[int, ...]:
        return tuple(sorted(self.activated_loras))


@dataclass(frozen=True)
class DeviceProfile:
    """Simple latency model for a device. / 设备延迟模型。"""

    name: str
    setup_ms: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    lora_ms_per_token: float
    batch_alpha: float
    max_batch_size: int
    decode_padding_penalty: float
    signature_mismatch_penalty: float


@dataclass
class SimulationMetrics:
    strategy: str
    workload: str
    num_requests: int
    throughput_rps: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    makespan_ms: float
    gpu_utilization: float
    cpu_utilization: float
    npu_utilization: float
    gpu_share: float
    cpu_share: float
    npu_share: float
    avg_batch_size_gpu: float
    avg_batch_size_cpu: float
    avg_batch_size_npu: float


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, max(0, int(math.ceil(q * len(arr))) - 1))
    return float(arr[idx])


def _length_bucket(request: ServingRequest) -> str:
    total = request.total_tokens
    if total <= 160:
        return "short"
    if total <= 320:
        return "medium"
    return "long"


def _service_time_ms(device: DeviceProfile, batch: Sequence[ServingRequest]) -> float:
    if not batch:
        return 0.0

    batch_size = len(batch)
    batch_scale = min(device.max_batch_size, batch_size) ** max(device.batch_alpha, 0.0)
    batch_scale = max(batch_scale, 1.0)

    prompt_sum = sum(req.prompt_tokens for req in batch)
    decode_sum = sum(req.decode_tokens for req in batch)
    total_sum = sum(req.total_tokens * max(1, len(req.signature)) for req in batch)
    unique_signatures = len({req.signature for req in batch})

    mean_prompt = prompt_sum / batch_size
    max_prompt = max(req.prompt_tokens for req in batch)
    prompt_padding_ratio = max_prompt / max(mean_prompt, 1e-6)
    mean_decode = decode_sum / batch_size
    max_decode = max(req.decode_tokens for req in batch)
    decode_padding_ratio = max_decode / max(mean_decode, 1e-6)

    prefill_ms = prompt_sum * device.prefill_ms_per_token / batch_scale
    prefill_ms *= 1.0 + 0.15 * max(0.0, prompt_padding_ratio - 1.0)
    decode_ms = decode_sum * device.decode_ms_per_token / batch_scale
    decode_ms *= 1.0 + device.decode_padding_penalty * max(0.0, decode_padding_ratio - 1.0)

    lora_ms = total_sum * device.lora_ms_per_token / batch_scale
    lora_ms *= 1.0 + device.signature_mismatch_penalty * max(0, unique_signatures - 1) * (1.0 + 0.12 * max(0, batch_size - 1))

    setup_ms = device.setup_ms * (1.0 + 0.45 * max(0, unique_signatures - 1))
    return float(setup_ms + prefill_ms + decode_ms + lora_ms)


def _default_profiles() -> Dict[str, DeviceProfile]:
    return {
        "cpu": DeviceProfile(
            name="cpu",
            setup_ms=0.20,
            prefill_ms_per_token=0.22,
            decode_ms_per_token=0.34,
            lora_ms_per_token=0.050,
            batch_alpha=0.08,
            max_batch_size=2,
            decode_padding_penalty=0.05,
            signature_mismatch_penalty=0.0,
        ),
        "gpu": DeviceProfile(
            name="gpu",
            setup_ms=0.65,
            prefill_ms_per_token=0.07,
            decode_ms_per_token=0.12,
            lora_ms_per_token=0.020,
            batch_alpha=0.72,
            max_batch_size=16,
            decode_padding_penalty=0.65,
            signature_mismatch_penalty=0.55,
        ),
        "npu_base": DeviceProfile(
            name="npu_base",
            setup_ms=0.45,
            prefill_ms_per_token=0.05,
            decode_ms_per_token=0.08,
            lora_ms_per_token=0.0,
            batch_alpha=0.70,
            max_batch_size=16,
            decode_padding_penalty=0.25,
            signature_mismatch_penalty=0.0,
        ),
        "gpu_lora": DeviceProfile(
            name="gpu_lora",
            setup_ms=0.35,
            prefill_ms_per_token=0.0,
            decode_ms_per_token=0.0,
            lora_ms_per_token=0.017,
            batch_alpha=0.72,
            max_batch_size=16,
            decode_padding_penalty=0.45,
            signature_mismatch_penalty=0.35,
        ),
        "cpu_lora": DeviceProfile(
            name="cpu_lora",
            setup_ms=0.10,
            prefill_ms_per_token=0.0,
            decode_ms_per_token=0.0,
            lora_ms_per_token=0.045,
            batch_alpha=0.02,
            max_batch_size=1,
            decode_padding_penalty=0.0,
            signature_mismatch_penalty=0.0,
        ),
    }


def generate_synthetic_requests(
    num_requests: int = 512,
    num_signatures: int = 8,
    hot_signature_ratio: float = 0.55,
    long_request_ratio: float = 0.30,
    multi_lora_ratio: float = 0.18,
    arrival_rate_rps: float = 18.0,
    seed: int = 42,
) -> List[ServingRequest]:
    """
    Build a synthetic request stream with hot/cold LoRA signatures.
    构造带有冷热 LoRA 签名的合成请求流。
    """

    rng = random.Random(seed)
    if num_signatures <= 1:
        signature_pool = [(0,)]
    else:
        signature_pool = [(idx,) for idx in range(num_signatures)]

    weights = []
    hot_cutoff = max(1, int(round(num_signatures * hot_signature_ratio)))
    for idx in range(num_signatures):
        if idx < hot_cutoff:
            weights.append(4.0)
        else:
            weights.append(1.0)

    requests: List[ServingRequest] = []
    current_time_ms = 0.0
    for idx in range(num_requests):
        inter_arrival_sec = rng.expovariate(max(arrival_rate_rps, 1e-6))
        current_time_ms += inter_arrival_sec * 1000.0

        base_sig = rng.choices(signature_pool, weights=weights, k=1)[0]
        if rng.random() < multi_lora_ratio and num_signatures >= 3:
            extra = rng.randrange(num_signatures)
            merged = tuple(sorted(set(base_sig + (extra,))))
            signature = merged
        else:
            signature = base_sig

        if rng.random() < long_request_ratio:
            prompt_tokens = rng.randint(180, 360)
            decode_tokens = rng.randint(90, 180)
        else:
            prompt_tokens = rng.randint(48, 160)
            decode_tokens = rng.randint(24, 96)

        requests.append(
            ServingRequest(
                request_id=f"req_{idx:05d}",
                arrival_time_ms=current_time_ms,
                activated_loras=signature,
                prompt_tokens=prompt_tokens,
                decode_tokens=decode_tokens,
            )
        )

    return requests


def build_workload_profile(name: str, seed: int = 42) -> List[ServingRequest]:
    """
    Predefined workload presets. / 预定义工作负载配置。
    """

    if name == "simple":
        return generate_synthetic_requests(
            num_requests=480,
            num_signatures=2,
            hot_signature_ratio=1.0,
            long_request_ratio=0.12,
            multi_lora_ratio=0.03,
            arrival_rate_rps=16.0,
            seed=seed,
        )
    if name == "complex":
        return generate_synthetic_requests(
            num_requests=640,
            num_signatures=8,
            hot_signature_ratio=0.50,
            long_request_ratio=0.34,
            multi_lora_ratio=0.20,
            arrival_rate_rps=18.0,
            seed=seed,
        )
    if name == "fragmented":
        return generate_synthetic_requests(
            num_requests=640,
            num_signatures=12,
            hot_signature_ratio=0.25,
            long_request_ratio=0.38,
            multi_lora_ratio=0.28,
            arrival_rate_rps=20.0,
            seed=seed,
        )
    if name == "bursty_hotspot":
        return generate_synthetic_requests(
            num_requests=1024,
            num_signatures=4,
            hot_signature_ratio=0.50,
            long_request_ratio=0.30,
            multi_lora_ratio=0.12,
            arrival_rate_rps=80.0,
            seed=seed,
        )
    raise ValueError(f"Unknown workload profile: {name}")


def _init_state():
    return {
        "gpu_available_ms": 0.0,
        "cpu_available_ms": 0.0,
        "gpu_busy_ms": 0.0,
        "cpu_busy_ms": 0.0,
        "gpu_batches": [],
        "cpu_batches": [],
        "gpu_request_count": 0,
        "cpu_request_count": 0,
        "gpu_signature_cache": [],
        "completion_times": {},
    }


def _record_batch(state, device_name: str, batch: Sequence[ServingRequest], start_ms: float, end_ms: float):
    state[f"{device_name}_busy_ms"] += max(0.0, end_ms - start_ms)
    state[f"{device_name}_available_ms"] = end_ms
    state[f"{device_name}_batches"].append(len(batch))
    state[f"{device_name}_request_count"] += len(batch)
    for req in batch:
        state["completion_times"][req.request_id] = end_ms


def _dispatch_batch(state, device_name: str, batch: Sequence[ServingRequest], start_ms: float, profile: DeviceProfile):
    service_ms = _service_time_ms(profile, batch)
    if device_name == "gpu":
        service_ms += _gpu_cache_penalty_ms(state, batch)
    end_ms = start_ms + service_ms
    _record_batch(state, device_name, batch, start_ms, end_ms)


def _gpu_cache_penalty_ms(state, batch: Sequence[ServingRequest], cache_size: int = 2, load_ms: float = 4.5) -> float:
    signatures = list({req.signature for req in batch})
    cache: List[Tuple[int, ...]] = list(state.get("gpu_signature_cache", []))
    penalty = 0.0
    for signature in signatures:
        if signature not in cache:
            penalty += load_ms
        if signature in cache:
            cache.remove(signature)
        cache.insert(0, signature)
        cache = cache[:cache_size]
    state["gpu_signature_cache"] = cache
    return penalty


def _pop_oldest(pending: List[ServingRequest]) -> ServingRequest:
    pending.sort(key=lambda req: req.arrival_time_ms)
    return pending.pop(0)


def _pop_gpu_fifo_batch(pending: List[ServingRequest], max_batch_size: int) -> List[ServingRequest]:
    pending.sort(key=lambda req: req.arrival_time_ms)
    batch = pending[:max_batch_size]
    del pending[: len(batch)]
    return batch


def _pop_best_signature_batch(
    pending: List[ServingRequest],
    max_batch_size: int,
    require_length_bucket: bool,
    gpu_min_batch: int,
    gpu_min_tokens: int,
) -> List[ServingRequest]:
    groups: Dict[Tuple[Tuple[int, ...], str], List[ServingRequest]] = defaultdict(list)
    for req in pending:
        bucket = _length_bucket(req) if require_length_bucket else "any"
        groups[(req.signature, bucket)].append(req)

    scored = []
    for (signature, bucket), group in groups.items():
        group.sort(key=lambda req: req.arrival_time_ms)
        batch = group[:max_batch_size]
        total_tokens = sum(req.total_tokens for req in batch)
        if len(batch) < gpu_min_batch and total_tokens < gpu_min_tokens:
            continue
        scored.append((len(batch), total_tokens, -batch[0].arrival_time_ms, signature, bucket, batch))

    if not scored:
        return []

    _, _, _, signature, bucket, batch = max(scored)
    chosen: List[ServingRequest] = []
    remaining: List[ServingRequest] = []
    need_ids = {req.request_id for req in batch}
    for req in pending:
        req_bucket = _length_bucket(req) if require_length_bucket else "any"
        if req.request_id in need_ids and req.signature == signature and req_bucket == bucket:
            chosen.append(req)
        else:
            remaining.append(req)
    pending[:] = remaining
    chosen.sort(key=lambda req: req.arrival_time_ms)
    return chosen


def simulate_strategy(
    requests: Sequence[ServingRequest],
    strategy: str,
    gpu_min_batch: int = 4,
    gpu_min_tokens: int = 256,
    gpu_batch_size: int = 8,
) -> SimulationMetrics:
    profiles = _default_profiles()
    pending = sorted(list(requests), key=lambda req: req.arrival_time_ms)
    waiting: List[ServingRequest] = []
    state = _init_state()
    clock_ms = 0.0

    while pending or waiting:
        next_arrival_ms = pending[0].arrival_time_ms if pending else float("inf")
        if not waiting:
            future = [t for t in [next_arrival_ms, state["gpu_available_ms"], state["cpu_available_ms"]] if t > clock_ms]
            if future:
                clock_ms = min(future)
        while pending and pending[0].arrival_time_ms <= clock_ms:
            waiting.append(pending.pop(0))

        dispatched_any = False

        while waiting:
            dispatched = False
            if strategy == "all_cpu":
                if state["cpu_available_ms"] <= clock_ms:
                    batch = [_pop_oldest(waiting)]
                    _dispatch_batch(state, "cpu", batch, clock_ms, profiles["cpu"])
                    dispatched = True

            elif strategy == "fifo_gpu":
                if state["gpu_available_ms"] <= clock_ms:
                    batch = _pop_gpu_fifo_batch(waiting, gpu_batch_size)
                    _dispatch_batch(state, "gpu", batch, clock_ms, profiles["gpu"])
                    dispatched = True

            elif strategy in {"signature_gpu_cpu", "signature_length_gpu_cpu"}:
                if state["gpu_available_ms"] <= clock_ms:
                    batch = _pop_best_signature_batch(
                        waiting,
                        max_batch_size=gpu_batch_size,
                        require_length_bucket=(strategy == "signature_length_gpu_cpu"),
                        gpu_min_batch=gpu_min_batch,
                        gpu_min_tokens=gpu_min_tokens,
                    )
                    if batch:
                        _dispatch_batch(state, "gpu", batch, clock_ms, profiles["gpu"])
                        dispatched = True

                if waiting and state["cpu_available_ms"] <= clock_ms:
                    batch = [_pop_oldest(waiting)]
                    _dispatch_batch(state, "cpu", batch, clock_ms, profiles["cpu"])
                    dispatched = True or dispatched
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if not dispatched:
                break
            dispatched_any = True

        if not dispatched_any:
            future = [t for t in [next_arrival_ms, state["gpu_available_ms"], state["cpu_available_ms"]] if t > clock_ms]
            clock_ms = min(future) if future else (clock_ms + 0.1)

    return _finalize_metrics(state, requests=requests, strategy=strategy, workload="custom")


def simulate_npu_hybrid_strategy(
    requests: Sequence[ServingRequest],
    gpu_min_batch: int = 4,
    gpu_min_tokens: int = 256,
    npu_batch_size: int = 8,
    gpu_batch_size: int = 8,
) -> SimulationMetrics:
    profiles = _default_profiles()
    arrivals = sorted(list(requests), key=lambda req: req.arrival_time_ms)
    pre_npu: List[ServingRequest] = []
    post_npu: List[ServingRequest] = []
    gpu_available_ms = 0.0
    cpu_available_ms = 0.0
    npu_available_ms = 0.0
    gpu_busy_ms = 0.0
    cpu_busy_ms = 0.0
    npu_busy_ms = 0.0
    gpu_batches: List[int] = []
    cpu_batches: List[int] = []
    npu_batches: List[int] = []
    gpu_request_count = 0
    cpu_request_count = 0
    gpu_signature_cache: List[Tuple[int, ...]] = []
    completion_times: Dict[str, float] = {}
    npu_release_times: Dict[str, float] = {}
    clock_ms = 0.0

    while arrivals or pre_npu or post_npu:
        while arrivals and arrivals[0].arrival_time_ms <= clock_ms:
            pre_npu.append(arrivals.pop(0))

        for req_id, release_time in list(npu_release_times.items()):
            if release_time <= clock_ms:
                match = next((req for req in pre_npu if req.request_id == req_id), None)
                if match is not None:
                    pre_npu.remove(match)
                    post_npu.append(match)
                del npu_release_times[req_id]

        dispatched = False

        if pre_npu and npu_available_ms <= clock_ms:
            pre_npu.sort(key=lambda req: req.arrival_time_ms)
            batch = pre_npu[:npu_batch_size]
            service_ms = _service_time_ms(profiles["npu_base"], batch)
            start_ms = clock_ms
            end_ms = start_ms + service_ms
            npu_busy_ms += service_ms
            npu_available_ms = end_ms
            npu_batches.append(len(batch))
            for req in batch:
                npu_release_times[req.request_id] = end_ms
            dispatched = True

        if post_npu and gpu_available_ms <= clock_ms:
            batch = _pop_best_signature_batch(
                post_npu,
                max_batch_size=gpu_batch_size,
                require_length_bucket=True,
                gpu_min_batch=gpu_min_batch,
                gpu_min_tokens=gpu_min_tokens,
            )
            if batch:
                service_ms = _service_time_ms(profiles["gpu_lora"], batch)
                temp_state = {"gpu_signature_cache": gpu_signature_cache}
                service_ms += _gpu_cache_penalty_ms(temp_state, batch)
                gpu_signature_cache = temp_state["gpu_signature_cache"]
                start_ms = clock_ms
                end_ms = start_ms + service_ms
                gpu_busy_ms += service_ms
                gpu_available_ms = end_ms
                gpu_batches.append(len(batch))
                gpu_request_count += len(batch)
                for req in batch:
                    completion_times[req.request_id] = end_ms
                dispatched = True

        if post_npu and cpu_available_ms <= clock_ms:
            post_npu.sort(key=lambda req: req.arrival_time_ms)
            batch = [post_npu.pop(0)]
            service_ms = _service_time_ms(profiles["cpu_lora"], batch)
            start_ms = clock_ms
            end_ms = start_ms + service_ms
            cpu_busy_ms += service_ms
            cpu_available_ms = end_ms
            cpu_batches.append(1)
            cpu_request_count += 1
            completion_times[batch[0].request_id] = end_ms
            dispatched = True or dispatched

        if dispatched:
            next_candidates = [
                arrivals[0].arrival_time_ms if arrivals else float("inf"),
                npu_available_ms,
                gpu_available_ms,
                cpu_available_ms,
                min(npu_release_times.values()) if npu_release_times else float("inf"),
            ]
            next_future = min(t for t in next_candidates if t > clock_ms) if any(t > clock_ms for t in next_candidates) else clock_ms + 0.1
            clock_ms = min(clock_ms + 0.1, next_future)
            continue

        candidates = [
            arrivals[0].arrival_time_ms if arrivals else float("inf"),
            npu_available_ms,
            gpu_available_ms,
            cpu_available_ms,
            min(npu_release_times.values()) if npu_release_times else float("inf"),
        ]
        clock_ms = min(t for t in candidates if t > clock_ms) if any(t > clock_ms for t in candidates) else clock_ms + 0.1

    makespan_ms = max(completion_times.values()) if completion_times else 0.0
    latencies = [completion_times[req.request_id] - req.arrival_time_ms for req in requests]
    throughput_rps = (len(requests) / makespan_ms) * 1000.0 if makespan_ms > 0 else 0.0
    return SimulationMetrics(
        strategy="npu_hybrid",
        workload="custom",
        num_requests=len(requests),
        throughput_rps=float(throughput_rps),
        mean_latency_ms=float(sum(latencies) / len(latencies)) if latencies else 0.0,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        makespan_ms=float(makespan_ms),
        gpu_utilization=float(gpu_busy_ms / makespan_ms) if makespan_ms > 0 else 0.0,
        cpu_utilization=float(cpu_busy_ms / makespan_ms) if makespan_ms > 0 else 0.0,
        npu_utilization=float(npu_busy_ms / makespan_ms) if makespan_ms > 0 else 0.0,
        gpu_share=float(gpu_request_count / len(requests)) if requests else 0.0,
        cpu_share=float(cpu_request_count / len(requests)) if requests else 0.0,
        npu_share=1.0 if requests else 0.0,
        avg_batch_size_gpu=float(sum(gpu_batches) / len(gpu_batches)) if gpu_batches else 0.0,
        avg_batch_size_cpu=float(sum(cpu_batches) / len(cpu_batches)) if cpu_batches else 0.0,
        avg_batch_size_npu=float(sum(npu_batches) / len(npu_batches)) if npu_batches else 0.0,
    )


def _finalize_metrics(state, requests: Sequence[ServingRequest], strategy: str, workload: str) -> SimulationMetrics:
    completion_times = state["completion_times"]
    makespan_ms = max(completion_times.values()) if completion_times else 0.0
    latencies = [completion_times[req.request_id] - req.arrival_time_ms for req in requests]
    throughput_rps = (len(requests) / makespan_ms) * 1000.0 if makespan_ms > 0 else 0.0

    gpu_count = state["gpu_request_count"]
    cpu_count = state["cpu_request_count"]
    total_count = max(1, gpu_count + cpu_count)

    return SimulationMetrics(
        strategy=strategy,
        workload=workload,
        num_requests=len(requests),
        throughput_rps=float(throughput_rps),
        mean_latency_ms=float(sum(latencies) / len(latencies)) if latencies else 0.0,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        makespan_ms=float(makespan_ms),
        gpu_utilization=float(state["gpu_busy_ms"] / makespan_ms) if makespan_ms > 0 else 0.0,
        cpu_utilization=float(state["cpu_busy_ms"] / makespan_ms) if makespan_ms > 0 else 0.0,
        npu_utilization=0.0,
        gpu_share=float(gpu_count / total_count),
        cpu_share=float(cpu_count / total_count),
        npu_share=0.0,
        avg_batch_size_gpu=float(sum(state["gpu_batches"]) / len(state["gpu_batches"])) if state["gpu_batches"] else 0.0,
        avg_batch_size_cpu=float(sum(state["cpu_batches"]) / len(state["cpu_batches"])) if state["cpu_batches"] else 0.0,
        avg_batch_size_npu=0.0,
    )
