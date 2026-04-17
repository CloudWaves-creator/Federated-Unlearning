# -*- coding: utf-8 -*-
"""
FAIR-VUE 逐轮增量构造 & 缓存工具。
从 main.py 中提取，保持逻辑完全一致。
"""

import os
import re
import gc
import hashlib
import json as _json

import torch


# ====================== 逐轮目录扫描 ======================

def list_iteration_dirs(train_path: str):
    """返回按轮排序的 [(轮次,int_dir_path), ...]"""
    items = []
    if not os.path.isdir(train_path):
        return items
    for name in os.listdir(train_path):
        m = re.match(r'^iteration_(\d+)$', name)
        if m:
            items.append((int(m.group(1)), os.path.join(train_path, name)))
    items.sort(key=lambda x: x[0])
    return items


def load_client_sd(iter_dir: str, cid: int):
    """加载某一轮的 client_{cid}.pth（返回state_dict或None）"""
    p = os.path.join(iter_dir, f"client_{cid}.pth")
    if not os.path.isfile(p):
        return None
    ckpt = torch.load(p, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt  # 默认就是state_dict


# ====================== 全量构造（兼容旧调用） ======================

def build_round_deltas(train_path: str, total_clients: int):
    """
    返回:
      round_client_deltas: dict[round_idx] -> dict[cid] -> Δ_i^t
    其中 Δ_i^t = client_{i,t} - mean_clients_{t-1}  (用上一轮客户端权重的均值近似 global_{t-1})
    round 从 1 开始有意义；0 轮没有上一轮。
    """
    rounds = list_iteration_dirs(train_path)
    round_client_deltas = {}
    for r_idx in range(1, len(rounds)):
        r_prev, dir_prev = rounds[r_idx-1]
        r_cur,  dir_cur  = rounds[r_idx]

        # 先收集上一轮所有客户端权重，求均值 g_prev
        prev_states = []
        for cid in range(total_clients):
            sd_prev = load_client_sd(dir_prev, cid)
            if sd_prev is not None:
                prev_states.append(sd_prev)
        if len(prev_states) == 0:
            continue
        # 逐 key 求均值
        keys = prev_states[0].keys()
        g_prev = {k: sum(sd[k] for sd in prev_states) / float(len(prev_states)) for k in keys}

        # 用 g_prev 做基线，构造本轮每个客户端的 Δ_i^t
        deltas_this_round = {}
        for cid in range(total_clients):
            sd_cur = load_client_sd(dir_cur, cid)
            if sd_cur is None:
                continue
            delta = {k: sd_cur[k] - g_prev[k] for k in g_prev.keys() if k in sd_cur}
            deltas_this_round[cid] = delta

        if len(deltas_this_round) > 0:
            round_client_deltas[r_cur] = deltas_this_round
    return round_client_deltas


# ====================== 流式逐轮构造（省内存） ======================

def iter_round_deltas_stream(
    train_path: str,
    total_clients: int,
    param_keys: list = None,
    max_rounds: int = None,
):
    """
    逐轮产生 {cid -> Δ_i^t}，避免一次性攒全量到内存。
    Δ_i^t = client_{i,t} - mean_clients_{t-1}

    优化点：
      1) 只在给定的 param_keys 上累积/构造 Δ，避免对整个 ResNet 全参数做差。
      2) 复用上一轮的均值 g_prev：每轮只加载"当前轮" client_{cid}.pth，
         不再为每一轮重复读取上一轮权重，减少约一半 ckpt 读盘。
      3) 可选 max_rounds：只解析最近 max_rounds 个"轮"的 Δ，进一步节省时间。
    """
    rounds = list_iteration_dirs(train_path)
    num_rounds = len(rounds)
    if num_rounds <= 1:
        return

    # 需要的"当前轮"个数（Δ_i^t 的轮数），不能超过 num_rounds-1
    max_rounds = int(max_rounds) if (max_rounds is not None and max_rounds > 0) else None
    if max_rounds is not None:
        max_rounds = min(max_rounds, num_rounds - 1)
        start_idx = num_rounds - max_rounds   # 从这里开始作为 "当前轮"
    else:
        start_idx = 1  # 兼容旧行为：从第 1 轮开始（需要第 0 轮作 g_prev）

    # 要用哪些参数 key
    keys_filter = set(param_keys) if param_keys is not None and len(param_keys) > 0 else None

    # --- 先在 start_idx-1 那一轮上构造 g_prev（mean_clients_{start_idx-1}） ---
    _, dir_prev = rounds[start_idx - 1]
    g_prev = None
    cnt_prev = 0
    for cid in range(total_clients):
        sd_prev = load_client_sd(dir_prev, cid)
        if sd_prev is None:
            continue
        if g_prev is None:
            # 初始化：只保留需要的 keys（若未指定 param_keys，则保留全部）
            if keys_filter is None:
                g_prev = {
                    k: v.detach().to('cpu', dtype=torch.float32).clone()
                    for k, v in sd_prev.items()
                }
            else:
                g_prev = {}
                for k in keys_filter:
                    if k in sd_prev:
                        g_prev[k] = sd_prev[k].detach().to('cpu', dtype=torch.float32).clone()
        else:
            if keys_filter is None:
                for k in g_prev.keys():
                    if k in sd_prev:
                        g_prev[k].add_(sd_prev[k].to(dtype=torch.float32))
            else:
                for k in g_prev.keys():
                    if k in sd_prev:
                        g_prev[k].add_(sd_prev[k].to(dtype=torch.float32))
        cnt_prev += 1
        del sd_prev

    if not cnt_prev or g_prev is None:
        return

    for k in g_prev.keys():
        g_prev[k].div_(float(cnt_prev))

    # --- 从 start_idx 开始，边算 Δ 边更新下一轮的 g_prev ---
    for r_idx in range(start_idx, num_rounds):
        r_cur, dir_cur = rounds[r_idx]

        deltas_this_round = {}
        sums_cur = None
        cnt_cur = 0

        for cid in range(total_clients):
            sd_cur = load_client_sd(dir_cur, cid)
            if sd_cur is None:
                continue

            # 1) 只在需要的 keys 上构造 Δ
            if keys_filter is None:
                keys_now = g_prev.keys()
            else:
                keys_now = [k for k in g_prev.keys() if k in keys_filter]

            d = {}
            for k in keys_now:
                if k in sd_cur and k in g_prev:
                    d[k] = sd_cur[k].to('cpu', dtype=torch.float32) - g_prev[k]
            if d:
                deltas_this_round[cid] = d

            # 2) 顺便累积本轮均值，用于下一轮的 g_prev
            if sums_cur is None:
                if keys_filter is None:
                    sums_cur = {
                        k: v.detach().to('cpu', dtype=torch.float32).clone()
                        for k, v in sd_cur.items()
                    }
                else:
                    sums_cur = {}
                    for k in keys_filter:
                        if k in sd_cur:
                            sums_cur[k] = sd_cur[k].detach().to('cpu', dtype=torch.float32).clone()
            else:
                if keys_filter is None:
                    for k in sums_cur.keys():
                        if k in sd_cur:
                            sums_cur[k].add_(sd_cur[k].to(dtype=torch.float32))
                else:
                    for k in sums_cur.keys():
                        if k in sd_cur:
                            sums_cur[k].add_(sd_cur[k].to(dtype=torch.float32))

            cnt_cur += 1
            del sd_cur

        if deltas_this_round:
            yield (r_cur, deltas_this_round)

        # 更新 g_prev → 当前轮的均值
        if sums_cur is not None and cnt_cur > 0:
            for k in sums_cur.keys():
                sums_cur[k].div_(float(cnt_cur))
            g_prev = sums_cur

        # 释放
        del deltas_this_round
        if sums_cur is not None:
            del sums_cur
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ====================== 增量缓存 ======================

def delta_cache_key(train_path: str,
                    total_clients: int,
                    target_id: int,
                    param_keys: list,
                    fair_target_rounds: int,
                    layer_mode: str = 'classifier'):
    meta = {
        "train_path": os.path.abspath(train_path),
        "total_clients": int(total_clients),
        "target_id": int(target_id),
        "param_keys": list(param_keys or []),
        "layer_mode": str(layer_mode),
        "fair_target_rounds": int(fair_target_rounds),
        "version": 1,  # 以后改格式可以+1，强制失效旧 cache
    }
    key_str = _json.dumps(meta, sort_keys=True)
    h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:16]
    return h, meta


def delta_cache_load(cache_dir: str, cache_key: str, meta_expected: dict):
    path = os.path.join(cache_dir, f"fairvue_delta_{cache_key}.pt")
    if not os.path.isfile(path):
        return None
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict) or "meta" not in obj:
        return None
    meta = obj.get("meta", {})
    # 简单比对：关键字段一致才算命中
    for k in ("train_path", "total_clients", "target_id",
              "fair_target_rounds", "param_keys", "version"):
        if meta.get(k) != meta_expected.get(k):
            return None
    return obj


def delta_cache_save(cache_dir: str, cache_key: str, payload: dict, verbose: bool = False):
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"fairvue_delta_{cache_key}.pt")
        torch.save(payload, path)
        if verbose:
            print(f"[FV-CACHE] delta cache SAVED to {path}")
    except Exception as e:
        if verbose:
            print(f"[FV-CACHE][WARN] save failed: {repr(e)}")
