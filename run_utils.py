# -*- coding: utf-8 -*-
"""
运行时通用工具：日志代理、种子锁定、内存跟踪、MIA 瘦身等。
从 main.py 中提取，供 main.py 及其它模块共用。
"""

import os
import re
import sys
import random
import argparse

import numpy as np
import torch

# ====================== str2bool ======================

def str2bool(v):
    """命令行显式布尔解析函数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ====================== 种子锁定 ======================

def seed_everything(seed: int):
    """一站式锁定所有随机源，确保完全可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """DataLoader worker 进程种子初始化：确保多 worker 下数据增强可复现。
    PyTorch 会自动为每个 worker 设置 torch 种子（base_seed + worker_id），
    但 numpy/random 不受控，需要手动派生。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ====================== 精度快捷工具 ======================

def get_accuracy_only(model, dataloader, device):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ====================== ProxyLog ======================

class ProxyLog:
    """
    代理 stdout/stderr：
    1) 可写入多个日志文件（同一份日志复制到多处）
    2) 控制台照常显示（保留 tqdm 进度条等）
    3) 暴露 isatty/fileno 等，让 tqdm 识别为 TTY
    """
    def __init__(self, stream, log_paths):
        self.stream = stream
        # 允许传入字符串或列表
        if isinstance(log_paths, (str, os.PathLike)):
            self.log_paths = [str(log_paths)]
        else:
            self.log_paths = [str(p) for p in log_paths]
        # 进度条/吞吐的过滤（保持与原逻辑一致）
        self.re_bar = re.compile(
            r"(?:\r)?(?:(?=.*\d{1,3}%\|.+\|).*|.*\|[#█░▉▊▋▌▍▎▏]+\|.*|.*\b(?:it/s|s/it|ETA|elapsed|remaining)\b.*)"
        )
        self.re_client = re.compile(r"^\s*Client:\s*\d+\s*$")
        self._buf = ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _should_skip(self, text: str) -> bool:
        if "\r" in text:
            return True
        if self.re_bar.search(text):
            return True
        if self.re_client.search(text.strip()):
            return True
        return False

    def _write_all(self, text: str):
        for p in self.log_paths:
            # 目录可能尚未创建；这里稳妥创建
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(text)

    def write(self, message: str):
        self._buf += message
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line_out = line + "\n"
            if not self._should_skip(line_out):
                self._write_all(line_out)
            self.stream.write(line_out)

    def flush(self):
        if self._buf:
            if not self._should_skip(self._buf):
                self._write_all(self._buf)
            self.stream.write(self._buf)
            self._buf = ""
        if hasattr(self.stream, "flush"):
            self.stream.flush()

    def isatty(self):
        try:
            return self.stream.isatty()
        except Exception:
            return True

    def fileno(self):
        try:
            return self.stream.fileno()
        except Exception:
            import io as _io
            raise _io.UnsupportedOperation("fileno")

    def writable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ====================== MIA 结果瘦身 ======================

def shrink_mia_result(res):
    """
    接受 evaluate_mia_attack 的任意返回（dict/tuple/ndarray/tensor/str），
    仅抽取四个标量：accuracy / precision / recall / f1。
    """
    if res is None:
        return None
    def _to_float(x):
        try:
            if isinstance(x, (list, tuple)) and x:
                return float(sum(_to_float(v) for v in x)/len(x))
            if isinstance(x, torch.Tensor):
                return float(x.mean().item())
            if isinstance(x, np.ndarray):
                return float(x.mean())
            return float(x)
        except Exception:
            return None
    acc=prec=rec=f1=None
    if isinstance(res, dict):
        m={k.lower().replace('-','_'):k for k in res.keys()}
        def g(*names):
            for n in names:
                if n in m: return res[m[n]]
            # 包含匹配：支持 mia_attacker_precision 等
            for lk, ok in m.items():
                if any(n in lk for n in names):
                    return res[ok]
            return None
        acc = g('accuracy','acc')
        prec= g('precision','precision_score')
        rec = g('recall','recall_score')
        f1  = g('f1','f1_score')
    elif isinstance(res, (list, tuple)):
        if len(res)>0: prec=res[0]
        if len(res)>1: rec =res[1]
        if len(res)>2: f1  =res[2]
    # 返回统一瘦身结果
    return {
        'accuracy': _to_float(acc),
        'precision': _to_float(prec),
        'recall': _to_float(rec),
        'f1': _to_float(f1),
    }


# ====================== PeakMem ======================

try:
    import resource as _resource  # not available on Windows
except Exception:
    _resource = None


class PeakMem:
    def __init__(self, device):
        self.device = device
        self.cpu_peak_mb = None
        self.gpu_peak_mb = None
        self._base_ru = None
        self._is_cuda = False
    def __enter__(self):
        import gc as _gc
        self._is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        _gc.collect()
        if self._is_cuda:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch.device(self.device))
            except Exception:
                pass
        if _resource is not None:
            try:
                self._base_ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            except Exception:
                self._base_ru = None
        return self
    def __exit__(self, exc_type, exc, tb):
        # CPU peak (delta ru_maxrss)
        if (_resource is not None) and (self._base_ru is not None):
            try:
                ru1 = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
                delta = max(0, ru1 - self._base_ru)
                # Linux returns KB; macOS returns bytes
                bytes_used = delta if sys.platform == "darwin" else (delta * 1024)
                self.cpu_peak_mb = bytes_used / (1024 * 1024)
            except Exception:
                self.cpu_peak_mb = None
        # GPU peak (allocated)
        if self._is_cuda:
            try:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(torch.device(self.device))
                self.gpu_peak_mb = peak / (1024 * 1024)
            except Exception:
                self.gpu_peak_mb = None
        return False


def print_mem_overhead(label: str, pm: PeakMem, summary: dict):
    cpu_str = f"{pm.cpu_peak_mb:.2f} MB" if pm.cpu_peak_mb is not None else "NA"
    gpu_str = f"{pm.gpu_peak_mb:.2f} MB" if pm.gpu_peak_mb is not None else "NA"
    print(f"[Memory] {label}: peak CPU={cpu_str}; peak GPU={gpu_str}")
    try:
        summary.setdefault('space_overhead', {})[label] = {
            'cpu_peak_mb': pm.cpu_peak_mb,
            'gpu_peak_mb': pm.gpu_peak_mb
        }
    except Exception:
        pass
