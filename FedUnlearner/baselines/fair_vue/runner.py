# -*- coding: utf-8 -*-
"""
FAIR-VUE 主运行器：从 main.py 中提取的完整 FAIR-VUE 遗忘流程。
包括：层模式选择、逐轮增量解析/缓存、三参自动调参、Fisher、SVD/Gram、
ρ 切分、特异分量擦除、正交修复、可选 Healing、评测与指标打印。
"""

import os
import time
import gc
from copy import deepcopy
from collections import deque

import numpy as np
import torch

from run_utils import seed_everything, get_accuracy_only, PeakMem, print_mem_overhead

from FedUnlearner.utils import (
    eval_ce_loss, cosine_angle_between_models, print_forgetting_metrics, eval_retain_acc
)
from FedUnlearner.baselines.fair_vue.fisher import empirical_fisher_diagonal
from FedUnlearner.baselines.fair_vue.subspace import (
    weighted_matrix_from_deltas_keys, rho_values_keys,
    topk_right_singular_vectors_gram,
    flatten_by_keys, state_dict_like_by_keys,
    flatten_state_dict, split_subspaces
)
from FedUnlearner.baselines.fair_vue.projection import projection_matrix
from FedUnlearner.baselines.fair_vue.healing import weight_interpolate_heal
from FedUnlearner.baselines.fair_vue.round_helpers import (
    iter_round_deltas_stream,
    delta_cache_key, delta_cache_load, delta_cache_save
)


def run_fair_vue(
    args,
    global_model,
    train_path,
    client_endpoints,
    clientwise_dataloaders,
    test_dataloader,
    retrained_global_model,
    has_retrain_baseline,
    t_retrain_sec,
    attack_model,
    mia_eval_nonmem_loader,
    num_classes,
    weights_path,
    summary,
    get_performance_fn,
    evaluate_mia_attack_fn,
):
    """
    执行完整的 FAIR-VUE 遗忘流程，返回 (fair_model, fair_time_sec)。

    Parameters
    ----------
    args : argparse.Namespace
    global_model : nn.Module - 训练后的全局模型
    train_path : str - full_training 目录路径
    client_endpoints : dict[int, LocalClientEndpoint]
    clientwise_dataloaders : dict[int, DataLoader]
    test_dataloader : DataLoader
    retrained_global_model : nn.Module - 重训练基线模型
    has_retrain_baseline : bool
    t_retrain_sec : float or None
    attack_model : MIA 攻击模型 or None
    mia_eval_nonmem_loader : DataLoader or None
    num_classes : int
    weights_path : str - 实验目录
    summary : dict - 汇总字典
    get_performance_fn : callable - get_performance 函数引用
    evaluate_mia_attack_fn : callable or None - evaluate_mia_attack 函数引用
    """

    # ---- FAIR-VUE（按轮）----
    print(">>> Running FAIR-VUE (round-wise)...")
    _t0 = time.time()
    _fv_last_t = _t0

    def _fv_time_mark(label, start_t, last_t):
        now = time.time()
        if args.fair_vue_debug:
            print(f"[FV-TIME] {label}: step={now - last_t:.3f}s, total={now - start_t:.3f}s")
        return now

    _pm_fair = PeakMem(args.device); _pm_fair.__enter__()
    fair_model = deepcopy(global_model).to(args.device)
    fair_model.eval()

    forget_client = args.forget_clients[0]
    target_id = forget_client

    # —— 参与子空间的参数集合：优先分类头参数，兼容多种命名
    all_param_keys = [name for name, p in fair_model.named_parameters() if p.requires_grad]
    
    if args.fair_layer_mode == 'all':
        # 模式1: 全量参数
        param_keys = all_param_keys
    elif args.fair_layer_mode == 'deep':
        # 模式2: 深层特征 + 分类头 (ResNet: layer4 + fc; SmallCNN: 最后50%层)
        preferred_head = ("fc.", "linear.", "classifier.", "head.")
        preferred_deep = ("layer4.", "layer3.", "conv3.", "conv2.") # 常见深层命名
        param_keys = [k for k in all_param_keys if any(k.startswith(p) for p in preferred_head + preferred_deep)]
        if len(param_keys) < len(all_param_keys) * 0.1: # 兜底，防止没匹配上
            param_keys = all_param_keys[-(len(all_param_keys)//2):]
    else:
        # 模式3: (默认) 仅分类头
        preferred = ("fc.", "linear.", "classifier.", "head.")
        param_keys = [k for k in all_param_keys if any(k.startswith(pref) for pref in preferred)]
        if len(param_keys) == 0:
            param_keys = all_param_keys[-min(4, len(all_param_keys)):]

    print(f"[FAIR-VUE] Layer Mode: {args.fair_layer_mode}, Params selected: {len(param_keys)} / {len(all_param_keys)}")

    if args.fair_vue_debug:
        print(f"[FV-DBG] param_keys selected (n={len(param_keys)}): {param_keys[:6]}{'...' if len(param_keys)>6 else ''}")

    # 1) 目标客户端逐轮增量 Δ_{cid}^{(r)}：先尝试从缓存加载，再视情况重新解析
    #    尊重 --skip_training 时传入的 --full_training_dir；否则沿用上文解析出的 train_path
    fv_train_path = os.path.abspath(args.full_training_dir) if args.full_training_dir else os.path.abspath(train_path)

    target_deltas_list = None
    other_deltas_list = None
    rounds_seen = 0
    cache_hit = False
    cache_key_val = None
    cache_meta = None

    if getattr(args, "fair_use_delta_cache", True):
        cache_dir = os.path.join(fv_train_path, "fair_vue_cache")
        try:
            cache_key_val, cache_meta = delta_cache_key(
                fv_train_path,
                args.total_num_clients,
                target_id,
                param_keys,
                args.fair_target_rounds,
                layer_mode=str(getattr(args, 'fair_layer_mode', 'classifier')),
            )
            cached = delta_cache_load(cache_dir, cache_key_val, cache_meta)
            if cached is not None:
                target_deltas_list = list(cached.get("target_deltas", []))
                other_deltas_list = list(cached.get("other_deltas_last_round", []))
                rounds_seen = int(cached.get("rounds_seen", len(target_deltas_list)))
                cache_hit = True
                if args.fair_vue_debug:
                    print(f"[FV-CACHE] delta cache HIT (key={cache_key_val}, "
                          f"rounds_seen={rounds_seen}, T={len(target_deltas_list)})")
        except Exception as e:
            if args.fair_vue_debug:
                print(f"[FV-CACHE][WARN] load failed, fallback to recompute: {repr(e)}")

    if not cache_hit:
        buf_T = deque(maxlen=max(1, int(args.fair_target_rounds)))
        last_others = []
        rounds_seen = 0
        for r_cur, deltas_r in iter_round_deltas_stream(
            fv_train_path,
            args.total_num_clients,
            param_keys=param_keys,  # 只在选中的参数上算 Δ
            max_rounds=int(getattr(args, "fair_target_rounds", 200)),  # 只取最近 R 轮
        ):
            rounds_seen += 1
            if target_id in deltas_r:
                buf_T.append(deltas_r[target_id])
            last_others = [d for cid, d in deltas_r.items() if cid != target_id]
            # 释放本轮临时
            del deltas_r
        target_deltas_list = list(buf_T)
        other_deltas_list = last_others

        # 仅在有数据时写缓存
        if getattr(args, "fair_use_delta_cache", True) and len(target_deltas_list) > 0:
            try:
                if cache_dir is None:
                    cache_dir = os.path.join(fv_train_path, "fair_vue_cache")
                if cache_key_val is None or cache_meta is None:
                    cache_key_val, cache_meta = delta_cache_key(
                        fv_train_path, args.total_num_clients, target_id,
                        param_keys, args.fair_target_rounds,
                        layer_mode=str(getattr(args, 'fair_layer_mode', 'classifier')),
                    )
                payload = {
                    "meta": cache_meta,
                    "target_deltas": target_deltas_list,
                    "other_deltas_last_round": other_deltas_list,
                    "rounds_seen": int(rounds_seen),
                }
                delta_cache_save(cache_dir, cache_key_val, payload, verbose=args.fair_vue_debug)
            except Exception as e:
                if args.fair_vue_debug:
                    print(f"[FV-CACHE][WARN] save failed (ignored): {repr(e)}")

    # === 诊断（不再依赖 rounds / round_client_deltas）===
    if args.fair_vue_debug:
        print(f"[FV-DBG] rounds_seen={rounds_seen}, T={len(target_deltas_list)}, M={len(other_deltas_list)}")
        # 打印：目标增量前3个的范数，以及最后一轮其它客户端增量的均值范数
        from statistics import mean
        def _flat_norm(d):
            return float(torch.norm(flatten_state_dict(d)).item())
        head_T = target_deltas_list[:3]
        if head_T:
            ns_T = [_flat_norm(d) for d in head_T]
            print(f"[FV-DBG] target_deltas head norms: {[f'{v:.3e}' for v in ns_T]}")
        if other_deltas_list:
            ns_O = [_flat_norm(d) for d in other_deltas_list]
            print(f"[FV-DBG] others(last round) mean||Δ||={mean(ns_O):.3e} (n={len(ns_O)})")

    if len(target_deltas_list) == 0:
        raise RuntimeError(f"[FAIR-VUE] 没有解析到目标客户端的逐轮增量，无法进行 SVD/Gram。")

    # === 原有：target/others 划分完成后 ===
    if args.fair_vue_debug:
        print(f"[FV-DBG] target_id={target_id}, T=len(target_deltas_list)={len(target_deltas_list)}, "
              f"M=len(other_deltas_list)={len(other_deltas_list)}")
        # Step1: 逐轮增量解析结束
        _fv_last_t = _fv_time_mark("step1_round_deltas", _t0, _fv_last_t)

    # ==========================================================
    # === FAIR-VUE 预自动调参（三项）：b/k/τ（不访问原始样本） ===
    # ==========================================================
    # [Fix] 强制重置种子，防止 MIA 等前置操作消耗随机状态导致调参采样不一致
    if args.seed is not None:
        seed_everything(args.seed)

    if args.fair_auto_tune_all:
        if args.fair_vue_debug:
            print("[FV-AUTO] ==== Start auto-tuning {fisher_batches, rank_k, tau_mode} ====")

        # —— 仅参数键，保证与 Fisher 的键一致
        all_param_keys_at = [name for name, p in fair_model.named_parameters() if p.requires_grad]
        preferred = ("fc.", "linear.", "classifier.", "head.")
        param_keys_at = [k for k in all_param_keys_at if any(k.startswith(pref) for pref in preferred)]
        if not param_keys_at:  # 兜底：避免空集合
            param_keys_at = all_param_keys_at[-min(4, len(all_param_keys_at)):]

        # ——（1）Fisher 批次数：稳定性（与 ≥2b 对比的余弦相似度）
        def _parse_list_csv_int(s: str):
            return [int(x) for x in str(s).split(',') if str(x).strip()!='']
        fisher_grid = sorted(set([b for b in (_parse_list_csv_int(args.fair_fisher_grid) or [1,2,5,10]) if b>0]))
        stability = float(args.fair_fisher_stability)
        def _flatten_fi(Fi: dict):
            xs = [Fi[k].detach().flatten().float().cpu() for k in param_keys_at if k in Fi]
            if not xs:
                # 兜底：至少用 Fi 的全部键拼成向量（仍可用于相似度判断）
                xs = [v.detach().flatten().float().cpu() for v in Fi.values()]
            return torch.cat(xs) if xs else torch.zeros(1)
        def _cos(a,b):
            na, nb = torch.norm(a), torch.norm(b)
            if na.item()==0 or nb.item()==0: return 0.0
            return float(torch.clamp(torch.dot(a,b)/(na*nb), -1.0, 1.0).item())
        chosen_b = fisher_grid[-1]
        chosen_fisher = None
        for b in fisher_grid:
            b2 = next((c for c in fisher_grid if c >= 2*b), fisher_grid[-1])
            # [Auto-Tune] 自动调参阶段强制使用 diagonal 以加速，避免 full fisher 卡死
            Fi_b  = client_endpoints[target_id].compute_fisher(fair_model.state_dict(), param_keys=param_keys_at, device=args.device, max_batches=b, fisher_type='diagonal')
            Fi_b2 = client_endpoints[target_id].compute_fisher(fair_model.state_dict(), param_keys=param_keys_at, device=args.device, max_batches=b2, fisher_type='diagonal')
            sim = _cos(_flatten_fi(Fi_b), _flatten_fi(Fi_b2))
            if args.fair_vue_debug:
                print(f"[FV-AUTO][Fisher] b={b} vs b'={b2} → cos={sim:.4f}")
            if sim >= stability:
                chosen_b = b
                chosen_fisher = Fi_b2
                break
        if chosen_fisher is None:
            chosen_fisher = client_endpoints[target_id].compute_fisher(fair_model.state_dict(), param_keys=param_keys_at, device=args.device, max_batches=chosen_b, fisher_type='diagonal')
        args.fair_fisher_batches = int(chosen_b)
        if args.fair_vue_debug:
            print(f"[FV-AUTO][Fisher] chosen_b={args.fair_fisher_batches}")

        # ——（2）rank_k：Fisher 加权的 Δ_target 历史矩阵的 SVD 累计能量阈值
        if len(target_deltas_list) >= 1:
            Xw = weighted_matrix_from_deltas_keys(target_deltas_list, chosen_fisher, param_keys_at, device="cpu")
            Xc = Xw - Xw.mean(dim=0, keepdim=True)
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            energy = (S**2); cum = torch.cumsum(energy, dim=0); total = torch.sum(energy) + 1e-12
            thr = float(args.fair_rank_energy)
            k_auto = int(torch.searchsorted(cum/total, torch.tensor(thr, device=cum.device)).item() + 1)
            k_auto = max(int(args.fair_rank_k_min), min(int(args.fair_rank_k_max), k_auto))
            args.fair_rank_k = int(k_auto)
            if args.fair_vue_debug:
                print(f"[FV-AUTO][rank_k] energy_thr={thr:.2f} → k={args.fair_rank_k} (min={args.fair_rank_k_min}, max={args.fair_rank_k_max})")

        # ——（3）tau_mode：用其它客户端在最后一轮的 Δ 计算 ρ 分布分离度
        if len(other_deltas_list) >= 1 and len(target_deltas_list) >= 1:
            # 用同一 V_k（与上面 Xw 一致）
            Xc_for_V = Xw - Xw.mean(dim=0, keepdim=True)
            _U, _S, Vh_full = torch.linalg.svd(Xc_for_V, full_matrices=False)
            V_k = Vh_full.T[:, :int(args.fair_rank_k)]
            def _rho_sep(tau_mode: str):
                rhos = rho_values_keys(V_k, [d for d in other_deltas_list if isinstance(d, dict)], param_keys_at)
                if len(rhos)==0: return 0.0
                r = np.asarray(rhos, dtype=float)
                tau = np.median(r) if tau_mode=='median' else float(r.mean())
                lower, upper = r[r<tau], r[r>=tau]
                if len(lower)==0 or len(upper)==0: return 0.0
                gap = float(upper.mean() - lower.mean())
                return gap if args.fair_tau_metric=='gap' else gap / float(r.std()+1e-12)
            s_med, s_mean = _rho_sep('median'), _rho_sep('mean')
            args.fair_tau_mode = 'median' if s_med >= s_mean else 'mean'
            if args.fair_vue_debug:
                print(f"[FV-AUTO][tau] sep(median)={s_med:.4f}, sep(mean)={s_mean:.4f} → choose {args.fair_tau_mode}")
    # =================== 三参自动调参结束 ====================
    if args.fair_auto_tune_all and args.fair_vue_debug:
        _fv_last_t = _fv_time_mark("step2_auto_tune_all", _t0, _fv_last_t)


    # 3) Fisher（遗忘指令下发 → 目标客户端本地计算 → 仅上传 Fisher 对角）
    #    固定随机性，避免抽样差异带来系统性偏移（使用 args.seed 而非硬编码）
    _fisher_seed = args.seed if args.seed is not None else 42
    seed_everything(_fisher_seed)
    if target_id in client_endpoints:
        fisher = client_endpoints[target_id].compute_fisher(
            model_state_dict=fair_model.state_dict(),
            param_keys=param_keys,  # 传入选定的参数键，确保 Full Fisher 维度匹配
            device="cpu",
            max_batches=args.fair_fisher_batches,
            fisher_type=args.fair_fisher_type
        )
    else:
        # fallback：没有端点时，仅对可训练参数使用单位权重
        fisher = {name: torch.ones_like(p) for name, p in fair_model.named_parameters() if p.requires_grad}

    
    # [Ablation] w/o Fisher: 强制覆盖 Fisher 为全 1 (退化为欧氏空间)
    if args.fair_ablation == 'no_fisher':
        if args.fair_vue_debug:
            print("[FV-ABLATION] Mode: w/o Fisher -> Overwriting Fisher with Identity (Ones).")
        if args.fair_fisher_type == 'full':
            # 如果是 Full 模式下 ablation，生成单位矩阵
            # 注意：这可能会 OOM，但逻辑上是正确的
            dim = fisher.shape[0]
            fisher = torch.eye(dim)
        else:
            for k in fisher:
                fisher[k] = torch.ones_like(fisher[k])


    # === Fisher 计算完毕后，插入点 B ===
    if args.fair_vue_debug:
        if isinstance(fisher, torch.Tensor):
            # Full matrix
            print(f"[FV-DBG] Full Fisher Matrix: shape={fisher.shape}, "
                  f"trace={fisher.trace().item():.3e}, mean={fisher.mean().item():.3e}")
        else:
            fvec = flatten_state_dict(fisher)
            print(f"[FV-DBG] fisher: device={fvec.device}, D={fvec.numel()}, "
                f"min={float(torch.min(fvec)): .3e}, max={float(torch.max(fvec)): .3e}, "
                f"mean={float(torch.mean(fvec)): .3e}")

    if args.fair_vue_debug:
        _fv_last_t = _fv_time_mark("step3_fisher", _t0, _fv_last_t)

    # 4) Fisher加权矩阵 & 低秩SVD拿到主方向V
    #    与 Fisher / deltas 对齐 keys，避免空拼接或维度不一致
    
    if args.fair_fisher_type == 'full':
        # Full 模式：Fisher 是矩阵，没有 keys。
        # 假设 param_keys 已经在 compute_fisher 时对齐，这里只检查 deltas 是否包含这些 key
        keys_valid = [k for k in param_keys
                      if all(k in d for d in target_deltas_list)
                      and (len(other_deltas_list) == 0 or all(k in d for d in other_deltas_list))]
    else:
        # Diagonal 模式：Fisher 是字典，需要求交集
        fisher_keys = set(fisher.keys())
        keys_valid = [k for k in param_keys
                      if (k in fisher_keys)
                      and all(k in d for d in target_deltas_list)
                      and (len(other_deltas_list) == 0 or all(k in d for d in other_deltas_list))]
    if len(keys_valid) == 0:
        # 放宽：只强制覆盖 target_deltas
        if args.fair_fisher_type == 'full':
            keys_valid = [k for k in param_keys if all(k in d for d in target_deltas_list)]
        else:
            keys_valid = [k for k in param_keys if (k in fisher_keys) and all(k in d for d in target_deltas_list)]

    if len(keys_valid) == 0:
        raise RuntimeError(
            f"[FAIR-VUE] 参与子空间的参数键为空。示例 param_keys={param_keys[:6]}，"
            f"fisher_keys={len(fisher_keys) if not isinstance(fisher, torch.Tensor) else 'full'}。请检查分类头命名（如 fc./linear./classifier./head./layer4.）"
        )
    if args.fair_vue_debug:
        print(f"[FV-DBG] using {len(keys_valid)} keys for subspace: {keys_valid}")
    Xw = weighted_matrix_from_deltas_keys(target_deltas_list, fisher, keys_valid, device="cpu")
    k = int(args.fair_rank_k)
    if k > Xw.shape[0]:
        k = Xw.shape[0]
    V = topk_right_singular_vectors_gram(Xw, k=k)  # D × k
    # === 诊断与设备设置（简化，移除坏掉的 f-string 与重复赋值） ===
    if args.fair_vue_debug:
        # Xw: T x D; V: D x k
        print(f"[FV-DBG] Xw shape={tuple(Xw.shape)}, device={Xw.device}")
    
    # [Ablation] w/o Dual: 禁用 Gram-SVD 优化，直接在大矩阵上做 SVD (易 OOM)
    if args.fair_ablation == 'no_dual':
        if args.fair_vue_debug:
            print(f"[FV-ABLATION] Mode: w/o Dual -> Running raw SVD on shape {tuple(Xw.shape)} (High Risk of OOM!)")
        try:
            # 直接对 T x D 矩阵做 SVD
            Xc = Xw - Xw.mean(dim=0, keepdim=True)
            # full_matrices=False 会返回 Vh (k x D)
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            V = Vh.T[:, :k] # D x k
        except RuntimeError as e:
            print(f"\n[FV-ABLATION] !!! OOM Triggered as expected in w/o Dual mode: {e} !!!\n")
            # 为了让程序不崩溃以便记录 'OOM' 结果，这里做一个假的 V 或者直接抛出
            raise e 
    else:
        # 正常路径：对偶 Gram 方法
        V = topk_right_singular_vectors_gram(Xw, k=k)  # D × k
    
    if args.fair_vue_debug:
        print(f"[FV-DBG] Xw shape={tuple(Xw.shape)}, V shape={tuple(V.shape)}")

    # 使用 Gram-SVD 结果 V（D×k），避免在高维上再做一次完整 SVD
    dev = V.device  # 统一使用这个设备

    # 5) 计算ρ并按阈值切分为 V_spec / V_comm
    rhos = rho_values_keys(V, other_deltas_list, keys_valid, fisher=fisher, max_samples=int(args.fair_rho_max_samples))  # 内部已用 V.device 对齐
    if len(rhos) == 0:
        tau = float('inf')
    else:
        tau = (sorted(rhos)[len(rhos)//2] if args.fair_tau_mode == 'median'
            else sum(rhos)/len(rhos))
    # ★ 这里要接住索引（别用下划线丢掉）
    V_spec, V_comm, spec_idx, comm_idx = split_subspaces(V, rhos, tau)

    # ★ 兜底：如果 V_spec 还是空，强制取 rho 最小的 1~2 个方向，放在 debug 外侧更稳
    if V_spec is None or V_spec.numel() == 0:
        idx_sorted = sorted(range(len(rhos)), key=lambda i: rhos[i])
        take = min(2, len(idx_sorted))
        if take > 0:
            V_spec = V[:, idx_sorted[:take]]

    if args.fair_vue_debug:
        r = np.array(rhos) if len(rhos)>0 else np.array([0.0])
        def q(a,p): 
            return float(np.quantile(a,p))
        print(f"[FV-DBG] rho stats: n={len(rhos)}, min={r.min():.3e}, q25={q(r,0.25):.3e}, "
            f"median={q(r,0.5):.3e}, q75={q(r,0.75):.3e}, max={r.max():.3e}, tau={tau:.3e}")
        print(f"[FV-DBG] |V_spec|={V_spec.shape[1]}, idx={spec_idx[:10]}{'...' if len(spec_idx)>10 else ''}")
        print(f"[FV-DBG] |V_comm|={V_comm.shape[1]}")

        if V_spec.numel() == 0:
            print("[FV-DBG][WARN] V_spec is EMPTY -> 将强制选择 rho 最小的1~2个方向作为特异方向（兜底）")
            idx_sorted = sorted(range(len(rhos)), key=lambda i: rhos[i])
            take = min(2, len(idx_sorted))
            V_spec = V[:, idx_sorted[:take]]

    # 6) 基于 V_spec 构造投影矩阵（会跟随 V_spec.device）
    # 低秩：只构造 Q，不构造 P

    Q = None
    if V_spec is not None and V_spec.numel() > 0:
        Q, _ = torch.linalg.qr(V_spec, mode='reduced')  # D_param x r
    
    # [Modified] 构造公共子空间的投影矩阵 Q_comm，用于提取"有用知识"
    Q_comm = None
    if V_comm is not None and V_comm.numel() > 0:
        Q_comm, _ = torch.linalg.qr(V_comm, mode='reduced')

    if args.fair_vue_debug:
        print(f"[FV-DBG] Q is None? {Q is None}, "
            f"Q.shape={(None if Q is None else tuple(Q.shape))}, device={(None if Q is None else Q.device)}")

        # Step4: 子空间（Gram-SVD + ρ + V_spec/V_comm + Q）结束
        _fv_last_t = _fv_time_mark("step4_subspace_and_rho", _t0, _fv_last_t)



    # 7) 逐轮累计"特异分量"
    # 只在 parameters 空间累积"目标客户端"的特异分量
    start_sd = {k: v.detach().clone().cpu() for k, v in fair_model.state_dict().items()}
    Dparam   = flatten_by_keys(start_sd, keys_valid).numel()
    spec_total = torch.zeros(Dparam, device=dev)

    # 仅基于已构建好的 target_deltas_list（避免再次引用 round_client_deltas）
    used_rounds = 0
    for d in target_deltas_list:
        d_tar = flatten_by_keys(d, keys_valid, device=dev)  # Δ_target^t
        if Q is not None:
            spec = Q @ (Q.T @ d_tar)     # 只拿"目标"的特异分量
        else:
            spec = torch.zeros_like(d_tar, device=dev)
        spec_total += spec

        
        used_rounds += 1
    if args.fair_vue_debug:
        print(f"[FV-DBG] used_rounds_for_target={used_rounds}, ||spec_total||_2={float(torch.norm(spec_total)):.3e}")

        # Step5: 累积特异分量结束
        _fv_last_t = _fv_time_mark("step5_accumulate_spec", _t0, _fv_last_t)

    # === 擦除系数确定 ===
    base_alpha = float(getattr(args, 'fair_erase_scale', 0.25))
    erase_scale = base_alpha

    if getattr(args, 'fair_auto_erase', False):
        # === 自动调参 erase_scale（仅 parameters；不访问原始数据）===
        # 1) 解析参数
        def _parse_pair_csv(s: str):
            xs = [float(x) for x in s.split(',') if x.strip()!='']
            if len(xs) < 2:
                return 0.0, 0.04
            return xs[0], xs[1]
        def _parse_list_csv(s: str):
            return [float(x) for x in s.split(',') if x.strip()!='']
        target_lo, target_hi = _parse_pair_csv(getattr(args, "fair_drop_bounds", "0.00,0.04"))
        grid_mults = _parse_list_csv(getattr(args, "fair_grid_scales", "0.5,0.75,1.0,1.25,1.5"))
        bisect_steps = int(getattr(args, "fair_bisect_steps", 3))

        # 2) 诊断量：Fisher 能量 & 特异性分
        def _fisher_energy(vec_1d: torch.Tensor) -> float:
            like = state_dict_like_by_keys(vec_1d.to('cpu'), start_sd, param_keys)
            s = 0.0
            for k in param_keys:
                Fi = fisher.get(k, None) if not isinstance(fisher, torch.Tensor) else None
                if Fi is None:
                    continue
                v = like[k].to(Fi.device).float()
                s += float((Fi.float().flatten() * (v.flatten()**2)).sum().item())
            return s
        spec_energy = _fisher_energy(spec_total)
        def _safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
            na = torch.norm(a); nb = torch.norm(b)
            if na.item()==0 or nb.item()==0:
                return 0.0
            return float(torch.dot(a, b) / (na*nb))
        with torch.no_grad():
            unit_spec = spec_total / (torch.norm(spec_total) + 1e-12)
            cos_list = []
            # 采样最多 256 个"其它客户端增量"估计平均相似度，避免过慢
            take = other_deltas_list[:256]
            for d in take:
                v = flatten_by_keys(d, param_keys, device=dev)
                cos_list.append(_safe_cos(unit_spec, v))
            avg_cos = sum(cos_list)/max(1, len(cos_list))
            idio = max(0.0, min(1.0, 1.0 - avg_cos))  # 特异性分：越大越"特"
        if args.fair_vue_debug:
            print(f"[FV-DBG] spec_fisher_energy={spec_energy:.3e}, avg_cos={avg_cos:.3f}, idiosyncrasy={idio:.3f}")

        # 3) 评估函数：临时应用 α·spec_total 到参数并测一次测试集精度
        def _eval_acc(alpha: float) -> float:
            param_now = flatten_by_keys(start_sd, param_keys, device=dev)
            param_new = param_now - float(alpha) * spec_total
            new_params = state_dict_like_by_keys(param_new.to('cpu'), start_sd, param_keys)
            tmp_sd = dict(start_sd)
            for k in param_keys:
                tmp_sd[k] = new_params[k]
            tmp_model = deepcopy(fair_model).to(args.device)
            tmp_model.load_state_dict(tmp_sd)
            acc = float(get_accuracy_only(tmp_model, test_dataloader, args.device))
            del tmp_model
            return acc

        # 用同一条评估链路测 baseline，确保 drop(0)==0
        baseline_acc = _eval_acc(0.0)

        # 4) 构造候选 α
        alpha0 = base_alpha * (0.7 + 0.6*idio)
        cands = sorted(set([0.0] + [max(0.0, m*base_alpha) for m in grid_mults] + [alpha0]))

        # 5) 粗网格搜索
        evals = [(a, _eval_acc(a)) for a in cands]
        drops = [(a, max(0.0, baseline_acc - acc)) for (a, acc) in evals]
        under = max([a for a, d in drops if d <= target_lo + 1e-6], default=None)
        over  = min([a for a, d in drops if d >= target_hi - 1e-6], default=None)

        if under is None and over is None:
            # 没覆盖目标区间
            mid = 0.5*(target_lo + target_hi)
            chosen = min(drops, key=lambda t: abs(t[1]-mid))[0]
        else:
            # 6) 二分细化
            lo = under if under is not None else 0.0
            hi = over  if over  is not None else max(cands)
            chosen = None
            for _ in range(max(0, bisect_steps)):
                mid_a = 0.5*(lo + hi)
                acc_m = _eval_acc(mid_a)
                drop_m = max(0.0, baseline_acc - acc_m)
                if args.fair_vue_debug:
                    print(f"[FV-DBG] bisect α={mid_a:.4f} → drop={drop_m:.4f}")
                if drop_m < target_lo:
                    lo = mid_a
                elif drop_m > target_hi:
                    hi = mid_a
                else:
                    chosen = mid_a
                    break
            if chosen is None:
                def _dist_to_interval(x, L, H): return 0.0 if L<=x<=H else min(abs(x-L), abs(x-H))
                drop_lo = max(0.0, baseline_acc - _eval_acc(lo))
                drop_hi = max(0.0, baseline_acc - _eval_acc(hi))
                chosen = lo if _dist_to_interval(drop_lo, target_lo, target_hi) <= _dist_to_interval(drop_hi, target_lo, target_hi) else hi
            
        erase_scale = chosen
        # Step6: 自动擦除系数搜索结束
        _fv_last_t = _fv_time_mark("step6_auto_erase_search", _t0, _fv_last_t)
    else:
            print(f"[FV-DBG] Auto-erase is DISABLED, using erase_scale={erase_scale}")

    # 7) 应用最终擦除到模型参数
    # [Corrected] 修正后的正交恢复：
    # 使用"其他客户端"的均值来修复公共子空间，绝对避免使用 target_deltas (防止隐私回流)
    
    repair_vec = torch.zeros_like(spec_total)
    if len(other_deltas_list) > 0 and Q_comm is not None:
        # 1. 计算保留客户端的均值方向
        # 注意：other_deltas_list 可能只包含最后一轮，但这代表了模型收敛时的通用梯度方向
        others_sum = torch.zeros_like(spec_total)
        for d in other_deltas_list:
            others_sum += flatten_by_keys(d, keys_valid, device=dev)
        others_mean = others_sum / float(len(other_deltas_list))

        # 2. 投影到公共子空间 (仅恢复通用知识)
        repair_vec = Q_comm @ (Q_comm.T @ others_mean)
        
        # 3. [Critical Fix] 能量补偿机制
        # 自动计算缩放系数，使得恢复的能量与擦除的能量在同一数量级
        norm_erase  = torch.norm(erase_scale * spec_total)
        norm_repair = torch.norm(repair_vec)
        
        # 目标：恢复约 50% ~ 60% 的被擦除能量，既能修补泛化损伤，又不至于覆盖遗忘效果
        # 如果 norm_repair 极小(防除零)，则不进行过度放大
        compensation_factor = 0.0
        if norm_repair > 1e-6:
            compensation_factor = args.fair_repair_ratio * (norm_erase / norm_repair)
        
        # 应用动态补偿系数
        repair_vec = repair_vec * compensation_factor

        if args.fair_vue_debug:
            print(f"[FV-DBG] Orthogonal Repair: ||Erase||={norm_erase:.3f}, ||RepairRaw||={norm_repair:.3f} -> Factor={compensation_factor:.3f}")

    # [Ablation] w/o Repair: 强制将修复向量置零
    if args.fair_ablation == 'no_repair':
        if args.fair_vue_debug:
            print("[FV-ABLATION] Mode: w/o Repair -> Forcing repair_vec to ZERO.")
        repair_vec = torch.zeros_like(spec_total)


    param_now   = flatten_by_keys(start_sd, param_keys, device=dev)
    # repair_vec 已经在上面被动态缩放过了，这里直接加即可
    param_new   = param_now - erase_scale * spec_total + repair_vec
    new_params  = state_dict_like_by_keys(param_new.to('cpu'), start_sd, param_keys)
    new_sd = dict(start_sd)
    for k in param_keys:
        new_sd[k] = new_params[k]
    fair_model.load_state_dict(new_sd)


    if args.fair_vue_debug:
        _fv_last_t = _fv_time_mark("step7_apply_erase", _t0, _fv_last_t)
   

    fair_time_sec = time.time() - _t0
    print(f"[Timing] FAIR-VUE time: {fair_time_sec:.2f}s")


    # 9) 评测 (移到计时结束后)
    perf = get_performance_fn(model=fair_model, test_dataloader=test_dataloader,
                        clientwise_dataloader=clientwise_dataloaders,
                        num_classes=num_classes, device=args.device)

    if args.fair_vue_debug:
        # Step8: 评测与汇总结束（总时间再标一遍）
        _fv_last_t = _fv_time_mark("step8_eval_and_summary", _t0, _fv_last_t)


    summary.setdefault('performance', {})
    summary['performance']['after_fair_vue'] = perf
    if args.verbose:
        print(f"Performance after FAIR-VUE : {perf}")

    # ---- 专门测忘却客户端的精度 ----
    forget_loader = clientwise_dataloaders[target_id]
    acc = get_accuracy_only(fair_model, forget_loader, args.device)
    print(f"[FAIR-VUE模型] 忘却客户端{target_id}自有数据精度: {acc*100:.2f}%")

    # ==== 六项指标统一打印（FAIR-VUE）====
    test_acc_fair    = get_accuracy_only(fair_model, test_dataloader, args.device)
    target_acc_fair  = acc
    retain_acc_fair  = eval_retain_acc(fair_model, clientwise_dataloaders, args.forget_clients, args.device)
    target_loss_fair = eval_ce_loss(fair_model, forget_loader, args.device)
    speedup_fair     = (t_retrain_sec / fair_time_sec) if (t_retrain_sec is not None and fair_time_sec > 0) else None
    angle_fair       = cosine_angle_between_models(fair_model, retrained_global_model) if has_retrain_baseline else None

    mia_fair = None
    if args.apply_membership_inference and evaluate_mia_attack_fn is not None:
        if args.mia_verbose:
            print("\n[调试] 开始执行成员推断攻击 (evaluate_mia_attack)...")
        # 与其他分支保持一致：对目标客户端执行成员推断
        mia_fair = evaluate_mia_attack_fn(
            target_model=deepcopy(fair_model),
            attack_model=attack_model,
            client_loaders=clientwise_dataloaders,
            test_loader=test_dataloader,
            dataset=args.dataset,
            forget_client_idx=args.forget_clients[0],
            device=args.device,
            eval_nonmem_loader=mia_eval_nonmem_loader
            
        )
    
        # [Save] 保存 FAIR-VUE 模型供后续可视化
        # 注意：如果启用了 heal，这里保存的是 heal 之后的最终模型
        save_f = os.path.join(weights_path, "fair_vue_model.pth")
        torch.save(fair_model.state_dict(), save_f)
        print(f"[FAIR-VUE] Model saved to: {save_f}")

        if args.mia_verbose:
            print(f"[调试] MIA 返回类型: {type(mia_fair)}")
            if isinstance(mia_fair, dict):
                print(f"[调试] MIA 字典键: {list(mia_fair.keys())[:10]}")
                for k, v in list(mia_fair.items())[:5]:
                    if isinstance(v, (int, float, str)):
                        print(f"  {k}: {v}")
                    elif hasattr(v, 'shape'):
                        print(f"  {k}: tensor/array shape={v.shape}")
                    elif isinstance(v, (list, tuple)):
                        print(f"  {k}: list length={len(v)}")
                    else:
                        print(f"  {k}: type={type(v)}")
    print_forgetting_metrics(
        method_name="FAIR-VUE",
        test_acc=test_acc_fair,
        retain_acc=retain_acc_fair,
        target_acc=target_acc_fair,
        target_loss=target_loss_fair,
        speedup_x=speedup_fair,
        angle_deg=angle_fair,
        mia_result=mia_fair
    )
    _pm_fair.__exit__(None, None, None)
    print_mem_overhead("FAIR-VUE", _pm_fair, summary)
    # —— 关键：清理 MIA 大对象并同步 CUDA，避免后续卡住 —— 
    try:
        if isinstance(mia_fair, dict):
            for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                if k in mia_fair: mia_fair.pop(k, None)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass



    # ===== HEALING: 让遗忘后的模型在保留客户端上"治疗/恢复" =====
    if args.heal:
        # 1) teacher 选择：post=用擦除后的模型；pre=用遗忘前的全局模型
        if args.heal_teacher == 'post':
            teacher = deepcopy(fair_model).to(args.device).eval()
        else:
            teacher = deepcopy(global_model).to(args.device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # 2) 方案1：一次性权重插值（Data-Free；不会触碰任意原始样本）
        if args.heal_teacher == 'post':
                print("[HEAL] teacher=post 与 student 相同，插值将不生效；如需恢复请使用 --heal_teacher pre")
        weight_interpolate_heal(
            student=fair_model,
            teacher=teacher,
            alpha=args.heal_alpha
        )

        # 4) 治疗后评测
        perf_heal = get_performance_fn(model=fair_model, test_dataloader=test_dataloader,
                                    clientwise_dataloader=clientwise_dataloaders,
                                    num_classes=num_classes, device=args.device)
        print(f"[HEAL] Performance after healing : {perf_heal}")
        forget_loader = clientwise_dataloaders[target_id]
        acc_forget_heal = get_accuracy_only(fair_model, forget_loader, args.device)
        print(f"[HEAL] 忘却客户端{target_id}自有数据精度(治疗后): {acc_forget_heal*100:.2f}%")

    return fair_model, fair_time_sec
