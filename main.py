import torch
import torchvision
import json
import time
import os
from copy import deepcopy
import random
import numpy as np
import sys
from datetime import datetime
import re
import gc

# === 从重构模块导入 ===
from run_utils import (
    str2bool, seed_everything, worker_init_fn as _worker_init_fn,
    get_accuracy_only, ProxyLog, shrink_mia_result as _shrink_mia_result,
    PeakMem, print_mem_overhead as _print_mem_overhead,
)
from config import build_parser

from FedUnlearner.utils import (
    print_exp_details, print_clientwise_class_distribution,
    eval_ce_loss, cosine_angle_between_models, print_forgetting_metrics,
    eval_retain_acc
)
from FedUnlearner.data_utils import get_dataset, create_dirichlet_data_distribution, create_iid_data_distribution, create_class_exclusive_distribution
from FedUnlearner.fed_learn import fed_train, get_performance
from FedUnlearner.models import AllCNN, ResNet18, SmallCNN
from FedUnlearner.attacks.backdoor import create_backdoor_dataset, evaluate_backdoor_attack
from FedUnlearner.baselines import run_pga, run_fed_eraser
from FedUnlearner.baselines import run_conda
from FedUnlearner.attacks.mia import train_attack_model, evaluate_mia_attack
from FedUnlearner.attacks.poisoning import create_poisoning_dataset, evaluate_poisoning_attack

from FedUnlearner.baselines.fair_vue.fisher import empirical_fisher_diagonal

# === 日志目录 ===
os.makedirs("./logs", exist_ok=True)

# === 参数解析器（从 config.py 导入） ===
parser = build_parser()




if __name__ == "__main__":

    args = parser.parse_args()
    # 将开关传递给 mia.py（用环境变量最省事）
    import os as _os
    _os.environ["MIA_VERBOSE"] = "1" if args.mia_verbose else "0"
    weights_path = os.path.abspath(os.path.join(args.exp_path, args.exp_name))
    
    # === 两份日志路径（时间命名 + 参数命名） ===
    LOG_ROOT = "./logs"
    TIME_DIR = os.path.join(LOG_ROOT, "by_time")
    PARAM_DIR = os.path.join(LOG_ROOT, "by_params")
    os.makedirs(TIME_DIR, exist_ok=True)
    os.makedirs(PARAM_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 只支持单个忘却客户端；若为空就用 NA 占位
    cid = (args.forget_clients[0] if getattr(args, "forget_clients", None) else "NA")

    param_basename = (
        f"client{cid}"
        f"_k{args.fair_rank_k}"
        f"_tau{args.fair_tau_mode}"
        f"_fb{args.fair_fisher_batches}"
        f"_es{args.fair_erase_scale}.log"
    )

    log_path_time   = os.path.join(TIME_DIR,   f"run_{ts}.log")
    log_path_param  = os.path.join(PARAM_DIR,  param_basename)

    # 安装到 stdout / stderr（同样的输出写两份日志）
    sys.stdout = ProxyLog(sys.stdout, [log_path_time, log_path_param])
    sys.stderr = ProxyLog(sys.stderr, [log_path_time, log_path_param])

    # 之后再打印实验详情，确保写入两份日志
    print_exp_details(args)
    summary = {}
    # get the dataset
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset)

    # create client groups
    client_groups = None

    # 设置随机种子
    if args.seed is not None:
        seed_everything(args.seed)

    if args.client_data_distribution == 'dirichlet':
        clientwise_dataset = create_dirichlet_data_distribution(train_dataset,
                                                                num_clients=args.total_num_clients, num_classes=num_classes, alpha=0.5)
    elif args.client_data_distribution == 'iid':
        clientwise_dataset = create_iid_data_distribution(train_dataset, num_clients=args.total_num_clients,
                                                          num_classes=num_classes)
    elif args.client_data_distribution == 'exclusive':
        # [修改] 多类主导分布：Client 0 独占 0, 1, 2, 3 的全量数据
        # 其他客户端只有这四个类别的 20% 副本 (稀释后)
        target_classes = [0, 1, 2, 3]
        print(f"[Setup] 创建多类主导分布：Client {cid} 独占 Classes {target_classes}")
        clientwise_dataset = create_class_exclusive_distribution(train_dataset, 
                                                                 num_clients=args.total_num_clients,
                                                                 num_classes=num_classes,
                                                                 exclusive_client=int(cid) if cid != "NA" else 0,
                                                                 exclusive_classes=target_classes)
    else:
        raise "Invalid client data distribution"

    # print the clientwise class distribution
    print_clientwise_class_distribution(clientwise_dataset, num_classes)


    if args.num_participating_clients > 1:
        print(
            f"Cutting of num participating client to: {args.num_participating_clients}")
        clientwise_dataset = {i: clientwise_dataset[i] for i in range(
            args.num_participating_clients)}
        print("Clientwise distribution after cutting: ")
        print_clientwise_class_distribution(clientwise_dataset, num_classes)
    # get the forget client

    if len(args.forget_clients) > 1:
        raise "Only one client forgetting supported at the moment."
    forget_client = args.forget_clients[0]

    if args.apply_backdoor:
        backdoor_dataset = None
        backdoor_pixels = None
        image_size = 224
        patch_size = 30
        if args.backdoor_position == 'corner':
            # [top left corner of patch, bottom right corner of patch]
            backdoor_pixels = [(0, 0), (patch_size, patch_size)]
        elif args.backdoor_position == 'center':
            backdoor_pixels = [(image_size//2 - patch_size//2, image_size//2 - patch_size//2),
                               (image_size//2 + patch_size//2, image_size//2 + patch_size//2)]
        else:
            raise "Invalid backdoor position"

        print(
            f"Size of client dataset before backdoor ingestion: {len(clientwise_dataset[args.forget_clients[0]])}")
        clientwise_dataset, backdoor_context = create_backdoor_dataset(clientwise_dataset=clientwise_dataset,
                                                                       forget_clients=args.forget_clients,
                                                                       backdoor_pixels=backdoor_pixels,
                                                                       backdoor_label=args.backdoor_label,
                                                                       num_samples=args.num_backdoor_samples_per_forget_client
                                                                       )

        print(
            f"Size of client dataset after backdoor ingestion: {len(clientwise_dataset[args.forget_clients[0]])}")

    if args.apply_label_poisoning:
        clientwise_dataset, poisoning_context = create_poisoning_dataset(clientwise_dataset=clientwise_dataset,
                                                                         forget_clients=args.forget_clients,
                                                                         test_split=0.2,
                                                                         num_poisoning_samples=args.num_label_poison_samples)

    # create dataloaders for the clients
    clientwise_dataloaders = {}
    for client_id, client_dataset in clientwise_dataset.items():
        print(f"Creating data loader for client: {client_id}")
        # [种子锁定] 为每个客户端创建独立的 Generator，确保 shuffle 顺序可复现
        _dl_gen = torch.Generator()
        _dl_seed = (args.seed + client_id) if args.seed is not None else client_id
        _dl_gen.manual_seed(_dl_seed)
        client_dataloader = torch.utils.data.DataLoader(
            client_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, drop_last=True, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            generator=_dl_gen,
            worker_init_fn=_worker_init_fn)
        clientwise_dataloaders[client_id] = client_dataloader
    
    # === 本地 Fisher 端点：客户端持有数据；服务端仅“请求 Fisher”，不触碰原始样本 ===
    def _build_fresh_model_for_args(args):
        # 与上面创建全局模型的分支保持一致，客户端本地重建同构模型
        if args.model == 'allcnn':
            if args.dataset == 'mnist':
                return AllCNN(num_classes=num_classes, num_channels=1)
            else:
                return AllCNN(num_classes=num_classes)
        elif args.model == 'resnet18':
            if args.dataset == 'mnist':
                return ResNet18(num_classes=num_classes, pretrained=args.pretrained, num_channels=1, dataset=args.dataset)
            else:
                return ResNet18(num_classes=num_classes, pretrained=args.pretrained, dataset=args.dataset)
        elif args.model == 'smallcnn':
            in_ch = 1 if args.dataset == 'mnist' else 3
            return SmallCNN(num_channels=in_ch, num_classes=num_classes)
        else:
            raise ValueError("Invalid model name")

    class LocalClientEndpoint:
        """
        轻量“RPC”端点：模拟把模型权重下发到客户端，
        由客户端在本地数据上计算 Fisher 对角并上传（仅上传统计量，不上传原始样本）。
        """
        def __init__(self, cid, dataloader, args):
            self.cid = cid
            self.loader = dataloader
            self.args = args

        def compute_fisher(self, model_state_dict, param_keys=None, device="cpu", max_batches=10, fisher_type='diagonal'):
            # 客户端本地重建同构模型并载入服务端下发的参数
            model = _build_fresh_model_for_args(self.args)
            # 若服务端权重来自 DataParallel，自动去除 'module.' 前缀以与本地裸模型对齐
            ref_keys = list(model.state_dict().keys())
            if all(k.startswith("module.") for k in model_state_dict.keys()) and \
               not any(k.startswith("module.") for k in ref_keys):
                model_state_dict = {k.replace("module.", "", 1): v for k, v in model_state_dict.items()}
            ret = model.load_state_dict(model_state_dict, strict=True)
            # 严格校验，避免 BN buffer / 参数名不一致导致的静默偏差
            assert len(ret.missing_keys) == 0 and len(ret.unexpected_keys) == 0, \
                f"Incompatible keys when loading endpoint model: {ret}"
            
            # [关键修复] 临时创建一个全新的 DataLoader，彻底隔离 MIA 的影响
            # 无论之前 MIA 是否遍历过 self.loader，这里都强制使用固定种子生成新的迭代顺序
            g_fisher = torch.Generator()
            _fisher_dl_seed = self.args.seed if self.args.seed is not None else 42
            g_fisher.manual_seed(_fisher_dl_seed)
            
            # 复用原 loader 的参数，但注入 generator + worker_init_fn
            temp_loader = torch.utils.data.DataLoader(
                self.loader.dataset,
                batch_size=self.loader.batch_size,
                shuffle=True,
                num_workers=self.loader.num_workers,
                drop_last=self.loader.drop_last,
                generator=g_fisher,
                worker_init_fn=_worker_init_fn
            )

            if fisher_type == 'full':
                from FedUnlearner.baselines.fair_vue.fisher import empirical_fisher_full
                return empirical_fisher_full(
                    model=model,
                    dataloader=temp_loader,
                    param_keys=param_keys, # 必须传入 key 列表以对齐维度
                    device=device,
                    max_batches=max_batches
                )
            else:
                # 用原始算法计算经验 Fisher 对角近似
                return empirical_fisher_diagonal(
                    model=model,
                    dataloader=temp_loader, # 使用临时 loader，而非 self.loader
                    device=device,
                    max_batches=max_batches
                )

    # 为每个客户端建立一个端点（仅保存回调，不暴露原始数据给服务端使用）
    client_endpoints = {
        cid: LocalClientEndpoint(cid, loader, args)
        for cid, loader in clientwise_dataloaders.items()
    }    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        worker_init_fn=_worker_init_fn)

    # train the model
    global_model = None
    retrained_global_model = None
    if args.model == 'allcnn': 
        if args.dataset == 'mnist':
            global_model = AllCNN(num_classes=num_classes, num_channels=1)
        else:
            global_model = AllCNN(num_classes=num_classes)

    elif args.model == 'resnet18':
        if args.dataset == 'mnist':
            global_model = ResNet18(num_classes=num_classes,
                                    pretrained=args.pretrained, num_channels=1, dataset=args.dataset)
        else:
            global_model = ResNet18(num_classes=num_classes,
                                    pretrained=args.pretrained, dataset=args.dataset)

    elif args.model == 'smallcnn':
        in_ch = 1 if args.dataset == 'mnist' else 3
        global_model = SmallCNN(num_channels=in_ch, num_classes=num_classes)

    else:
        raise ValueError("Invalid model name")

    retrained_global_model = deepcopy(global_model)
    print(f"Model: {global_model}")

    # =========================================================================
    #                               阶段 1: Full Training
    # =========================================================================
    
    # 如果处于 'retraining' 或 'unlearning' 阶段，强制跳过训练（变为加载模式）
    if args.execution_stage in ['retraining', 'unlearning']:
        args.skip_training = True
        if args.verbose:
            print(f"[{args.execution_stage}] 模式：自动跳过 Full Training 训练，尝试加载权重...")

    # 原来：
    # train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
    # global_model = fed_train(...)

    # 改为：
    if args.skip_training:
        # 复用已有训练产物
        train_path = os.path.abspath(
            args.full_training_dir if args.full_training_dir
            else os.path.join(weights_path, "full_training")
        )
        if not os.path.isdir(train_path):
            raise RuntimeError(f"[Unlearn-Only] 找不到 full_training 目录：{train_path}")

        # 选择要加载的全局模型权重
        import os, torch
        candidates = [
            os.path.join(train_path, "final_model.pth"),
            os.path.join(train_path, f"iteration_{max([int(d.split('_')[-1]) for d in os.listdir(train_path) if d.startswith('iteration_')], default=-1)}", "global_model.pth")
        ]
        ckpt_path = args.global_ckpt if args.global_ckpt else next((p for p in candidates if os.path.isfile(p)), None)
        if not ckpt_path:
            raise RuntimeError("[Unlearn-Only] 未找到可用的全局模型权重（final_model.pth 或最后一轮的 global_model.pth）")

        state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
        global_model.load_state_dict(state_dict)
        print(f"[Unlearn-Only] 复用训练结果：{ckpt_path}")
    else:
        # 照常训练（注意 fed_train 会清空 weights_path）
        train_path = os.path.abspath(os.path.join(weights_path, "full_training"))
        global_model = fed_train(num_training_iterations=args.num_training_iterations,
                                test_dataloader=test_dataloader,
                                clientwise_dataloaders=clientwise_dataloaders,
                                weights_path=train_path,
                                global_model=global_model,
                                num_local_epochs=args.num_local_epochs,
                                lr=args.lr,
                                optimizer_name=args.optimizer,
                                device=args.device)


    # [修改] 仅在 'all' 或 'full_training' 阶段才执行 Full Training 的测评
    if args.execution_stage in ['all', 'full_training']:
        perf = get_performance(model=global_model, test_dataloader=test_dataloader, num_classes=num_classes,
                            clientwise_dataloader=clientwise_dataloaders, device=args.device)
        summary['performance'] = {}
        summary['performance']['after_training'] = perf
        if args.verbose:
            print(f"Performance after training : {perf}")

            forget_loader = clientwise_dataloaders[forget_client]
            acc = get_accuracy_only(global_model, forget_loader, args.device)
            print(f"[Training模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")
    else:
        # 如果跳过测评，给个空字典防止报错
        summary['performance'] = {}
        if args.verbose: print("[Skip] 跳过 Full Training 测评")

    # === [MIA-INIT] 在任何 evaluate_mia_attack 调用之前，先准备好分割与攻击器 ===
    # 幂等：若后面已有同名对象，这里不会重复构造
    attack_model = locals().get("attack_model", None)
    mia_shadow_nonmem_loader = locals().get("mia_shadow_nonmem_loader", None)
    mia_eval_nonmem_loader   = locals().get("mia_eval_nonmem_loader", None)
    should_train_mia = args.apply_membership_inference

    if should_train_mia:
        # 1) 准备“互斥”的非成员集（shadow/eval）
        if mia_eval_nonmem_loader is None or mia_shadow_nonmem_loader is None:
            from torch.utils.data import random_split, DataLoader as _DL
            _n_test = len(test_dataloader.dataset)
            _n_shadow_nonmem = int(0.8 * _n_test)
            _n_eval_nonmem   = _n_test - _n_shadow_nonmem
            _gen = torch.Generator().manual_seed(args.seed if getattr(args, "seed", None) is not None else 0)
            _shadow_nm_ds, _eval_nm_ds = random_split(
                test_dataloader.dataset, [_n_shadow_nonmem, _n_eval_nonmem], generator=_gen
            )
            mia_shadow_nonmem_loader = _DL(_shadow_nm_ds, batch_size=test_dataloader.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
            mia_eval_nonmem_loader   = _DL(_eval_nm_ds,    batch_size=test_dataloader.batch_size,
                                           shuffle=False, num_workers=args.num_workers)
            # 诊断打印：仅在 --mia_verbose 时输出
            if args.mia_verbose:
                try:
                    _s_idx = getattr(_shadow_nm_ds, "indices", None)
                    _e_idx = getattr(_eval_nm_ds, "indices", None)
                    _overlap = (set(_s_idx) & set(_e_idx)) if (_s_idx is not None and _e_idx is not None) else set()
                    print(f"[MIA-SPLIT] shadow_nonmem={_n_shadow_nonmem} eval_nonmem={_n_eval_nonmem} "
                          f"shadow_id={id(_shadow_nm_ds)} eval_id={id(_eval_nm_ds)} overlap={len(_overlap)}")
                    if _s_idx is not None and _e_idx is not None:
                        print(f"[MIA-SPLIT] shadow_head={list(_s_idx[:5])} ... tail={list(_s_idx[-5:])}")
                        print(f"[MIA-SPLIT] eval__head={list(_e_idx[:5])} ... tail={list(_e_idx[-5:])}")
                except Exception as _e:
                    print(f"[MIA-SPLIT][WARN] split diagnostics failed: {_e}")

        # 2) 训练一次攻击器（基于 full-training shadow 模型）
        if attack_model is None:
            shadow_model = deepcopy(global_model)
            attack_model = train_attack_model(
                shadow_global_model=shadow_model,
                shadow_client_loaders=clientwise_dataloaders,
                shadow_test_loader=mia_shadow_nonmem_loader,
                dataset=args.dataset, device=args.device)



    # ==== 六项指标统一打印（Training 基线）+ MIA：只要 --apply_membership_inference 就默认跑 ====
    mia_training = None
    if args.apply_membership_inference:
        # 在“训练好的完整模型”上执行成员推断（评估集非成员与 shadow 非成员互斥）
        mia_training = evaluate_mia_attack(
            target_model=deepcopy(global_model),
            attack_model=attack_model,
            client_loaders=clientwise_dataloaders,
            test_loader=test_dataloader,
            dataset=args.dataset,
            forget_client_idx=forget_client,
            device=args.device,
            eval_nonmem_loader=mia_eval_nonmem_loader
        )

    # 统一六个指标：测试集准确率、遗忘客户端准确率、遗忘客户端交叉熵、加速比(Training 无)、参数夹角(Training 无)、MIA（三元组）
    if args.execution_stage in ['all', 'full_training']:
        _forget_loader = clientwise_dataloaders[forget_client]
        test_acc_tr    = get_accuracy_only(global_model, test_dataloader, args.device)
        target_acc_tr  = get_accuracy_only(global_model, _forget_loader, args.device)
        retain_acc_tr  = eval_retain_acc(global_model, clientwise_dataloaders, args.forget_clients, args.device)
        target_loss_tr = eval_ce_loss(global_model, _forget_loader, args.device)
        speedup_tr     = None   # 以 retrain 为基线，此处不计
        angle_tr       = None   # 需相对 retrain 的夹角，这里留空
        print_forgetting_metrics(
            method_name="Training",
            test_acc=test_acc_tr,
            retain_acc=retain_acc_tr,
            target_acc=target_acc_tr,
            target_loss=target_loss_tr,
            speedup_x=speedup_tr,
            angle_deg=angle_tr,
            mia_result=mia_training
        )

    # [逻辑中断] 如果只跑 full_training，到此结束
    if args.execution_stage == 'full_training':
        print(">>> [Finish] Full Training stage completed. Exiting.")
        sys.exit(0)

    # 清理 MIA 大对象与 CUDA 缓存，避免后续卡住 (通用清理)
    try:
        import torch, gc
        if isinstance(mia_training, dict):
            for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                mia_training.pop(k, None)
        # [修改] 在 unlearning 阶段跳过强制同步和清理，节省 2-5 秒
        if args.execution_stage != 'unlearning' and torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    # （删除：这里不再做三参自动调参，改到 fair_vue 分支内“按轮增量解析后”执行）



    # -------------------------------------------------------
    # === MIA: 准备“互斥”的非成员数据（避免泄漏）【若前面已构造，这里跳过】===
    from torch.utils.data import random_split, DataLoader as _DL
    if args.apply_membership_inference and (locals().get("mia_eval_nonmem_loader") is None
                                            or locals().get("mia_shadow_nonmem_loader") is None):
        _n_test = len(test_dataloader.dataset)
        _n_shadow_nonmem = int(0.8 * _n_test)
        _n_eval_nonmem   = _n_test - _n_shadow_nonmem
        _gen = torch.Generator().manual_seed(args.seed if args.seed is not None else 0)
        _shadow_nm_ds, _eval_nm_ds = random_split(test_dataloader.dataset, [_n_shadow_nonmem, _n_eval_nonmem], generator=_gen)
        mia_shadow_nonmem_loader = _DL(_shadow_nm_ds, batch_size=test_dataloader.batch_size, shuffle=False,
                                       num_workers=args.num_workers, worker_init_fn=_worker_init_fn)
        mia_eval_nonmem_loader   = _DL(_eval_nm_ds,    batch_size=test_dataloader.batch_size, shuffle=False,
                                       num_workers=args.num_workers, worker_init_fn=_worker_init_fn)

    # —— 自检日志只在 --mia_verbose 时打印 —— 
    if args.mia_verbose:
        try:
            _s_idx = getattr(_shadow_nm_ds, "indices", None)
            _e_idx = getattr(_eval_nm_ds, "indices", None)
            _overlap = (set(_s_idx) & set(_e_idx)) if (_s_idx is not None and _e_idx is not None) else set()
            print(f"[MIA-SPLIT] shadow_nonmem={_n_shadow_nonmem} eval_nonmem={_n_eval_nonmem} "
                  f"shadow_id={id(_shadow_nm_ds)} eval_id={id(_eval_nm_ds)} overlap={len(_overlap)}")
            if _s_idx is not None and _e_idx is not None:
                print(f"[MIA-SPLIT] shadow_head={list(_s_idx[:5])} ... tail={list(_s_idx[-5:])}")
                print(f"[MIA-SPLIT] eval__head={list(_e_idx[:5])} ... tail={list(_e_idx[-5:])}")
        except Exception as _e:
            print(f"[MIA-SPLIT][WARN] split diagnostics failed: {_e}")

    # train mia attack model（使用“不与评估重叠”的非成员子集；若已存在则跳过）
    if args.apply_membership_inference and (locals().get("attack_model") is None):
        shadow_model = deepcopy(global_model)
        attack_model = train_attack_model(
            shadow_global_model=shadow_model,
            shadow_client_loaders=clientwise_dataloaders,
            shadow_test_loader=mia_shadow_nonmem_loader,
            dataset=args.dataset, device=args.device)
    # ---------------------------------------------------------
    # evaluate attack accuracy
    # [修改] 仅在 'all' 或 'full_training' 阶段评估初始模型的后门，避免调参时卡顿
    if args.apply_backdoor and args.execution_stage in ['all', 'full_training']:
        # ------------------------------------------
        # TO-DO: Implement data poisoning and eval from https://arxiv.org/abs/2402.14015 (give credit in code!)

        ''' 
        -- Create data loaders with poison/clean data --
        To be implemented in here (till ca. line 44): https://github.com/drimpossible/corrective-unlearning-bench/blob/main/src/main.py
        FYI kep corrupt size at 3 for the pixel attack patch, manip_set_size=opt.forget_set_size determines how many poisoned samples
        From line 80 onwards keep opt.deletion_size == opt.forget_set_size to have all poison samples known. Unlearning unkown samples is
        a different hard problem beyond the scope of this FL-UL paper
        Helper functions in here: https://github.com/drimpossible/corrective-unlearning-bench/blob/main/src/datasets.py

        -- Eval --
        For evaluation, report the accuracies on the poisoned data with the clean labels 
        (i.e., what they should be, not what the manipulated poisoned sample says it is) and
        the accuracy on the remaining clean data. See figures 2 & 3
        '''
        # ------------------------------------------

        backdoor_results = evaluate_backdoor_attack(model=global_model, backdoor_context=backdoor_context,
                                                    device=args.device)
        summary['backdoor_results'] = {}
        summary['backdoor_results']['global_model_after_training'] = backdoor_results

        backdoor_client = deepcopy(global_model)
        backdoor_client_path = os.path.abspath(os.path.join(
            train_path, f"iteration_{args.num_training_iterations - 1}", f"client_{args.forget_clients[0]}.pth"))
        backdoor_client.load_state_dict(torch.load(backdoor_client_path))

        backdoor_results_client = evaluate_backdoor_attack(model=backdoor_client, backdoor_context=backdoor_context,
                                                           device=args.device)
        summary['backdoor_results']['backdoor_client_after_training'] = backdoor_results_client

        if args.verbose:
            print(
                f"Backdoor results after training : {summary['backdoor_results']}")

    # evaluate poisoning accuracy
    # [修改] 同上，跳过投毒评估
    if args.apply_label_poisoning and args.execution_stage in ['all', 'full_training']:
        poisoning_results = evaluate_poisoning_attack(model=global_model,
                                                      poisoning_context=poisoning_context,
                                                      device=args.device)
        summary['poisoning_results'] = {}
        summary['poisoning_results']['global_model_after_training'] = poisoning_results

        poisoning_client = deepcopy(global_model)
        poisoning_client_path = os.path.abspath(os.path.join(
            train_path, f"iteration_{args.num_training_iterations - 1}", f"client_{args.forget_clients[0]}.pth"))
        poisoning_client.load_state_dict(torch.load(poisoning_client_path))

        poisoning_results_client = evaluate_poisoning_attack(model=poisoning_client,
                                                             poisoning_context=poisoning_context,
                                                             device=args.device)
        summary['poisoning_results']['poisoning_client_after_training'] = poisoning_results_client

        if args.verbose:
            print(
                f"Poisoning results after training : {summary['poisoning_results']}")

    # =========================================================================
    #                               阶段 2: Retraining
    # =========================================================================
    

    retrain_path = os.path.join(weights_path, "retraining")
    # train the model on retain data
    retain_clientwise_dataloaders = {key: value for key, value in clientwise_dataloaders.items()
                                     if key not in args.forget_clients}
    # [修改] 注释掉这个打印，防止 Client 很多时刷屏几千行
    # print(f"Retain Client wise Loaders: {retain_clientwise_dataloaders}")

    # 如果处于 'unlearning' 阶段，我们只需要加载 Retrain 模型算指标，不需要重新跑训练流程
    if args.execution_stage == 'unlearning':
        args.skip_retraining = True
        if args.verbose: print("[unlearning] 模式：跳过 Retraining 训练，将尝试加载基线用于计算 Speedup/Angle")


    # === 计时：重训基线（供 Speedup 对比） ===
    t_retrain_sec = None
    has_retrain_baseline = False
    if not args.skip_retraining:
        # 如果当前是 'full_training' 阶段，这里根本不会执行到（上面已 exit）
        # 所以这里一定是 'all' 或 'retraining'
        _t0 = time.time()
        retrained_global_model = fed_train(num_training_iterations=args.num_training_iterations, test_dataloader=test_dataloader,
                                        clientwise_dataloaders=retain_clientwise_dataloaders,
                                        global_model=retrained_global_model, num_local_epochs=args.num_local_epochs,
                                        device=args.device, weights_path=retrain_path, lr=args.lr, optimizer_name=args.optimizer)
        t_retrain_sec = time.time() - _t0
        has_retrain_baseline = True

        if args.execution_stage in ['all', 'retraining']:
            perf = get_performance(model=retrained_global_model, test_dataloader=test_dataloader,
                                clientwise_dataloader=clientwise_dataloaders,
                                num_classes=num_classes, device=args.device)
            summary['performance']['after_retraining'] = perf
            if args.verbose:
                print(f"Performance after retraining : {perf}")
                print(f"[Timing] Retrain baseline time: {t_retrain_sec:.2f}s" if t_retrain_sec is not None else "[Timing] Retrain baseline time: NA")
            # evaluate attack accuracy on retrained model

            # ---- 专门测忘却客户端的精度 ----
                forget_loader = clientwise_dataloaders[forget_client]
                acc = get_accuracy_only(retrained_global_model, forget_loader, args.device)
                print(f"[Retrain模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")


        # ==== 统一打印（Retrain Baseline）+ MIA：只要开启 MIA 就默认跑 ====
        mia_retrain = None
        if args.apply_membership_inference:
            mia_retrain = evaluate_mia_attack(
                target_model=deepcopy(retrained_global_model),
                attack_model=attack_model,
                client_loaders=clientwise_dataloaders,
                test_loader=test_dataloader,
                dataset=args.dataset,
                forget_client_idx=forget_client,
                device=args.device,
                eval_nonmem_loader=mia_eval_nonmem_loader
            )
        test_acc_rt    = get_accuracy_only(retrained_global_model, test_dataloader, args.device)
        target_acc_rt  = get_accuracy_only(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
        retain_acc_rt  = eval_retain_acc(retrained_global_model, clientwise_dataloaders, args.forget_clients, args.device)
        target_loss_rt = eval_ce_loss(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
        speedup_rt     = 1.0  # retrain 作为基线
        angle_rt       = 0.0
        
        if args.execution_stage in ['all', 'retraining']:
            print_forgetting_metrics("Retrain", test_acc_rt, retain_acc_rt, target_acc_rt, target_loss_rt, speedup_rt, angle_rt, mia_retrain)
        # 清理大对象
        try:
            import torch, gc
            if isinstance(mia_retrain, dict):
                for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                    mia_retrain.pop(k, None)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    else:
        # 跳过重训：若用户提供了 retraining_dir / retrained_ckpt，则直接载入基线权重
        import os, torch
        ckpt_path = None
        if args.retrained_ckpt:
            ckpt_path = os.path.abspath(args.retrained_ckpt)
            if not os.path.isfile(ckpt_path):
                raise RuntimeError(f"[Skip-Retrain] 找不到指定的 --retrained_ckpt：{ckpt_path}")
        elif args.retraining_dir:
            rdir = os.path.abspath(args.retraining_dir)
            if not os.path.isdir(rdir):
                raise RuntimeError(f"[Skip-Retrain] 找不到指定的 --retraining_dir：{rdir}")
            # 先尝试 final_model.pth，其次尝试最后一轮的 global_model.pth
            last_iter = -1
            for name in os.listdir(rdir):
                if name.startswith("iteration_"):
                    try:
                        idx = int(name.split("_")[-1])
                        last_iter = max(last_iter, idx)
                    except Exception:
                        pass
            candidates = [
                os.path.join(rdir, "final_model.pth"),
                os.path.join(rdir, f"iteration_{last_iter}", "global_model.pth") if last_iter >= 0 else None
            ]
            ckpt_path = next((p for p in candidates if p and os.path.isfile(p)), None)
            if not ckpt_path:
                raise RuntimeError(f"[Skip-Retrain] 在 {rdir} 未找到 final_model.pth 或最后一轮 global_model.pth")

        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            retrained_global_model.load_state_dict(state_dict)
            has_retrain_baseline = True
            print(f"[Skip-Retrain] 复用重训练基线：{ckpt_path}")
            
            # [修改] 如果是 'unlearning' 阶段，我们只加载权重不算指标，节省时间
            # 只有 'all' 或 显式 'retraining' (但跳过训练?) 时才测评
            if args.execution_stage in ['all', 'retraining']:
                # 既然有了基线，也一起评测便于对照
                perf = get_performance(model=retrained_global_model, test_dataloader=test_dataloader,
                                    clientwise_dataloader=clientwise_dataloaders,
                                    num_classes=num_classes, device=args.device)
                summary['performance']['after_retraining'] = perf
                if args.verbose:
                    print(f"Performance after (loaded) retraining : {perf}")
                forget_loader = clientwise_dataloaders[forget_client]
                acc = get_accuracy_only(retrained_global_model, forget_loader, args.device)
                print(f"[Retrain(loaded)模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")
            


            # ==== 统一打印（Retrain Baseline，Loaded）+ MIA：只要开启 MIA 就默认跑 ====
            mia_retrain = None
            # 只有当 attack_model 真正被训练了 (not None) 才跑 MIA
            if args.apply_membership_inference and attack_model is not None:
                mia_retrain = evaluate_mia_attack(
                    target_model=deepcopy(retrained_global_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=forget_client,
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            
            if args.execution_stage in ['all', 'retraining']:
                test_acc_rt    = get_accuracy_only(retrained_global_model, test_dataloader, args.device)
                target_acc_rt  = get_accuracy_only(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
                retain_acc_rt  = eval_retain_acc(retrained_global_model, clientwise_dataloaders, args.forget_clients, args.device)
                target_loss_rt = eval_ce_loss(retrained_global_model, clientwise_dataloaders[forget_client], args.device)
                speedup_rt     = None   # 此分支没计时，就打印 NA
                angle_rt       = 0.0
                print_forgetting_metrics("Retrain", test_acc_rt, retain_acc_rt, target_acc_rt, target_loss_rt, speedup_rt, angle_rt, mia_retrain)
            
            try:
                import torch, gc
                if isinstance(mia_retrain, dict):
                    for k in ['mia_attacker_predictions','mia_attacker_probabilities','predictions','probabilities','scores']:
                        mia_retrain.pop(k, None)
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

        else:
            if args.verbose:
                print("[Skip] 跳过重训练基线（--skip_retraining），且未提供 --retraining_dir / --retrained_ckpt") 

    # [逻辑中断] 如果只跑 retraining，到此结束
    if args.execution_stage == 'retraining':
        print(">>> [Finish] Retraining stage completed. Exiting.")
        sys.exit(0)




    # =========================================================================
    #                               阶段 3: Unlearning Baselines
    # =========================================================================

    if args.apply_backdoor and not args.skip_retraining and args.execution_stage in ['all', 'retraining']:
        retrained_backdoor_results = evaluate_backdoor_attack(model=retrained_global_model,
                                                              backdoor_context=backdoor_context, device=args.device)
        summary['backdoor_results']['after_retraining'] = retrained_backdoor_results
        if args.verbose:
            print(
                f"Backdoor results after retraining : {retrained_backdoor_results}")

    if args.apply_label_poisoning and not args.skip_retraining and args.execution_stage in ['all', 'retraining']:
        retrained_poisoning_results = evaluate_poisoning_attack(model=retrained_global_model,
                                                                poisoning_context=poisoning_context,
                                                                device=args.device)
        summary['poisoning_results']['after_retraining'] = retrained_poisoning_results

        if args.verbose:
            print(
                f"Poisoning results after retraining : {retrained_poisoning_results}")

    # Run Baseline methods and check the performance on them
    baselines_methods = args.baselines
    for baseline in baselines_methods:
        if baseline == 'pga':
            _t0 = time.time()
            global_model_pga = deepcopy(global_model)
            # [修复] 显式移动到 device，防止 PGA 内部报错
            global_model_pga = global_model_pga.to(args.device)
           
            _pm_pga = PeakMem(args.device); _pm_pga.__enter__()
            unlearned_pga_model = run_pga(global_model=global_model_pga,
                                          weights_path=train_path,
                                          clientwise_dataloaders=clientwise_dataloaders,
                                          forget_client=args.forget_clients,
                                          model=args.model,
                                          dataset=args.dataset,
                                          num_clients=args.total_num_clients,
                                          num_classes=num_classes,
                                          pretrained=args.pretrained,
                                          num_training_iterations=args.num_training_iterations,
                                          device=args.device,
                                          lr=(args.pga_unlearn_lr if args.pga_unlearn_lr is not None else args.lr),
                                          optimizer_name=args.optimizer,
                                          num_local_epochs=args.num_local_epochs,
                                          num_unlearn_rounds=args.pga_unlearn_rounds,
                                          num_post_training_rounds=1,
                                          alpha=args.pga_alpha)

            perf = get_performance(model=unlearned_pga_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            pga_time_sec = time.time() - _t0
            print(f"[Timing] PGA time: {pga_time_sec:.2f}s")
            summary['performance']['after_pga'] = perf
            if args.verbose:
                print(f"Performance after pga : {perf}")
            # check backdoor on PGA model
            if args.apply_backdoor:
                forget_backdoor_pga = evaluate_backdoor_attack(model=unlearned_pga_model, backdoor_context=backdoor_context,
                                                               device=args.device)
                summary['backdoor_results']['after_pga'] = forget_backdoor_pga

                if args.verbose:
                    print(
                        f"Backdoor results after pga : {forget_backdoor_pga}")
            if args.apply_label_poisoning:
                forget_poisoning_pga = evaluate_poisoning_attack(model=unlearned_pga_model,
                                                                 poisoning_context=poisoning_context,
                                                                 device=args.device)
                summary['poisoning_results']['after_pga'] = forget_poisoning_pga

                if args.verbose:
                    print(
                        f"Poisoning results after pga : {forget_poisoning_pga}")


            # ==== 六项指标统一打印（PGA）====
            test_acc_pga    = get_accuracy_only(unlearned_pga_model, test_dataloader, args.device)
            target_acc_pga  = get_accuracy_only(unlearned_pga_model, clientwise_dataloaders[forget_client], args.device)
            retain_acc_pga  = eval_retain_acc(unlearned_pga_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_pga = eval_ce_loss(unlearned_pga_model, clientwise_dataloaders[forget_client], args.device)
            speedup_pga     = (t_retrain_sec / pga_time_sec) if (t_retrain_sec is not None and pga_time_sec > 0) else None
            angle_pga       = cosine_angle_between_models(unlearned_pga_model, retrained_global_model) if has_retrain_baseline else None
            mia_pga = None
            if args.apply_membership_inference and attack_model is not None:
                mia_pga = evaluate_mia_attack(
                    target_model=deepcopy(unlearned_pga_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )


            # [Save] 保存 PGA 模型供后续可视化
            save_p = os.path.join(weights_path, "pga_model.pth")
            torch.save(unlearned_pga_model.state_dict(), save_p)
            print(f"[PGA] Model saved to: {save_p}")

            print_forgetting_metrics("PGA", test_acc_pga, retain_acc_pga, target_acc_pga, target_loss_pga, speedup_pga, angle_pga, mia_pga)
            _pm_pga.__exit__(None, None, None)
            _print_mem_overhead("PGA", _pm_pga, summary)
        elif baseline == 'fed_eraser':
            _t0 = time.time()
            global_model_federaser = deepcopy(global_model)
            # Debug: 在主流程里先把 FedEraser 关键配置打印出来
            print(
                "[FedEraser] main: starting unlearning with "
                f"forget_clients={args.forget_clients}, "
                f"total_num_clients={args.total_num_clients}, "
                f"num_training_iterations={args.num_training_iterations}, "
                f"lr={args.lr}, optimizer={args.optimizer}, "
                f"fe_strength={args.fe_strength}, "
                f"fe_scale_from={args.fe_scale_from}, "
                f"fe_normalize={args.fe_normalize}, "
                f"fe_max_step_ratio={args.fe_max_step_ratio}, "
                f"fe_apply_regex={args.fe_apply_regex}, "
                f"fe_eps={args.fe_eps}"
            )
            _pm_fe = PeakMem(args.device); _pm_fe.__enter__()
            unlearned_federaser_model = run_fed_eraser(global_model=global_model_federaser,
                                                       weights_path=train_path,
                                                       clientwise_dataloaders=clientwise_dataloaders,
                                                       forget_clients=args.forget_clients,
                                                       num_clients=args.total_num_clients,
                                                       num_rounds=args.num_training_iterations,
                                                       device=args.device,
                                                       lr=args.lr,
                                                       optimizer_name=args.optimizer,
                                                       local_cali_round=1,
                                                       num_unlearn_rounds=1,
                                                       num_post_training_rounds=1,
                                                       # 透传强度参数
                                                       fe_strength=args.fe_strength,
                                                       fe_scale_from=args.fe_scale_from,
                                                       fe_normalize=args.fe_normalize,
                                                       fe_max_step_ratio=args.fe_max_step_ratio,
                                                       fe_apply_regex=args.fe_apply_regex,
                                                       fe_eps=args.fe_eps)
            perf = get_performance(model=unlearned_federaser_model, test_dataloader=test_dataloader,
                                   clientwise_dataloader=clientwise_dataloaders, num_classes=num_classes,
                                   device=args.device)
            federaser_time_sec = time.time() - _t0
            print(f"[Timing] FedEraser time: {federaser_time_sec:.2f}s")
            summary['performance']['after_federaser'] = perf
            if args.verbose:
                print(f"Performance after federaser : {perf}")
            # check backdoor on Federaser model
            if args.apply_backdoor:
                forget_backdoor_federaser = evaluate_backdoor_attack(model=unlearned_federaser_model, backdoor_context=backdoor_context,
                                                                     device=args.device)
                summary['backdoor_results']['after_federaser'] = forget_backdoor_federaser

                if args.verbose:
                    print(
                        f"Backdoor results after federaser : {forget_backdoor_federaser}")
            if args.apply_label_poisoning:
                forget_poisoning_federaser = evaluate_poisoning_attack(model=unlearned_federaser_model,
                                                                       poisoning_context=poisoning_context,
                                                                       device=args.device)
                summary['poisoning_results']['after_federaser'] = forget_poisoning_federaser

                if args.verbose:
                    print(
                        f"Poisoning results after federaser : {forget_poisoning_federaser}")


            # ==== 六项指标统一打印（FedEraser）====
            test_acc_fe    = get_accuracy_only(unlearned_federaser_model, test_dataloader, args.device)
            target_acc_fe  = get_accuracy_only(unlearned_federaser_model, clientwise_dataloaders[forget_client], args.device)
            retain_acc_fe  = eval_retain_acc(unlearned_federaser_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_fe = eval_ce_loss(unlearned_federaser_model, clientwise_dataloaders[forget_client], args.device)
            speedup_fe     = (t_retrain_sec / federaser_time_sec) if (t_retrain_sec is not None and federaser_time_sec > 0) else None
            angle_fe       = cosine_angle_between_models(unlearned_federaser_model, retrained_global_model) if has_retrain_baseline else None
            mia_fe = None
            if args.apply_membership_inference:
                mia_fe = evaluate_mia_attack(
                    target_model=deepcopy(unlearned_federaser_model),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            print_forgetting_metrics("FedEraser", test_acc_fe, retain_acc_fe, target_acc_fe, target_loss_fe, speedup_fe, angle_fe, mia_fe)
            _pm_fe.__exit__(None, None, None)
            _print_mem_overhead("FedEraser", _pm_fe, summary)
        elif baseline == 'fair_vue':
            from FedUnlearner.baselines.fair_vue.runner import run_fair_vue
            fair_model, fair_time_sec = run_fair_vue(
                args=args,
                global_model=global_model,
                train_path=train_path,
                client_endpoints=client_endpoints,
                clientwise_dataloaders=clientwise_dataloaders,
                test_dataloader=test_dataloader,
                retrained_global_model=retrained_global_model,
                has_retrain_baseline=has_retrain_baseline,
                t_retrain_sec=t_retrain_sec,
                attack_model=locals().get("attack_model", None),
                mia_eval_nonmem_loader=locals().get("mia_eval_nonmem_loader", None),
                num_classes=num_classes,
                weights_path=weights_path,
                summary=summary,
                get_performance_fn=get_performance,
                evaluate_mia_attack_fn=evaluate_mia_attack if args.apply_membership_inference else None,
            )

        elif baseline == 'fast_fu':
            # ---- fast-fU（round-wise） ----
            # quick adapter that runs the FastFUServer over the existing full_training outputs
            from FedUnlearner.baselines.fast_fu.fast_fu import run_fast_fu
            print(">>> Running fast-fU (adapter).")
            # 1) 映射：forget_clients -> attackers（fast-fU 使用 attackers 触发擦除）
            if (not hasattr(args, 'attacker')) or (args.attacker is None) or (len(args.attacker) == 0):
                args.attacker = list(sorted(set(args.forget_clients)))

            # 2) 训练日志路径：指向直接包含 iteration_* 的目录
            fv_train_path = os.path.abspath(args.full_training_dir) if args.full_training_dir else os.path.abspath(train_path)
            if os.path.isdir(os.path.join(fv_train_path, 'full_training')):
                fv_train_path = os.path.join(fv_train_path, 'full_training')
            if (not os.path.isdir(fv_train_path)) or (not any(n.startswith('iteration_') for n in os.listdir(fv_train_path))):
                raise FileNotFoundError(
                    f"[fast-fU] iteration_* not found under: {fv_train_path}. "
                    "Please set --full_training_dir to the folder that directly contains iteration_*/client_*.pth."
                )

            # 3) 与 loader 数量对齐，避免扫描不存在的 client id
            try:
                args.total_num_clients = max(int(args.total_num_clients), len(clientwise_dataloaders))
            except Exception:
                args.total_num_clients = len(clientwise_dataloaders)

            if args.verbose:
                print(f">>> fast-fU config: attackers={args.attacker}, path='{fv_train_path}'")
            # call runner — 传入 attack_model 与 eval_nonmem_loader，让 fast-fU 分支内也能跑 MIA（与 PGA 一致）
            _pm_ff = PeakMem(args.device); _pm_ff.__enter__()
            run_fast_fu(args=args,
                        clientwise_dataloaders=clientwise_dataloaders,
                        train_path=fv_train_path,
                        global_model=deepcopy(global_model),
                        test_dataloader=test_dataloader,
                        retrained_global_model=deepcopy(retrained_global_model),
                        attack_model=locals().get("attack_model", None),
                        eval_nonmem_loader=locals().get("mia_eval_nonmem_loader", None),
                        # 把 retrain 基线用时传给 fast-FU 以输出 Speedup
                        retrain_time_sec=locals().get("t_retrain_sec", None))
            print(">>> fast-fU run finished.")
            _pm_ff.__exit__(None, None, None)
            _print_mem_overhead("fast-fU", _pm_ff, summary)

        elif baseline == 'quickdrop':
            # ---- QuickDrop（严格贴近原法的 DC/梯度匹配蒸馏 + 合成集本地更新）----
            from FedUnlearner.baselines import run_quickdrop
            print(">>> Running QuickDrop (baseline).")
            _t0 = time.time()
            _pm_qd = PeakMem(args.device); _pm_qd.__enter__()
            qd_model, qd_info = run_quickdrop(
                args=args,
                global_model=deepcopy(global_model),
                # 只在“保留客户端”上参与（与 retraining 保持一致）
                clientwise_dataloaders=retain_clientwise_dataloaders,
                test_dataloader=test_dataloader,
                num_classes=num_classes,
                device=args.device,
            )
            # 优先使用内部记录的纯遗忘时间（不含合成图像的迭代过程）
            qd_time_sec = qd_info.get("unlearn_time", time.time() - _t0)
            print(f"[Timing] QuickDrop pure unlearn time: {qd_time_sec:.2f}s")

            # ==== 六项指标统一打印（QuickDrop）====
            test_acc_qd    = get_accuracy_only(qd_model, test_dataloader, args.device)
            target_acc_qd  = get_accuracy_only(qd_model, clientwise_dataloaders[forget_client], args.device)
            retain_acc_qd  = eval_retain_acc(qd_model, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_qd = eval_ce_loss(qd_model, clientwise_dataloaders[forget_client], args.device)
            speedup_qd     = (t_retrain_sec / qd_time_sec) if (locals().get('t_retrain_sec', None) is not None and qd_time_sec > 0) else None
            angle_qd       = cosine_angle_between_models(qd_model, retrained_global_model) if locals().get('has_retrain_baseline', False) else None
            mia_qd = None
            if args.apply_membership_inference:
                mia_qd = evaluate_mia_attack(
                    target_model=deepcopy(qd_model),
                    attack_model=locals().get("attack_model", None),
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=locals().get("mia_eval_nonmem_loader", None)
                )
            print_forgetting_metrics("QuickDrop", test_acc_qd, retain_acc_qd, target_acc_qd, target_loss_qd, speedup_qd, angle_qd, mia_qd)
            # 供后续可能的二次评测使用
            unlearned_quickdrop_model = deepcopy(qd_model)
            _pm_qd.__exit__(None, None, None)
            _print_mem_overhead("QuickDrop", _pm_qd, summary)

        elif baseline == 'conda':
            # === Contribution Dampening（原 LEGACY_UNLEARN）作为标准 baseline ===
            _t0 = time.time()
            model_conda = deepcopy(global_model)
            # 注意：该 baseline 期望传入实验根路径（其下含 full_training），保持与旧版一致
            _pm_conda = PeakMem(args.device); _pm_conda.__enter__()

            # 如果命令行指定了 --conda_weights_path，就优先用；否则退回默认的 weights_path
            conda_weights_root = args.conda_weights_path or weights_path

            model_conda = run_conda(                
                global_model=model_conda,
                weights_path=conda_weights_root, 
                forget_clients=args.forget_clients,
                total_num_clients=len(clientwise_dataloaders),
                dampening_constant=args.dampening_constant,
                dampening_upper_bound=args.dampening_upper_bound,
                ratio_cutoff=args.ratio_cutoff,
                dampening_lower_bound=args.conda_lower_bound,
                eps=args.conda_eps,
                device=args.device
            )
            perf = get_performance(
                model=model_conda,
                test_dataloader=test_dataloader,
                clientwise_dataloader=clientwise_dataloaders,
                num_classes=num_classes,
                device=args.device
            )
            conda_time_sec = time.time() - _t0
            print(f"[Timing] CONDA time: {conda_time_sec:.2f}s")
            summary['performance']['after_conda'] = perf
            if args.verbose:
                print(f"Performance after conda : {perf}")

            # 攻击评测保持与其他 baseline 一致
            if args.apply_backdoor:
                forget_backdoor_conda = evaluate_backdoor_attack(
                    model=model_conda, backdoor_context=backdoor_context, device=args.device
                )
                summary['backdoor_results']['after_conda'] = forget_backdoor_conda
                if args.verbose:
                    print(f"Backdoor results after conda : {forget_backdoor_conda}")
            if args.apply_label_poisoning:
                forget_poisoning_conda = evaluate_poisoning_attack(
                    model=model_conda, poisoning_context=poisoning_context, device=args.device
                )
                summary['poisoning_results']['after_conda'] = forget_poisoning_conda
                if args.verbose:
                    print(f"Poisoning results after conda : {forget_poisoning_conda}")

            # 六项统一指标
            test_acc_conda    = get_accuracy_only(model_conda, test_dataloader, args.device)
            target_acc_conda  = get_accuracy_only(model_conda, clientwise_dataloaders[forget_client], args.device)
            retain_acc_conda  = eval_retain_acc(model_conda, clientwise_dataloaders, args.forget_clients, args.device)
            target_loss_conda = eval_ce_loss(model_conda, clientwise_dataloaders[forget_client], args.device)
            speedup_conda     = (t_retrain_sec / conda_time_sec) if (t_retrain_sec is not None and conda_time_sec > 0) else None
            angle_conda       = cosine_angle_between_models(model_conda, retrained_global_model) if has_retrain_baseline else None
            mia_conda = None
            if args.apply_membership_inference:
                mia_conda = evaluate_mia_attack(
                    target_model=deepcopy(model_conda),
                    attack_model=attack_model,
                    client_loaders=clientwise_dataloaders,
                    test_loader=test_dataloader,
                    dataset=args.dataset,
                    forget_client_idx=args.forget_clients[0],
                    device=args.device,
                    eval_nonmem_loader=mia_eval_nonmem_loader
                )
            print_forgetting_metrics("CONDA", test_acc_conda, retain_acc_conda, target_acc_conda, target_loss_conda, speedup_conda, angle_conda, mia_conda)
            _pm_conda.__exit__(None, None, None)
            _print_mem_overhead("CONDA", _pm_conda, summary)
        # ----------------- FedFIM -----------------
    if "fedfim" in args.baselines:
        from FedUnlearner.baselines import run_fedfIM
        fedfim_model = deepcopy(global_model)

        _pm_fim = PeakMem(args.device); _pm_fim.__enter__()
        _result = run_fedfIM(
            model=fedfim_model,
            client_loaders=clientwise_dataloaders,
            forget_clients=args.forget_clients,
            device=args.device,
            num_classes=num_classes,
            fim_max_passes=args.fim_max_passes,
            fim_max_batches=args.fim_max_batches,
            fim_mode=args.fim_mode,
            fim_topk=args.fim_topk,
            dampening_constant=args.fim_gamma,
            ratio_cutoff=args.fim_ratio_cutoff,
            upper_bound=args.fim_upper_bound,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            finetune_weight_decay=args.finetune_wd,
        )
# 兼容：run_fedfIM 可能返回 model 或 (model, F_r, F_f)
        if isinstance(_result, tuple):
            fedfim_model = _result[0]
        else:
            fedfim_model = _result

        # 保底断言，防止类型再出错
        import torch as _torch
        assert isinstance(fedfim_model, _torch.nn.Module), \
            f"run_fedfIM must return an nn.Module as first item, got {type(fedfim_model)}"

        perf = get_performance(model=fedfim_model, test_dataloader=test_dataloader,
                            clientwise_dataloader=clientwise_dataloaders,
                            num_classes=num_classes, device=args.device)
        summary['performance']['after_fedfim'] = perf
        if args.verbose:
            print(f"Performance after FedFIM : {perf}")

        forget_loader = clientwise_dataloaders[forget_client]
        acc = get_accuracy_only(fedfim_model, forget_loader, args.device)
        print(f"[FedFIM模型] 忘却客户端{forget_client}自有数据精度: {acc*100:.2f}%")
        
        mia_fedfim = None
        if args.apply_membership_inference:
            mia_fedfim = evaluate_mia_attack(
                target_model=deepcopy(fedfim_model),
                attack_model=attack_model,
                client_loaders=clientwise_dataloaders,
                test_loader=test_dataloader,
                dataset=args.dataset,
                forget_client_idx=args.forget_clients[0],
                device=args.device,
                eval_nonmem_loader=mia_eval_nonmem_loader
            )
        # 补充打印 FedFIM 的完整指标
        speedup_fim = None # FedFIM 暂未在主流程计时
        retain_acc_fim = eval_retain_acc(fedfim_model, clientwise_dataloaders, args.forget_clients, args.device)
        angle_fim = cosine_angle_between_models(fedfim_model, retrained_global_model) if has_retrain_baseline else None
        print_forgetting_metrics("FedFIM", perf['test_acc'], retain_acc_fim, acc, eval_ce_loss(fedfim_model, forget_loader, args.device), speedup_fim, angle_fim, mia_fedfim)

        _pm_fim.__exit__(None, None, None)
        _print_mem_overhead("FedFIM", _pm_fim, summary)



    # check mia precision and recall on all model
    summary['mia_attack'] = {}


    # Add configurations to the summary
    summary['config'] = vars(args)

    # Create a timestamp for the summary file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Dump the summary into a file with the summary-timestamp name
    with open(os.path.join(weights_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
