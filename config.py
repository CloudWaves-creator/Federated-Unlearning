# -*- coding: utf-8 -*-
"""
命令行参数定义。从 main.py 中提取，保持所有参数完全一致。
"""

import argparse
from run_utils import str2bool

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='FedUnlearner')

    # add arguments
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--exp_path", default="./experiments/", type=str)
    parser.add_argument('--model', type=str, default='allcnn', choices=["allcnn", 'resnet18', 'smallcnn'],
                        help='model name')
    parser.add_argument('--pretrained', type=str2bool,
                        default=False, help='use pretrained model')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=["mnist", "cifar10", "cifar100", "tinyimagenet"],
                        help='dataset name')
    parser.add_argument('--optimizer', type=str, default='adam', choices=["sgd", "adam"],
                        help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_local_epochs', type=int,
                        default=1, help='number of local epochs')

    parser.add_argument('--num_training_iterations', type=int, default=1,
                        help='number of training iterations for global model')
    parser.add_argument('--num_participating_clients', type=int, default=-1, help='number of users participating in trainig, \
                                                                                    -1 if all are required to participate')

    # baselines
    parser.add_argument('--baselines', type=str, nargs="*", default=[], 
        choices=['pga', 'fed_eraser', 'fedfim', 'fair_vue', 'fast_fu', 'quickdrop', 'conda'],
        help='baseline methods for unlearning')

    # ===== PGA 超参（显式控制遗忘强度） =====
    parser.add_argument('--pga_alpha', type=float, default=20000,
        help='核心：PGA: unlearning strength factor (scales both distance threshold and gradient-ascent step size)')
    parser.add_argument('--pga_unlearn_rounds', type=int, default=5,
        help='PGA: number of gradient-ascent epochs on the forget client data')
    parser.add_argument('--pga_unlearn_lr', type=float, default=0.2,
        help='核心：PGA: learning rate used during PGA unlearning (default: use --lr)')


    # ===== FedEraser 可调强度参数 =====
    parser.add_argument('--fe_strength', type=float, default=10,
        help='FedEraser: overall multiplier on geometry step size')
    parser.add_argument('--fe_scale_from', type=str, default='new', choices=['old','new','none'],
        help='FedEraser: scale source; old=||Σ(oldCM-oldGM)||, new=||Σ(newCM-oldCM)||, none=1')
    parser.add_argument('--fe_normalize', type=str2bool, default=True,
        help='FedEraser: divide by direction L2 norm')
    parser.add_argument('--fe_max_step_ratio', type=float, default=3,
        help='FedEraser: clip per-layer step norm to ratio * ||newGM[layer]||')
    parser.add_argument('--fe_apply_regex', type=str, default=None,
        help='FedEraser: only apply to params whose name matches this regex (e.g., "fc|classifier")')
    parser.add_argument('--fe_eps', type=float, default=1e-12,
        help='FedEraser: small epsilon for numeric stability')

    # ---------- fast-fU 超参（与原实现同名语义） ----------
    parser.add_argument('--fast_expected_saving', type=int, default=5,
        help='fast-fU: expected number of saved client updates (m)')
    parser.add_argument('--fast_alpha', type=float, default=1,
        help='核心：fast-fU: alpha coefficient')
    parser.add_argument('--fast_theta', type=float, default=35.475,
        help='优先核心：fast-fU: theta scaling for unlearning term')

    # ---------- QuickDrop 超参（贴近原实现命名/语义） ----------
    parser.add_argument('--qd_scale', type=float, default=0.01,
        help='核心：QuickDrop: 每类合成样本比例（如 0.01 表示每类约 1%）')
    parser.add_argument('--qd_method', type=str, default='dc', choices=['dc'],
        help='QuickDrop: 蒸馏方法（此实现提供 DC/gradient matching 变体）')
    parser.add_argument('--qd_syn_steps', type=int, default=5,
        help='QuickDrop: 蒸馏外循环步数（优化合成图像）')
    parser.add_argument('--qd_lr_img', type=float, default=0.3,
        help='QuickDrop: 合成图像的学习率')
    parser.add_argument('--qd_batch_real', type=int, default=64,
        help='QuickDrop: 真实批大小（用来计算目标梯度）')
    parser.add_argument('--qd_batch_syn', type=int, default=128,
        help='QuickDrop: 合成批大小（用来计算匹配梯度）')
    parser.add_argument('--qd_local_epochs', type=int, default=1,
        help='QuickDrop: 本地训练轮数（默认沿用 num_local_epochs）')
    parser.add_argument('--qd_save_affine', type=str2bool, default=False,
        help='QuickDrop: 是否保存各客户端合成（affine/synthetic）数据张量')
    parser.add_argument('--qd_affine_dir', type=str, default='quickdrop_affine',
        help='QuickDrop: 合成数据保存目录（位于 experiments/exp_name 下）')
    parser.add_argument('--qd_log_interval', type=int, default=25,
        help='QuickDrop: 蒸馏外循环日志步长（每多少步打印一次进度）')
    # [新增] 专属遗忘参数，解耦训练参数
    parser.add_argument('--qd_unlearn_lr', type=float, default=None,
        help='QuickDrop: 遗忘阶段的专用学习率 (默认使用全局 lr)')
    parser.add_argument('--qd_unlearn_wd', type=float, default=None,
        help='QuickDrop: 遗忘阶段的专用权重衰减 (默认使用全局 weight_decay)')

    # 若已存在合成集缓存，则直接加载并跳过蒸馏（默认开启）
    parser.add_argument('--qd_use_affine_cache', type=str2bool, default=True,
        help='QuickDrop: 发现缓存则复用合成集，避免重复蒸馏')



    # FAIR-VUE 参数
    parser.add_argument('--fair_rank_k', type=int, default=16, help='SVD 主成分数')
    parser.add_argument('--fair_tau_mode', type=str, default='median', choices=['median','mean'], help='ρ阈值模式')
    parser.add_argument('--fair_fisher_batches', type=int, default=5, help='Fisher估计的批次数')
    parser.add_argument('--fair_vue_debug', type=str2bool, default=False,
                        help='是否打印 FAIR-VUE 调试信息（True/False）')
    parser.add_argument('--fair_erase_scale', type=float, default=0.25,
                        help='特异分量擦除强度 (0,1]，默认0.25，建议先小后大')
    parser.add_argument('--fair_auto_erase', type=str2bool, default=False,
                        help='自动调参 erase_scale 以拟合重训练（默认开启）')
    parser.add_argument('--fair_auto_tune_all', type=str2bool, default=False,
                        help='联合自动调参 Fisher批次 / rank_k / tau_mode（默认开启）')
    parser.add_argument('--fair_fisher_grid', type=str, default='1,2,5,10',
                        help='Fisher 批次数候选（逗号分隔），用于稳定性搜索')
    parser.add_argument('--fair_fisher_stability', type=float, default=0.98,
                        help='Fisher 对角稳定阈值（与更大批次数的余弦相似度阈值）')
    parser.add_argument('--fair_rank_energy', type=float, default=0.90,
                        help='选取 fair_rank_k 的累计奇异值能量阈值（默认 90%）')
    parser.add_argument('--fair_rank_k_min', type=int, default=4, help='rank_k 下界')
    parser.add_argument('--fair_rank_k_max', type=int, default=64, help='rank_k 上界')
    parser.add_argument('--fair_tau_metric', type=str, default='stdgap', choices=['stdgap','gap'],
                        help='τ 选择的分离度指标：stdgap=(均值差/总体std)，gap=上下组均值差')
    parser.add_argument('--fair_drop_bounds', type=str, default='0.00,0.04',
                        help='期望全局精度下降区间，形如 "min,max"（默认 0~4%）')
    parser.add_argument('--fair_grid_scales', type=str, default='0.5,0.75,1.0,1.25,1.5',
                        help='粗网格倍数（相对 fair_erase_scale）')
    parser.add_argument('--fair_bisect_steps', type=int, default=3,
                        help='命中区间后的二分细化步数')
    parser.add_argument('--skip_retraining', type=str2bool, default=False,
                        help='是否跳过重训练阶段（True/False）')
    # —— 新增：严格控制 FAIR-VUE 的内存/样本规模 ——
    parser.add_argument('--fair_target_rounds', type=int, default=200,
                        help='仅取最近 R 轮目标客户端增量')
    parser.add_argument('--fair_rho_max_samples', type=int, default=128,
                        help='ρ 的其它客户端子采样上限（默认 128）')
    parser.add_argument('--fair_layer_mode', type=str, default='classifier',
                        choices=['classifier', 'deep', 'all'],
                        help='FAIR-VUE: 指定参与子空间构建的层 (classifier: 仅头, deep: 头+深层, all: 全参数)')
    parser.add_argument('--fair_use_delta_cache', type=str2bool, default=True,
                        help='FAIR-VUE: 是否启用逐轮增量缓存（默认开启）')
    parser.add_argument('--fair_fisher_type', type=str, default='diagonal', choices=['diagonal', 'full'],
                        help='Fisher 矩阵类型：diagonal(对角近似,默认), full(全量矩阵,仅限小模型!)')

    # ==== FAIR-VUE 消融实验参数 ====
    parser.add_argument('--fair_ablation', type=str, default='none',
                        choices=['none', 'no_fisher', 'no_repair', 'no_dual'],
                        help='消融实验变体: none(完整), no_fisher(欧氏距离), no_repair(无修复), no_dual(无对偶优化/易OOM)')
    parser.add_argument('--fair_repair_ratio', type=float, default=0.4,
                        help='FAIR-VUE: 能量补偿比例 (修复能量/擦除能量)，默认 0.4')


    # backdoor attack related arguments
    parser.add_argument('--apply_backdoor', type=str2bool, default=False,
                        help='是否启用后门攻击（True/False）')
    parser.add_argument('--backdoor_position', type=str, default='corner', choices=["corner", "center"],
                        help='backdoor position')
    parser.add_argument('--num_backdoor_samples_per_forget_client', type=int, default=10,
                        help='number of backdoor samples per forget client')
    parser.add_argument('--backdoor_label', type=int,
                        default=0, help='backdoor label')

    # membership inference attack related arguments
    parser.add_argument('--apply_membership_inference', type=str2bool, default=False,
                        help='是否启用成员推理攻击（True/False）')
    # 是否打印 MIA 详细调试信息（默认不打印）
    parser.add_argument('--mia_verbose', type=str2bool, default=False,
                            help='Print detailed diagnostics for MIA (default: false)')
    parser.add_argument('--attack_type', type=str, default='blackbox', choices=["blackbox", "whitebox"],
                        help='attack type')

    # label posioning attack related arguments
    parser.add_argument('--apply_label_poisoning', type=str2bool, default=False,
                        help='是否启用标签投毒（True/False）')
    parser.add_argument('--num_label_poison_samples', type=int, default=10,
                        help='number of label poisoning samples')

    # —— CONDA provide indexes of clients which are to be forgotten, allow multiple clients to be forgotten
    parser.add_argument('--forget_clients', type=int, nargs='+',
                        default=[0], help='forget clients')
    parser.add_argument('--total_num_clients', type=int,
                        default=10, help='total number of clients')
    parser.add_argument('--client_data_distribution', type=str, default='dirichlet',
                        choices=["dirichlet", "iid", "exclusive"], help='client data distribution')
    parser.add_argument('--dampening_constant', type=float,
                        default=0.5, help='dampening constant')
    parser.add_argument('--dampening_upper_bound', type=float,
                        default=0.5, help='dampening upper bound')
    parser.add_argument('--ratio_cutoff', type=float,
                        default=0.75, help='ratio cutoff,conda核心')
    # —— CONDA 额外安全阈值
    parser.add_argument('--conda_lower_bound', type=float, default=0,
                        help='CONDA: 乘子下界（避免把权重乘成接近 0）')
    parser.add_argument('--conda_eps', type=float, default=1e-6,
                        help='CONDA: 防 0 除的数值稳定项')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=["cpu", "cuda"], help='device name')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')
    parser.add_argument("--num_workers", type=int, default=0,
                        help="number of workers for data loading")
    parser.add_argument('--conda_weights_path', type=str, default=None,
                        help='CONDA: 手动指定读取权重的目录（实验根目录或 full_training 目录）')

    # create argument parser ...
    parser.add_argument('--skip_training', type=str2bool, default=False,
                        help='是否仅执行遗忘流程（True/False）')
    parser.add_argument('--full_training_dir', type=str, default='',
                        help='已有的 full_training 目录（含 iteration_*/client_*.pth 和 final_model.pth）')
    parser.add_argument('--global_ckpt', type=str, default='',
                        help='可选：显式指定要加载的全局模型权重路径（.pth）')
    parser.add_argument('--retraining_dir', type=str, default='',
                        help='当 --skip_retraining 时可用：已有的 retraining 目录（含 iteration_*/global_model.pth 或 final_model.pth）')
    parser.add_argument('--retrained_ckpt', type=str, default='',
                        help='当 --skip_retraining 时可用：显式指定重训练基线的全局模型 .pth 路径')

    # FedFIM 参数
    parser.add_argument("--fim_max_batches", type=int, default=2)
    parser.add_argument("--fim_max_passes", type=int, default=1)
    parser.add_argument("--fim_mode", type=str, default="prob", choices=["prob", "label"])
    parser.add_argument("--fim_topk", type=int, default=None)
    parser.add_argument("--fim_ratio_cutoff", type=float, default=1.0)
    parser.add_argument("--fim_gamma", type=float, default=0.2)
    parser.add_argument("--fim_upper_bound", type=float, default=1.0)
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--finetune_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_wd", type=float, default=0.0)
    parser.add_argument("--global_weight", type=str, default="")
    parser.add_argument("--output_weight_path", type=str, default="")


    # ==== 执行阶段控制 (新增) ====
    parser.add_argument('--execution_stage', type=str, default='all',
                        choices=['all', 'full_training', 'retraining', 'unlearning'],
                        help='指定运行阶段：all(全流程), full_training(仅训练), retraining(仅重训练), unlearning(仅遗忘)')



    # ==== HEAL / 模型治疗参数 ====
    parser.add_argument('--heal', type=str2bool, default=False,
                        help='是否启用治疗阶段（True/False）')
    parser.add_argument('--heal_alpha', type=float, default=0.05,
                        help='权重插值系数 α，student←(1-α)student+α·teacher，建议 0.02~0.10')
    parser.add_argument('--heal_steps', type=int, default=80,
                        help='治疗步数（迭代批次数）')
    parser.add_argument('--heal_lr', type=float, default=1e-4,
                        help='治疗学习率')
    parser.add_argument('--heal_T', type=float, default=2.0,
                        help='KD 蒸馏温度')
    parser.add_argument('--heal_lambda_kd', type=float, default=0.3,
                        help='KD loss 权重')
    parser.add_argument('--heal_lambda_ortho', type=float, default=1e-3,
                        help='正交惩罚项权重（抑制回流到被遗忘子空间）')
    parser.add_argument('--heal_teacher', type=str, default='post',
                        choices=['pre','post'],
                        help='选择 teacher：pre=遗忘前全局模型，post=遗忘后模型（默认，避免把目标知识拉回）')
    parser.add_argument('--no_heal_grad_proj', action='store_true',
                        help='关闭梯度正交投影（默认开启）')
    parser.add_argument('--heal_shuffle', action='store_true',
                        help='治疗 DataLoader 是否 shuffle（默认 False 更可复现）')
    parser.add_argument('--heal_batch_size', type=int, default=None,
                        help='治疗 batch_size；不填则用训练时 batch_size')

    return parser
