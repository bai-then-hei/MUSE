import sys
import os
import logging

import torch
import torch.distributed as dist

from model.base_model.simtier import cosine_simtier_list
from model.base_model.layers import multi_head_att, multi_head_att_v2, fc_repeats
from model.base_model.bls import HyBroadFusion

from utils.utils import clip_prop, write_info_to_file

class MUSE_DIN(torch.nn.Module):
    def __init__(self, args=dict(), D=32, RT_STEPS=50, UNI_STEPS=50):
        super().__init__()
        self.args = args
        self.D = D
        self.RT_STEPS = RT_STEPS
        self.UNI_STEPS = UNI_STEPS
        
        # simtier
        self.all_seq_image_res = cosine_simtier_list(
            steps=[RT_STEPS, UNI_STEPS],
            n_dim=128,
            scope_list=["rt", "uni_v2"],
            eps_list=[0.1, 0.1],
            dim_list=[1, 1],
        )

        self.realtime_att = multi_head_att(
            2 * self.D, 2 * self.D, [2 * self.D], [2 * self.D]
        )
        self.attn_score_cross = True # whether to use SA-TA
        self.uni_att_v2 = multi_head_att_v2(
            2 * self.D, 2 * self.D, [2 * self.D], [2 * self.D], attn_score_cross=self.attn_score_cross
        )

        # self.uni_att_item = multi_head_att_v2(
        #     self.D, self.D, [self.D], [self.D], attn_score_cross=self.attn_score_cross
        # )
        # self.uni_att_cate = multi_head_att_v2(
        #     self.D, self.D, [self.D], [self.D], attn_score_cross=self.attn_score_cross
        # )
        
        self.bls_out_dim = 64

        # 原始 MUSE 的基础拼接维度（不包含 BLS）
        # base_din_dim = 15 * D + 3 * (D // 2) + 44
        base_din_dim = 15 * self.D + 3 * (self.D // 2) + 44
        
        # 当前数据流里 ad + user 的实际拼接宽度为 102。
        # 即使远端暂时没有同步到 LazyLinear 版本，这里也不能传 0，
        # 否则会直接初始化出 0x128 的线性层权重。
        self.bls_non_seq_dim = int(self.args.get("bls_non_seq_dim", 102))
        self.bls_num_layers = int(self.args.get("bls_num_layers", 3))

        self.bls_fusion = HyBroadFusion(
            seq_dim=8 * self.D,   # 8 * 32 = 256
            non_seq_dim=self.bls_non_seq_dim,
            num_feat_nodes=128, 
            num_cross_nodes=128, 
            out_dim=self.bls_out_dim,
            num_layers=self.bls_num_layers
        )
        self.bls_gate = torch.nn.Parameter(torch.tensor(0.1))

        # 最终 MLP 输入维度 = 原始 base_din_dim (宽度侧特征已剥离)
        din_dim = base_din_dim
        self.fc_tower = fc_repeats(din_dim, shape=[256, 128, 64, 2], acts=['dice', 'dice', 'dice', 'dice', 'dice', None])

        # Side wide branch: consumes BLS output only for ablation.
        self.wide_layer = torch.nn.Linear(self.bls_out_dim, 2)
        torch.nn.init.zeros_(self.wide_layer.weight)
        torch.nn.init.zeros_(self.wide_layer.bias)
        
        self.use_aux_loss = self.args["use_aux_loss"]
        if self.use_aux_loss:
            self.fc_tower_aux = fc_repeats(self.D * 4, shape=[200, 80, 2], acts=['dice', 'dice', None])

        self.use_kl_loss = self.args.get("use_kl_loss", True)
        self.kl_loss_weight = float(self.args.get("kl_loss_weight", 0.05))
        self.kl_temperature = max(float(self.args.get("kl_temperature", 1.0)), 1e-6)
        self.kl_eps = float(self.args.get("kl_eps", 1e-8))

        self.reset_parameters()

    def reset_parameters(self): 
        for name, module in self.named_children(): 
            module.reset_parameters()
        torch.nn.init.zeros_(self.wide_layer.weight)
        torch.nn.init.zeros_(self.wide_layer.bias)

    def forward(
        self,
        user_embs,
        ad_embs,
        uni_seq_embs,
        short_seq_fn,
        label
    ):
        
        item_content = torch.reshape(ad_embs[-1], [-1, 1, 128])
        ad_embs = ad_embs[:-1]

        eval_flag = (uni_seq_embs[0].requires_grad == False)

        all_seq_image_res = self.all_seq_image_res(
            item_content,
            [
                short_seq_fn[-1],
                uni_seq_embs[-1],
            ],
            [None, None],
        )
        # print(f"avg image cosine sim {all_seq_image_res[0][1].mean().item()}, avg-max image cosine sim {(all_seq_image_res[0][1].mean(dim=1)).max(dim=0).values.item()}, all-max image cosine sim {torch.max(all_seq_image_res[0][1]).item()}", file=sys.stderr)
        # write_info_to_file(f"avg image cosine sim {all_seq_image_res[0][1].mean().item()}, avg-max image cosine sim {(all_seq_image_res[0][1].mean(dim=1)).max(dim=0).values.item()}, all-max image cosine sim {torch.max(all_seq_image_res[0][1]).item()}", file_path=f"{self.args['exp_name']}_info.txt")
        
        # ablate content
        # all_seq_image_res[0][1] = torch.zeros_like(all_seq_image_res[0][1]).detach()
        # all_seq_image_res[1][1] = torch.zeros_like(all_seq_image_res[1][1]).detach()

        rt_att = torch.concat(
            [
                torch.reshape(short_seq_fn[k], [-1, self.RT_STEPS, self.D])
                for k in range(2)
            ],
            dim=2,
        )
        uni_seq_att_v2 = torch.concat(
            [
                torch.reshape(uni_seq_embs[k], [-1, self.UNI_STEPS, self.D]) 
                for k in range(2)
            ], 
            dim=2
        )

        # uni_seq_att_v2 = torch.concat(
        #     [
        #         torch.nn.functional.normalize(torch.reshape(uni_seq_embs[0], [-1, self.UNI_STEPS, self.D]), dim=-1),
        #         torch.reshape(uni_seq_embs[1], [-1, self.UNI_STEPS, self.D])
        #     ], 
        #     dim=2
        # )

        # uni_seq_att_v2 = torch.concat(
        #     [
        #         torch.reshape(torch.zeros_like(uni_seq_embs[0]).detach(), [-1, self.UNI_STEPS, self.D]),
        #         torch.reshape(uni_seq_embs[1], [-1, self.UNI_STEPS, self.D]),
        #     ], 
        #     dim=2
        # )

        att_ad_2 = torch.reshape(
            torch.concat(ad_embs[:2], dim=1),
            [-1, 1, 2 * self.D],
        )

        rt_att_out = self.realtime_att(att_ad_2, rt_att)

        # use DIN for ESU
        uni_seq_att_out_v2 = self.uni_att_v2(att_ad_2, uni_seq_att_v2, mm_cosine=[all_seq_image_res[0][1]])

        # KL regularization: force ID attention distribution to align with image-similarity distribution
        loss_kl = torch.zeros((), dtype=label.dtype, device=label.device)
        if self.use_kl_loss and self.args["method"] not in ["din"]:
            id_attn_dist = self.uni_att_v2.calc_attn_score(att_ad_2, uni_seq_att_v2)
            id_attn_dist = id_attn_dist / (id_attn_dist.sum(dim=1, keepdim=True) + self.kl_eps)

            image_attn_dist = torch.nn.functional.softmax(
                all_seq_image_res[0][1] / self.kl_temperature,
                dim=1,
            ).detach()
            loss_kl = torch.nn.functional.kl_div(
                torch.log(id_attn_dist + self.kl_eps),
                image_attn_dist,
                reduction="batchmean",
            )

        # uni_seq_att_out_v2 = [
        #     self.uni_att_item(att_ad_2[..., :16], uni_seq_att_v2[..., :16], mm_cosine=[all_seq_image_res[0][1]]),
        #     self.uni_att_cate(att_ad_2[..., 16:], uni_seq_att_v2[..., 16:], mm_cosine=[all_seq_image_res[0][1]])
        # ]
        # uni_seq_att_out_v2 = torch.concat(uni_seq_att_out_v2, dim=1)

        ad, user = torch.concat(ad_embs, dim=1), torch.concat(user_embs, dim=1)

        # 保持与初始化传入维度一致
        # rt_att_out: [B, 1, 2D]
        # uni_seq_att_out_v2: [B, 1, 2D]
        uni_seq_att_v2 = uni_seq_att_v2.mean(dim=1)  # [B, 2D]
        rt_att = rt_att.mean(dim=1)                  # [B, 2D]

        # squeeze 1 维，使其变为 [B, 2D]
        rt_att_out = rt_att_out.squeeze(1)
        uni_seq_att_out_v2 = uni_seq_att_out_v2.squeeze(1)
        
        # if eval_flag:
        #     norm_list = [
        #         torch.norm(uni_seq_att_v2, dim=-1).mean().item(),
        #         torch.norm(uni_seq_att_out_v2, dim=-1).mean().item(),
        #         torch.norm(rt_att, dim=-1).mean().item(),
        #         torch.norm(rt_att_out, dim=-1).mean().item(),
        #     ]
        #     write_info_to_file(f"{norm_list}", file_path=f"{self.args['exp_name']}_info.txt")

        # TODO: add branches to ablate seq features
        if self.args["method"] in ["din"]:
            # eclude long seq features
            uni_seq_att_v2 = torch.zeros_like(uni_seq_att_v2).detach()
            uni_seq_att_out_v2 = torch.zeros_like(uni_seq_att_out_v2).detach()
            all_seq_image_res[1][1] = torch.zeros_like(all_seq_image_res[1][1]).detach()

        # ablate rt_seq
        # rt_att = torch.zeros_like(rt_att).detach()
        # rt_att_out = torch.zeros_like(rt_att_out).detach()
        # uni_seq_att_v2 = torch.zeros_like(uni_seq_att_v2).detach()
        # uni_seq_att_out_v2 = torch.zeros_like(uni_seq_att_out_v2).detach()
        # all_seq_image_res[1][0] = torch.zeros_like(all_seq_image_res[1][0]).detach()

        # ad = torch.zeros_like(ad).detach()
        # user = torch.zeros_like(user).detach()

        # 准备 BLS 的输入特征
        # rt_att_out (2*D), uni_seq_att_out_v2 (2*D), uni_seq_att_v2 (2*D), rt_att (2*D) => 8*D
        seq_features_for_bls = torch.concat(
            [rt_att_out, uni_seq_att_out_v2, uni_seq_att_v2, rt_att], 
            dim=1
        )
        non_seq_features_for_bls = torch.concat(
            [ad, user], 
            dim=1
        )

        if seq_features_for_bls.shape[1] != self.bls_fusion.seq_feature_mapper.in_features:
            raise RuntimeError(
                f"BLS seq dim mismatch: got {seq_features_for_bls.shape[1]}, "
                f"expect {self.bls_fusion.seq_feature_mapper.in_features}"
            )
        # 通过 BLS 计算交叉增强特征
        bls_cross_feature = self.bls_gate * self.bls_fusion(seq_features_for_bls, non_seq_features_for_bls)
        bls_cross_feature = torch.nan_to_num(bls_cross_feature, nan=0.0, posinf=1e4, neginf=-1e4)

        # Feed width-network output into the new Wide branch input.
        wide_branch_input = bls_cross_feature

        # 加入到原有 din 拼接中 (剥离 BLS 特征，让其只走宽分支)
        din = torch.concat(
            [rt_att_out, uni_seq_att_out_v2, uni_seq_att_v2, rt_att, ad, user] + all_seq_image_res[1],
            dim=1
        )
        expected_fc_in = self.fc_tower.linears[0].in_features
        if din.shape[1] != expected_fc_in:
            raise RuntimeError(
                f"fc_tower input dim mismatch: got {din.shape[1]}, expect {expected_fc_in}"
            )
        # print(f"ad shape: {ad.shape}, user shape: {user.shape}, all shape: {din.shape}")

        deep_logits = self.fc_tower(din)
        if wide_branch_input.shape[1] != self.wide_layer.in_features:
            raise RuntimeError(
                f"wide_layer input dim mismatch: got {wide_branch_input.shape[1]}, expect {self.wide_layer.in_features}"
            )
        wide_logits = self.wide_layer(wide_branch_input)
        item_fc6 = deep_logits + wide_logits
        prop = clip_prop(
            torch.nn.functional.softmax(item_fc6, dim=-1) + 0.0000001
        )
        loss_ce = -(torch.log(prop) * label).sum(dim=1, keepdim=True)

        if self.use_aux_loss:
            # add aux loss to help training of GSU/ESU
            din_aux = torch.cat([uni_seq_att_out_v2, uni_seq_att_v2], dim=1)
            fc_out_aux = self.fc_tower_aux(din_aux)
            prop_aux = clip_prop(
                torch.nn.functional.softmax(fc_out_aux, dim=-1) + 0.0000001
            )
            loss_aux = -(torch.log(prop_aux) * label).sum(dim=1, keepdim=True)
            total_loss = loss_ce.mean(dim=0) + loss_aux.mean(dim=0) + self.kl_loss_weight * loss_kl
        else:
            total_loss = loss_ce.mean(dim=0) + self.kl_loss_weight * loss_kl

        return total_loss, prop
    
    def save_ckpt(self, ckpt_path: str, rank=None):
        if rank is not None and rank != 0:
            return

        state_dict = {
            'all_seq_image_res': self.all_seq_image_res.state_dict(),
            'realtime_att': self.realtime_att.state_dict(),
            'uni_att_v2': self.uni_att_v2.state_dict(),
            'fc_tower': self.fc_tower.state_dict(),
            'wide_layer': self.wide_layer.state_dict(),
            'bls_fusion': self.bls_fusion.state_dict(),
            'bls_gate': self.bls_gate.detach().cpu()
        }

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(state_dict, ckpt_path)
        logging.info(f"Checkpoint saved to {ckpt_path}")
    
    def load_ckpt(self, ckpt_path: str, map_location=None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        device_id = torch.cuda.current_device()

        if map_location is None:
            map_location = lambda storage, loc: storage.cuda(device_id)

        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.all_seq_image_res.load_state_dict(ckpt['all_seq_image_res'])
        self.realtime_att.load_state_dict(ckpt['realtime_att'])
        self.uni_att_v2.load_state_dict(ckpt['uni_att_v2'])
        self.fc_tower.load_state_dict(ckpt['fc_tower'])
        if 'wide_layer' in ckpt:
            wide_layer_state = ckpt['wide_layer']
            if (
                wide_layer_state.get('weight', None) is not None
                and wide_layer_state['weight'].shape == self.wide_layer.weight.shape
                and wide_layer_state.get('bias', None) is not None
                and wide_layer_state['bias'].shape == self.wide_layer.bias.shape
            ):
                self.wide_layer.load_state_dict(wide_layer_state)
            else:
                logging.warning(
                    f"Skip loading wide_layer due to shape mismatch. ckpt weight {wide_layer_state.get('weight', None).shape if wide_layer_state.get('weight', None) is not None else 'None'} vs model weight {self.wide_layer.weight.shape}"
                )
        if 'bls_fusion' in ckpt:
            self.bls_fusion.load_state_dict(ckpt['bls_fusion'], strict=False)
        if 'bls_gate' in ckpt:
            self.bls_gate.data.copy_(ckpt['bls_gate'].to(self.bls_gate.device))

        logging.info(f"[Rank {device_id}] Checkpoint loaded from {ckpt_path}")
        return self