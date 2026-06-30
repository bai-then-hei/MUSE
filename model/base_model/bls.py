import torch
import torch.nn as nn

class HyBroadFusion(nn.Module):
    def __init__(self, seq_dim, non_seq_dim, num_feat_nodes=128, num_cross_nodes=128, out_dim=64, num_layers=1):
        super().__init__()
        self.num_layers = max(1, int(num_layers))
        
        # 1. 对应全局特征生成：宽泛特征节点映射 (取代压缩 token)
        self.seq_feature_mapper = nn.Linear(seq_dim, num_feat_nodes)
        if non_seq_dim is None or non_seq_dim <= 0:
            self.non_seq_feature_mapper = nn.LazyLinear(num_feat_nodes)
        else:
            self.non_seq_feature_mapper = nn.Linear(non_seq_dim, num_feat_nodes)

        # 2. 可配置层数的宽度融合块
        self.cross_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes)
            for _ in range(self.num_layers)
        ])
        self.seq_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes)
            for _ in range(self.num_layers)
        ])
        self.non_seq_layers = nn.ModuleList([
            nn.Linear(num_feat_nodes, num_cross_nodes)
            for _ in range(self.num_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.Linear(3 * num_cross_nodes, num_feat_nodes)
            for _ in range(self.num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(num_feat_nodes)
            for _ in range(self.num_layers)
        ])
        self.layer_gates = nn.Parameter(torch.full((self.num_layers,), 0.1))
        
        # 3. 对应最终增强与层级输出：全局扁平拼接
        total_width = (num_feat_nodes * 2) + (num_cross_nodes * 3)
        self.readout_layer = nn.Linear(total_width, out_dim)

    def reset_parameters(self):
        for name, module in self.named_children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, seq_input, non_seq_input):
        # Step 1: 映射到宽泛节点映射空间
        Z_seq = torch.relu(self.seq_feature_mapper(seq_input))
        Z_non_seq = torch.relu(self.non_seq_feature_mapper(non_seq_input))
        g = torch.zeros_like(Z_seq)

        # Step 2: 多层宽度融合
        H_cross = None
        H_seq = None
        H_non_seq = None
        for layer_idx in range(self.num_layers):
            seq_ctx = Z_seq + g
            non_seq_ctx = Z_non_seq + g

            # 用非序列特征作为'门/Query'去增强序列特征
            interacted_Z = seq_ctx * non_seq_ctx  # 乘性交互
            H_cross = torch.tanh(self.cross_layers[layer_idx](interacted_Z))

            # 自身增强分支
            H_seq = torch.relu(self.seq_layers[layer_idx](seq_ctx))
            H_non_seq = torch.relu(self.non_seq_layers[layer_idx](non_seq_ctx))

            # 层间状态更新（残差 + 归一化）
            update = torch.relu(self.update_layers[layer_idx](torch.cat([H_cross, H_seq, H_non_seq], dim=-1)))
            gate = torch.sigmoid(self.layer_gates[layer_idx])
            g = self.norm_layers[layer_idx](g + gate * update)

        Z_seq = Z_seq + g
        Z_non_seq = Z_non_seq + g
        
        # Step 3: 扁平拼出超级宽度向量 (Mixer 融合的平替)
        Broad_State = torch.cat([
            Z_seq,      # 纯序列记忆
            Z_non_seq,  # 纯非序列记忆
            H_cross,    # 序列-非序列交叉模式
            H_seq,      # 序列高阶模式
            H_non_seq   # 非序列高阶模式
        ], dim=-1)

        Broad_State = torch.nan_to_num(Broad_State, nan=0.0, posinf=1e4, neginf=-1e4)
        output = self.readout_layer(Broad_State)
        return output
