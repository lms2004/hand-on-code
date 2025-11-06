# 从 LlamaAttention 修改而来，适配 DeepseekV2 模型的注意力模块，简单版本不带 kv cache
class DeepseekV2MLA(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        # MHA 初始化相关
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.v_head_dim = config.v_head_dim

        self.o_proj = nn.Linear(
            self.v_head_dim * self.num_heads, 
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.attention_dropout = config.attention_dropout
        self.training = False
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        # MLA 相关 part1: 压缩
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank)
        self.q_down_rmsnorm = DeepseekV2RMSNorm(self.q_lora_rank)
        
        self.kv_down_proj = nn.Linear(
            self.hidden_size, 
            self.kv_lora_rank + config.qk_rope_head_dim
        )
        self.kv_down_rmsnorm = DeepseekV2RMSNorm(self.kv_lora_rank)
        
        # MLA 相关 part2: 解压缩. # W^{WQ} 和 W^{QR} 权重是合并再一起的。
        self.qk_head_dim = self.qk_nope_head_dim  + self.qk_rope_head_dim
        self.q_up_proj = nn.Linear(
            self.q_lora_rank, 
            self.num_heads * self.qk_head_dim,
            bias=False,
        )
        
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank, 
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        
        # MLA 相关 part3: 切片 q k 张量，以及 rope 旋转位置编码
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        ) 

    def forward(self, hidden_states, position_ids, casual_mask=None):
        batch_size, q_len, hidden_size = hidden_states.shape

        # 1，q 压缩和解压缩，以及 split to q_nope, q_rope
        q = self.q_up_proj(
            self.q_down_rmsnorm(self.q_down_proj(hidden_states))
        )

        q = q.view(batch_size, q_len, self.num_heads, self.qk_head_dim).transpose(1,2)
        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim = -1,
        )

        # 2, kv 压缩和解压缩
        kv_down = self.kv_down_proj(hidden_states)
        
        # compressed_kv 压缩后的 kv 张量
        compressed_kv, k_rope = torch.split(
            kv_down,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim = -1,
        )
        # num_heads = 1 后续广播其它 heads 上
        k_rope = k_rope.view(batch_size, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        # 对 compressed_kv 解压缩
        kv = (
            self.kv_up_proj(self.kv_down_rmsnorm(compressed_kv))
            .view(batch_size, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim = -1,
        )

        # 3, 计算 cos 和 sin，并应用 rope 旋转位置编码
        kv_seq_len = value_states.shape[-2] # shape (b, nums_head, seq_len, v_head_dim)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids)

        # 4, 执行 self-attention 计算
        query_states = torch.concat([q_nope, q_rope], dim=-1)
        key_states = torch.concat(
            [k_nope, k_rope.expand(-1, self.num_heads, -1, -1)], 
            dim=-1
        )
        # qk^t
        scores = (
            torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.qk_head_dim)
        )

        if casual_mask is not None:
            scores = scores.masked_fill(casual_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        ) # attn_weights shape: [bs, num_heads, seq_len, seq_len]
        
        attn_output = torch.matmul(attn_weights, value_states) # shape: [bs, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.num_heads * self.v_head_dim)

        # 5, MLA 输出映射
        output = self.o_proj(attn_output)

        return output, attn_weights