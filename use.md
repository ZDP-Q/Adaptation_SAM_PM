# 环境配置
## CPU
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install lightning
```
##  GPU
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xformers lightning
```

## 其它依赖
```shell
pip install -r requirements.txt
```

# 适配cpu
```python
# 取消数据在内存中固定
{
  "pin_memory": False,
}
```
# 不使用xformers
```python
# 在modules.py中注释
# from xformers import ops
# if self.training:
    #     values = xops.memory_efficient_attention(q, k, v, p=self.dropout) # (B, 64*64, num_heads, self.head_dim*num_obj=3)
    # else:
    #     values = xops.memory_efficient_attention(q, k, v, p=0) # (B, 64*64, num_heads, self.head_dim*num_obj=3)
# 改为
q = q.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.key_head_dim)
k = k.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len * num_frames, self.key_head_dim)
v = v.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len * num_frames, self.value_head_dim * num_objects)
# if self.training:
    #     values = xops.memory_efficient_attention(qk, mk, mv, p=self.dropout) # (B*P, 64*64, embed_dim=128)
    # else:
    #     values = xops.memory_efficient_attention(qk, mk, mv, p=0) # (B*P, 64*64, embed_dim=128)
# 改为
values = F.scaled_dot_product_attention(qk, mk, mv, dropout_p=self.dropout if self.training else 0)
```