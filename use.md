# 环境配置
## CPU
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
##  GPU
```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 其它依赖
```shell
pip install -r requirements.txt
```

# 适配cpu
```python
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
# if self.training:
    #     values = xops.memory_efficient_attention(qk, mk, mv, p=self.dropout) # (B*P, 64*64, embed_dim=128)
    # else:
    #     values = xops.memory_efficient_attention(qk, mk, mv, p=0) # (B*P, 64*64, embed_dim=128)
```