import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange
from diffusers.utils import logging

logger = logging.get_logger(__name__)

class PromptLearner(nn.Module):
    def __init__(
        self,
        tokenizer,
        text_encoder_instance,
        n_ctx=8,
        n_prompts=4,
        dtype=torch.float32,
        customize_prefix="",
        customize_suffix="",
        reparam_samples=4
    ):
        super().__init__()
        self.dtype = dtype
        self.n_prompts = n_prompts
        self.n_cls = 1
        self.customize_prefix = "" if customize_prefix is None else customize_prefix
        self.customize_suffix = "" if customize_suffix is None else customize_suffix
        self.reparam_samples = reparam_samples
        self.n_ctx = n_ctx

        self.tokenizer = tokenizer
        # 這裡 text_encoder_instance 預期是完整的 CLIPTextModel
        self.text_encoder = CustomTextEncoder(text_encoder_instance.text_model, dtype=self.dtype)
        self.text_encoder.requires_grad_(False)
        self.embedder = self.text_encoder.embeddings
        
        ctx_dim = self.text_encoder.final_layer_norm.weight.shape[0]
        self.ctx_dim = ctx_dim

        logger.info(f"Initializing single-concept PromptLearner: {n_prompts} prompts, length {n_ctx}")
        
        # 初始化學習向量 ctx: (1, n_prompts, n_ctx, ctx_dim)
        ctx_vectors = torch.empty(self.n_cls, n_prompts, n_ctx, ctx_dim, dtype=self.dtype)
        # 使用較小的標準差初始化，有利於在大資料集下收斂
        nn.init.normal_(ctx_vectors, std=0.01)
        self.ctx = nn.Parameter(ctx_vectors)

        # 預先計算並儲存靜態的 Prefix & Suffix Embeddings
        self._prepare_static_embeddings()

        # 準備用於 Pooled Output 的 Token IDs 模板
        prompt_placeholder = " ".join(["X"] * n_ctx)
        template = (self.customize_prefix + " " + prompt_placeholder + " " + self.customize_suffix).strip()
        
        self.tokenized_prompts = self.tokenizer(
            template,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.embedder.position_ids.device)

        self.n_tokens = self.tokenizer.model_max_length
        self.means = None
        self.stds = None
        self.prompt_texts = ["<learned_fingerprint_style>"]

    def _prepare_static_embeddings(self):
        """預計算 Prefix/Suffix Embeddings，含位置編碼，避免重複運算"""
        device = self.embedder.position_ids.device
        max_len = self.tokenizer.model_max_length

        prefix_ids = self.tokenizer(self.customize_prefix, add_special_tokens=False, return_tensors="pt", truncation=True).input_ids.to(device)
        suffix_ids = self.tokenizer(self.customize_suffix, add_special_tokens=False, return_tensors="pt", truncation=True).input_ids.to(device)

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # 組合 ID 序列: [BOS] + [Prefix] + [Placeholders] + [Suffix] + [EOS] + [PADs]
        placeholder_id = eos_id 
        ids = [bos_id] + prefix_ids[0].tolist() + [placeholder_id] * self.n_ctx + suffix_ids[0].tolist() + [eos_id]
        ids += [pad_id] * (max_len - len(ids))

        token_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) 
        
        with torch.no_grad():
            # 這裡 embedder 會自動加上位置編碼
            self.register_buffer("static_embeddings", self.embedder(token_ids).type(self.dtype))

        self.ctx_start = 1 + prefix_ids.shape[1]
        self.ctx_end = self.ctx_start + self.n_ctx

    def concat_custom_v2(self, ctx):
        """將學習到的 ctx 塞入預算好的 Embeddings 模板中"""
        B, P, _, _ = ctx.shape
        # 擴展模板維度
        full_embeddings = repeat(self.static_embeddings, "1 l d -> b p l d", b=B, p=P).clone()
        # 替換 Placeholder 區域
        full_embeddings[:, :, self.ctx_start:self.ctx_end, :] = ctx
        return full_embeddings

    def orthogonal_loss(self, pooled_prompts):
        """計算多組 Prompt 之間的正交性損失，防止特徵坍縮"""
        if pooled_prompts.shape[1] <= 1:
            return torch.tensor(0.0, device=pooled_prompts.device, dtype=self.dtype)
            
        pooled_prompts = torch.nn.functional.normalize(pooled_prompts, dim=-1)
        cos_sim = pooled_prompts @ pooled_prompts.transpose(1, 2) 
        
        diag_mask = torch.eye(cos_sim.shape[1], device=cos_sim.device).bool()
        cos_sim.masked_fill_(diag_mask.unsqueeze(0), 0)
        
        # 分母防止除以零
        denom = cos_sim.shape[1] * (cos_sim.shape[1] - 1)
        loss_per_batch = (cos_sim ** 2).sum(dim=(1, 2)) / denom
        return loss_per_batch.mean()

    def reparameterize(self, prompts, n_samples=1):
        """執行 Reparameterization Trick 以獲得連續的潛在空間採樣"""
        if prompts.shape[1] <= 1: # 如果只有一個 prompt，直接擴展維度回傳
            return repeat(prompts, "b 1 l d -> b n l d", n=n_samples)
            
        # 計算跨 Prompt 維度 (dim=1) 的均值與標準差
        mu = torch.mean(prompts, dim=1) 
        # 使用 unbiased=False 確保小樣本下不會產生 NaN
        std = torch.std(prompts, dim=1, unbiased=False) 
        
        eps = torch.randn_like(repeat(std, "b l d -> b n l d", n=n_samples)) 
        return mu.unsqueeze(1) + eps * std.unsqueeze(1)

    def forward(self, cls_idx=None, imgs=None):
        batch_size = cls_idx.shape[0] if cls_idx is not None else 1
        # 取得學習中的 ctx 並擴展到 batch
        ctx = self.ctx[0].unsqueeze(0).expand(batch_size, -1, -1, -1) 
        
        prompts = self.concat_custom_v2(ctx)
        tokenized_prompts = self.tokenized_prompts.expand(batch_size * self.n_prompts, -1)
        
        # 經過 Text Encoder 得到 Hidden States (L, D) 與 Pooled Output (D)
        prompts_hidden_state, pooled_prompts = self.text_encoder(
            prompts,
            pooled=True,
            tokenized_prompts=tokenized_prompts
        )
        
        ortho_loss = self.orthogonal_loss(pooled_prompts)
        # 這裡會輸出 (Batch, reparam_samples, Max_Len, Dim)
        prompts_hidden_state = self.reparameterize(prompts_hidden_state, n_samples=self.reparam_samples) 
        return prompts_hidden_state, ortho_loss

    def fit(self, prefix=None, suffix=None):
        """訓練結束後，計算最終的統計分佈供推論使用"""
        self.means = np.empty((self.n_cls, self.n_tokens, self.ctx_dim)) 
        self.stds = np.empty(self.means.shape) 
        
        cls_ctx = self.ctx[0].unsqueeze(0) # (1, P, n_ctx, D)
        cls_prompts = self.concat_custom_v2(cls_ctx)
        # 取得不含隨機採樣的原始 Embeddings
        cls_prompts = self.text_encoder(cls_prompts)[0] # (1, P, L, D)
        cls_prompts = cls_prompts.squeeze(0) # (P, L, D)
        
        # 儲存平均值
        self.means[0] = cls_prompts.mean(dim=0).detach().float().cpu().numpy()
        
        # 嚴謹處理標準差，防止 P=1 時的 NaN
        if cls_prompts.shape[0] <= 1:
            self.stds[0] = np.zeros_like(self.means[0])
        else:
            self.stds[0] = cls_prompts.std(dim=0, unbiased=False).detach().float().cpu().numpy()


class CustomTextEncoder(nn.Module):
    def __init__(self, text_transformer_model, dtype=torch.float32):
        super().__init__()
        self.encoder = text_transformer_model.encoder
        self.final_layer_norm = text_transformer_model.final_layer_norm
        self.embeddings = text_transformer_model.embeddings
        self.dtype = dtype

    def forward(self, prompts, pooled=False, tokenized_prompts=None):
        # 注意：prompts 傳入時應已包含位置編碼 (來自 PromptLearner 的 static_embeddings)
        
        b, p, l, d = prompts.shape
        prompts = rearrange(prompts, "b p l d -> (b p) l d")
            
        # 建立 Causal Attention Mask (CLIP 預設為下三角矩陣)
        attention_mask = torch.empty(prompts.shape[0], l, l, dtype=prompts.dtype, device=prompts.device)
        attention_mask.fill_(torch.tensor(torch.finfo(prompts.dtype).min, device=prompts.device))
        attention_mask.triu_(1) 
        attention_mask = attention_mask.unsqueeze(1) 
        
        # 通過 Transformer Encoder 層
        outputs = self.encoder(inputs_embeds=prompts, causal_attention_mask=attention_mask) 
        last_hidden_state = self.final_layer_norm(outputs[0]).type(self.dtype)

        pooled_output = None
        if pooled:
            # 根據 EOS Token 的位置提取 Pooled Output (SD 1.5 慣例)
            # tokenized_prompts 應為 ( (B*P), Max_Len )
            eos_indices = tokenized_prompts.argmax(dim=-1)
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), eos_indices] 
            pooled_output = rearrange(pooled_output, "(b p) d -> b p d", p=p)

        last_hidden_state = rearrange(last_hidden_state, "(b p) l d -> b p l d", p=p)

        if pooled:
            return last_hidden_state, pooled_output
        return last_hidden_state