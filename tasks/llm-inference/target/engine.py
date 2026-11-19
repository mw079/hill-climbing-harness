"""Plain Qwen3.5 (dense, text-only) greedy inference in MLX core ops. Starting point.

Written for clarity, not speed: no custom Metal kernels, no mx.compile, a Python loop
over prompt tokens in the linear-attention layers, and a concatenating KV cache.

Interface (fixed by the driver):
    Engine(model_dir)                       load bf16 weights from model_dir
    Engine.generate(prompt_ids, n) -> list  n greedy token ids; state is not kept
                                            between calls
"""
import json
from pathlib import Path

import mlx.core as mx


class Engine:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        c = json.loads((model_dir / "config.json").read_text())
        c = c.get("text_config", c)
        self.hidden = c["hidden_size"]
        self.n_layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.n_heads, self.n_kv, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        rope = c.get("rope_parameters", {})
        self.rope_dims = int(self.head_dim * rope.get("partial_rotary_factor", 0.25))
        self.rope_base = float(rope.get("rope_theta", 10000.0))
        self.hk, self.hv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        self.dk, self.dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        self.conv_k = c["linear_conv_kernel_dim"]
        self.interval = c["full_attention_interval"]

        weights = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            for k, v in mx.load(str(f)).items():
                if k.startswith("model.language_model."):
                    weights[k[len("model.language_model."):]] = v
        # Qwen3.5 stores these RMSNorm scales as offsets from 1.
        for k in list(weights):
            if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                           "q_norm.weight", "k_norm.weight")) or k == "norm.weight":
                weights[k] = weights[k] + 1.0
            if k.endswith("conv1d.weight"):                  # (C, 1, K) -> (C, K)
                weights[k] = weights[k].reshape(weights[k].shape[0], -1)
        self.w = weights
        mx.eval(self.w)

    # ------------------------------------------------------------------ layers

    def linear(self, x, name):
        return x @ self.w[name + ".weight"].T

    def rms_norm(self, x, name):
        return mx.fast.rms_norm(x, self.w[name + ".weight"], self.eps)

    def mlp(self, x, p):
        g = self.linear(x, p + ".gate_proj")
        u = self.linear(x, p + ".up_proj")
        return self.linear(g * mx.sigmoid(g) * u, p + ".down_proj")

    def attention(self, x, p, cache, offset):
        B, L, _ = x.shape
        q = self.linear(x, p + ".q_proj").reshape(B, L, self.n_heads, 2 * self.head_dim)
        q, gate = q[..., :self.head_dim], q[..., self.head_dim:].reshape(B, L, -1)
        k = self.linear(x, p + ".k_proj").reshape(B, L, self.n_kv, self.head_dim)
        v = self.linear(x, p + ".v_proj").reshape(B, L, self.n_kv, self.head_dim)
        q = self.rms_norm(q, p + ".q_norm").transpose(0, 2, 1, 3)
        k = self.rms_norm(k, p + ".k_norm").transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.rope_dims, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
        if cache["k"] is not None:
            k = mx.concatenate([cache["k"], k], axis=2)
            v = mx.concatenate([cache["v"], v], axis=2)
        cache["k"], cache["v"] = k, v
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5,
                                                 mask="causal" if L > 1 else None)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.linear(o * mx.sigmoid(gate), p + ".o_proj")

    def gated_delta_net(self, x, p, cache):
        B, L, _ = x.shape
        key_dim, value_dim = self.hk * self.dk, self.hv * self.dv
        qkv = self.linear(x, p + ".in_proj_qkv")
        z = self.linear(x, p + ".in_proj_z").reshape(B, L, self.hv, self.dv)
        b = self.linear(x, p + ".in_proj_b")
        a = self.linear(x, p + ".in_proj_a")

        # Causal depthwise convolution over the last conv_k inputs.
        if cache["conv"] is None:
            cache["conv"] = mx.zeros((B, self.conv_k - 1, qkv.shape[-1]), dtype=qkv.dtype)
        window = mx.concatenate([cache["conv"], qkv], axis=1)
        cache["conv"] = window[:, -(self.conv_k - 1):]
        w = self.w[p + ".conv1d.weight"]                     # (C, K)
        conv = sum(window[:, j:j + L] * w[:, j] for j in range(self.conv_k))
        conv = conv * mx.sigmoid(conv)

        q = conv[..., :key_dim].reshape(B, L, self.hk, self.dk).astype(mx.float32)
        k = conv[..., key_dim:2 * key_dim].reshape(B, L, self.hk, self.dk).astype(mx.float32)
        v = conv[..., 2 * key_dim:].reshape(B, L, self.hv, self.dv).astype(mx.float32)
        q = q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6) * (self.dk ** -0.5)
        k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        if self.hv != self.hk:
            q = mx.repeat(q, self.hv // self.hk, axis=2)
            k = mx.repeat(k, self.hv // self.hk, axis=2)

        beta = mx.sigmoid(b.astype(mx.float32))                                  # (B, L, Hv)
        A = mx.exp(self.w[p + ".A_log"].astype(mx.float32))
        s = a.astype(mx.float32) + self.w[p + ".dt_bias"].astype(mx.float32)
        g = mx.exp(-A * mx.logaddexp(s, 0.0))                                   # (B, L, Hv)

        # Recurrence, one token at a time. state: (B, Hv, Dv, Dk), float32.
        state = cache["state"]
        if state is None:
            state = mx.zeros((B, self.hv, self.dv, self.dk), dtype=mx.float32)
        ys = []
        for t in range(L):
            state = state * g[:, t, :, None, None]
            kv_mem = (state @ k[:, t, :, :, None])[..., 0]                     # (B, Hv, Dv)
            delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
            state = state + delta[..., None] * k[:, t, :, None, :]
            ys.append((state @ q[:, t, :, :, None])[..., 0])
        cache["state"] = state
        y = mx.stack(ys, axis=1).astype(x.dtype)                                # (B, L, Hv, Dv)

        # Gated RMSNorm, then the output projection.
        y = mx.fast.rms_norm(y, self.w[p + ".norm.weight"], self.eps).astype(mx.float32)
        zf = z.astype(mx.float32)
        y = (y * zf * mx.sigmoid(zf)).astype(x.dtype)
        return self.linear(y.reshape(B, L, value_dim), p + ".out_proj")

    # ------------------------------------------------------------------ model

    def forward(self, ids, caches, offset):
        x = self.w["embed_tokens.weight"][ids][None]                            # (1, L, D)
        for i in range(self.n_layers):
            p = f"layers.{i}"
            h = self.rms_norm(x, p + ".input_layernorm")
            if (i + 1) % self.interval == 0:
                h = self.attention(h, p + ".self_attn", caches[i], offset)
            else:
                h = self.gated_delta_net(h, p + ".linear_attn", caches[i])
            x = x + h
            x = x + self.mlp(self.rms_norm(x, p + ".post_attention_layernorm"), p + ".mlp")
        x = self.rms_norm(x[:, -1:], "norm")
        return x @ self.w["embed_tokens.weight"].T                              # tied head

    def generate(self, prompt_ids, max_new_tokens):
        caches = [{"k": None, "v": None, "conv": None, "state": None} for _ in range(self.n_layers)]
        logits = self.forward(mx.array(prompt_ids), caches, 0)
        out, offset = [], len(prompt_ids)
        for _ in range(max_new_tokens):
            token = mx.argmax(logits[0, -1]).item()
            out.append(token)
            if len(out) == max_new_tokens:
                break
            logits = self.forward(mx.array([token]), caches, offset)
            offset += 1
        return out
