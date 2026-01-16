import torch
import torch.nn as nn
import torch.nn.functional as F

# === Activaciones disponibles ===
ACTS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "elu": nn.ELU(),
    "prelu": nn.PReLU(),
    "leakyrelu": nn.LeakyReLU(),
    "hardswish": nn.Hardswish(),
}

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, learned=False):
        super().__init__()
        if learned:
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                 (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# === Capa encoder flexible ===
class TransformerEncoderLayerVarDim(nn.Module):
    def __init__(self, d_in, d_out, nhead, dim_feedforward=512,
                 dropout=0.1, activation="gelu", norm_type="prenorm"):
        super().__init__()
        self.norm_type = norm_type

        # ✅ Comprobar divisibilidad
        if d_out % nhead != 0:
            raise ValueError(f"[Error] d_out={d_out} no es divisible por num_heads={nhead}. "
                             f"Cada cabeza recibiría {d_out / nhead:.2f} dimensiones.")

        # Proyección de entrada si cambia la dimensión
        self.input_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

        # Multihead Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_out, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_out, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_out)
        self.dropout = nn.Dropout(dropout)

        # Normalización
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)

        # Activación modular
        act_key = activation.lower()
        if act_key not in ACTS:
            raise ValueError(f"Activación '{activation}' no soportada.")
        self.activation = ACTS[act_key]

        # Guardar atención
        self.last_attn_map = None

    def forward(self, src, mask=None):
        x = self.input_proj(src)

        # Bloque de autoatención
        if self.norm_type == "prenorm":
            x = x + self._self_attn_block(self.norm1(x), mask)
        else:
            x = self.norm1(x + self._self_attn_block(x, mask))

        # Bloque feed-forward
        if self.norm_type == "prenorm":
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm2(x + self._ff_block(x))

        return x

    def _self_attn_block(self, x, mask=None):
        out, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=(mask == False) if mask is not None else None,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn_map = attn_weights.detach().cpu()
        return self.dropout(out)

    def _ff_block(self, x):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))

# === Transformer flexible con soporte de dims variables ===
class FlexibleTransformer(nn.Module):
    def __init__(self,
                 input_dim=768,
                 model_dim=256,            # Puede ser int o lista [512, 256, 128]
                 num_heads=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 activation="gelu",
                 norm_type="prenorm",
                 pos_encoding="learned",
                 use_cls_token=True,
                 pooling="cls",
                 num_classes=3,
                 max_seq_len=128):
        super().__init__()
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        # Aceptar lista de dimensiones o un único valor
        if isinstance(model_dim, int):
            self.model_dims = [model_dim] * num_layers
        elif isinstance(model_dim, (list, tuple)):
            assert len(model_dim) == num_layers, \
                f"Debe haber {num_layers} dimensiones (una por capa)."
            self.model_dims = list(model_dim)
        else:
            raise TypeError("model_dim debe ser int o lista de int.")

        # Proyección inicial
        self.input_proj = nn.Linear(input_dim, self.model_dims[0])

        # Positional Encoding
        self.pe = PositionalEncoding(self.model_dims[0], max_seq_len + 1,
                                     learned=(pos_encoding == "learned"))

        # Token [CLS]
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dims[0]))

        # ✅ Verificar divisibilidad en todas las capas antes de construirlas
        for i, dim in enumerate(self.model_dims):
            if dim % num_heads != 0:
                raise ValueError(
                    f"[Error en capa {i+1}] model_dim={dim} no es divisible por num_heads={num_heads} "
                    f"(cada cabeza tendría {dim / num_heads:.2f} dims)."
                )

        # Capas encoder
        self.layers = nn.ModuleList([
            TransformerEncoderLayerVarDim(
                d_in=self.model_dims[i - 1] if i > 0 else self.model_dims[0],
                d_out=self.model_dims[i],
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_type=norm_type
            )
            for i in range(num_layers)
        ])

        # Normalización final y clasificador
        self.norm = nn.LayerNorm(self.model_dims[-1])
        self.classifier = nn.Linear(self.model_dims[-1], num_classes)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        x = self.input_proj(x)

        # Añadir token CLS
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)

        # Codificación posicional
        x = self.pe(x)

        # Pasar por las capas encoder
        for layer in self.layers:
            x = layer(x, mask)

        # Normalización final
        x = self.norm(x)

        # Pooling final
        if self.pooling == "cls":
            out = x[:, 0]
        elif self.pooling == "mean":
            out = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True) if mask is not None else x.mean(1)
        elif self.pooling == "max":
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                out, _ = x_masked.max(1)
            else:
                out, _ = x.max(1)
        else:
            raise ValueError(f"Pooling desconocido: {self.pooling}")

        return self.classifier(out)
