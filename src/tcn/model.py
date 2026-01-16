import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

# ---- Activaciones ----
ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "leakyrelu": nn.LeakyReLU,
    "hardswish": nn.Hardswish,
}

# ---- helper: padding SAME 1D (siempre simétrico, NO causal) ----
def same_pad_1d(kernel_size: int, dilation: int) -> tuple[int, int]:
    total = (kernel_size - 1) * dilation
    left = total // 2
    right = total - left
    return (left, right)

# -------------------- Bloque TCN (no-causal) --------------------
class TemporalBlock(nn.Module):
    """
    Bloque residual TCN (SIN modo causal):
      - convs dilatadas con padding simétrico
      - activación -> dropout (opcional)
      - residual 1x1 si cambian canales
      - LayerNorm al final del bloque (opcional)
      - WeightNorm en las convs (opcional)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        convs_per_block: int,
        dropout: Optional[float],     # None o 0.0 lo desactiva
        activation: str,
        use_weight_norm: bool,
        wn_on_skip: bool,
        ln_block_end: bool,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad_tuple = same_pad_1d(kernel_size, dilation)

        act_ctor = ACTS[(activation or "relu").lower()]
        self.convs = nn.ModuleList()
        self.acts  = nn.ModuleList()

        def make_conv(ci, co):
            # padding simétrico (SAME) directamente en la Conv si kernel impar
            pad_conv = ((kernel_size - 1) // 2) * dilation if (kernel_size % 2 == 1) else 0
            m = nn.Conv1d(ci, co, kernel_size, stride=1, dilation=dilation, padding=pad_conv, bias=True)
            return nn.utils.parametrizations.weight_norm(m) if use_weight_norm else m

        c_in = in_ch
        for _ in range(convs_per_block):
            self.convs.append(make_conv(c_in, out_ch))
            self.acts.append(act_ctor())
            c_in = out_ch

        self.dropout = nn.Identity() if (dropout is None or dropout <= 0.0) else nn.Dropout(dropout)

        # Skip connection
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if wn_on_skip and use_weight_norm:
                self.downsample = nn.utils.parametrizations.weight_norm(self.downsample)
        else:
            self.downsample = nn.Identity()

        self.ln_end = nn.LayerNorm(out_ch) if ln_block_end else nn.Identity()

        # Init si NO usamos WeightNorm (igual que tenías)
        if not use_weight_norm:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        out = x
        for conv, a in zip(self.convs, self.acts):
            # Si el padding interno es 0 (kernel par), aplica SAME manual
            pad_attr = conv.padding
            pad_is_zero = (isinstance(pad_attr, int) and pad_attr == 0) or \
                          (isinstance(pad_attr, tuple) and all(p == 0 for p in pad_attr))
            if pad_is_zero:
                out = F.pad(out, self.pad_tuple)  # (left, right)
            out = conv(out)
            out = a(out)
            out = self.dropout(out)

        out = F.relu(out + self.downsample(x), inplace=True)
        if isinstance(self.ln_end, nn.LayerNorm):
            out = self.ln_end(out.transpose(1, 2)).transpose(1, 2)
        return out

# -------------------- Pila TCN (no-causal) --------------------
class TemporalConvNet(nn.Module):
    """
    Pila de TemporalBlock con dilataciones crecientes o personalizadas (SIN modo causal).
      - channels: lista de canales por bloque (len = #bloques)
      - dilations: lista del mismo tamaño; si None -> [1,2,4,...]
      - dropout: float global o lista[Optional[float]] por bloque
      - ln_block_end: bool global o lista[bool] por bloque
    """
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        kernel_size: int,
        dilations: Optional[List[int]],
        convs_per_block: int,
        dropout: Union[float, List[Optional[float]]],
        activation: str,
        use_weight_norm: bool,
        wn_on_skip: bool,
        ln_block_end: Union[bool, List[bool]],
    ):
        super().__init__()
        assert len(channels) >= 1, "channels debe tener al menos un valor"
        if dilations is None:
            dilations = [2**i for i in range(len(channels))]
        assert len(dilations) == len(channels), "dilations y channels deben coincidir"

        L = len(channels)
        dropouts = dropout if isinstance(dropout, list) else [dropout] * L
        ln_flags = ln_block_end if isinstance(ln_block_end, list) else [ln_block_end] * L

        blocks = []
        c_in = in_ch
        for i, (c_out, d) in enumerate(zip(channels, dilations)):
            blocks.append(TemporalBlock(
                in_ch=c_in, out_ch=c_out,
                kernel_size=kernel_size, dilation=d, convs_per_block=convs_per_block,
                dropout=dropouts[i], activation=activation,
                use_weight_norm=use_weight_norm, wn_on_skip=wn_on_skip,
                ln_block_end=ln_flags[i]
            ))
            c_in = c_out

        self.net = nn.Sequential(*blocks)
        self.out_ch = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.net(x)

# -------------------- Clasificador TCN (no-causal) --------------------
class FlexibleTCN(nn.Module):
    """
    Entrada:  x (B,T,D), lengths (B,)
    Salida:   logits (B,num_classes)
    Sin modo causal; si pooling='attention', se crea proyección interna automáticamente.
    """
    def __init__(
        self,
        input_dim: int,
        channels: Optional[List[int]],
        kernel_size: int,
        dilations: Optional[List[int]],
        convs_per_block: int,
        dropout: Union[float, List[Optional[float]]],
        activation: str,
        use_weight_norm: bool,
        wn_on_skip: bool,
        ln_block_end: Union[bool, List[bool]],
        pooling: str,       # 'last' | 'mean' | 'max' | 'attention'
        fc_hidden: Optional[int],
        num_classes: int = 3,
    ):
        super().__init__()
        if channels is None:
            channels = [128, 128, 128, 128]
        self.pooling = (pooling or "mean").lower()

        # Proyección D -> C0
        self.stem = nn.Conv1d(input_dim, channels[0], kernel_size=1)

        # TCN (no-causal)
        self.tcn = TemporalConvNet(
            in_ch=channels[0], channels=channels, kernel_size=kernel_size,
            dilations=dilations, convs_per_block=convs_per_block, dropout=dropout,
            activation=activation, use_weight_norm=use_weight_norm, wn_on_skip=wn_on_skip,
            ln_block_end=ln_block_end
        )
        feat_dim = self.tcn.out_ch

        # Atención (solo si pooling='attention'; sin parámetro attention_proj)
        if self.pooling == "attention":
            self.att_proj = nn.Linear(feat_dim, feat_dim)
            self.att_vec  = nn.Linear(feat_dim, 1, bias=False)
        else:
            self.att_proj = None
            self.att_vec  = None

        # Cabeza FC
        if fc_hidden is not None:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, fc_hidden),
                ACTS[(activation or "relu").lower()](),
                nn.Linear(fc_hidden, num_classes)
            )
        else:
            self.head = nn.Linear(feat_dim, num_classes)

    # --------- helpers de máscara/pooling ---------
    @staticmethod
    def _make_mask(x_btd: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_btd.shape
        idx = torch.arange(T, device=x_btd.device)[None, :]
        return idx < lengths[:, None]  # (B,T)

    def _masked_mean(self, x_btd, lengths):
        lengths = lengths.clamp_min(1)
        mask = self._make_mask(x_btd, lengths).unsqueeze(-1)
        x_sum = (x_btd * mask).sum(dim=1)
        denom = lengths.float().unsqueeze(-1)
        return x_sum / denom

    def _masked_max(self, x_btd, lengths):
        lengths = lengths.clamp_min(1)
        mask = self._make_mask(x_btd, lengths).unsqueeze(-1)
        x_masked = x_btd.masked_fill(~mask, float('-inf'))
        return x_masked.max(dim=1).values

    def _attn_pool(self, x_btd, lengths):
        # Se llama solo si pooling == 'attention'
        lengths = lengths.clamp_min(1)
        mask = self._make_mask(x_btd, lengths)
        h = torch.tanh(self.att_proj(x_btd))     # proyección + no linealidad
        scores = self.att_vec(h).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x_btd * attn).sum(dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_ct = x.transpose(1, 2).contiguous()   # (B, D, T)
        y = self.stem(x_ct)                     # (B, C0, T)
        y = self.tcn(y)                         # (B, C,  T)
        y_btd = y.transpose(1, 2)               # (B, T,  C)

        lengths = lengths.to(x.device)

        if self.pooling == "last":
            idx = (lengths - 1).clamp_min(0)
            feats = y_btd[torch.arange(B, device=x.device), idx]
        elif self.pooling == "mean":
            feats = self._masked_mean(y_btd, lengths)
        elif self.pooling == "max":
            feats = self._masked_max(y_btd, lengths)
        elif self.pooling == "attention":
            feats = self._attn_pool(y_btd, lengths)
        else:
            raise ValueError(f"Pooling desconocido: {self.pooling}")

        return self.head(feats)

# --------- Campo receptivo (útil para elegir niveles) ---------
def tcn_receptive_field(kernel_size: int, dilations: List[int], convs_per_block: int = 2) -> int:
    """RF = 1 + sum_bloques (convs_per_block * (kernel-1) * dilation)"""
    rf = 1
    for d in dilations:
        rf += convs_per_block * (kernel_size - 1) * d
    return rf

# (opcional) Utilidad para eliminar weight norm al exportar
from torch.nn.utils import parametrize as P
def remove_weight_norm_(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            try:
                P.remove_parametrizations(m, "weight", leave_parametrized=True)
            except ValueError:
                pass
