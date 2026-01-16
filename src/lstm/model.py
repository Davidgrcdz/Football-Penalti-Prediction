import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FlexibleLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
        dropout_layers=None,
        norm_layers=None,
        fc_hidden=None,
        num_classes=3,
        activation="relu",
        pooling="last",
        norm_post_lstm=False,
        layer_sizes=None,
        native_mode=True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.pooling = pooling.lower()
        self.num_directions = 2 if bidirectional else 1

        # ==========================================================
        #            NORMALIZAR LISTAS DE ENTRADA
        # ==========================================================
        dropout_layers = [] if dropout_layers is None else list(dropout_layers)
        norm_layers = [] if norm_layers is None else list(norm_layers)

        # ==========================================================
        #                   VALIDACIONES ESTRICTAS
        # ==========================================================

        # --- 1) num_layers debe ser >= 1
        if num_layers < 1:
            raise ValueError("num_layers debe ser >= 1")

        # --- 2) Si se usa layer_sizes, debe coincidir
        if layer_sizes is None:
            layer_sizes = [hidden_dim] * num_layers
        else:
            if len(layer_sizes) != num_layers:
                raise ValueError(
                    f"layer_sizes debe tener longitud {num_layers}, recibido {len(layer_sizes)}"
                )

        # --- 3) pooling válido
        if self.pooling not in ("last", "mean", "max", "attention"):
            raise ValueError(f"Pooling inválido: {self.pooling}")

        # --- 4) VALIDACIÓN DE native_mode
        uniform = all(h == layer_sizes[0] for h in layer_sizes)

        if native_mode:
            # 4.1: No se puede usar native_mode con tamaños no uniformes
            if not uniform:
                raise ValueError(
                    f"native_mode=True requiere layer_sizes uniformes. Recibido: {layer_sizes}"
                )
            # 4.2: dropout_layers y norm_layers deben ser vacías
            if len(dropout_layers) > 0:
                raise ValueError(
                    "dropout_layers no debe usarse con native_mode=True (no tienen efecto)."
                )
            if len(norm_layers) > 0:
                raise ValueError(
                    "norm_layers no debe usarse con native_mode=True (no tienen efecto)."
                )

        # --- 5) VALIDACIÓN DE DROPOUT
        if num_layers == 1:
            # LSTM 1 capa no puede tener dropout
            if dropout != 0.0:
                raise ValueError(
                    "dropout solo es válido para num_layers > 1, porque LSTM no aplica dropout en una capa."
                )
            if len(dropout_layers) > 0:
                raise ValueError("dropout_layers debe estar vacío si num_layers=1")
        else:
            # num_layers > 1
            if not native_mode:
                # --- MODO MODULAR ---
                if dropout > 0.0 and len(dropout_layers) == 0:
                    raise ValueError(
                        "dropout > 0 requiere especificar dropout_layers en modo modular."
                    )

                # Validar índices de dropout_layers
                for dl in dropout_layers:
                    if not (1 <= dl <= num_layers - 1):
                        raise ValueError(
                            f"Índice inválido en dropout_layers={dropout_layers}. "
                            f"Solo se permite 1..{num_layers-1}"
                        )

        # --- 6) VALIDACIÓN DE norm_layers EN MODO MODULAR
        if not native_mode:
            for nl in norm_layers:
                if not (1 <= nl <= num_layers - 1):
                    raise ValueError(
                        f"Índice inválido en norm_layers={norm_layers}. "
                        f"Solo se permite 1..{num_layers-1}"
                    )

        # ==========================================================
        #          GUARDAR PARÁMETROS VERIFICADOS
        # ==========================================================
        self.layer_sizes = layer_sizes
        self.uniform = uniform
        self.use_native = (native_mode and uniform)
        self.dropout_layers = dropout_layers
        self.norm_layers = norm_layers

        # ==========================================================
        #          DEFINICIÓN DE CAPAS (NO CAMBIADA)
        # ==========================================================

        # ----- NATIVO -----
        if self.use_native:
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=layer_sizes[0],
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.layer_norms = None
            lstm_feat_dim = layer_sizes[0] * self.num_directions

        # ----- MODULAR -----
        else:
            self.lstm_layers = nn.ModuleList()
            self.lstm_layers.append(nn.LSTM(
                input_size=input_dim,
                hidden_size=layer_sizes[0],
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            ))

            for i in range(1, num_layers):
                self.lstm_layers.append(nn.LSTM(
                    input_size=layer_sizes[i - 1] * self.num_directions,
                    hidden_size=layer_sizes[i],
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                ))

            # Norm
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(layer_sizes[i] * self.num_directions)
                for i in range(num_layers - 1)
            ])

            # Dropout entre capas
            self.dropout_mod = nn.Dropout(dropout) if dropout > 0 else None
            lstm_feat_dim = layer_sizes[-1] * self.num_directions

        # Norm global opcional
        self.lstm_norm = nn.LayerNorm(lstm_feat_dim) if norm_post_lstm else None

        # Atención
        if self.pooling == "attention":
            self.attn_proj = nn.Linear(lstm_feat_dim, lstm_feat_dim)
            self.attn_vec = nn.Linear(lstm_feat_dim, 1, bias=False)

        # FC
        acts = {
            'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU,
            'selu': nn.SELU, 'prelu': nn.PReLU,
            'hardswish': nn.Hardswish, 'elu': nn.ELU,
            'leakyrelu': nn.LeakyReLU, 'silu': nn.SiLU, 'mish': nn.Mish
        }
        if fc_hidden is not None:
            self.fc = nn.Sequential(
                nn.Linear(lstm_feat_dim, fc_hidden),
                acts[activation.lower()](),
                nn.Linear(fc_hidden, num_classes)
            )
        else:
            self.fc = nn.Linear(lstm_feat_dim, num_classes)

    # ==========================================================
    #                   POOLING HELPERS
    # ==========================================================
    @staticmethod
    def _make_mask(x, lengths_dev):
        T = x.size(1)
        return torch.arange(T, device=x.device)[None, :] < lengths_dev[:, None]

    def _masked_mean(self, x, lengths_dev):
        mask = self._make_mask(x, lengths_dev).unsqueeze(-1)
        return (x * mask).sum(dim=1) / lengths_dev.float().clamp_min(1).unsqueeze(-1)

    def _masked_max(self, x, lengths_dev):
        mask = self._make_mask(x, lengths_dev).unsqueeze(-1)
        return x.masked_fill(~mask, float('-inf')).max(dim=1).values

    def _attn_pool(self, x, lengths_dev):
        mask = self._make_mask(x, lengths_dev)
        scores = self.attn_vec(torch.tanh(self.attn_proj(x))).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * attn).sum(dim=1)

    # ==========================================================
    #                      FORWARD
    # ==========================================================
    def forward(self, x, lengths):
        lengths_cpu = lengths.cpu()
        lengths_dev = lengths.to(x.device)

        # ---- NATIVO ----
        if self.use_native:
            packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, (h_n, _) = self.lstm(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True)

            if self.pooling == "last":
                feats = (
                    torch.cat([h_n[-2], h_n[-1]], dim=1)
                    if self.bidirectional
                    else h_n[-1]
                )
            elif self.pooling == "mean":
                feats = self._masked_mean(x, lengths_dev)
            elif self.pooling == "max":
                feats = self._masked_max(x, lengths_dev)
            else:
                feats = self._attn_pool(x, lengths_dev)

        # ---- MODULAR ----
        else:
            for i, lstm_layer in enumerate(self.lstm_layers):
                packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
                packed_out, (h_n, _) = lstm_layer(packed)
                x, _ = pad_packed_sequence(packed_out, batch_first=True)

                if i < self.num_layers - 1:
                    if (i + 1) in self.norm_layers:
                        x = self.layer_norms[i](x)
                    if (i + 1) in self.dropout_layers and self.dropout_mod:
                        x = self.dropout_mod(x)

            if self.pooling == "last":
                feats = (
                    torch.cat([h_n[0], h_n[1]], dim=1)
                    if self.bidirectional
                    else h_n[0]
                )
            elif self.pooling == "mean":
                feats = self._masked_mean(x, lengths_dev)
            elif self.pooling == "max":
                feats = self._masked_max(x, lengths_dev)
            else:
                feats = self._attn_pool(x, lengths_dev)

        if self.lstm_norm:
            feats = self.lstm_norm(feats)

        return self.fc(feats)
