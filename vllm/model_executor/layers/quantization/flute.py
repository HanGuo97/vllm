from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

import flute
import flute.integrations

# Hack
flute.TEMPLATE_TUNED_WITHOUT_M_CONFIGS[(108, 4, 128, 28672, 4096)] = (
    flute.TEMPLATE_TUNED_WITHOUT_M_CONFIGS[(108, 4, 128, 28672, 8192)])


class FluteConfig(QuantizationConfig):
    """Config class for FLUTE Quantization."""

    def __init__(
        self,
        num_bits: int,
        group_size: int,
    ) -> None:
        if num_bits not in [2, 3, 4]:
            raise ValueError
        if num_bits == 3:
            raise NotImplementedError

        self.num_bits = num_bits
        self.group_size = group_size
        self.pack_factor = int(16 / num_bits)

    def __repr__(self) -> str:
        return f"FluteConfig(num_bits={self.num_bits}, group_size={self.group_size})"

    @classmethod
    def get_name(cls) -> str:
        return "flute"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["flute_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FluteConfig":
        num_sms = cls.get_from_keys(config, ["num_sms"])
        if num_sms != flute.NUM_SMS:
            raise ValueError(
                f"SMs mismatch: the model was quantized with "
                f"{num_sms}, but running with {flute.NUM_SMS} SMs.")

        num_bits = cls.get_from_keys(config, ["num_bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(num_bits=num_bits, group_size=group_size)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
    ) -> Optional["FluteLinearMethod"]:
        if isinstance(layer, LinearBase):
            return FluteLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class FluteLinearMethod(LinearMethodBase):
    """Linear method for Flute.

    Args:
        quant_config: The Flute quantization config.
    """

    def __init__(self, quant_config: FluteConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:

        # sharding is not supported
        if input_size != input_size_per_partition:
            raise NotImplementedError
        if output_size != sum(output_partition_sizes):
            raise NotImplementedError

        if params_dtype not in [torch.float16]:
            raise TypeError

        K = input_size_per_partition
        N = sum(output_partition_sizes)
        P = int(N / 16 * self.quant_config.num_bits)
        G = int(K / self.quant_config.group_size)
        device = "cuda"

        weight = Parameter(
            torch.empty(
                (P, K),
                dtype=torch.int16,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        scales = Parameter(
            torch.empty(
                (N, G),
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        tables = Parameter(
            torch.arange(
                2 ** self.quant_config.num_bits,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            tables,
            {
                **extra_weight_attrs,
                "input_dim": None,
                "output_dim": None,
                "ignore_warning": True,
            },
        )

        tables2 = Parameter(
            flute.utils.make_qmap2_from_qmap(tables),
            requires_grad=False,
        )
        set_weight_attrs(
            tables2,
            {
                **extra_weight_attrs,
                "input_dim": None,
                "output_dim": None,
                "ignore_warning": True,
            },
        )

        layer.num_bits = self.quant_config.num_bits
        layer.group_size = self.quant_config.group_size
        layer.workspace = flute.integrations._WORKSPACE

        layer.register_parameter("weight", weight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("tables", tables)
        layer.register_parameter("tables2", tables2)

        layer.needs_repacking = True
        layer.output_partition_sizes = output_partition_sizes

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (not hasattr(layer, "needs_repacking")
                or not layer.needs_repacking):
            return

        # vLLM occasionally fuses a few parameters into a single tensor, and
        # as such, we need to potentially re-pack the tensor.
        if len(layer.output_partition_sizes) == 1:
            return

        # split the combined tensors into individual tensors
        # weight: [P, K]
        # scales: [N, G]
        Ns = layer.output_partition_sizes
        Ps = [int(N / 16 * layer.num_bits) for N in Ns]
        Qs = torch.split(layer.weight, Ps, dim=0)
        Ss = torch.split(layer.scales, Ns, dim=0)

        Qs_unpacked = []
        for Q, S in zip(Qs, Ss):
            # hack: we reconstruct the unpacked data using the fact that
            # `W.T = I @ W.T` and thus using the `qgemm` routine
            _X = torch.eye(
                Q.shape[1],
                dtype=S.dtype,
                device=S.device)
            # the scales needs to be just ones
            _S = torch.ones_like(S)
            # the tables need to return the original values
            _T = torch.arange(
                2 ** layer.num_bits,
                dtype=S.dtype,
                device=S.device)
            _T2 = flute.utils.make_qmap2_from_qmap(_T)

            # unpack
            _Q = flute.qgemm_simple(
                _X,
                Q,
                _S,
                _T,
                _T2,
                layer.workspace,
                layer.num_bits,
                layer.group_size)
            Qs_unpacked.append(_Q.T)

        # re-pack the tensors
        Q_unpacked = torch.cat(Qs_unpacked, dim=0)
        template_id = flute.TEMPLATE_TUNED_WITHOUT_M_CONFIGS[(
            flute.NUM_SMS,
            layer.num_bits,
            layer.group_size,
            Q_unpacked.shape[0],   # N
            Q_unpacked.shape[1])]  # K
        Q_repacked = flute.utils.pack(
            Q_unpacked.T.contiguous(),
            num_bits=layer.num_bits,
            template_ids=[template_id])
        layer.weight = Parameter(
            Q_repacked,
            requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = flute.qgemm_simple(
            x,
            layer.weight,
            layer.scales,
            layer.tables,
            layer.tables2,
            layer.workspace,
            layer.num_bits,
            layer.group_size,
        )

        if bias is not None:
            output.add_(bias)  # In-place add

        return output
