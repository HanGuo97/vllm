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
import flute.utils
import flute.integrations


class FluteConfig(QuantizationConfig):
    """Config class for FLUTE Quantization."""

    def __init__(
        self,
        num_bits: int = 4,
        group_size: int = 128,
    ) -> None:
        if num_bits not in [2, 3, 4]:
            raise ValueError
        if num_bits == 3:
            raise NotImplementedError

        self.num_bits = num_bits
        self.group_size = group_size
        self.pack_factor = int(16 / self.num_bits)

    def __repr__(self) -> str:
        return (
            f"FluteConfig(num_bits={self.num_bits}, group_size={self.group_size})"
        )

    @classmethod
    def get_name(self) -> str:
        return "flute"

    @classmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(self) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["flute_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FluteConfig":
        num_bits = cls.get_from_keys(config, ["num_bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(num_bits, group_size)

    def get_quant_method(
            self,
            layer: torch.nn.Module) -> Optional["FluteLinearMethod"]:
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

    def __init__(self, quant_config: FluteConfig):
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

        if params_dtype != torch.float16:
            raise TypeError
        if extra_weight_attrs:
            raise ValueError

        # Quantized weights packed into Int16.
        K = input_size_per_partition
        N = sum(output_partition_sizes)
        P = int(N / 16 * self.quant_config.num_bits)
        G = int(K / self.quant_config.group_size)
        device = torch.device("cuda")

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
                device=device,
                dtype=params_dtype,
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
            },
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("tables", tables)
        layer.register_parameter("tables2", tables2)
        layer.workspace = flute.integrations._WORKSPACE

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
