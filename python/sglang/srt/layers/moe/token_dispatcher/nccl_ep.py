"""NCCL EP dispatcher for SGLang MoE expert parallelism.

Supports both Low-Latency (LL) and High-Throughput (HT) modes.
Outputs DeepEP-compatible NamedTuples so ``DeepEPMoE.run_moe_core`` works unchanged.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
    DispatcherBaseHooks,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
    DeepEPPDispatchHooks,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

logger = logging.getLogger(__name__)

try:
    from nccl_ep.nccl_wrapper import (
        NCCL_EP_AUTO,
        NCCLLibrary,
        get_nccl_comm_from_group,
        ncclDataTypeEnum,
        ncclEpAlgorithm_t,
        ncclEpDispatchConfig_t,
        ncclEpGroupConfig_t,
        ncclEpTensorTag_t,
        ncclNDTensor_t,
    )

    _HAVE_NCCL_EP = True
except ImportError:
    _HAVE_NCCL_EP = False

DT = ncclDataTypeEnum if _HAVE_NCCL_EP else None
TAG = ncclEpTensorTag_t if _HAVE_NCCL_EP else None


def _make_nd_tensor(t: torch.Tensor, tag: int, datatype: int) -> ncclNDTensor_t:
    """Wrap a contiguous CUDA tensor as ``ncclNDTensor_t`` with all-1 strides."""
    t = t.contiguous()
    ndim = t.dim()
    sizes = (ctypes.c_uint * ndim)(*[int(s) for s in t.shape])
    strides = (ctypes.c_uint * ndim)(*([1] * ndim))
    desc = ncclNDTensor_t()
    desc.version = 1
    desc.ndim = ndim
    desc.sizes = ctypes.cast(sizes, ctypes.POINTER(ctypes.c_uint))
    desc.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_uint))
    desc.datatype = datatype
    desc.data = ctypes.c_void_p(t.data_ptr())
    desc.tag = tag
    desc.flags = 0
    # prevent GC of underlying arrays
    desc._sizes_buf = sizes
    desc._strides_buf = strides
    desc._torch_ref = t
    return desc


def _as_ptr_array(tensors: Sequence[ncclNDTensor_t]) -> Any:
    n = len(tensors)
    if n == 0:
        return None
    arr = (ctypes.POINTER(ncclNDTensor_t) * n)()
    for i, t in enumerate(tensors):
        arr[i] = ctypes.pointer(t)
    # Return the array directly (NOT ctypes.cast). ctypes.cast creates a new pointer
    # object that does NOT hold a reference to `arr`, causing `arr` to be immediately
    # garbage-collected by Python's reference counting when the caller gets the cast
    # result. The C library then receives a dangling pointer. Returning the array
    # directly keeps it alive in the caller's scope; ctypes auto-converts
    # (POINTER(T) * n) to POINTER(POINTER(T)) when passing to a C function.
    return arr


def _stream_ptr(stream: torch.cuda.Stream) -> ctypes.c_void_p:
    return ctypes.c_void_p(stream.cuda_stream)


class _NcclEpState:
    """Lazily initialised per-process singleton holding NCCL EP long-lived objects."""

    _instance: Optional[_NcclEpState] = None

    def __init__(self):
        self.lib: NCCLLibrary = NCCLLibrary()
        self.comm = None
        self.stream: Optional[torch.cuda.Stream] = None
        self._ll_group = None
        self._ht_group = None
        self._group_configs: dict = {}

    @classmethod
    def get(cls) -> _NcclEpState:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_comm(self, group: torch.distributed.ProcessGroup):
        if self.comm is not None:
            return
        self.comm = get_nccl_comm_from_group(group=group, nccl_lib=self.lib)
        self.stream = torch.cuda.Stream()

    def get_group(
        self,
        is_ht: bool,
        num_experts: int,
        num_local_experts: int,
        max_tokens_per_rank: int,
        hidden_size: int,
    ):
        attr = "_ht_group" if is_ht else "_ll_group"
        grp = getattr(self, attr)
        if grp is not None:
            return grp

        cfg = ncclEpGroupConfig_t()
        cfg.version = 1
        cfg.algorithm = (
            ncclEpAlgorithm_t.NCCL_EP_ALGO_HIGH_THROUGHPUT
            if is_ht
            else ncclEpAlgorithm_t.NCCL_EP_ALGO_LOW_LATENCY
        )
        cfg.num_experts = num_experts
        cfg.max_tokens_per_rank = max_tokens_per_rank
        cfg.token_size_bytes = hidden_size * 2  # bfloat16
        cfg.rdma_buffer_size = NCCL_EP_AUTO
        cfg.num_qp_per_rank = (
            NCCL_EP_AUTO if is_ht else num_local_experts
        )
        cfg.num_channels = NCCL_EP_AUTO

        grp = self.lib.ncclEpCreateGroup(
            self.comm, cfg, _stream_ptr(self.stream)
        )
        setattr(self, attr, grp)
        self._group_configs[attr] = cfg
        return grp


class _NcclEpImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not _HAVE_NCCL_EP:
            raise ImportError(
                "nccl_ep package is not installed. "
                "Install from nccl/contrib/nccl_ep/python (pip install -e .) "
                "and ensure NCCL_HOME points to an EP build."
            )

        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.world_size = dist.get_world_size(group)

        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None
        self._combine_num_tokens: Optional[int] = None
        self._orig_num_tokens: Optional[int] = None
        self._skip_nccl_ep: bool = False
        self.quant_config: Optional[dict] = None
        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        state = _NcclEpState.get()
        state.ensure_comm(group)
        self._state = state

    def dispatch_a(self, hidden_states, topk_output):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(self, hidden_states, topk_ids, topk_weights):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None

    def _maybe_dummy_zero_tokens(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """NCCL EP requires non-empty token/routing tensors (see Flashinfer dummy-token pattern)."""
        self._orig_num_tokens = int(hidden_states.shape[0])
        if self._orig_num_tokens != 0:
            return hidden_states, topk_ids, topk_weights
        device = hidden_states.device
        ep_rank = dist.get_rank(self.group)
        base = ep_rank * self.num_local_experts
        k = int(topk_ids.shape[1])
        tw_k = int(topk_weights.shape[1])
        # NCCL EP dispatch uses bf16 token layout (see get_group token_size_bytes); always bf16.
        # Build dummies on CPU then copy: direct torch.zeros(..., device=cuda) can raise
        # cudaErrorInvalidConfiguration on some driver/PyTorch paths even for tiny tensors.
        hidden_states = torch.zeros(
            1, self.hidden_size, dtype=torch.bfloat16, device="cpu"
        ).to(device, non_blocking=False)
        topk_ids = torch.full((1, k), base, dtype=torch.int64, device="cpu").to(
            device, non_blocking=False
        )
        topk_weights = torch.zeros(
            (1, tw_k), dtype=topk_weights.dtype, device="cpu"
        ).to(device, non_blocking=False)
        return hidden_states, topk_ids, topk_weights


class _NcclEpImplNormal(_NcclEpImplBase):
    """High-Throughput mode implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dispatch_descs = None

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        hidden_states, topk_ids, topk_weights = self._maybe_dummy_zero_tokens(
            hidden_states, topk_ids, topk_weights
        )
        return hidden_states, topk_ids, topk_weights

    def dispatch_b(self, hidden_states, topk_ids, topk_weights):
        lib = self._state.lib
        stream_c = _stream_ptr(self._state.stream)

        event = torch.cuda.current_stream().record_event()
        self._state.stream.wait_event(event)

        ep_group = self._state.get_group(
            is_ht=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            max_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden_size=self.hidden_size,
        )

        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        ep_dev = hidden_states.device
        num_recv_tokens = self.num_max_dispatch_tokens_per_rank * self.world_size

        # True idle batch (0 tokens on this rank): skip NCCL EP. Dummy 1-token tensors
        # satisfy routing shape, but ncclEpCreateHandle/dispatch can still fail on some
        # stacks; async errors then surface at torch.empty. Feed zeros and skip combine.
        if self._orig_num_tokens == 0:
            self._skip_nccl_ep = True
            self.handle = None
            self._topk_desc = None
            recv_hidden = torch.zeros(
                (num_recv_tokens, self.hidden_size),
                dtype=torch.bfloat16,
                device="cpu",
            ).to(ep_dev, non_blocking=False)
            recv_topk_weights = torch.zeros(
                (num_recv_tokens, top_k), dtype=torch.float32, device="cpu"
            ).to(ep_dev, non_blocking=False)
            recv_topk_ids = torch.zeros(
                (num_recv_tokens, top_k), dtype=torch.int64, device="cpu"
            ).to(ep_dev, non_blocking=False)
            num_recv_per_expert = [0] * self.num_local_experts
            self._dispatch_descs = None
            self._ep_group_ref = ep_group
            self._combine_num_tokens = num_tokens
            return DeepEPNormalDispatchOutput(
                recv_hidden, None, recv_topk_ids, recv_topk_weights, num_recv_per_expert,
            )

        topk_desc = _make_nd_tensor(topk_ids, TAG.NCCL_EP_TENSOR_TAG_TOPK_IDX, DT.ncclInt64)
        self._topk_desc = topk_desc
        self.handle = lib.ncclEpCreateHandle(
            ep_group, topk_desc, None, stream_c, local_tensors=None, use_fp8=False
        )

        hs_bf16 = hidden_states.to(dtype=torch.bfloat16).contiguous()
        inp_tokens_desc = _make_nd_tensor(
            hs_bf16, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )
        tw_contig = topk_weights.float().contiguous()
        inp_tw_desc = _make_nd_tensor(
            tw_contig, TAG.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, DT.ncclFloat32
        )

        recv_hidden = torch.empty(
            (num_recv_tokens, self.hidden_size), dtype=torch.bfloat16, device=ep_dev
        )
        out_tokens_desc = _make_nd_tensor(
            recv_hidden, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )
        recv_topk_weights = torch.empty(
            (num_recv_tokens, top_k), dtype=torch.float32, device=ep_dev
        )
        out_tw_desc = _make_nd_tensor(
            recv_topk_weights, TAG.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, DT.ncclFloat32
        )
        recv_topk_ids = torch.empty(
            (num_recv_tokens, top_k), dtype=torch.int64, device=ep_dev
        )
        out_ti_desc = _make_nd_tensor(
            recv_topk_ids, TAG.NCCL_EP_TENSOR_TAG_TOPK_IDX, DT.ncclInt64
        )

        topk_desc.tag = TAG.NCCL_EP_TENSOR_TAG_TOPK_IDX
        dispatch_inputs = [inp_tokens_desc, inp_tw_desc, topk_desc]
        dispatch_outputs = [out_tokens_desc, out_tw_desc, out_ti_desc]

        disp_cfg = ncclEpDispatchConfig_t()
        disp_cfg.round_scales = 0

        lib.ncclEpDispatch(
            self.handle,
            _as_ptr_array(dispatch_inputs), len(dispatch_inputs),
            _as_ptr_array(dispatch_outputs), len(dispatch_outputs),
            None, 0,
            0, disp_cfg, stream_c,
        )
        lib.ncclEpComplete(self.handle, None, stream_c)
        self._state.stream.synchronize()

        num_recv_per_expert = [0] * self.num_local_experts
        for i in range(num_recv_tokens):
            eid = int(recv_topk_ids[i, 0].item())
            local_eid = eid - (dist.get_rank(self.group) * self.num_local_experts)
            if 0 <= local_eid < self.num_local_experts:
                num_recv_per_expert[local_eid] += 1

        self._dispatch_descs = [
            inp_tokens_desc, inp_tw_desc, topk_desc,
            out_tokens_desc, out_tw_desc, out_ti_desc,
        ]
        self._ep_group_ref = ep_group
        # HT combine asserts combined_x->sizes[0] == handle->num_tokens (dispatch batch size).
        self._combine_num_tokens = num_tokens

        return DeepEPNormalDispatchOutput(
            recv_hidden, None, recv_topk_ids, recv_topk_weights, num_recv_per_expert,
        )

    def combine_a(self, hidden_states, topk_ids, topk_weights):
        return (hidden_states,)

    def combine_b(self, hidden_states):
        if self._skip_nccl_ep:
            self._skip_nccl_ep = False
            dev = hidden_states.device
            n_orig = self._orig_num_tokens
            self._orig_num_tokens = None
            self._combine_num_tokens = None
            return torch.empty(
                (n_orig, self.hidden_size), dtype=torch.bfloat16, device=dev
            )

        lib = self._state.lib
        stream_c = _stream_ptr(self._state.stream)
        ep_group = self._ep_group_ref

        event = torch.cuda.current_stream().record_event()
        self._state.stream.wait_event(event)

        num_recv = hidden_states.shape[0]
        num_combined = self._combine_num_tokens

        hs_contig = hidden_states.to(dtype=torch.bfloat16).contiguous()
        comb_in_desc = _make_nd_tensor(
            hs_contig, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )

        result = torch.empty(
            (num_combined, self.hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        comb_out_desc = _make_nd_tensor(
            result, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )

        lib.ncclEpCombine(
            self.handle,
            _as_ptr_array([comb_in_desc]), 1,
            _as_ptr_array([comb_out_desc]), 1,
            None, 0,
            0, None, stream_c,
        )
        lib.ncclEpComplete(self.handle, None, stream_c)
        self._state.stream.synchronize()

        self._dispatch_descs = None
        self._combine_num_tokens = None
        lib.ncclEpHandleDestroy(self.handle)
        self.handle = None
        self._topk_desc = None

        n_orig = self._orig_num_tokens
        self._orig_num_tokens = None
        return result[:n_orig]


class _NcclEpImplLowLatency(_NcclEpImplBase):
    """Low-Latency mode implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dispatch_descs = None

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        hidden_states, topk_ids, topk_weights = self._maybe_dummy_zero_tokens(
            hidden_states, topk_ids, topk_weights
        )
        expected_m = (
            hidden_states.shape[0] * self.world_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        return hidden_states, topk_ids, topk_weights, expected_m

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, expected_m):
        lib = self._state.lib
        stream_c = _stream_ptr(self._state.stream)

        event = torch.cuda.current_stream().record_event()
        self._state.stream.wait_event(event)

        ep_group = self._state.get_group(
            is_ht=False,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            max_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden_size=self.hidden_size,
        )

        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        num_recv_tokens = self.num_max_dispatch_tokens_per_rank * self.world_size
        ep_dev = hidden_states.device

        if self._orig_num_tokens == 0:
            self._skip_nccl_ep = True
            self.handle = None
            self._topk_desc = None
            recv_hidden = torch.zeros(
                (self.num_local_experts, num_recv_tokens, self.hidden_size),
                dtype=torch.bfloat16,
                device="cpu",
            ).to(ep_dev, non_blocking=False)
            masked_m = torch.zeros(
                self.num_local_experts, dtype=torch.int32, device="cpu"
            ).to(ep_dev, non_blocking=False)
            self._dispatch_descs = None
            self._ep_group_ref = ep_group
            self._num_recv_tokens = num_recv_tokens
            self._combine_num_tokens = num_tokens
            return DeepEPLLDispatchOutput(
                recv_hidden, None, topk_ids, topk_weights, masked_m, expected_m,
            )

        topk_desc = _make_nd_tensor(topk_ids, TAG.NCCL_EP_TENSOR_TAG_TOPK_IDX, DT.ncclInt64)
        self._topk_desc = topk_desc
        self.handle = lib.ncclEpCreateHandle(
            ep_group, topk_desc, None, stream_c, local_tensors=None, use_fp8=False
        )

        hs_bf16 = hidden_states.to(dtype=torch.bfloat16).contiguous()
        inp_desc = _make_nd_tensor(
            hs_bf16, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )

        recv_hidden = torch.empty(
            (self.num_local_experts, num_recv_tokens, self.hidden_size),
            dtype=torch.bfloat16, device=ep_dev,
        )
        out_desc = _make_nd_tensor(
            recv_hidden, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )

        masked_m = torch.empty(self.num_local_experts, dtype=torch.int32, device=ep_dev)
        counter_desc = _make_nd_tensor(
            masked_m, TAG.NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE, DT.ncclInt32
        )

        disp_cfg = ncclEpDispatchConfig_t()
        disp_cfg.round_scales = 0

        lib.ncclEpDispatch(
            self.handle,
            _as_ptr_array([inp_desc]), 1,
            _as_ptr_array([out_desc]), 1,
            _as_ptr_array([counter_desc]), 1,
            0, disp_cfg, stream_c,
        )
        lib.ncclEpComplete(self.handle, None, stream_c)
        self._state.stream.synchronize()

        self._dispatch_descs = [inp_desc, out_desc, counter_desc]
        self._ep_group_ref = ep_group
        self._num_recv_tokens = num_recv_tokens
        self._combine_num_tokens = num_tokens

        return DeepEPLLDispatchOutput(
            recv_hidden, None, topk_ids, topk_weights, masked_m, expected_m,
        )

    def combine_a(self, hidden_states, topk_ids, topk_weights):
        return (hidden_states, topk_ids, topk_weights)

    def combine_b(self, hidden_states, topk_ids, topk_weights):
        if self._skip_nccl_ep:
            self._skip_nccl_ep = False
            dev = hidden_states.device
            h = self.hidden_size
            n_orig = self._orig_num_tokens
            self._orig_num_tokens = None
            self._combine_num_tokens = None
            return torch.empty((n_orig, h), dtype=torch.bfloat16, device=dev)

        lib = self._state.lib
        stream_c = _stream_ptr(self._state.stream)
        ep_group = self._ep_group_ref

        event = torch.cuda.current_stream().record_event()
        self._state.stream.wait_event(event)

        if hidden_states.dim() == 3:
            n_exp, n_recv, h = hidden_states.shape
        else:
            n_recv = hidden_states.shape[0]
            h = self.hidden_size
            n_exp = self.num_local_experts

        # LL ncclEpCombine asserts out->sizes[0] == topk_weights->sizes[0] (see nccl_ep.cc).
        num_combined = int(topk_weights.shape[0])

        hs_contig = hidden_states.contiguous()
        if hs_contig.dim() == 2:
            hs_contig = hs_contig.view(n_exp, n_recv, h)
        comb_in_desc = _make_nd_tensor(
            hs_contig, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )

        result = torch.empty(
            (num_combined, h), dtype=torch.bfloat16, device="cuda"
        )
        comb_out_desc = _make_nd_tensor(
            result, TAG.NCCL_EP_TENSOR_TAG_TOKENS, DT.ncclBfloat16
        )

        tw_contig = topk_weights.float().contiguous()
        tw_desc = _make_nd_tensor(
            tw_contig, TAG.NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, DT.ncclFloat32
        )

        lib.ncclEpCombine(
            self.handle,
            _as_ptr_array([comb_in_desc]), 1,
            _as_ptr_array([comb_out_desc]), 1,
            _as_ptr_array([tw_desc]), 1,
            0, None, stream_c,
        )
        lib.ncclEpComplete(self.handle, None, stream_c)
        self._state.stream.synchronize()

        self._dispatch_descs = None
        self._combine_num_tokens = None
        lib.ncclEpHandleDestroy(self.handle)
        self.handle = None
        self._topk_desc = None

        n_orig = self._orig_num_tokens
        self._orig_num_tokens = None
        return result[:n_orig]


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class NcclEpDispatcher(BaseDispatcher):
    """Top-level NCCL EP dispatcher supporting LL and HT modes."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _NcclEpImplLowLatency(**common_kwargs)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _NcclEpImplNormal(**common_kwargs)

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _NcclEpImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)
