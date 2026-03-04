# to be combined with the sparse coordinator class and sparse algorithm family

from typing import List, NamedTuple, Optional

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseNSATokenToKVPool,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool_host import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module

device_module = get_device_module()

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: HiSparseTokenToKVPoolAllocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group: torch.distributed.ProcessGroup,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device

        self.mem_pool_device: HiSparseNSATokenToKVPool = (
            self.token_to_kv_pool_allocator.get_kvcache()
        )
        self.mem_pool_host = MLATokenToKVPoolHost(
            device_pool=self.mem_pool_device,
            host_to_device_ratio=2,
            host_size=0,
            page_size=1,  # for simplicity, we set page size to 1 to enable backup one token at a time
            layout="layer_first",
            override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
        )

        max_num_reqs = req_to_token_pool.size
        max_context_len = req_to_token_pool.max_context_len

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_reqs, self.padded_buffer_size), dtype=torch.int64, device=device
        )
        self.req_to_host_pool = torch.zeros(
            (max_num_reqs, max_context_len),
            dtype=torch.int64,
            device=device,
        )

        self.write_staging_stream = device_module.Stream()
        self.write_decoding_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.ack_decoding_queue: List[HiSparseAct] = []

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (max_num_reqs, layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (max_num_reqs, layer_num, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.bitmap = torch.full(
            (max_num_reqs, max_context_len),
            -1,
            dtype=torch.int16,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(max_num_reqs, layer_num, 1)
            .contiguous()
        )

    def admit_request_into_staging(self, req: Req) -> None:
        req.staging = True
        logical_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        device_indices = self.mem_pool_device._translate_loc_to_hisparse_device(
            logical_indices
        )

        prefill_len = len(device_indices)
        host_indices = self.mem_pool_host.alloc(prefill_len).to(device=self.device)
        assert (
            host_indices is not None
        ), "Host mem pool alloc failed, this should not happen"
        self.req_to_host_pool[req.req_pool_idx, :prefill_len] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device, host_indices, device_indices, io_backend="kernel"
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def alloc_device_buffer(self, req: Req) -> None:
        allocated_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(req.fill_ids)
        ]
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            allocated_indices,
            self.padded_buffer_size,
        )
        self.req_to_device_buffer[req.req_pool_idx, : self.padded_buffer_size] = (
            buffer_indices
        )

        # initialize the token locs for the device buffer
        self.req_device_buffer_tokens[
            req.req_pool_idx, :, : self.device_buffer_size
        ] = torch.arange(self.device_buffer_size, device=self.device)
        self.req_device_buffer_token_locs[
            req.req_pool_idx, :, : self.padded_buffer_size
        ] = buffer_indices[: self.padded_buffer_size]

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_batch(self) -> Optional[ScheduleBatch]:
        ready_batch = None
        if len(self.ack_staging_queue) == 0:
            return ready_batch

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            req.staging = False
            finish_count -= 1
            if (
                len(self.ack_staging_queue) == 0
                or self.ack_staging_queue[0][2].batch != req.batch
            ):
                if ready_batch is None:
                    ready_batch = req.batch
                else:
                    ready_batch.merge_batch(req.batch)
            # to break the circular reference
            req.batch = None
        return ready_batch

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ) -> None:
        reserved_buffer_loc = self.req_to_device_buffer[
            req_pool_indices, self.device_buffer_size
        ]

        short_reqs = seq_lens <= self.device_buffer_size
        if torch.any(short_reqs):
            reserved_buffer_loc[short_reqs] = self.req_to_device_buffer[
                req_pool_indices[short_reqs], seq_lens[short_reqs] - 1
            ]

        # todo, clear the prior mapping as well
        self.mem_pool_device.full_to_hisparse_device_index_mapping[out_cache_loc] = (
            reserved_buffer_loc
        )
        # proceed only if the backup is finished for new generated tokens
        self.wait_for_decode_writes()

    def get_front_topk_tokens(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        # a dummy selection for testing
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )
        for i in range(num_reqs):
            top_n = min(
                seq_lens[i],
                self.top_k,
            )
            if top_n == 0:
                continue
            top_k_indices[i, :top_n] = self.req_to_device_buffer[req_pool_indices[i]][
                :top_n
            ]
        return top_k_indices

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        # a dummy selection for testing
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )
        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)
            assert torch.all(selected_tokens >= 0), "Selected invalid token positions"
            assert torch.all(
                selected_tokens < seq_len
            ), "Selected token positions out of range"
            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                special_tokens_mask = selected_tokens == seq_len
                host_indices = self.req_to_host_pool[
                    req_idx, selected_tokens[~special_tokens_mask]
                ]
                assert torch.all(
                    host_indices > 0
                ), "Host indices should be valid for non-special tokens"
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
                device_indices[special_tokens_mask] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices[~special_tokens_mask],
                    layer_id - self.mem_pool_device.start_layer,
                    io_backend="kernel",
                )
            top_k_indices[i][:top_n] = device_indices.to(torch.int32)
        return top_k_indices

    def wait_for_decode_writes(self) -> None:
        if len(self.ack_decoding_queue) == 0:
            return
        _, finish_event, _ = self.ack_decoding_queue.pop(0)
        finish_event.synchronize()

    def retract_req(self, req: Req) -> None:
        # release resources for the request
        # todo, cancel ongoing data transfer for the request if any
        self.request_finished(req)
        return

    def update_requests_after_decode(self, reqs: List[Req]) -> None:
        if len(reqs) == 0:
            return
        # todo, adjust the granularity to mitigate the overhead
        req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], device=self.device
        )
        req_kv_lens = torch.tensor(
            [r.kv_allocated_len for r in reqs],
            device=self.device,
        )
        last_token_indices = self.req_to_device_buffer[
            req_pool_indices, self.device_buffer_size
        ]

        short_reqs = req_kv_lens <= self.device_buffer_size
        if torch.any(short_reqs):
            last_token_indices[short_reqs] = self.req_to_device_buffer[
                req_pool_indices[short_reqs], req_kv_lens[short_reqs] - 1
            ]

        # backup the new KV cache to host for future use
        host_indices = self.mem_pool_host.alloc(len(last_token_indices)).to(
            device=self.device
        )
        assert (
            host_indices is not None
        ), "Host mem pool alloc failed, this should not happen"
        self.req_to_host_pool[req_pool_indices, req_kv_lens - 1] = host_indices

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_decoding_stream):
            start_event.wait(self.write_decoding_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                last_token_indices.contiguous(),
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_decoding_stream)
            if last_token_indices.is_cuda:
                last_token_indices.record_stream(self.write_decoding_stream)

        self.ack_decoding_queue.append(HiSparseAct(start_event, finish_event, None))

    def request_finished(self, req: Req):
        # release memory
        buffer_indices = self.req_to_device_buffer[req.req_pool_idx]
        self.token_to_kv_pool_allocator.free_hisparse_indices(buffer_indices)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv_allocated_len
        ]
        self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            allocated_locs
        ] = 0

        host_indices = self.req_to_host_pool[req.req_pool_idx, : req.kv_allocated_len]
        host_indices = host_indices[host_indices > 0]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        # clear req info
        self.req_device_buffer_tokens[req.req_pool_idx, :, :] = -1
        self.req_device_buffer_token_locs[req.req_pool_idx, :, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = 0
        self.lru_slots[req.req_pool_idx].copy_(self._lru_init)
