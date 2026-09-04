"""Asynchronous committed-suffix proposals for DFlash.

The target result from round N is copied to pinned host memory on a private
stream.  Round N+1 consumes that completed copy to query the suffix corpus,
then stages candidates and a hit mask back to the device.  DFlash remains the
fallback for every row; proposal arbitration itself is a device-side select.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class _PendingCommit:
    req_ids: tuple[str, ...]
    tokens: torch.Tensor
    lengths: torch.Tensor
    ready: object


class DFlashSuffixPipeline:
    """Double-buffer committed suffix lookup without a same-round D2H wait."""

    def __init__(self, *, corpus, device, width: int, max_depth: int) -> None:
        self.corpus = corpus
        self.device = device
        self.width = int(width)
        self.max_depth = int(max_depth)
        self._copy_stream = torch.get_device_module(device).Stream()
        self._pending: Optional[_PendingCommit] = None
        self._previous_req_ids: set[str] = set()

    def reset(self) -> None:
        if self._pending is not None:
            self._pending.ready.synchronize()
        self._pending = None
        self._previous_req_ids.clear()
        self.corpus.reset()

    def stage_committed(
        self,
        *,
        req_ids: Sequence[str],
        tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        """Queue round-N committed output for round N+1 corpus lookup."""
        if self._pending is not None:
            # Batch discontinuities can leave a staged copy unconsumed.  This
            # slow path never runs in the steady decode loop.
            self._pending.ready.synchronize()
        token_host = torch.empty_like(tokens, device="cpu", pin_memory=True)
        length_host = torch.empty_like(lengths, device="cpu", pin_memory=True)
        device_module = torch.get_device_module(self.device)
        copy_done = device_module.Event()
        current_stream = device_module.current_stream()
        with device_module.stream(self._copy_stream):
            self._copy_stream.wait_stream(current_stream)
            token_host.copy_(tokens, non_blocking=True)
            length_host.copy_(lengths, non_blocking=True)
            copy_done.record()
        self._pending = _PendingCommit(
            req_ids=tuple(req_ids),
            tokens=token_host,
            lengths=length_host,
            ready=copy_done,
        )

    def candidates(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return staged candidates and a device hit mask for this batch."""
        pending = self._pending
        committed_by_req: dict[str, list[int]] = {}
        if pending is not None:
            # The copy was issued one complete model round earlier.  Waiting
            # here does not serialize against the current proposal/verify work.
            pending.ready.synchronize()
            for row, req_id in enumerate(pending.req_ids):
                length = int(pending.lengths[row])
                committed_by_req[req_id] = pending.tokens[row, :length].tolist()
            self._pending = None

        req_ids = [req.rid for req in batch.reqs]
        contexts: list[list[int]] = []
        total_lens: list[int] = []
        eligible = np.zeros((len(req_ids),), dtype=np.bool_)
        committed: list[list[int]] = []
        for row, req in enumerate(batch.reqs):
            relayed = committed_by_req.get(req.rid, []) if batch.enable_overlap else []
            output_tail = list(req.output_ids[-self.max_depth :]) + relayed
            input_tail = list(req.origin_input_ids[-self.max_depth :])
            combined = (input_tail + output_tail)[-self.max_depth :]
            contexts.append(combined)
            total_lens.append(
                len(req.origin_input_ids) + len(req.output_ids) + len(relayed)
            )
            eligible[row] = bool(combined) and (
                not batch.enable_overlap or req.rid in committed_by_req
            )
            if len(relayed) >= 2:
                committed.append(relayed[-self.max_depth :])

        self.corpus.synchronize()
        ids, masks = self.corpus.batch_get(req_ids, contexts, total_lens)
        # Learn only tokens already committed by target verification, and only
        # after querying, so a row cannot manufacture its own continuation.
        if committed:
            self.corpus.batch_put(committed)

        tree = masks.reshape(len(req_ids), self.width, self.width)
        hits = tree[:, self.width - 1, : self.width].all(axis=1) & eligible
        current_req_ids = set(req_ids)
        departed = self._previous_req_ids - current_req_ids
        if departed:
            self.corpus.erase_match_state(list(departed))
        self._previous_req_ids = current_req_ids

        token_host = torch.as_tensor(ids.reshape(len(req_ids), self.width)).pin_memory()
        hit_host = torch.as_tensor(hits).pin_memory()
        return (
            token_host.to(self.device, dtype=torch.int64, non_blocking=True),
            hit_host.to(self.device, dtype=torch.bool, non_blocking=True),
        )
