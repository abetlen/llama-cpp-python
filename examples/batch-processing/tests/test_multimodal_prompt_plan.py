from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_server_module():
    module_name = "batch_processing_server"
    if module_name in sys.modules:
        return sys.modules[module_name]

    server_path = Path(__file__).resolve().parents[1] / "server.py"
    spec = importlib.util.spec_from_file_location(module_name, server_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


server = load_server_module()


def make_prompt_plan(non_causal: bool = False):
    text_a = server.PromptSegment(
        kind="text",
        start_pos=0,
        n_pos=2,
        identity_tokens=[10, 11],
        decode_start_pos=0,
        decode_n_pos=2,
        text_tokens=[10, 11],
    )
    media = server.PromptSegment(
        kind="image",
        start_pos=2,
        n_pos=4,
        identity_tokens=[-1, -2, -3, -4],
        decode_start_pos=2,
        decode_n_pos=2,
        embeddings=np.zeros((4, 3), dtype=np.float32),
        positions=np.array(
            [
                2,
                2,
                3,
                3,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.int32,
        ),
        non_causal=non_causal,
    )
    text_b = server.PromptSegment(
        kind="text",
        start_pos=6,
        n_pos=2,
        identity_tokens=[12, 13],
        decode_start_pos=4,
        decode_n_pos=2,
        text_tokens=[12, 13],
    )
    return server.PromptPlan(
        text="",
        generation_prompt="",
        text_tokens=[10, 11, 12, 13],
        identity_tokens=[10, 11, -1, -2, -3, -4, 12, 13],
        segments=[text_a, media, text_b],
        text_token_index_by_pos={0: 0, 1: 1, 6: 2, 7: 3},
    )


def make_request(plan):
    return server.CompletionRequest(
        payload=server.CreateCompletionRequest(prompt="", max_tokens=1),
        prompt_text="",
        prompt_tokens=list(plan.identity_tokens),
        prompt_plan=plan,
        effective_max_len=len(plan.identity_tokens) + 1,
        internal_completion_count=1,
        prompt_visible_start=0,
        base_seq_id=0,
    )


def test_prompt_plan_reusable_boundaries_and_decoder_positions():
    plan = make_prompt_plan()

    assert plan.is_reusable_boundary(1)
    assert plan.is_reusable_boundary(2)
    assert not plan.is_reusable_boundary(3)
    assert not plan.is_reusable_boundary(5)
    assert plan.is_reusable_boundary(6)
    assert plan.is_reusable_boundary(7)

    assert plan.clamp_to_reusable_boundary(5) == 2
    assert plan.decoder_pos_up_to(0) == 0
    assert plan.decoder_pos_up_to(1) == 1
    assert plan.decoder_pos_up_to(2) == 2
    assert plan.decoder_pos_up_to(6) == 4
    assert plan.decoder_pos_up_to(8) == 6

    with pytest.raises(RuntimeError, match="inside"):
        plan.decoder_pos_up_to(3)


def test_media_identity_is_row_expanded_but_decoder_advance_is_atomic():
    plan = make_prompt_plan()

    assert plan.eval_token_count == 8
    assert plan.position_increments_up_to(6) == [1, 1, 0, 0, 0, 2]

    history = server.SequenceHistory()
    history.extend(0, plan.identity_tokens, plan.position_increments_up_to(plan.length))

    assert history.size == 8
    assert history.position_length(0) == 6
    assert history.position_length_for_prefix(0, 2) == 2
    assert history.position_length_for_prefix(0, 5) == 2
    assert history.position_length_for_prefix(0, 6) == 4


def test_mrope_positions_do_not_need_contiguous_grouping():
    media = make_prompt_plan().segments[1]

    assert media.rows_for_capacity(0, 2) == 2
    assert media.rows_for_capacity(2, 2) == 2

    embeddings, positions, increments = media.media_slice(1, 2)

    assert embeddings.shape == (2, 3)
    assert positions.shape == (8,)
    assert increments == [0, 0]


def test_sequence_cache_match_rejects_media_interior_prefix():
    plan = make_prompt_plan()
    request = make_request(plan)
    scheduler = server.CompletionScheduler.__new__(server.CompletionScheduler)

    assert not scheduler.is_sequence_cache_match_usable(
        request,
        server.SequenceCache.Match(tokens=tuple(plan.identity_tokens[:3])),
        resident_reuse_len=0,
    )
    assert scheduler.is_sequence_cache_match_usable(
        request,
        server.SequenceCache.Match(tokens=tuple(plan.identity_tokens[:6])),
        resident_reuse_len=0,
    )


def test_non_causal_media_schedules_atomically():
    plan = make_prompt_plan(non_causal=True)
    request = make_request(plan)
    request.prompt_cursor = 2
    scheduler = server.CompletionScheduler.__new__(server.CompletionScheduler)

    assert scheduler._pending_allocation_for_capacity(request, 0, 3) == 0
    assert scheduler._pending_allocation_for_capacity(request, 0, 4) == 4


def test_multimodal_prefill_is_not_mtp_eligible():
    plan = make_prompt_plan()
    request = make_request(plan)
    scheduler = server.CompletionScheduler.__new__(server.CompletionScheduler)
    scheduler.requests = {request.id: request}
    item = server.CompletionScheduler.BatchItem.prefill(
        request_id=request.id,
        seq_id=0,
        start_pos=0,
        llama_start_pos=0,
        tokens=[],
        identity_tokens=plan.identity_tokens,
        output_indices=[None] * len(plan.identity_tokens),
        output_positions=list(range(len(plan.identity_tokens))),
        position_increments=plan.position_increments_up_to(plan.length),
    )

    assert not scheduler.item_allows_mtp_processing(item)

    text_plan = server.PromptPlan.from_tokens("", [1, 2, 3])
    text_request = make_request(text_plan)
    scheduler.requests = {text_request.id: text_request}
    text_item = server.CompletionScheduler.BatchItem.prefill(
        request_id=text_request.id,
        seq_id=0,
        start_pos=0,
        llama_start_pos=0,
        tokens=[1, 2, 3],
        identity_tokens=[1, 2, 3],
        output_indices=[None, None, 0],
        output_positions=[0, 1, 2],
        position_increments=[1, 1, 1],
    )

    assert scheduler.item_allows_mtp_processing(text_item)
