"""Tests for cloud runner job planning."""

from neuropress.cloud.vast_runner import resolve_gpu_name

# ---------------------------------------------------------------------------
# resolve_gpu_name — in sweep mode the parent process rents the instance, so
# whatever GPU it resolves is the GPU the job actually gets.  Before this was
# centralized, the parent read only the per-entry `gpu:` field and ignored
# --gpu-name, so the parent could rent a different GPU than the subprocess was
# told to use.  On 2026-08-03 that meant --gpu-name RTX_4090 silently kept
# renting the cheapest RTX 3090.
# ---------------------------------------------------------------------------


def test_entry_gpu_overrides_cli_flag():
    """A per-entry `gpu:` is more specific than a sweep-wide flag."""
    assert resolve_gpu_name("RTX_4090", "RTX_3090") == "RTX_4090"


def test_cli_flag_applies_when_entry_has_no_gpu():
    assert resolve_gpu_name(None, "RTX_4090") == "RTX_4090"


def test_none_when_neither_specified():
    assert resolve_gpu_name(None, None) is None


def test_entry_gpu_applies_without_cli_flag():
    assert resolve_gpu_name("RTX_A5000", None) == "RTX_A5000"


# ---------------------------------------------------------------------------
# sweep_instance_override — --instance-id / --resume were silently ignored in
# sweep mode: _run_sweep always called _connect(), which rents.  On 2026-08-05
# that turned a resume of a preserved instance into a fresh rent attempt that
# then failed on "No GPU offers match query", while the preserved instance kept
# billing.  Same defect class as --gpu-name being ignored in sweep mode.
# ---------------------------------------------------------------------------


def test_no_override_means_rent_normally():
    from neuropress.cloud.vast_runner import sweep_instance_override

    assert sweep_instance_override(3, None) is None


def test_single_entry_sweep_attaches_to_given_instance():
    from neuropress.cloud.vast_runner import sweep_instance_override

    assert sweep_instance_override(1, 46882704) == 46882704


# ---------------------------------------------------------------------------
# select_offer — retrying after an SSH timeout must not re-rent the machine
# that just failed.  _rent took offers[0] (cheapest) with no memory of past
# failures, and the offer search is deterministic, so all 3 attempts landed on
# the same bad host.  On 2026-08-07 offer 43228806 was rented twice in a row
# after a 40-minute SSH timeout: 3 attempts would have burned ~2 h of billing
# on one unreachable machine and returned no result.
# ---------------------------------------------------------------------------


def _offers(*specs):
    return [{"id": oid, "dph_total": dph} for oid, dph in specs]


def test_picks_cheapest_when_nothing_excluded():
    from neuropress.cloud.vast_runner import select_offer

    offers = _offers((1, 0.40), (2, 0.50))
    assert select_offer(offers, set())["id"] == 1


def test_skips_excluded_offer():
    from neuropress.cloud.vast_runner import select_offer

    offers = _offers((1, 0.40), (2, 0.50))
    assert select_offer(offers, {1})["id"] == 2


def test_returns_none_when_all_excluded():
    """Caller decides whether to fall back or fail; the picker does not guess."""
    from neuropress.cloud.vast_runner import select_offer

    offers = _offers((1, 0.40), (2, 0.50))
    assert select_offer(offers, {1, 2}) is None


def test_no_offers_at_all_is_none():
    from neuropress.cloud.vast_runner import select_offer

    assert select_offer([], set()) is None


def test_offer_ids_compare_across_int_and_str():
    """vast returns ids as ints here and strings there; exclusion must survive."""
    from neuropress.cloud.vast_runner import select_offer

    offers = [{"id": "1", "dph_total": 0.40}, {"id": 2, "dph_total": 0.50}]
    assert select_offer(offers, {1})["id"] == 2


# ---------------------------------------------------------------------------
# check_local_cache_dir — a sweep's `cache_dir:` is a LOCAL path (the remote is
# always _REMOTE_CACHE).  Writing a remote-looking path there, e.g.
# `/workspace/cache_vl8`, makes every sync raise PermissionError('/workspace')
# because a non-root user cannot create it.  The run then dies before writing
# eval_report.json and before teardown, leaving the instance billing.
#
# This silently cost three runs their report (2026-08-05..07: v16 vl8_probe 243
# errors, v17 vl8_hadamard 2350, v18 vl8_gptq1 232); the one run that omitted
# cache_dir and got the auto-derived local path had 0.  Fail at parse time with
# one clear message instead of thousands of sync errors two hours later.
# ---------------------------------------------------------------------------


def test_writable_existing_dir_is_accepted(tmp_path):
    from neuropress.cloud.vast_runner import check_local_cache_dir

    assert check_local_cache_dir(tmp_path) is None


def test_creatable_dir_under_writable_parent_is_accepted(tmp_path):
    """The cache dir itself need not exist yet - sync creates it."""
    from neuropress.cloud.vast_runner import check_local_cache_dir

    assert check_local_cache_dir(tmp_path / "cache_v19") is None


def test_uncreatable_root_level_dir_is_rejected():
    from neuropress.cloud.vast_runner import check_local_cache_dir

    msg = check_local_cache_dir("/workspace/cache_vl8")
    assert msg is not None
    assert "/workspace" in msg


def test_rejection_message_names_the_remote_confusion(tmp_path):
    """The message must say cache_dir is local, or the fix is not obvious."""
    from neuropress.cloud.vast_runner import check_local_cache_dir

    msg = check_local_cache_dir("/workspace/cache_vl8")
    assert "local" in msg.lower()


def test_disabled_cache_is_not_a_path(tmp_path):
    from neuropress.cloud.vast_runner import check_local_cache_dir

    assert check_local_cache_dir(None) is None
    assert check_local_cache_dir(False) is None


def test_multi_entry_sweep_with_instance_id_is_rejected():
    """One instance cannot serve N entries; failing loudly beats renting silently."""
    import pytest

    from neuropress.cloud.vast_runner import sweep_instance_override

    with pytest.raises(ValueError, match="instance-id"):
        sweep_instance_override(2, 46882704)
