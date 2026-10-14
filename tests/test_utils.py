import os

import pytest

import llama_cpp._utils as utils


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _proc_cgroup(tmp_path, text):
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(text)
    return str(proc)


def test_cgroup_v2_quota(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "cpu.max", "200000 100000\n")
    proc = _proc_cgroup(tmp_path, "0::/\n")
    assert utils._cgroup_cpu_limit(proc, str(root)) == 2.0


def test_cgroup_v2_no_quota(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "cpu.max", "max 100000\n")
    proc = _proc_cgroup(tmp_path, "0::/\n")
    assert utils._cgroup_cpu_limit(proc, str(root)) is None


def test_cgroup_v2_uses_tightest_ancestor(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "kubepods.slice" / "cpu.max", "150000 100000\n")
    _write(root / "kubepods.slice" / "pod" / "cpu.max", "max 100000\n")
    _write(root / "other.slice" / "cpu.max", "50000 100000\n")
    proc = _proc_cgroup(tmp_path, "0::/kubepods.slice/pod\n")
    assert utils._cgroup_cpu_limit(proc, str(root)) == 1.5


def test_cgroup_v1_quota(tmp_path):
    root = tmp_path / "cgroup"
    # Without a cgroup namespace the container's own cgroup is the mount root
    # and the host path from /proc/self/cgroup does not exist under it.
    _write(root / "cpu,cpuacct" / "cpu.cfs_quota_us", "50000\n")
    _write(root / "cpu,cpuacct" / "cpu.cfs_period_us", "100000\n")
    proc = _proc_cgroup(tmp_path, "12:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n")
    assert utils._cgroup_cpu_limit(proc, str(root)) == 0.5


def test_cgroup_v1_no_quota(tmp_path):
    root = tmp_path / "cgroup"
    _write(root / "cpu" / "cpu.cfs_quota_us", "-1\n")
    _write(root / "cpu" / "cpu.cfs_period_us", "100000\n")
    proc = _proc_cgroup(tmp_path, "4:cpu,cpuacct:/\n")
    assert utils._cgroup_cpu_limit(proc, str(root)) is None


def test_cgroup_missing_files(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    assert utils._cgroup_cpu_limit(str(tmp_path / "missing"), str(root)) is None
    proc = _proc_cgroup(tmp_path, "0::/\n")
    assert utils._cgroup_cpu_limit(proc, str(root)) is None


@pytest.mark.parametrize(
    "limit, expected",
    [(None, 20), (2.0, 2), (1.5, 2), (0.5, 1), (64.0, 20)],
)
def test_default_cpu_count_caps_affinity_at_quota(monkeypatch, limit, expected):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(20)), False)
    monkeypatch.setattr(utils, "_cgroup_cpu_limit", lambda: limit)
    assert utils.default_cpu_count() == expected


def test_default_cpu_count_uses_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, False)
    monkeypatch.setattr(utils.multiprocessing, "cpu_count", lambda: 20)
    monkeypatch.setattr(utils, "_cgroup_cpu_limit", lambda: None)
    assert utils.default_cpu_count() == 2


def test_default_cpu_count_without_affinity(monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(utils.multiprocessing, "cpu_count", lambda: 8)
    monkeypatch.setattr(utils, "_cgroup_cpu_limit", lambda: None)
    assert utils.default_cpu_count() == 8
