"""Unit tests for everyrow.task — progress polling, callbacks, ETA, JSONL logging."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from everyrow.constants import EveryrowError
from everyrow.generated.models import (
    PublicTaskType,
    TaskProgressInfo,
    TaskStatus,
    TaskStatusResponse,
)
from everyrow.task import (
    _format_eta,
    _get_progress,
    await_task_completion,
)


def _make_status(
    status: TaskStatus = TaskStatus.PENDING,
    progress: TaskProgressInfo | None = None,
    error: str | None = None,
) -> TaskStatusResponse:
    return TaskStatusResponse(
        task_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        status=status,
        task_type=PublicTaskType.AGENT,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        progress=progress,
        error=error,
    )


class TestGetProgress:
    def test_returns_none_when_missing(self):
        status = _make_status()
        assert _get_progress(status) is None

    def test_returns_progress_info(self):
        status = _make_status(
            progress=TaskProgressInfo(
                pending=0,
                running=1,
                completed=9,
                failed=0,
                total=10,
            )
        )
        info = _get_progress(status)
        assert info is not None
        assert info.completed == 9
        assert info.running == 1


# --- ETA tests ---


class TestFormatEta:
    def test_no_eta_when_zero_completed(self):
        assert _format_eta(0, 10, 5.0) == ""

    def test_no_eta_when_zero_elapsed(self):
        assert _format_eta(5, 10, 0.0) == ""

    def test_eta_calculation(self):
        # 5 of 10 done in 10s → rate = 0.5/s → 5 remaining → ~10s
        result = _format_eta(5, 10, 10.0)
        assert result == "~10s remaining"

    def test_eta_rounds(self):
        # 3 of 10 done in 9s → rate = 0.333/s → 7 remaining → ~21s
        result = _format_eta(3, 10, 9.0)
        assert result == "~21s remaining"


# --- await_task_completion tests ---


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Replace asyncio.sleep with a no-op to avoid real waits."""
    monkeypatch.setattr("everyrow.task.asyncio.sleep", AsyncMock())


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def jsonl_tmp(tmp_path, monkeypatch):
    """Redirect JSONL log to a temp file."""
    everyrow_dir = tmp_path / ".everyrow"
    everyrow_dir.mkdir()
    log_path = everyrow_dir / "progress.jsonl"
    monkeypatch.setattr(
        "everyrow.task.os.path.expanduser",
        lambda p: str(log_path) if "progress.jsonl" in p else p,
    )
    return log_path


@pytest.mark.asyncio
async def test_immediate_completion(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """Task already completed on first poll — no progress output."""
    mock_status = mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
    )
    mock_status.return_value = _make_status(TaskStatus.COMPLETED)

    result = await await_task_completion(uuid.uuid4(), mock_client)
    assert result.status == TaskStatus.COMPLETED
    mock_status.assert_called_once()


@pytest.mark.asyncio
async def test_progress_callback_fires_on_change(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """on_progress callback fires when snapshot changes."""
    task_id = uuid.uuid4()
    callback = MagicMock()

    statuses = [
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=5, running=0, completed=0, failed=0, total=5),
        ),
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=3, running=2, completed=0, failed=0, total=5),
        ),
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=1, running=2, completed=2, failed=0, total=5),
        ),
        _make_status(
            TaskStatus.COMPLETED,
            TaskProgressInfo(pending=0, running=0, completed=5, failed=0, total=5),
        ),
    ]

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        side_effect=statuses,
    )

    await await_task_completion(task_id, mock_client, on_progress=callback)

    # All 4 statuses have different snapshots, so callback fires 4 times
    assert callback.call_count == 4
    # Verify the first call got the initial progress
    first_call = callback.call_args_list[0][0][0]
    assert isinstance(first_call, TaskProgressInfo)
    assert first_call.pending == 5
    # Verify the last call got the final progress
    last_call = callback.call_args_list[-1][0][0]
    assert last_call.completed == 5


@pytest.mark.asyncio
async def test_callback_skips_duplicate_snapshot(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """on_progress callback does NOT fire when snapshot is unchanged."""
    task_id = uuid.uuid4()
    callback = MagicMock()

    same_progress = TaskProgressInfo(
        pending=3,
        running=2,
        completed=0,
        failed=0,
        total=5,
    )
    statuses = [
        _make_status(TaskStatus.PENDING, same_progress),
        _make_status(TaskStatus.PENDING, same_progress),  # duplicate
        _make_status(TaskStatus.PENDING, same_progress),  # duplicate
        _make_status(
            TaskStatus.COMPLETED,
            TaskProgressInfo(pending=0, running=0, completed=5, failed=0, total=5),
        ),
    ]

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        side_effect=statuses,
    )

    await await_task_completion(task_id, mock_client, on_progress=callback)

    # Only 2 unique snapshots (initial + final), so 2 calls
    assert callback.call_count == 2


@pytest.mark.asyncio
async def test_jsonl_log_written(mocker, mock_client, jsonl_tmp):
    """Progress entries are logged to the JSONL file."""
    task_id = uuid.uuid4()

    statuses = [
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=5, running=0, completed=0, failed=0, total=5),
        ),
        _make_status(
            TaskStatus.COMPLETED,
            TaskProgressInfo(pending=0, running=0, completed=5, failed=0, total=5),
        ),
    ]

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        side_effect=statuses,
    )

    await await_task_completion(task_id, mock_client)

    assert jsonl_tmp.exists()
    lines = [json.loads(line) for line in jsonl_tmp.read_text().strip().split("\n")]
    # Expect: start entry, 2 progress entries (different snapshots), done entry
    assert len(lines) >= 3
    assert lines[0]["step"] == "start"
    assert lines[0]["total"] == 5
    assert lines[-1]["step"] == "done"
    assert lines[-1]["succeeded"] == 5


@pytest.mark.asyncio
async def test_stderr_output_format(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
    capsys,
):
    """Default progress output writes to stderr in expected format."""
    task_id = uuid.uuid4()

    statuses = [
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=3, running=2, completed=0, failed=0, total=5),
        ),
        _make_status(
            TaskStatus.COMPLETED,
            TaskProgressInfo(pending=0, running=0, completed=5, failed=0, total=5),
        ),
    ]

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        side_effect=statuses,
    )

    await await_task_completion(task_id, mock_client)

    captured = capsys.readouterr()
    # Progress goes to stderr
    assert "[0/5]" in captured.err or "[5/5]" in captured.err
    assert "running" in captured.err
    assert "Done" in captured.err


@pytest.mark.asyncio
async def test_failed_task_raises(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """A task that ends in FAILED raises EveryrowError."""
    task_id = uuid.uuid4()

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        return_value=_make_status(TaskStatus.FAILED, error="Something went wrong"),
    )

    with pytest.raises(EveryrowError, match="Task failed"):
        await await_task_completion(task_id, mock_client)


@pytest.mark.asyncio
async def test_revoked_task_raises(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """A task that ends in REVOKED raises EveryrowError."""
    task_id = uuid.uuid4()

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        return_value=_make_status(TaskStatus.REVOKED),
    )

    with pytest.raises(EveryrowError, match="revoked"):
        await await_task_completion(task_id, mock_client)


@pytest.mark.asyncio
async def test_retries_on_transient_error(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """Transient status fetch errors are retried up to max_retries."""
    task_id = uuid.uuid4()

    call_count = 0

    async def status_with_error(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("network blip")
        return _make_status(TaskStatus.COMPLETED)

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        side_effect=status_with_error,
    )

    result = await await_task_completion(task_id, mock_client)
    assert result.status == TaskStatus.COMPLETED
    assert call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_retries_exhausted_raises(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
):
    """Exceeding max_retries raises EveryrowError."""
    task_id = uuid.uuid4()

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        side_effect=ConnectionError("persistent failure"),
    )

    with pytest.raises(EveryrowError, match="retries"):
        await await_task_completion(task_id, mock_client)


@pytest.mark.asyncio
async def test_session_url_in_output(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
    capsys,
):
    """Session URL is printed to stderr when provided."""
    task_id = uuid.uuid4()

    statuses = [
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=5, running=0, completed=0, failed=0, total=5),
        ),
        _make_status(
            TaskStatus.COMPLETED,
            TaskProgressInfo(pending=0, running=0, completed=5, failed=0, total=5),
        ),
    ]

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        side_effect=statuses,
    )

    await await_task_completion(
        task_id, mock_client, session_url="https://everyrow.io/sessions/abc123"
    )

    captured = capsys.readouterr()
    assert "https://everyrow.io/sessions/abc123" in captured.err


@pytest.mark.asyncio
async def test_final_summary_line(
    mocker,
    mock_client,
    jsonl_tmp,  # noqa: ARG001
    capsys,
):
    """Completion prints a summary with succeeded/failed counts."""
    task_id = uuid.uuid4()

    statuses = [
        _make_status(
            TaskStatus.PENDING,
            TaskProgressInfo(pending=2, running=1, completed=0, failed=0, total=3),
        ),
        _make_status(
            TaskStatus.COMPLETED,
            TaskProgressInfo(pending=0, running=0, completed=2, failed=1, total=3),
        ),
    ]

    mocker.patch(
        "everyrow.task.get_task_status_tasks_task_id_status_get.asyncio",
        new_callable=AsyncMock,
        side_effect=statuses,
    )

    await await_task_completion(task_id, mock_client)

    captured = capsys.readouterr()
    assert "2 succeeded" in captured.err
    assert "1 failed" in captured.err
