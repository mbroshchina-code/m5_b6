import uuid
from unittest.mock import AsyncMock

import pytest

from bot.services.broadcast_worker import BroadcastWorker


@pytest.mark.asyncio
async def test_broadcast_worker_sends_and_marks_job():
    job_id = uuid.uuid4()
    backend = AsyncMock()
    backend.get_pending_broadcasts.return_value = [
        {
            "id": str(job_id),
            "message": "Готово",
            "interface": "telegram",
            "recipients": ["42", "43"],
        }
    ]
    bot = AsyncMock()

    await BroadcastWorker(backend).process_once(bot)

    assert bot.send_message.await_count == 2
    backend.mark_broadcast.assert_awaited_once_with(job_id, "sent")
