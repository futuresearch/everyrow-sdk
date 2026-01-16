import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from everyrow.api_utils import handle_response
from everyrow.generated.api.default import (
    create_session_endpoint_sessions_create_post,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models.create_session_request import CreateSessionRequest


class Session:
    """Session object containing client and session_id."""

    def __init__(self, client: AuthenticatedClient, session_id: UUID):
        self.client = client
        self.session_id = session_id

    def get_url(self) -> str:
        """Get the URL to view this session in the web interface.

        Returns:
            str: URL to the session in the format {EVERYROW_BASE_URL}/sessions/{session_id}
                 Defaults to https://everyrow.io/sessions/{session_id} if EVERYROW_BASE_URL
                 is not set in environment variables.
        """
        base_url = os.environ.get("EVERYROW_BASE_URL", "https://everyrow.io")
        return f"{base_url}/sessions/{self.session_id}"


@asynccontextmanager
async def create_session(
    client: AuthenticatedClient,
    name: str | None = None,
) -> AsyncGenerator[Session, None]:
    """Create a new session and yield it as an async context manager.

    Args:
        client: Authenticated client to use for session creation.
                The client should already be in an async context manager.
        name: Name for the session. If not provided, defaults to
              "everyrow-sdk-session-{timestamp}".
    """
    response = await create_session_endpoint_sessions_create_post.asyncio(
        client=client,
        body=CreateSessionRequest(name=name or f"everyrow-sdk-session-{datetime.now().isoformat()}"),
    )
    response = handle_response(response)
    session = Session(client=client, session_id=response.session_id)
    yield session
