import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

from everyrow.api_utils import create_client, handle_response
from everyrow.constants import DEFAULT_EVERYROW_APP_URL
from everyrow.generated.api.sessions import (
    create_session_endpoint_sessions_post,
)
from everyrow.generated.client import AuthenticatedClient
from everyrow.generated.models.create_session import CreateSession


def get_session_url(session_id: UUID) -> str:
    base_url = os.environ.get("EVERYROW_APP_URL", DEFAULT_EVERYROW_APP_URL).rstrip("/")
    return f"{base_url}/sessions/{session_id}"


class Session:
    """Session object containing client and session_id."""

    def __init__(self, client: AuthenticatedClient, session_id: UUID):
        self.client = client
        self.session_id = session_id

    def get_url(self) -> str:
        """Get the URL to view this session in the web interface."""
        return get_session_url(self.session_id)


@asynccontextmanager
async def create_session(
    client: AuthenticatedClient | None = None,
    name: str | None = None,
) -> AsyncGenerator[Session, None]:
    """Create a new session and yield it as an async context manager.

    Args:
        client: Optional authenticated client. If not provided, one will be created
                automatically using the EVERYROW_API_KEY environment variable and
                managed within this context manager.
        name: Name for the session. If not provided, defaults to
              "everyrow-sdk-session-{timestamp}".

    Example:
        # With explicit client (client lifecycle managed externally)
        async with create_client() as client:
            async with create_session(client=client, name="My Session") as session:
                ...

        # Without client (client created and managed internally)
        async with create_session(name="My Session") as session:
            ...
    """
    owns_client = client is None
    if owns_client:
        client = create_client()
        await client.__aenter__()

    try:
        response = await create_session_endpoint_sessions_post.asyncio(
            client=client,
            body=CreateSession(
                name=name or f"everyrow-sdk-session-{datetime.now().isoformat()}"
            ),
        )
        response = handle_response(response)
        session = Session(client=client, session_id=response.session_id)
        yield session
    finally:
        if owns_client:
            await client.__aexit__()
