from typing import Optional

from phoenix.client import Client
from phoenix.client.types import PromptVersion

from app.config import settings


class PromptManager:
    """
    Phoenix-backed prompt manager.

    Responsibilities:
    - Connect to Phoenix
    - Create prompt versions
    - Retrieve prompts
    - Retrieve specific prompt versions/tags
    - Format prompts with runtime variables
    """

    def __init__(self):
        self.client = Client(
            base_url=self._get_phoenix_base_url()
        )

    def _get_phoenix_base_url(self) -> str:
        """
        Convert the configured Phoenix endpoint into the
        HTTP endpoint required by the Phoenix Python client.
        """

        endpoint = settings.phoenix_endpoint.rstrip("/")

        # OTLP endpoint:
        # http://localhost:4317
        #
        # Phoenix Client endpoint:
        # http://localhost:6006
        if ":4317" in endpoint:
            return endpoint.replace(":4317", ":6006")

        return endpoint

    def create(
        self,
        name: str,
        messages: list[dict],
        model_name: str = "qwen3:8b",
        description: Optional[str] = None,
    ):
        """
        Create a new prompt version in Phoenix.

        Calling this again with the same prompt name
        creates another version.
        """

        version = PromptVersion(
            messages,
            model_name=model_name,
            model_provider="OLLAMA",
            template_format="MUSTACHE",
        )

        return self.client.prompts.create(
            name=name,
            version=version,
            prompt_description=description,
        )

    def get(
        self,
        name: str,
        version_id: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        """
        Retrieve a prompt from Phoenix.

        If no version/tag is supplied, retrieve the prompt
        identified by name.
        """

        if version_id:
            return self.client.prompts.get(
                prompt_version_id=version_id
            )

        if tag:
            return self.client.prompts.get(
                prompt_identifier=name,
                tag=tag,
            )

        return self.client.prompts.get(
            prompt_identifier=name
        )

    def format(
        self,
        prompt,
        variables: dict[str, str],
    ):
        """
        Format a Phoenix prompt using runtime variables.
        """

        return prompt.format(
            variables=variables
        )