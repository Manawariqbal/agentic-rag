from dataclasses import dataclass
from datetime import datetime


@dataclass
class PromptVersion:

    name: str
    version: int
    template: str
    created_at: datetime


class PromptManager:

    def __init__(self):

        self._prompts: dict[
            str,
            list[PromptVersion]
        ] = {}

    def register(
        self,
        name: str,
        template: str,
    ) -> PromptVersion:

        versions = self._prompts.setdefault(
            name,
            []
        )

        version = PromptVersion(
            name=name,
            version=len(versions) + 1,
            template=template,
            created_at=datetime.utcnow(),
        )

        versions.append(version)

        return version

    def get(
        self,
        name: str,
        version: int | None = None,
    ) -> PromptVersion:

        versions = self._prompts.get(name)

        if not versions:
            raise KeyError(
                f"Prompt not found: {name}"
            )

        if version is None:
            return versions[-1]

        for prompt in versions:
            if prompt.version == version:
                return prompt

        raise KeyError(
            f"Prompt version not found: "
            f"{name}:{version}"
        )

    def list_versions(
        self,
        name: str,
    ) -> list[PromptVersion]:

        return self._prompts.get(
            name,
            []
        )