class TechMatchError(Exception):
    """Base exception for all TechMatch errors."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n  Hint: {self.hint}"
        return self.message



class InsufficientSkillsError(TechMatchError):
    """Raised when the user provides fewer skills than the required minimum."""

    def __init__(self, provided: int, required: int) -> None:
        super().__init__(
            message=f"Got {provided} skill(s), but at least {required} are required.",
            hint="Select more skills from the menu or type them manually.",
        )
        self.provided = provided
        self.required = required


class EmptyInputError(TechMatchError):
    """Raised when the user submits an empty or whitespace-only skill list."""

    def __init__(self) -> None:
        super().__init__(
            message="No skills were provided.",
            hint="Enter skill numbers (e.g. 1,3,7) or type skill names directly.",
        )


class InvalidSkillIndexError(TechMatchError):
    """Raised when a numeric index is out of range for the skill catalogue."""

    def __init__(self, index: int, max_index: int) -> None:
        super().__init__(
            message=f"Skill number {index} does not exist (valid range: 1–{max_index}).",
            hint="Check the skill menu and enter a number within the listed range.",
        )
        self.index = index
        self.max_index = max_index


class DuplicateSkillError(TechMatchError):
    """Raised when the same skill is added more than once (non-fatal, just a warning)."""

    def __init__(self, skill: str) -> None:
        super().__init__(
            message=f"'{skill}' has already been added.",
            hint="Each skill only needs to be listed once.",
        )
        self.skill = skill



class EmptyCorpusError(TechMatchError):
    """Raised when the job-role corpus is empty or None."""

    def __init__(self) -> None:
        super().__init__(
            message="The job-role corpus is empty.",
            hint="Ensure data.py contains at least one entry in JOB_CORPUS.",
        )


class MissingRoleFieldError(TechMatchError):
    """Raised when a job-role dict is missing a required key."""

    def __init__(self, role_id: str, field: str) -> None:
        super().__init__(
            message=f"Job role '{role_id}' is missing required field '{field}'.",
            hint=f"Add the '{field}' key to every entry in JOB_CORPUS inside data.py.",
        )
        self.role_id = role_id
        self.field = field


class ColdStartError(TechMatchError):
    """
    Raised when the user's vector is zero (no vocabulary overlap with corpus).
    The engine catches this internally and switches to the trending fallback,
    so this exception is only surfaced if the fallback itself also fails.
    """

    def __init__(self) -> None:
        super().__init__(
            message="Cold Start: your skills produced a zero-magnitude vector.",
            hint="Add more skills that appear in the job-role corpus, or check data.py.",
        )



class ConfigError(TechMatchError):
    """Raised for missing or malformed configuration values."""

    def __init__(self, key: str, detail: str = "") -> None:
        super().__init__(
            message=f"Configuration error for key '{key}'. {detail}".strip(),
            hint="Review the settings at the top of config.py.",
        )
        self.key = key


class TerminalNotSupportedError(TechMatchError):
    """Raised when the terminal cannot render ANSI colour codes."""

    def __init__(self) -> None:
        super().__init__(
            message="This terminal does not support ANSI colour codes.",
            hint="Run with --no-color flag, or use a modern terminal (Windows Terminal, iTerm2, etc.).",
        )
