"""Schema versioning. Bump the major part on any breaking change to a contract."""

SCHEMA_VERSION = "1.1.0"


def major(version: str) -> int:
    """Return the major component of a 'MAJOR.MINOR.PATCH' version string."""
    return int(version.split(".")[0])
