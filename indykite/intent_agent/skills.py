"""Discovery, parsing, and cataloging of SKILL.md files under skills/."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from a2a.types import AgentSkill

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_CATALOG_PREAMBLE_PATH = Path(__file__).resolve().parent / "skill_catalog_preamble.txt"


def _coerce_str_list(value: object, delimiter: str = ",") -> list[str]:
    """Normalize a YAML frontmatter value (list, delimited str, or absent) to a list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [s.strip() for s in value.split(delimiter) if s.strip()]
    return []


def _parse_frontmatter(raw: str, location: Path) -> tuple[dict[str, Any], str] | None:
    """Split a SKILL.md's raw text into its parsed YAML frontmatter and body."""
    parts = re.split(r"^---\s*$", raw.strip(), maxsplit=2, flags=re.MULTILINE)
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        logger.warning("Skill file %s has invalid YAML frontmatter", location)
        return None
    if not isinstance(meta, dict):
        return None
    return meta, parts[2].strip()


def _build_skill_record(meta: dict[str, Any], body: str, location: Path) -> dict[str, Any] | None:
    """Build a skill record from frontmatter/body, or None if fields are missing."""
    name = meta.get("name") or meta.get("title")
    description = meta.get("description")
    if not name or not description:
        logger.warning("Skill file %s missing name or description", location)
        return None
    name = str(name).strip()
    description = str(description).strip()
    if not description:
        return None
    return {
        "name": name,
        "description": description,
        "location": str(location),
        "body": body,
        "tags": _coerce_str_list(meta.get("tags")),
        "examples": _coerce_str_list(meta.get("examples"), delimiter="\n"),
    }


def _parse_skill_file(location: Path) -> dict[str, Any] | None:
    """Parse a SKILL.md file into a skill record, or None if invalid."""
    try:
        raw = location.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read skill file %s", location)
        return None
    if not raw.strip():
        return None
    parsed = _parse_frontmatter(raw, location)
    if parsed is None:
        return None
    meta, body = parsed
    return _build_skill_record(meta, body, location)


def discover_skills() -> list[dict[str, Any]]:
    """Discover skills under _SKILLS_DIR: subdirectories containing a SKILL.md."""
    skills: list[dict[str, Any]] = []
    if not _SKILLS_DIR.is_dir():
        logger.info("Skills directory not found: %s", _SKILLS_DIR)
        return skills
    seen: set[str] = set()
    try:
        entries = sorted(_SKILLS_DIR.iterdir())
    except OSError:
        logger.warning("Error scanning skills dir %s", _SKILLS_DIR)
        return skills
    for entry in entries:
        if entry.name.startswith(".") or entry.name in {"node_modules", ".git"}:
            continue
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        record = _parse_skill_file(skill_md)
        if not record:
            continue
        name = record["name"]
        if name in seen:
            logger.warning("Duplicate skill name %r, skipping %s", name, skill_md)
            continue
        seen.add(name)
        skills.append(record)
    skills.sort(key=lambda s: str(s["name"]))
    return skills


def build_skill_catalog_prompt(skills: list[dict[str, Any]]) -> str:
    """Build the tier-1 skill catalog for the system prompt (progressive disclosure)."""
    if not skills:
        return ""
    preamble = _CATALOG_PREAMBLE_PATH.read_text(encoding="utf-8").strip("\n")
    lines = [*preamble.splitlines(), "", "<available_skills>"]
    lines.extend(
        f"  <skill><name>{skill['name']}</name><description>{skill['description']}</description></skill>"
        for skill in skills
    )
    lines.append("</available_skills>")
    return "\n".join(lines)


def build_agent_skill_catalog(skills: list[dict[str, Any]]) -> list[AgentSkill]:
    """Map discovered skill records onto A2A AgentSkill entries for the agent card."""
    catalog = [
        AgentSkill(
            id=str(s["name"]).replace(" ", "-").lower()[:64].strip("-"),
            name=str(s["name"]).replace("-", " ").title(),
            description=s["description"],
            tags=s.get("tags") or ["intent", "nlp", "extraction"],
            examples=s.get("examples") or [],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        )
        for s in skills
    ]
    if catalog:
        return catalog
    return [
        AgentSkill(
            id="extract-intent",
            name="Extract Intent",
            description=(
                "Extract intent from a provided input prompt text. Returns a "
                "structured description of the user's goal, action, and key entities."
            ),
            tags=["intent", "nlp", "extraction"],
            examples=[],
            input_modes=["text/plain"],
            output_modes=["text/plain"],
        ),
    ]
