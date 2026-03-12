import re
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import aiofiles
import aiofiles.os
from utils import sanitize_filename
from shared_functions import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    DEFAULT_HALSTEAD_THRESHOLDS,
    DEFAULT_MAINTAINABILITY_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# Global write lock for thread safety
write_lock = asyncio.Lock()


class DocumentationError(Exception):
    pass


class FileWriteError(DocumentationError):
    pass


@dataclass
class BadgeConfig:
    """Configuration for badge generation."""
    metric_name: str
    value: Union[int, float]
    thresholds: Dict[str, int]
    logo: Optional[str] = None
    style: str = "flat-square"
    label_color: Optional[str] = None

    def get_color(self) -> str:
        low, medium, high = (
            self.thresholds["low"],
            self.thresholds["medium"],
            self.thresholds["high"],
        )
        if self.value <= low:
            return "success"
        elif self.value <= medium:
            return "yellow"
        else:
            return "critical"


class BadgeGenerator:
    """Badge generation for metrics."""

    _badge_template = (
        "![{label}](https://img.shields.io/badge/"
        "{encoded_label}-{value}-{color}"
        "?style={style}{logo_part}{label_color_part})"
    )

    @classmethod
    def generate_badge(cls, config: BadgeConfig) -> str:
        try:
            label = config.metric_name.replace("_", " ").title()
            encoded_label = label.replace(" ", "%20")
            color = config.get_color()
            value = f"{config.value:.2f}" if isinstance(config.value, float) else str(config.value)
            logo_part = f"&logo={config.logo}" if config.logo else ""
            label_color_part = f"&labelColor={config.label_color}" if config.label_color else ""
            return cls._badge_template.format(
                label=label, encoded_label=encoded_label, value=value,
                color=color, style=config.style, logo_part=logo_part,
                label_color_part=label_color_part,
            )
        except Exception as e:
            logger.error(f"Error generating badge: {e}")
            return ""

    @classmethod
    def generate_all_badges(cls, metrics: Dict[str, Any]) -> str:
        badges = []
        try:
            if (complexity := metrics.get("complexity")) is not None:
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Complexity", value=complexity,
                    thresholds=DEFAULT_COMPLEXITY_THRESHOLDS, logo="codeClimate",
                )))
            if halstead := metrics.get("halstead"):
                logo_map = {"volume": "stackOverflow", "difficulty": "codewars", "effort": "atlassian"}
                for name, key in [("Volume", "volume"), ("Difficulty", "difficulty"), ("Effort", "effort")]:
                    if (val := halstead.get(key)) is not None:
                        badges.append(cls.generate_badge(BadgeConfig(
                            metric_name=name, value=val,
                            thresholds=DEFAULT_HALSTEAD_THRESHOLDS[key],
                            logo=logo_map[key],
                        )))
            if (mi := metrics.get("maintainability_index")) is not None:
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Maintainability", value=mi,
                    thresholds=DEFAULT_MAINTAINABILITY_THRESHOLDS, logo="codeclimate",
                )))
            return " ".join(badges)
        except Exception as e:
            logger.error(f"Error generating badges: {e}")
            return ""


class MarkdownFormatter:
    """Markdown formatting utilities."""

    @staticmethod
    def truncate_description(description: str, max_length: int = 100, ellipsis: str = "...") -> str:
        if not description or len(description) <= max_length:
            return description
        truncated = description[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        return truncated + ellipsis

    @staticmethod
    def sanitize_text(text: str) -> str:
        special_chars = r'[`*_{}[$()#+\-.!|]'
        text = re.sub(special_chars, lambda m: '\\' + m.group(0), str(text))
        text = text.replace('\n', ' ').replace('\r', '')
        return ' '.join(text.split())

    def format_table(self, headers: List[str], rows: List[List[Any]], alignment: Optional[List[str]] = None) -> str:
        if not headers or not rows:
            return ""
        try:
            headers = [self.sanitize_text(str(h)) for h in headers]
            if not alignment:
                alignment = ['left'] * len(headers)
            align_map = {'left': ':---', 'center': ':---:', 'right': '---:'}
            separators = [align_map.get(a, ':---') for a in alignment]
            table_lines = [
                f"| {' | '.join(headers)} |",
                f"| {' | '.join(separators)} |",
            ]
            for row in rows:
                row = (row + [''] * len(headers))[:len(headers)]
                sanitized = [self.sanitize_text(str(c)) for c in row]
                table_lines.append(f"| {' | '.join(sanitized)} |")
            return '\n'.join(table_lines)
        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            return ""


def generate_markdown_report(
    structured_response: Dict[str, Any],
    file_path: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate Markdown documentation from structured LLM response."""
    sections = []
    formatter = MarkdownFormatter()

    # Header with badges
    sections.append(f"# {Path(file_path).name}")
    if metrics:
        badge_str = BadgeGenerator.generate_all_badges(metrics)
        if badge_str:
            sections.append(badge_str)

    # Summary
    if summary := structured_response.get("summary"):
        sections.append(f"## Summary\n\n{summary}")

    # Functions table
    if functions := structured_response.get("functions"):
        headers = ["Function", "Args", "Description", "Complexity"]
        rows = []
        for f in functions:
            args_list = f.get("args", [])
            args_str = ", ".join(
                a if isinstance(a, str) else a.get("name", "") for a in args_list
            )
            desc = formatter.truncate_description(f.get("docstring", ""), 100)
            rows.append([f.get("name", ""), args_str, desc, str(f.get("complexity", ""))])
        sections.append(f"## Functions\n\n{formatter.format_table(headers, rows)}")

    # Classes table
    if classes := structured_response.get("classes"):
        headers = ["Class", "Description", "Methods"]
        rows = []
        for cls in classes:
            methods = cls.get("methods", [])
            method_names = ", ".join(m.get("name", "") for m in methods)
            desc = formatter.truncate_description(cls.get("docstring", ""), 100)
            rows.append([cls.get("name", ""), desc, method_names])
        sections.append(f"## Classes\n\n{formatter.format_table(headers, rows)}")

    # Variables table
    if variables := structured_response.get("variables"):
        headers = ["Variable", "Type", "Description"]
        rows = [[v.get("name", ""), v.get("type", ""), v.get("description", "")] for v in variables]
        sections.append(f"## Variables\n\n{formatter.format_table(headers, rows)}")

    # Constants table
    if constants := structured_response.get("constants"):
        headers = ["Constant", "Type", "Description"]
        rows = [[c.get("name", ""), c.get("type", ""), c.get("description", "")] for c in constants]
        sections.append(f"## Constants\n\n{formatter.format_table(headers, rows)}")

    # Metrics section
    if metrics:
        metrics_lines = ["## Metrics\n"]
        if c := metrics.get("complexity"):
            metrics_lines.append(f"- **Cyclomatic Complexity:** {c}")
        if mi := metrics.get("maintainability_index"):
            metrics_lines.append(f"- **Maintainability Index:** {mi:.1f}")
        if h := metrics.get("halstead"):
            metrics_lines.append(f"- **Halstead Volume:** {h.get('volume', 0):.1f}")
            metrics_lines.append(f"- **Halstead Difficulty:** {h.get('difficulty', 0):.1f}")
            metrics_lines.append(f"- **Halstead Effort:** {h.get('effort', 0):.1f}")
        sections.append("\n".join(metrics_lines))

    return "\n\n".join(sections)


async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str,
    project_id: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Writes documentation to JSON and Markdown files."""
    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        async with write_lock:
            project_output_dir = Path(output_dir) / project_id
            await aiofiles.os.makedirs(project_output_dir, exist_ok=True)

            relative_path = Path(file_path).relative_to(repo_root)
            safe_parts = [sanitize_filename(part) for part in relative_path.parts]
            output_subdir = project_output_dir / Path(*safe_parts[:-1]) if len(safe_parts) > 1 else project_output_dir
            await aiofiles.os.makedirs(output_subdir, exist_ok=True)
            safe_filename = sanitize_filename(relative_path.name)
            base_path = output_subdir / safe_filename

            # Write JSON
            json_path = base_path.with_suffix(".json")
            try:
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(json.dumps(documentation, indent=2, sort_keys=True))
            except Exception as e:
                logger.error(f"Error writing JSON to {json_path}: {e}")
                raise FileWriteError(f"Failed to write JSON: {e}")

            # Write Markdown
            if documentation.get("generate_markdown", True):
                try:
                    markdown_content = generate_markdown_report(
                        documentation, file_path, metrics
                    )
                    md_path = base_path.with_suffix(".md")
                    async with aiofiles.open(md_path, "w") as f:
                        await f.write(markdown_content)
                except Exception as e:
                    logger.error(f"Error writing Markdown: {e}")
                    raise FileWriteError(f"Failed to write Markdown: {e}")

            logger.info(f"Documentation written to {json_path}")
            return documentation

    except FileWriteError:
        raise
    except Exception as e:
        logger.error(f"Error writing documentation report: {e}")
        raise DocumentationError(f"Documentation write failed: {e}")
