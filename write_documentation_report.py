import re
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from jinja2 import Environment, PackageLoader, select_autoescape, Template
import aiofiles
import aiofiles.os
from functools import lru_cache

from provider_config import load_provider_configs
from token_utils import TokenManager
from chunk import ChunkManager
from dependency_analyzer import DependencyAnalyzer
from context import HierarchicalContextManager
from utils import sanitize_filename

logger = logging.getLogger(__name__)

# Global write lock for thread safety
write_lock = asyncio.Lock()


class DocumentationError(Exception):
    """Base exception for documentation-related errors."""
    pass


class TemplateError(DocumentationError):
    """Raised when template processing fails."""
    pass


class FileWriteError(DocumentationError):
    """Raised when file writing fails."""
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
        """Determines badge color based on thresholds."""
        low, medium, high = (
            self.thresholds["low"],
            self.thresholds["medium"],
            self.thresholds["high"]
        )

        if self.value <= low:
            return "success"
        elif self.value <= medium:
            return "yellow"
        else:
            return "critical"


class BadgeGenerator:
    """Enhanced badge generation with caching and templates."""

    _badge_template = (
        "![{label}](https://img.shields.io/badge/"
        "{encoded_label}-{value}-{color}"
        "?style={style}{logo_part}{label_color_part})"
    )

    @classmethod
    @lru_cache(maxsize=128)
    def generate_badge(cls, config: BadgeConfig) -> str:
        """Generates a Markdown badge with caching."""
        try:
            label = config.metric_name.replace("_", " ").title()
            encoded_label = label.replace(" ", "%20")
            color = config.get_color()

            if isinstance(config.value, float):
                value = f"{config.value:.2f}"
            else:
                value = str(config.value)

            logo_part = f"&logo={config.logo}" if config.logo else ""
            label_color_part = (
                f"&labelColor={config.label_color}"
                if config.label_color
                else ""
            )

            return cls._badge_template.format(
                label=label,
                encoded_label=encoded_label,
                value=value,
                color=color,
                style=config.style,
                logo_part=logo_part,
                label_color_part=label_color_part
            )

        except Exception as e:
            logger.error(f"Error generating badge: {e}")
            return ""

    @classmethod
    def generate_all_badges(cls, metrics: Dict[str, Any]) -> str:
        """Generates all relevant badges for metrics."""
        badges = []

        try:
            if complexity := metrics.get("complexity"):
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Complexity",
                    value=complexity,
                    thresholds=DEFAULT_COMPLEXITY_THRESHOLDS,
                    logo="codeClimate"
                )))

            if halstead := metrics.get("halstead"):
                halstead_configs = [
                    BadgeConfig(
                        metric_name="Volume",
                        value=halstead["volume"],
                        thresholds=DEFAULT_HALSTEAD_THRESHOLDS["volume"],
                        logo="stackOverflow"
                    ),
                    BadgeConfig(
                        metric_name="Difficulty",
                        value=halstead["difficulty"],
                        thresholds=DEFAULT_HALSTEAD_THRESHOLDS["difficulty"],
                        logo="codewars"
                    ),
                    BadgeConfig(
                        metric_name="Effort",
                        value=halstead["effort"],
                        thresholds=DEFAULT_HALSTEAD_THRESHOLDS["effort"],
                        logo="atlassian"
                    )
                ]
                badges.extend(
                    cls.generate_badge(config)
                    for config in halstead_configs
                )

            if mi := metrics.get("maintainability_index"):
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Maintainability",
                    value=mi,
                    thresholds=DEFAULT_MAINTAINABILITY_THRESHOLDS,
                    logo="codeclimate"
                )))

            if coverage := metrics.get("test_coverage", {}).get("line_rate"):
                badges.append(cls.generate_badge(BadgeConfig(
                    metric_name="Coverage",
                    value=coverage,
                    thresholds={"low": 80, "medium": 60, "high": 0},
                    logo="testCoverage"
                )))

            return " ".join(badges)

        except Exception as e:
            logger.error(f"Error generating badges: {e}")
            return ""


class MarkdownFormatter:
    """Enhanced Markdown formatting with template support."""

    def __init__(self):
        """Initializes the MarkdownFormatter."""
        self.env = Environment(
            loader=PackageLoader('documentation', 'templates'),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

        self.env.filters['truncate_description'] = self.truncate_description
        self.env.filters['sanitize_text'] = self.sanitize_text

    @staticmethod
    def truncate_description(
        description: str,
        max_length: int = 100,
        ellipsis: str = "..."
    ) -> str:
        """Truncates description with word boundary awareness."""
        if not description or len(description) <= max_length:
            return description

        truncated = description[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + ellipsis

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitizes text for Markdown with improved character handling."""
        special_chars = r'[`*_{}[$()#+\-.!|]'
        text = re.sub(
            special_chars,
            lambda m: '\\' + m.group(0),
            str(text)
        )

        text = text.replace('\n', ' ').replace('\r', '')

        return ' '.join(text.split())

    def format_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        alignment: Optional[List[str]] = None
    ) -> str:
        """Formats data into a Markdown table with alignment support."""
        if not headers or not rows:
            return ""

        try:
            headers = [self.sanitize_text(str(header)) for header in headers]

            if not alignment:
                alignment = ['left'] * len(headers)

            align_map = {
                'left': ':---',
                'center': ':---:',
                'right': '---:'
            }
            separators = [
                align_map.get(align, ':---')
                for align in alignment
            ]

            table_lines = [
                f"| {' | '.join(headers)} |",
                f"| {' | '.join(separators)} |"
            ]

            for row in rows:
                row = (row + [''] * len(headers))[:len(headers)]
                sanitized_row = [
                    self.sanitize_text(str(cell))
                    for cell in row
                ]
                table_lines.append(
                    f"| {' | '.join(sanitized_row)} |"
                )

            return '\n'.join(table_lines)

        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            return ""


class DocumentationGenerator:
    """Enhanced documentation generation with template support."""

    def __init__(self):
        """Initializes the DocumentationGenerator."""
        self.formatter = MarkdownFormatter()
        self.badge_generator = BadgeGenerator()
        self._load_templates()

    def _load_templates(self):
        """Loads and validates templates."""
        try:
            self.templates = {
                'main': self.formatter.env.get_template('main.md.j2'),
                'function': self.formatter.env.get_template('function.md.j2'),
                'class': self.formatter.env.get_template('class.md.j2'),
                'metric': self.formatter.env.get_template('metric.md.j2'),
                'summary': self.formatter.env.get_template('summary.md.j2')
            }
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise TemplateError(f"Failed to load templates: {e}")

    async def generate_documentation(
        self,
        documentation: Dict[str, Any],
        language: str,
        file_path: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generates comprehensive documentation using templates."""
        try:
            badges = self.badge_generator.generate_all_badges(
                metrics or documentation.get("metrics", {})
            )

            language_info = self._get_language_info(language)
            functions_doc = await self._generate_functions_section(
                documentation.get("functions", [])
            )
            classes_doc = await self._generate_classes_section(
                documentation.get("classes", [])
            )
            metrics_doc = await self._generate_metrics_section(
                documentation.get("metrics", {}),
                metrics or {}
            )
            summary = await self._generate_summary_section(
                documentation,
                language_info
            )

            content = await self._render_template(
                self.templates['main'],
                {
                    'file_name': Path(file_path).name,
                    'badges': badges,
                    'language': language_info,
                    'summary': summary,
                    'functions': functions_doc,
                    'classes': classes_doc,
                    'metrics': metrics_doc,
                    'documentation': documentation
                }
            )

            toc = self._generate_toc(content)

            return f"# Table of Contents\n\n{toc}\n\n{content}"

        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            raise DocumentationError(f"Documentation generation failed: {e}")

    async def _generate_functions_section(
        self,
        functions: List[Dict[str, Any]]
    ) -> str:
        """Generates functions documentation using templates."""
        if not functions:
            return ""

        try:
            function_docs = []
            for func in functions:
                doc = await self._render_template(
                    self.templates['function'],
                    {'function': func}
                )
                function_docs.append(doc)

            return "\n\n".join(function_docs)
        except Exception as e:
            logger.error(f"Error generating functions section: {e}")
            return ""

    async def _generate_classes_section(
        self,
        classes: List[Dict[str, Any]]
    ) -> str:
        """Generates classes documentation using templates."""
        if not classes:
            return ""

        try:
            class_docs = []
            for cls in classes:
                doc = await self._render_template(
                    self.templates['class'],
                    {'class': cls}
                )
                class_docs.append(doc)

            return "\n\n".join(class_docs)
        except Exception as e:
            logger.error(f"Error generating classes section: {e}")
            return ""

    async def _generate_metrics_section(
        self,
        doc_metrics: Dict[str, Any],
        additional_metrics: Dict[str, Any]
    ) -> str:
        """Generates metrics documentation using templates."""
        try:
            combined_metrics = {**doc_metrics, **additional_metrics}
            return await self._render_template(
                self.templates['metric'],
                {'metrics': combined_metrics}
            )
        except Exception as e:
            logger.error(f"Error generating metrics section: {e}")
            return ""

    async def _generate_summary_section(
        self,
        documentation: Dict[str, Any],
        language_info: Dict[str, Any]
    ) -> str:
        """Generates summary section using templates."""
        try:
            return await self._render_template(
                self.templates['summary'],
                {
                    'documentation': documentation,
                    'language': language_info
                }
            )
        except Exception as e:
            logger.error(f"Error generating summary section: {e}")
            return ""

    @staticmethod
    def _get_language_info(language: str) -> Dict[str, Any]:
        """Gets language-specific information."""
        from utils import LANGUAGE_MAPPING

        for ext, info in LANGUAGE_MAPPING.items():
            if info["name"] == language:
                return info
        return {"name": language}

    async def _render_template(
        self,
        template: Template,
        context: Dict[str, Any]
    ) -> str:
        """Renders a template asynchronously."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                template.render,
                context
            )
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            raise TemplateError(f"Failed to render template: {e}")

    def _generate_toc(self, content: str) -> str:
        """Generates table of contents from content."""
        toc_entries = []
        current_level = 0

        for line in content.splitlines():
            if line.startswith('#'):
                level = 0
                while line.startswith('#'):
                    level += 1
                    line = line[1:]

                heading = line.strip()
                if not heading:
                    continue

                anchor = heading.lower()
                anchor = re.sub(r'[^\w\- ]', '', anchor)
                anchor = anchor.replace(' ', '-')

                indent = '  ' * (level - 1)
                toc_entries.append(
                    f"{indent}- [{heading}](#{anchor})"
                )

        return '\n'.join(toc_entries)


async def write_documentation_report(
    documentation: Optional[Dict[str, Any]],
    language: str,
    file_path: str,
    repo_root: str,
    output_dir: str,
    project_id: str,
    metrics: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Writes documentation to JSON and Markdown files."""
    if not documentation:
        logger.warning(f"No documentation to write for '{file_path}'")
        return None

    try:
        async with write_lock:
            project_output_dir = Path(output_dir) / project_id
            await aiofiles.os.makedirs(
                project_output_dir,
                exist_ok=True
            )

            relative_path = Path(file_path).relative_to(repo_root)
            safe_filename = sanitize_filename(relative_path.name)
            base_path = project_output_dir / safe_filename

            json_path = base_path.with_suffix(".json")
            try:
                async with aiofiles.open(json_path, "w") as f:
                    await f.write(json.dumps(
                        documentation,
                        indent=2,
                        sort_keys=True
                    ))
            except Exception as e:
                logger.error(f"Error writing JSON to {json_path}: {e}")
                raise FileWriteError(f"Failed to write JSON: {e}")

            if documentation.get("generate_markdown", True):
                try:
                    generator = DocumentationGenerator()
                    markdown_content = await generator.generate_documentation(
                        documentation,
                        language,
                        file_path,
                        metrics
                    )

                    md_path = base_path.with_suffix(".md")
                    async with aiofiles.open(md_path, "w") as f:
                        await f.write(markdown_content)
                except Exception as e:
                    logger.error(f"Error writing Markdown to {md_path}: {e}")
                    raise FileWriteError(f"Failed to write Markdown: {e}")

            logger.info(f"Documentation written to {json_path}")
            return documentation

    except Exception as e:
        logger.error(f"Error writing documentation report: {e}")
        raise DocumentationError(f"Documentation write failed: {e}")
