# src/core/template_manager.py

import os
from typing import Dict, Optional
from jinja2 import Environment, FileSystemLoader, Template
import aiofiles
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TemplateManager:
    def __init__(self, template_dir: str = "templates"):
        """
        Initialize the TemplateManager with a template directory.
        
        Args:
            template_dir: Directory containing the templates
        """
        self._template_dir = template_dir
        self._cache: Dict[str, Template] = {}
        self._env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Define standard templates
        self._templates = {
            'main': 'main.md.j2',
            'function': 'function.md.j2',
            'class': 'class.md.j2',
            'metric': 'metric.md.j2',
            'summary': 'summary.md.j2'
        }

    async def get_template(self, template_name: str) -> Template:
        """
        Get a template by name, using cache if available.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            The requested template
            
        Raises:
            ValueError: If template name is invalid
        """
        if template_name not in self._templates:
            raise ValueError(f"Invalid template name: {template_name}")
            
        if template_name not in self._cache:
            try:
                self._cache[template_name] = self._env.get_template(
                    self._templates[template_name]
                )
            except Exception as e:
                logger.error(f"Failed to load template {template_name}: {e}")
                raise

        return self._cache[template_name]

    async def render_template(self, template_name: str, context: dict) -> str:
        """
        Render a template with the given context.
        
        Args:
            template_name: Name of the template to render
            context: Dictionary of variables to pass to the template
            
        Returns:
            The rendered template string
        """
        template = await self.get_template(template_name)
        try:
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            raise

    async def add_custom_template(self, name: str, template_path: str) -> None:
        """
        Add a custom template to the manager.
        
        Args:
            name: Name to reference the template
            template_path: Path to the template file
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
            
        self._templates[name] = template_path
        if name in self._cache:
            del self._cache[name]
