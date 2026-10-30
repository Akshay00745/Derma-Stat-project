"""
Modular UI Components for Dermastat Air Vision Web Application.
"""

from .home import render_home_page
from .upload import render_upload_page
from .results import render_results_page
from .explainability import render_explainability_page
from .detailed import render_detailed_page
from .performance import render_performance_page
from .about import render_about_page
from .help import render_help_page
from .sidebar import render_sidebar

__all__ = [
    "render_home_page",
    "render_upload_page",
    "render_results_page",
    "render_explainability_page",
    "render_detailed_page",
    "render_performance_page",
    "render_about_page",
    "render_help_page",
    "render_sidebar"
]
