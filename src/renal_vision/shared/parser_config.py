import argparse
import sys
from typing import Optional


class TermStyle:
    """
    Manages ANSI escape codes for terminal styling, disabling them
    if the output stream is not a TTY (e.g., output is redirected to a file).
    """

    # Check if the output stream is a terminal (TTY)
    # We check stderr since that is where argparse prints help messages.
    _is_tty = sys.stderr.isatty()

    # --- Color Definitions ---
    if _is_tty:
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        # Define empty strings if not running in a terminal
        RED = GREEN = YELLOW = CYAN = BOLD = RESET = ""

    @staticmethod
    def style(text: str, color: str = "", bold: bool = False) -> str:
        """Helper method to wrap text with color and optional bolding."""
        prefix = color
        if bold:
            prefix = TermStyle.BOLD + prefix

        # This will return "text" if TermStyle is using empty strings (non-TTY)
        return f"{prefix}{text}{TermStyle.RESET}"


def get_base_parser(module_name: Optional[str] = None) -> argparse.ArgumentParser:
    """Base parser with custom formatting for RenalVision"""

    name = "RenalVision" if not module_name else f"RenalVision - {module_name}"
    name = TermStyle.style(name, bold=True)
    desc = TermStyle.style("Modular Framework for Lesion Classification", TermStyle.GREEN)

    epilog = (
        f"{TermStyle.BOLD}------  Radboudumc OncoAI & TUM AI in Radiology – 2025  ------{TermStyle.RESET}\n"
        f"Website: {TermStyle.CYAN}https://www.comfort-ai.eu/for-patients/kidney-cancer{TermStyle.RESET}\n"
    )

    parser = argparse.ArgumentParser(
        prog=name,
        description=desc,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return parser
