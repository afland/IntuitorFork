#!/usr/bin/env python3
"""
Standalone math evaluation utilities.
Adapted from verl math evaluation functions.
"""

import re
from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def remove_right_units(string: str) -> str:
    """Remove units from the right side of a string."""
    # Remove common units
    units = [
        "degrees", "degree", "°", "radians", "radian", "rad",
        "meters", "meter", "m", "cm", "mm", "km", "inches", "inch", "in", "ft", "feet",
        "seconds", "second", "s", "minutes", "minute", "min", "hours", "hour", "hr", "h",
        "dollars", "dollar", "$", "cents", "cent", "¢",
        "percent", "%", "percentage"
    ]
    
    for unit in units:
        if string.endswith(unit):
            string = string[:-len(unit)].strip()
            break
    
    return string


def fix_sqrt(string: str) -> str:
    """Fix sqrt expressions to proper LaTeX format."""
    _fix_sqrt_patterns = [
        (r"\\sqrt(\w+)", r"\\sqrt{\1}"),
        (r"\\sqrt{([^{}]+)}\s*(\w+)", r"\\sqrt{\1\2}"),
    ]
    
    for pattern, replacement in _fix_sqrt_patterns:
        string = re.sub(pattern, replacement, string)
    
    return string


def fix_fracs(string: str) -> str:
    """Fix fraction expressions to proper LaTeX format."""
    # Handle \frac followed by single characters
    string = re.sub(r"\\frac(\w)(\w)", r"\\frac{\1}{\2}", string)
    
    # Handle a/b format
    string = re.sub(r"(\w+)/(\w+)", r"\\frac{\1}{\2}", string)
    
    return string


def fix_a_slash_b(string: str) -> str:
    """Fix a/b format to LaTeX fractions."""
    return re.sub(r"(\w+)/(\w+)", r"\\frac{\1}{\2}", string)


def strip_string(string: str) -> str:
    """Normalize a string for comparison."""
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." 
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # fix fracs
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # fix a/b format
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    """Check if two strings are equivalent after normalization."""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(f"Comparing: '{ss1}' vs '{ss2}'")
        return ss1 == ss2
    except Exception:
        return str1 == str2


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Compute the MATH dataset score for a solution."""
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(f"Error in compute_score: {e}")

    return retval 