#!/usr/bin/env python
"""
Script to bundle metrics PNGs/HTML/CSV files into a single report directory.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bundle metrics PNGs/HTML/CSV files into a single report directory"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories containing PNG, HTML, or CSV files",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for the report (default: report_<timestamp>)",
    )
    parser.add_argument(
        "--include-patterns",
        nargs="+",
        default=["*.png", "*.html", "*.csv"],
        help="File patterns to include in the report",
    )
    return parser.parse_args()


def find_matching_files(paths: List[str], patterns: List[str]) -> List[Path]:
    """Find all files matching the given patterns in the specified paths."""
    matching_files = []
    
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path {path} does not exist, skipping", file=sys.stderr)
            continue
            
        if path.is_file():
            # Check if the file matches any of the patterns
            for pattern in patterns:
                if path.match(pattern):
                    matching_files.append(path)
                    break
        elif path.is_dir():
            # Recursively find all files matching any pattern in the directory
            for pattern in patterns:
                matching_files.extend(path.rglob(pattern))
    
    return matching_files


def create_report_directory(output_dir: Optional[str] = None) -> Path:
    """Create the report directory with a timestamp if not specified."""
    if output_dir:
        report_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(f"report_{timestamp}")
    
    # Create the directory if it doesn't exist
    report_dir.mkdir(parents=True, exist_ok=True)
    
    return report_dir


def organize_files_by_type(files: List[Path], report_dir: Path) -> None:
    """Organize files into subdirectories by file type."""
    # Create subdirectories for each file type
    png_dir = report_dir / "images"
    html_dir = report_dir / "html"
    csv_dir = report_dir / "data"
    
    png_dir.mkdir(exist_ok=True)
    html_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)
    
    # Copy files to appropriate subdirectories
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".png":
                dest_dir = png_dir
            elif file_path.suffix.lower() == ".html":
                dest_dir = html_dir
            elif file_path.suffix.lower() == ".csv":
                dest_dir = csv_dir
            else:
                # For any other file types, put them in a misc directory
                misc_dir = report_dir / "misc"
                misc_dir.mkdir(exist_ok=True)
                dest_dir = misc_dir
            
            # Copy the file, preserving the original filename
            dest_path = dest_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_path} to {dest_path}")
            
        except Exception as e:
            print(f"Error copying {file_path}: {e}", file=sys.stderr)


def create_index_html(report_dir: Path) -> None:
    """Create an index.html file to browse the report contents."""
    index_path = report_dir / "index.html"
    
    # Count files in each directory
    png_files = list((report_dir / "images").glob("*.png"))
    html_files = list((report_dir / "html").glob("*.html"))
    csv_files = list((report_dir / "data").glob("*.csv"))
    misc_files = list((report_dir / "misc").glob("*")) if (report_dir / "misc").exists() else []
    
    with open(index_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Metrics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
        ul {{ list-style-type: none; padding-left: 0; }}
        li {{ margin: 5px 0; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Metrics Report</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Images ({len(png_files)})</h2>
    <ul>
""")
        for png in png_files:
            f.write(f'        <li><a href="images/{png.name}" target="_blank">{png.name}</a></li>\n')
        
        f.write("""    </ul>
    
    <h2>HTML Files ({len(html_files)})</h2>
    <ul>
""")
        for html in html_files:
            f.write(f'        <li><a href="html/{html.name}" target="_blank">{html.name}</a></li>\n')
        
        f.write("""    </ul>
    
    <h2>Data Files ({len(csv_files)})</h2>
    <ul>
""")
        for csv in csv_files:
            f.write(f'        <li><a href="data/{csv.name}" target="_blank">{csv.name}</a></li>\n')
        
        if misc_files:
            f.write("""    </ul>
    
    <h2>Other Files ({len(misc_files)})</h2>
    <ul>
""")
            for misc in misc_files:
                f.write(f'        <li><a href="misc/{misc.name}" target="_blank">{misc.name}</a></li>\n')
        
        f.write("""    </ul>
</body>
</html>""")
    
    print(f"Created index file at {index_path}")


def main() -> None:
    """Main function to bundle metrics files into a report directory."""
    args = parse_arguments()
    
    try:
        # Find all matching files
        matching_files = find_matching_files(args.inputs, args.include_patterns)
        
        if not matching_files:
            print("No matching files found. Report will be empty.")
        
        # Create report directory
        report_dir = create_report_directory(args.output)
        
        # Organize files by type
        organize_files_by_type(matching_files, report_dir)
        
        # Create index.html
        create_index_html(report_dir)
        
        # Print the full path to the report directory
        print(f"\nReport created at: {report_dir.absolute()}")
        
    except Exception as e:
        print(f"Error creating report: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
