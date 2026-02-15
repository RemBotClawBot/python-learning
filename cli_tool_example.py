#!/usr/bin/env python3
"""
CLI Tool Development Example
=============================

This module demonstrates building command-line interfaces with:
- argparse (built-in)
- Click (third-party library)
- Typer (modern alternative)

Covers: subcommands, arguments, options, help text,
input validation, output formatting, and progress bars.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import csv
from datetime import datetime
from typing import Optional, List, Dict, Any
import textwrap

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("Note: Click not installed. Run: pip install click")

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False
    print("Note: Typer/Rich not installed. Run: pip install typer rich")


def argparse_example_cli():
    """
    Demonstrate argparse - Python's built-in CLI library.
    """
    parser = argparse.ArgumentParser(
        description="File and text manipulation CLI tool",
        epilog="Example: python cli_tool_example.py parse --files data.csv --format json"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Count command
    count_parser = subparsers.add_parser('count', help='Count lines/words/chars in text')
    count_parser.add_argument('text', help='Text to analyze')
    count_parser.add_argument('--words', '-w', action='store_true', help='Count words')
    count_parser.add_argument('--lines', '-l', action='store_true', help='Count lines')
    count_parser.add_argument('--chars', '-c', action='store_true', help='Count characters')
    
    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse and convert files')
    parse_parser.add_argument('files', nargs='+', help='Files to parse')
    parse_parser.add_argument('--format', '-f', choices=['json', 'csv', 'tsv'], 
                            default='json', help='Output format')
    parse_parser.add_argument('--output', '-o', help='Output file')
    parse_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Grep command
    grep_parser = subparsers.add_parser('grep', help='Search for patterns in files')
    grep_parser.add_argument('pattern', help='Pattern to search for')
    grep_parser.add_argument('files', nargs='+', help='Files to search')
    grep_parser.add_argument('--ignore-case', '-i', action='store_true', 
                           help='Case-insensitive search')
    grep_parser.add_argument('--context', '-C', type=int, default=0, 
                           help='Lines of context to show')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process data with transformations')
    process_parser.add_argument('input', help='Input file or data')
    process_parser.add_argument('--transform', '-t', choices=['upper', 'lower', 'reverse', 'sort'],
                               default='upper', help='Transformation to apply')
    process_parser.add_argument('--batch-size', type=int, default=1000,
                              help='Batch size for processing')
    process_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be done without executing')
    
    return parser


def execute_argparse_command(args):
    """Execute argparse commands."""
    if args.command == 'count':
        text = args.text
        
        if not (args.words or args.lines or args.chars):
            # Default: count all
            args.words = args.lines = args.chars = True
        
        results = {}
        if args.lines:
            results['lines'] = len(text.splitlines())
        if args.words:
            results['words'] = len(text.split())
        if args.chars:
            results['characters'] = len(text)
        
        print("Count results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
    elif args.command == 'parse':
        if args.verbose:
            print(f"Parsing {len(args.files)} files to {args.format} format")
        
        all_data = []
        for file_path in args.files:
            if not Path(file_path).exists():
                print(f"Warning: File '{file_path}' not found")
                continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Simple parsing - in reality you'd parse based on file type
                data = {
                    'filename': file_path,
                    'size': len(content),
                    'lines': len(content.splitlines()),
                    'modified': datetime.fromtimestamp(Path(file_path).stat().st_mtime).isoformat()
                }
                all_data.append(data)
                
                if args.verbose:
                    print(f"  Processed: {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Format output
        if args.format == 'json':
            output = json.dumps(all_data, indent=2)
        elif args.format == 'csv':
            import io
            output_buffer = io.StringIO()
            writer = csv.DictWriter(output_buffer, fieldnames=['filename', 'size', 'lines', 'modified'])
            writer.writeheader()
            writer.writerows(all_data)
            output = output_buffer.getvalue()
        else:  # tsv
            output = '\t'.join(['filename', 'size', 'lines', 'modified']) + '\n'
            for item in all_data:
                output += '\t'.join(str(item[field]) for field in ['filename', 'size', 'lines', 'modified']) + '\n'
        
        # Output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Output written to {args.output}")
        else:
            print(output)
    
    elif args.command == 'grep':
        pattern = args.pattern
        
        for file_path in args.files:
            if not Path(file_path).exists():
                print(f"Warning: File '{file_path}' not found")
                continue
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                matches = []
                for i, line in enumerate(lines):
                    text_to_check = line.lower() if args.ignore_case else line
                    pattern_to_check = pattern.lower() if args.ignore_case else pattern
                    
                    if pattern_to_check in text_to_check:
                        # Collect context
                        start = max(0, i - args.context)
                        end = min(len(lines), i + args.context + 1)
                        
                        for j in range(start, end):
                            prefix = '>' if j == i else ' '
                            matches.append(f"{prefix} {j+1:4d}: {lines[j].rstrip()}")
                        if end < len(lines):  # Add separator if not at end
                            matches.append('---')
                
                if matches:
                    print(f"\n=== {file_path} ===")
                    for match in matches:
                        print(match)
                        
            except Exception as e:
                print(f"Error searching {file_path}: {e}")
    
    elif args.command == 'process':
        input_data = args.input
        
        # Check if input is a file
        if Path(input_data).exists():
            with open(input_data, 'r') as f:
                data = f.read()
        else:
            data = input_data
        
        if args.dry_run:
            print(f"Dry run: Would process {len(data)} characters with '{args.transform}' transform")
            print(f"Batch size: {args.batch_size}")
            return
        
        # Apply transformation
        if args.transform == 'upper':
            result = data.upper()
        elif args.transform == 'lower':
            result = data.lower()
        elif args.transform == 'reverse':
            result = data[::-1]
        elif args.transform == 'sort':
            result = '\n'.join(sorted(data.splitlines()))
        else:
            result = data
        
        # Process in batches if large
        if len(data) > args.batch_size:
            print(f"Processing {len(data)} characters in batches of {args.batch_size}...")
            batches = [data[i:i+args.batch_size] for i in range(0, len(data), args.batch_size)]
            print(f"Created {len(batches)} batches")
        
        print(f"\nResult ({args.transform}):")
        print("=" * 40)
        print(result[:500] + ("..." if len(result) > 500 else ""))
        print("=" * 40)
    
    else:
        print("No command specified. Use --help for usage.")


@click.group()
def click_cli():
    """Click-based CLI with rich features."""
    pass


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['json', 'yaml', 'csv']), 
              default='json', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(file_path, format, output, verbose):
    """Analyze file contents with detailed statistics."""
    if verbose:
        click.echo(f"Analyzing {file_path}...")
    
    stats = get_file_stats(file_path)
    
    if format == 'json':
        result = json.dumps(stats, indent=2)
    elif format == 'yaml':
        try:
            import yaml
            result = yaml.dump(stats, default_flow_style=False)
        except ImportError:
            click.echo("YAML format requires PyYAML: pip install pyyaml")
            result = json.dumps(stats, indent=2)
    else:  # csv
        import io
        output_buffer = io.StringIO()
        writer = csv.DictWriter(output_buffer, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
        result = output_buffer.getvalue()
    
    if output:
        with open(output, 'w') as f:
            f.write(result)
        click.echo(f"Analysis saved to {output}")
    else:
        click.echo(result)


@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--extensions', '-e', multiple=True, help='File extensions to include')
@click.option('--min-size', type=int, help='Minimum file size in bytes')
@click.option('--max-size', type=int, help='Maximum file size in bytes')
@click.option('--sort-by', type=click.Choice(['name', 'size', 'modified']), 
              default='name', help='Sort order')
@click.option('--reverse', '-r', is_flag=True, help='Reverse sort order')
def scan(directory, extensions, min_size, max_size, sort_by, reverse):
    """Scan directory and list files with filtering options."""
    files = []
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            file_path = Path(root) / filename
            stat = file_path.stat()
            
            # Apply filters
            if extensions and not any(filename.endswith(ext) for ext in extensions):
                continue
            
            if min_size and stat.st_size < min_size:
                continue
                
            if max_size and stat.st_size > max_size:
                continue
            
            files.append({
                'path': str(file_path.relative_to(directory)),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'type': 'file'
            })
    
    # Sort files
    key_func = lambda x: x['path'] if sort_by == 'name' else x['size'] if sort_by == 'size' else x['modified']
    files.sort(key=key_func, reverse=reverse)
    
    # Display results
    if files:
        click.echo(f"Found {len(files)} files in {directory}:")
        for file in files[:50]:  # Limit output
            size_str = click.style(f"{file['size']:,} bytes", fg='cyan')
            modified_str = click.style(file['modified'].strftime('%Y-%m-%d %H:%M'), fg='yellow')
            click.echo(f"  {file['path']} - {size_str} - {modified_str}")
        
        if len(files) > 50:
            click.echo(f"  ... and {len(files) - 50} more files")
    else:
        click.echo(f"No files found in {directory} matching criteria")


@click.command()
@click.argument('text')
@click.option('--wrap', '-w', type=int, help='Wrap text at specified width')
@click.option('--prefix', '-p', default='', help='Prefix for each line')
@click.option('--suffix', '-s', default='', help='Suffix for each line')
@click.option('--color', '-c', type=click.Choice(['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']),
              help='Color for output')
def format_text(text, wrap, prefix, suffix, color):
    """Format text with wrapping and styling."""
    if wrap:
        wrapped_lines = textwrap.wrap(text, width=wrap)
        result = '\n'.join(f"{prefix}{line}{suffix}" for line in wrapped_lines)
    else:
        result = f"{prefix}{text}{suffix}"
    
    if color:
        result = click.style(result, fg=color)
    
    click.echo(result)


# Add Click commands if available
if CLICK_AVAILABLE:
    click_cli.add_command(analyze)
    click_cli.add_command(scan)
    click_cli.add_command(format_text)


class TyperCLI:
    """Typer-based CLI with rich output."""
    
    def __init__(self):
        self.app = typer.Typer(help="Advanced CLI with rich formatting")
        self.console = Console()
        self.setup_commands()
    
    def setup_commands(self):
        @self.app.command()
        def stats(
            path: str = typer.Argument(..., help="File or directory path"),
            detail: bool = typer.Option(False, "--detail", "-d", help="Show detailed information"),
            recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursive directory scan")
        ):
            """Show statistics for a file or directory."""
            path_obj = Path(path)
            
            if not path_obj.exists():
                self.console.print(f"[red]Error:[/red] Path '{path}' does not exist")
                raise typer.Exit(code=1)
            
            if path_obj.is_file():
                self._show_file_stats(path_obj, detail)
            else:
                self._show_directory_stats(path_obj, detail, recursive)
        
        @self.app.command()
        def convert(
            input_file: str = typer.Argument(..., help="Input file path"),
            output_format: str = typer.Option("json", "--format", "-f", 
                                           help="Output format (json, csv, yaml)"),
            output_file: Optional[str] = typer.Option(None, "--output", "-o", 
                                                    help="Output file path")
        ):
            """Convert file between formats."""
            input_path = Path(input_file)
            
            if not input_path.exists():
                self.console.print(f"[red]Error:[/red] Input file '{input_file}' does not exist")
                raise typer.Exit(code=1)
            
            with self.console.status("[bold green]Converting file..."):
                data = self._read_file(input_path)
                converted = self._convert_data(data, output_format)
            
            if output_file:
                output_path = Path(output_file)
                output_path.write_text(converted)
                self.console.print(f"[green]Success:[/green] File converted and saved to '{output_file}'")
            else:
                self.console.print(Panel(converted, title=f"Converted to {output_format}"))
        
        @self.app.command()
        def progress(
            count: int = typer.Option(100, "--count", "-c", help="Number of items to process"),
            delay: float = typer.Option(0.1, "--delay", "-d", help="Delay between items")
        ):
            """Demonstrate progress tracking."""
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                typer.progressbar(length=count)
            ) as progress:
                task = progress.add_task("Processing...", total=count)
                
                for i in range(count):
                    progress.update(task, advance=1, description=f"Processing item {i+1}/{count}")
                    time.sleep(delay)
            
            self.console.print(f"[green]✓[/green] Processed {count} items")
    
    def _show_file_stats(self, file_path: Path, detail: bool):
        """Show statistics for a file."""
        stat = file_path.stat()
        
        table = Table(title=f"File Statistics: {file_path.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Size", f"{stat.st_size:,} bytes")
        table.add_row("Modified", datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))
        table.add_row("Created", datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'))
        
        if detail:
            try:
                content = file_path.read_text()
                table.add_row("Lines", str(len(content.splitlines())))
                table.add_row("Words", str(len(content.split())))
                table.add_row("Characters", str(len(content)))
            except:
                table.add_row("Content", "[red]Cannot read as text[/red]")
        
        self.console.print(table)
    
    def _show_directory_stats(self, dir_path: Path, detail: bool, recursive: bool):
        """Show statistics for a directory."""
        files = []
        total_size = 0
        
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    'path': file_path.relative_to(dir_path),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
                total_size += stat.st_size
        
        table = Table(title=f"Directory Statistics: {dir_path.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Files", str(len(files)))
        table.add_row("Total Size", f"{total_size:,} bytes")
        table.add_row("Average Size", f"{total_size//len(files) if files else 0:,} bytes")
        
        if detail and files:
            # Show largest files
            largest_files = sorted(files, key=lambda x: x['size'], reverse=True)[:5]
            table.add_row("", "")
            table.add_row("[bold]Largest Files[/bold]", "")
            
            for file in largest_files:
                table.add_row(f"  {file['path']}", f"{file['size']:,} bytes")
        
        self.console.print(table)
    
    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Read file and return data dictionary."""
        stat = file_path.stat()
        
        data = {
            'filename': file_path.name,
            'path': str(file_path),
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
        }
        
        try:
            content = file_path.read_text()
            data['content_preview'] = content[:100] + ("..." if len(content) > 100 else "")
            data['line_count'] = len(content.splitlines())
            data['word_count'] = len(content.split())
        except:
            data['content_preview'] = "[binary file]"
        
        return data
    
    def _convert_data(self, data: Dict[str, Any], format: str) -> str:
        """Convert data to specified format."""
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'csv':
            import io
            output_buffer = io.StringIO()
            writer = csv.DictWriter(output_buffer, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
            return output_buffer.getvalue()
        elif format == 'yaml':
            try:
                import yaml
                return yaml.dump(data, default_flow_style=False)
            except ImportError:
                return json.dumps(data, indent=2) + "\n# Note: Install PyYAML for YAML support"
        else:
            return str(data)
    
    def run(self, args=None):
        """Run the Typer CLI."""
        if TYPER_AVAILABLE:
            self.app()
        else:
            self.console.print("[red]Typer not available. Install with: pip install typer rich[/red]")


def get_file_stats(file_path):
    """Get file statistics."""
    stat = Path(file_path).stat()
    
    stats = {
        'filename': Path(file_path).name,
        'path': str(Path(file_path).absolute()),
        'size_bytes': stat.st_size,
        'size_human': human_readable_size(stat.st_size),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'permissions': oct(stat.st_mode)[-3:],
        'inode': stat.st_ino
    }
    
    # Try to read as text for additional stats
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            stats['line_count'] = len(content.splitlines())
            stats['word_count'] = len(content.split())
            stats['character_count'] = len(content)
    except:
        stats['line_count'] = 'N/A'
        stats['word_count'] = 'N/A'
        stats['character_count'] = 'N/A'
    
    return stats


def human_readable_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    """Run CLI examples."""
    print("="*60)
    print("CLI TOOL DEVELOPMENT EXAMPLES")
    print("="*60)
    print("\nThis module demonstrates three approaches to building CLIs:")
    print("1. argparse (built-in)")
    print("2. Click (feature-rich third-party)")
    print("3. Typer (modern with type hints)")
    print("="*60)
    
    # Parse arguments if any
    if len(sys.argv) > 1:
        parser = argparse_example_cli()
        args = parser.parse_args()
        
        if args.command:
            execute_argparse_command(args)
        else:
            parser.print_help()
        return
    
    # Interactive demo mode
    import time
    
    print("\nARGPARSE EXAMPLE (built-in):")
    print("="*40)
    
    # Simulate argparse commands
    test_args = type('Args', (), {
        'command': 'count',
        'text': 'Hello World\nThis is a test\nWith multiple lines',
        'words': True,
        'lines': True,
        'chars': True
    })()
    
    print("Running: count 'Hello World\\nThis is a test\\nWith multiple lines' --words --lines --chars")
    execute_argparse_command(test_args)
    
    print("\n" + "="*40)
    print("CLICK EXAMPLE (if installed):")
    print("="*40)
    
    if CLICK_AVAILABLE:
        print("Available commands:\n")
        
        # Show command help
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            click_cli(['--help'])
        
        click_help = f.getvalue()
        print(textwrap.indent(click_help, '  '))
        
        print("\nExample usage:")
        print("  click_cli analyze README.md --format json --verbose")
        print("  click_cli scan . --extensions .py --extensions .md --sort-by size")
        print("  click_cli format-text 'Lorem ipsum dolor sit amet' --wrap 20 --color blue")
    else:
        print("Click not installed. Install with: pip install click")
    
    print("\n" + "="*40)
    print("TYPER EXAMPLE (if installed):")
    print("="*40)
    
    if TYPER_AVAILABLE:
        print("Available commands:\n")
        
        # Create and show Typer help
        typer_cli = TyperCLI()
        
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                typer_cli.app(["--help"])
            except SystemExit:
                pass
        
        typer_help = f.getvalue()
        print(textwrap.indent(typer_help, '  '))
        
        print("\nExample usage:")
        print("  typer_cli stats . --detail --recursive")
        print("  typer_cli convert data.txt --format csv --output data.csv")
        print("  typer_cli progress --count 50 --delay 0.05")
    else:
        print("Typer not installed. Install with: pip install typer rich")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("\nargparse (built-in):")
    print("  • Pros: No dependencies, standard library, flexible")
    print("  • Cons: Verbose, boilerplate code, limited features")
    
    print("\nClick (third-party):")
    print("  • Pros: Clean API, command grouping, decorator syntax")
    print("  • Cons: Additional dependency, decorator-based (not everyone's preference)")
    
    print("\nTyper (modern):")
    print("  • Pros: Type hints, automatic --help, Rich integration")
    print("  • Cons: Requires Python 3.6+, newer dependency")
    
    print("\nRecommendations:")
    print("  • Simple scripts: argparse")
    print("  • Production tools: Click or Typer")
    print("  • Modern projects with type hints: Typer")
    print("  • Maximum compatibility: argparse")
    
    print("\n" + "="*60)
    print("To run these examples:")
    print("  python cli_tool_example.py count 'Hello World' --words")
    print("  python cli_tool_example.py parse README.md --format json")
    print("  python cli_tool_example.py grep 'def' *.py --context 2")
    print("="*60)


if __name__ == "__main__":
    import time  # Import here for Typer progress demo
    
    # Check if running in direct mode
    if len(sys.argv) > 1 and sys.argv[1] == 'typer':
        # Run Typer CLI directly
        if TYPER_AVAILABLE:
            typer_cli = TyperCLI()
            typer_cli.run()
        else:
            print("Typer not installed. Run: pip install typer rich")
    elif len(sys.argv) > 1 and sys.argv[1] == 'click':
        # Run Click CLI directly
        if CLICK_AVAILABLE:
            click_cli()
        else:
            print("Click not installed. Run: pip install click")
    else:
        # Run the demo
        main()