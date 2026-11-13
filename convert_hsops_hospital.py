#!/usr/bin/env python3
"""Convert an authorized HSOPS 2.0 hospital SAS dataset to a local CSV.
"""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path,
                        help='Authorized hospital-level .sas7bdat input file')
    parser.add_argument('--output', required=True, type=Path,
                        help='Local .csv destination; its parent directory must exist')
    args = parser.parse_args(argv)
    source = args.input.expanduser()
    destination = args.output.expanduser()
    if source.suffix.lower() != '.sas7bdat' or not source.is_file():
        parser.error('--input must be an existing .sas7bdat file')
    if destination.suffix.lower() != '.csv':
        parser.error('--output must have a .csv extension')
    if not destination.parent.is_dir():
        parser.error('The output parent directory does not exist')
    if destination.exists():
        parser.error('The output file already exists; choose a new destination')
    try:
        import pyreadstat
    except ImportError:
        parser.error('Install dependencies first: python -m pip install pandas pyreadstat')
    try:
        df, _ = pyreadstat.read_sas7bdat(str(source))
    except Exception:
        print('Could not read the SAS file. Check access and file format locally.', file=sys.stderr)
        return 1
    try:
        # Exclusive creation avoids silently overwriting an existing data file.
        with destination.open('x', encoding='utf-8', newline='') as output:
            df.to_csv(output, index=False)
    except FileExistsError:
        print('Output already exists; no file was overwritten.', file=sys.stderr)
        return 1
    except OSError:
        print('Could not finish writing the CSV. Check local permissions and disk space; '
              'a partial output may remain.', file=sys.stderr)
        return 1
    print('CSV conversion completed. No records were displayed. Keep the output private '
          'unless release is approved.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
