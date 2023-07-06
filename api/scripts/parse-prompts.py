from typing import List
from argparse import ArgumentParser
from sys import argv
from collections import Counter
from json import dumps


def parse_args(args: List[str]):
    parser = ArgumentParser(
        prog="onnx-web prompt parser",
        description="count phrase frequency in prompt books",
    )
    parser.add_argument("file", nargs="+", help="prompt files to parse")
    return parser.parse_args(args)


def main():
    args = parse_args(argv[1:])

    lines: List[str] = []
    for file in args.file:
        with open(file, "r") as f:
            lines.extend(f.readlines())

    phrases = []
    for line in lines:
        phrases.extend([p.lower().strip() for p in line.split(",")])

    count = Counter(phrases)
    print(dumps(dict(count.most_common())))


if __name__ == "__main__":
    main()