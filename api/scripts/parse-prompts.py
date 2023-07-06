from argparse import ArgumentParser
from collections import Counter
from json import dumps, loads
from sys import argv
from typing import List


def parse_args(args: List[str]):
    parser = ArgumentParser(
        prog="onnx-web prompt parser",
        description="count phrase frequency in prompt books",
    )
    parser.add_argument("file", nargs="+", help="prompt files to parse")
    return parser.parse_args(args)


def load_duck(file: str):
    import duckdb

    cursor = duckdb.connect()
    return [p[0] for p in cursor.sql(f"SELECT * FROM '{file}'").fetchall()]


def load_json(file: str):
    with open(file, "r") as f:
        data = loads(f.read())
        params = data.get("params", None)
        if params:
            prompt = params.get("input_prompt", None)
            if prompt:
                return prompt

            prompt = params.get("prompt", None)
            if prompt:
                return prompt

    return ""


def load_text(file: str):
    with open(file, "r") as f:
        return f.readlines()


def main():
    args = parse_args(argv[1:])

    lines: List[str] = []
    for file in args.file:
        if file.endswith(".parquet") or file.endswith(".duckdb"):
            lines.extend(load_duck(file))
        elif file.endswith(".json"):
            # json only contains a single prompt
            lines.append(load_json(file))
        else:
            lines.extend(load_text(file))

    phrases = []
    for line in lines:
        phrases.extend([p.lower().strip() for p in line.split(",")])

    count = Counter(phrases)
    print(dumps(dict(count.most_common())))


if __name__ == "__main__":
    main()