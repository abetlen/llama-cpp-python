from __future__ import annotations

import argparse
import copy
import cProfile
import importlib.util
import io
import json
import pstats
import random
import statistics
import string
import sys
import time
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, cast


SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_PATH = SCRIPT_DIR / "server.py"
GPT_OSS_CHAT_TEMPLATE_URL = "https://huggingface.co/openai/gpt-oss-20b/raw/main/chat_template.jinja"
TEMPLATE_CACHE_PATH = SCRIPT_DIR / ".cache" / "gpt_oss_chat_template.jinja"

# Source: Hugging Face Transformers "Response Parsing" docs, GPT-OSS schema example.
GPT_OSS_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {
            "type": "string",
            "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)",
        },
        "thinking": {
            "type": "string",
            "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>",
        },
        "tool_calls": {
            "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-regex": r"^to=functions\.(\w+)",
                            },
                            "arguments": {
                                "type": "object",
                                "x-regex": r"<\|message\|>(.*)",
                                "x-parser": "json",
                                "additionalProperties": True,
                            },
                        },
                    },
                },
            },
        },
    },
}


def load_server_module() -> Any:
    spec = importlib.util.spec_from_file_location("batch_processing_server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gpt_oss_chat_template() -> str:
    if TEMPLATE_CACHE_PATH.exists():
        return TEMPLATE_CACHE_PATH.read_text()
    TEMPLATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GPT_OSS_CHAT_TEMPLATE_URL) as response:
        template = response.read().decode("utf-8")
    TEMPLATE_CACHE_PATH.write_text(template)
    return template


@dataclass
class BenchmarkCase:
    messages: List[Any]
    prompt: str
    output: str
    chunks: List[str]
    tools: List[Dict[str, Any]]
    scenario: str
    prompt_chars: int
    output_chars: int


def random_word(rng: random.Random, min_len: int = 3, max_len: int = 10) -> str:
    alphabet = string.ascii_lowercase
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def random_sentence(rng: random.Random, min_words: int = 6, max_words: int = 16) -> str:
    words = [random_word(rng) for _ in range(rng.randint(min_words, max_words))]
    sentence = " ".join(words)
    return sentence.capitalize() + "."


def random_paragraph(rng: random.Random, min_sentences: int = 2, max_sentences: int = 5) -> str:
    return " ".join(
        random_sentence(rng) for _ in range(rng.randint(min_sentences, max_sentences))
    )


def build_tool_catalog(rng: random.Random, tool_count: int = 6) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    primitive_types = ["string", "integer", "number"]
    for tool_index in range(tool_count):
        name = f"{random_word(rng, 4, 8)}_{tool_index}"
        property_count = rng.randint(1, 4)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for property_index in range(property_count):
            property_name = f"{random_word(rng, 4, 8)}_{property_index}"
            type_name = rng.choice(primitive_types)
            properties[property_name] = {"type": type_name}
            if rng.random() < 0.7:
                required.append(property_name)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": random_sentence(rng, 5, 10),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return tools


def sample_tool_arguments(rng: random.Random, tool: Dict[str, Any]) -> Dict[str, Any]:
    parameters = tool["function"]["parameters"]
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    arguments: Dict[str, Any] = {}
    for key, schema in properties.items():
        if key not in required and rng.random() < 0.45:
            continue
        schema_type = schema.get("type")
        if schema_type == "string":
            arguments[key] = random_sentence(rng, 2, 5)
        elif schema_type == "integer":
            arguments[key] = rng.randint(0, 10_000)
        elif schema_type == "number":
            arguments[key] = round(rng.random() * 1000, 3)
        else:
            arguments[key] = random_sentence(rng, 2, 5)
    return arguments


def random_history_messages(
    server_module: Any,
    rng: random.Random,
    tools: List[Dict[str, Any]],
) -> List[Any]:
    message_cls = server_module.ChatCompletionRequestMessage
    messages: List[Any] = [
        message_cls(
            role="developer",
            content=(
                "Follow Harmony formatting exactly. Use analysis for reasoning, commentary for tool calls, "
                "and final for end-user visible text."
            ),
        )
    ]
    if rng.random() < 0.5:
        messages.append(
            message_cls(
                role="user",
                content=random_paragraph(rng, 1, 2),
            )
        )
        messages.append(
            message_cls(
                role="assistant",
                content=random_paragraph(rng, 1, 2),
                channel="final",
                phase="final_answer",
            )
        )
    messages.append(
        message_cls(
            role="user",
            content=random_paragraph(rng, 1, 3),
        )
    )
    return messages


def build_output_text(
    rng: random.Random,
    tools: List[Dict[str, Any]],
    scenario: str,
) -> str:
    parts: List[str] = []
    if scenario in {"analysis-final", "analysis-tool", "analysis-tool-final", "analysis-multi-tool"}:
        parts.append(
            "<|start|>assistant<|channel|>analysis<|message|>"
            + random_paragraph(rng, 2, 4)
            + "<|end|>"
        )
    if scenario in {"analysis-tool", "analysis-tool-final", "analysis-multi-tool"}:
        tool_count = 2 if scenario == "analysis-multi-tool" else 1
        for _ in range(tool_count):
            tool = rng.choice(tools)
            arguments = sample_tool_arguments(rng, tool)
            parts.append(
                "<|start|>assistant<|channel|>commentary "
                f"to=functions.{tool['function']['name']} "
                "<|constrain|>json<|message|>"
                + json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                + "<|call|>"
            )
    if scenario in {"final-only", "analysis-final", "analysis-tool-final"}:
        parts.append(
            "<|start|>assistant<|channel|>final<|message|>"
            + random_paragraph(rng, 2, 5)
            + "<|end|>"
        )
    return "".join(parts)


def chunk_output_text(rng: random.Random, text: str) -> List[str]:
    chunks: List[str] = []
    index = 0
    while index < len(text):
        if text[index] == "<":
            chunk_len = rng.randint(1, 4)
        else:
            chunk_len = rng.randint(1, 16)
        chunks.append(text[index : index + chunk_len])
        index += chunk_len
    return chunks


def build_cases(
    server_module: Any,
    *,
    case_count: int,
    seed: int,
) -> List[BenchmarkCase]:
    rng = random.Random(seed)
    template = load_gpt_oss_chat_template()
    formatter = server_module.Jinja2ChatFormatter(
        template,
        bos_token="<|startoftext|>",
        eos_token="<|return|>",
    )
    cases: List[BenchmarkCase] = []
    scenarios = [
        "final-only",
        "analysis-final",
        "analysis-tool",
        "analysis-tool-final",
        "analysis-multi-tool",
    ]
    for _ in range(case_count):
        tools = build_tool_catalog(rng, tool_count=rng.randint(3, 8))
        messages = random_history_messages(server_module, rng, tools)
        prompt, _ = formatter.format(messages=messages, tools=tools, tool_choice="auto")
        scenario = rng.choices(
            scenarios,
            weights=[1, 2, 3, 2, 1],
            k=1,
        )[0]
        output = build_output_text(rng, tools, scenario)
        cases.append(
            BenchmarkCase(
                messages=messages,
                prompt=prompt,
                output=output,
                chunks=chunk_output_text(rng, output),
                tools=tools,
                scenario=scenario,
                prompt_chars=len(prompt),
                output_chars=len(output),
            )
        )
    return cases


def time_stage(
    fn: Callable[[], int],
    *,
    repetitions: int,
) -> Dict[str, float]:
    timings: List[float] = []
    checksum = 0
    for _ in range(repetitions):
        start = time.perf_counter()
        checksum ^= fn()
        timings.append(time.perf_counter() - start)
    return {
        "mean_ms": statistics.mean(timings) * 1000.0,
        "median_ms": statistics.median(timings) * 1000.0,
        "min_ms": min(timings) * 1000.0,
        "max_ms": max(timings) * 1000.0,
        "checksum": float(checksum),
    }


def profile_stage(
    fn: Callable[[], int],
    *,
    sort_by: str = "cumtime",
    top_n: int = 25,
) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    checksum = fn()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(sort_by)
    stats_map = cast(Dict[Any, Any], getattr(stats, "stats"))
    filtered = [
        (func, stat)
        for func, stat in stats_map.items()
        if func[0].endswith("server.py") or func[0].endswith("benchmark_response_parser.py")
    ]
    filtered.sort(key=lambda item: item[1][3], reverse=True)
    stream.write(f"checksum={checksum}\n")
    stream.write(f"top {top_n} functions by cumulative time\n")
    for func, stat in filtered[:top_n]:
        cc, nc, tt, ct, _callers = stat
        filename, line_no, function_name = func
        stream.write(
            f"{ct:9.6f}s cum {tt:9.6f}s self {nc:7d} calls "
            f"{Path(filename).name}:{line_no} {function_name}\n"
        )
    return stream.getvalue()


def benchmark_template_render(server_module: Any, cases: Sequence[BenchmarkCase]) -> int:
    formatter = server_module.Jinja2ChatFormatter(
        load_gpt_oss_chat_template(),
        bos_token="<|startoftext|>",
        eos_token="<|return|>",
    )
    total = 0
    for case in cases:
        prompt, stop = formatter.format(
            messages=case.messages,
            tools=case.tools,
            tool_choice="auto",
        )
        total += len(prompt) + len(stop)
    return total


def benchmark_construct_warm(server_module: Any, cases: Sequence[BenchmarkCase]) -> int:
    total = 0
    for case in cases:
        parser = server_module.ResponseParser(GPT_OSS_RESPONSE_SCHEMA, tools=case.tools)
        total += int(parser.started)
    return total + len(cases)


def benchmark_construct_cold(server_module: Any, cases: Sequence[BenchmarkCase]) -> int:
    total = 0
    for case in cases:
        parser = server_module.ResponseParser(copy.deepcopy(GPT_OSS_RESPONSE_SCHEMA), tools=case.tools)
        total += int(parser.started)
    return total + len(cases)


def benchmark_full_parse(server_module: Any, cases: Sequence[BenchmarkCase]) -> int:
    total = 0
    for case in cases:
        parser = server_module.ResponseParser(GPT_OSS_RESPONSE_SCHEMA, tools=case.tools)
        message = parser.parse_completion_message(case.output)
        total += len(json.dumps(message, ensure_ascii=False, sort_keys=True))
    return total


def benchmark_stream(server_module: Any, cases: Sequence[BenchmarkCase]) -> int:
    total = 0
    for case in cases:
        parser = server_module.ResponseParser(
            GPT_OSS_RESPONSE_SCHEMA,
            tools=case.tools,
            completion_id="bench",
            choice_index=0,
        )
        for chunk_index, text in enumerate(case.chunks):
            payloads = parser.consume_completion_chunk(
                text,
                chunk_id="cmpl_bench",
                created=0,
                model="gpt-oss-bench",
                finish_reason="stop" if chunk_index == len(case.chunks) - 1 else None,
            )
            total += len(payloads)
    return total


def benchmark_stream_fastpath_eligibility(server_module: Any) -> int:
    parser = server_module.ResponseParser(GPT_OSS_RESPONSE_SCHEMA, tools=None)
    return 1 if getattr(parser, "_stream_plan", None) is not None else 0


def scenario_counts(cases: Sequence[BenchmarkCase]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in cases:
        counts[case.scenario] = counts.get(case.scenario, 0) + 1
    return counts


def print_stage_result(name: str, result: Dict[str, float]) -> None:
    print(
        f"{name:24s}"
        f" mean={result['mean_ms']:9.3f} ms"
        f" median={result['median_ms']:9.3f} ms"
        f" min={result['min_ms']:9.3f} ms"
        f" max={result['max_ms']:9.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile-top", type=int, default=20)
    args = parser.parse_args()

    server_module = load_server_module()
    cases = build_cases(server_module, case_count=args.cases, seed=args.seed)

    print(f"cases={len(cases)} seed={args.seed}")
    print(f"prompt_chars mean={statistics.mean(case.prompt_chars for case in cases):.1f}")
    print(f"output_chars mean={statistics.mean(case.output_chars for case in cases):.1f}")
    print(f"chunks mean={statistics.mean(len(case.chunks) for case in cases):.1f}")
    print(f"scenarios={json.dumps(scenario_counts(cases), sort_keys=True)}")
    print()

    stages = [
        ("template_render", lambda: benchmark_template_render(server_module, cases)),
        ("construct_warm", lambda: benchmark_construct_warm(server_module, cases)),
        ("construct_cold", lambda: benchmark_construct_cold(server_module, cases)),
        ("full_parse", lambda: benchmark_full_parse(server_module, cases)),
        ("stream", lambda: benchmark_stream(server_module, cases)),
    ]

    for name, fn in stages:
        result = time_stage(fn, repetitions=args.repetitions)
        print_stage_result(name, result)

    fastpath = benchmark_stream_fastpath_eligibility(server_module)
    print()
    print(f"gpt_oss_stream_plan_compiled={bool(fastpath)}")
    print()
    for name, fn in stages:
        print(f"[profile] {name}")
        print(profile_stage(fn, top_n=args.profile_top))


if __name__ == "__main__":
    main()
