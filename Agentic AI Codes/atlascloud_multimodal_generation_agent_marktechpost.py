# -*- coding: utf-8 -*-
"""Atlas Cloud multimodal generation agent tutorial.

This script demonstrates a conservative agent pattern for discovering Atlas Cloud
models, validating the current input schema, building an image/video generation
request, and optionally submitting and polling the async task.

Default behavior is a dry run. Use --submit only after reviewing the selected
model, generated request body, and expected cost in the printed preview.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


MODELS_URL = "https://api.atlascloud.ai/api/v1/models"
MEDIA_BASE_URL = "https://api.atlascloud.ai/api/v1"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "atlascloud-marktechpost-tutorial/1.0",
}


class AtlasCloudError(RuntimeError):
    """Raised for recoverable tutorial errors."""


def fetch_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={**DEFAULT_HEADERS, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AtlasCloudError(f"GET {url} failed with HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise AtlasCloudError(f"GET {url} failed: {exc}") from exc


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: int = 60) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            **DEFAULT_HEADERS,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AtlasCloudError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise AtlasCloudError(f"POST {url} failed: {exc}") from exc


@dataclass(frozen=True)
class ModelChoice:
    model: str
    display_name: str
    model_type: str
    schema_url: str
    price: Any


def list_console_models(model_type: str) -> list[dict[str, Any]]:
    catalog = fetch_json(MODELS_URL)
    models = catalog.get("data", [])
    return [
        item
        for item in models
        if item.get("display_console") is True and item.get("type") == model_type
    ]


def choose_model(model_type: str, keyword: str) -> ModelChoice:
    keyword_norm = keyword.lower()
    matches = []
    for item in list_console_models(model_type):
        haystack = " ".join(
            str(item.get(field, ""))
            for field in ("model", "displayName", "familyDisplayName", "profile", "tags")
        ).lower()
        if keyword_norm in haystack:
            matches.append(item)

    if not matches:
        raise AtlasCloudError(f"No display-console {model_type} model matched keyword: {keyword}")

    selected = matches[0]
    schema_url = selected.get("schema")
    if not schema_url:
        raise AtlasCloudError(
            f"Selected model {selected.get('model')} has no schema URL; choose another model."
        )

    return ModelChoice(
        model=selected["model"],
        display_name=selected.get("displayName") or selected["model"],
        model_type=selected["type"],
        schema_url=schema_url,
        price=selected.get("price"),
    )


def load_input_schema(choice: ModelChoice) -> tuple[dict[str, Any], list[str]]:
    schema_doc = fetch_json(choice.schema_url)
    input_schema = schema_doc.get("components", {}).get("schemas", {}).get("Input", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    if not properties:
        raise AtlasCloudError(f"Schema for {choice.model} did not expose Input.properties")
    return properties, required


def parse_extra_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AtlasCloudError(f"--extra-json must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AtlasCloudError("--extra-json must decode to an object")
    return value


def build_request(
    choice: ModelChoice,
    properties: dict[str, Any],
    required: list[str],
    prompt: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    request_body: dict[str, Any] = {"model": choice.model}

    if "prompt" in properties:
        request_body["prompt"] = prompt
    elif "text" in properties:
        request_body["text"] = prompt
    else:
        raise AtlasCloudError("Selected model schema has neither 'prompt' nor 'text' input.")

    allowed = set(properties.keys())
    ignored = sorted(set(extra) - allowed)
    for key, value in extra.items():
        if key in allowed:
            request_body[key] = value

    missing = [
        field
        for field in required
        if field not in request_body and field != "model"
    ]
    if missing:
        raise AtlasCloudError(
            "Request is missing schema-required fields: "
            + ", ".join(missing)
            + ". Pass them with --extra-json."
        )

    if ignored:
        print(f"Ignored fields not present in the live schema: {', '.join(ignored)}")

    return request_body


def submit_generation(choice: ModelChoice, request_body: dict[str, Any], api_key: str) -> str:
    endpoint = "generateImage" if choice.model_type == "Image" else "generateVideo"
    response = post_json(f"{MEDIA_BASE_URL}/model/{endpoint}", request_body, api_key)
    prediction_id = response.get("data", {}).get("id")
    if not prediction_id:
        raise AtlasCloudError(f"Generation response did not include data.id: {response}")
    return str(prediction_id)


def poll_prediction(prediction_id: str, api_key: str, interval_seconds: int = 5) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    while True:
        result = fetch_json(
            f"{MEDIA_BASE_URL}/model/prediction/{prediction_id}",
            headers=headers,
            timeout=30,
        )
        status = str(result.get("data", {}).get("status", ""))
        print(f"prediction={prediction_id} status={status}")
        if status in {"completed", "succeeded", "failed"}:
            return result
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Atlas Cloud multimodal generation agent")
    parser.add_argument("--type", choices=["Image", "Video"], default="Image")
    parser.add_argument("--keyword", default="text-to-image", help="Keyword used to select a live model")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--extra-json", help="Additional schema-validated request fields")
    parser.add_argument("--submit", action="store_true", help="Submit the task after preview")
    parser.add_argument("--poll", action="store_true", help="Poll until the submitted task finishes")
    args = parser.parse_args()

    try:
        choice = choose_model(args.type, args.keyword)
        properties, required = load_input_schema(choice)
        request_body = build_request(
            choice=choice,
            properties=properties,
            required=required,
            prompt=args.prompt,
            extra=parse_extra_json(args.extra_json),
        )

        preview = {
            "selected_model": choice.__dict__,
            "schema_fields": sorted(properties.keys()),
            "required_fields": required,
            "request_body": request_body,
            "submit": args.submit,
        }
        print(json.dumps(preview, indent=2, ensure_ascii=False))

        if not args.submit:
            print("Dry run only. Re-run with --submit after reviewing the request and cost.")
            return 0

        api_key = os.environ.get("ATLASCLOUD_API_KEY")
        if not api_key:
            raise AtlasCloudError("Set ATLASCLOUD_API_KEY before using --submit.")

        prediction_id = submit_generation(choice, request_body, api_key)
        print(f"Submitted prediction: {prediction_id}")

        if args.poll:
            result = poll_prediction(prediction_id, api_key)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        return 0
    except AtlasCloudError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
