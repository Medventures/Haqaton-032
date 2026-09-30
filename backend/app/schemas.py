"""A deliberately small, shared JSON Schema contract for forms and LLM output."""

import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


MAX_FIELDS = 40
MAX_TEXT_LENGTH = 8000


class InvalidForm(ValueError):
    pass


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formSchema: dict[str, Any]


class EditField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: StrictStr | None
    expectedRevision: StrictInt = Field(ge=0)


class UnlockField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expectedRevision: StrictInt = Field(ge=0)


class InsertTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=20000)
    speaker: StrictStr | None = Field(default=None, max_length=64)


def validate_form(schema: dict[str, Any]) -> None:
    """Reject unsupported UI/schema constructs instead of silently ignoring them."""
    if len(json.dumps(schema, ensure_ascii=False)) > 128_000:
        raise InvalidForm("Описание формы превышает 128 КБ")
    allowed = {"$schema", "$id", "title", "description", "type", "properties", "required", "additionalProperties"}
    if set(schema) - allowed:
        raise InvalidForm(f"Неподдерживаемые свойства схемы: {sorted(set(schema) - allowed)}")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise InvalidForm("Форма должна быть object с additionalProperties: false")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not 1 <= len(properties) <= MAX_FIELDS:
        raise InvalidForm(f"Форма должна содержать от 1 до {MAX_FIELDS} полей")
    required = schema.get("required")
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise InvalidForm("required должен перечислять все поля; незаполненные значения равны null")
    if set(required) != set(properties) or len(required) != len(properties):
        raise InvalidForm("required должен перечислять каждое поле ровно один раз")
    for key, field in properties.items():
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", key) or key in {"__proto__", "constructor", "prototype"}:
            raise InvalidForm(f"Недопустимый идентификатор поля: {key}")
        if not isinstance(field, dict):
            raise InvalidForm(f"{key}: ожидается описание поля")
        field_allowed = {"type", "title", "description", "maxLength", "enum", "x-ui", "x-enumLabels"}
        if set(field) - field_allowed:
            raise InvalidForm(f"{key}: неподдерживаемые свойства {sorted(set(field) - field_allowed)}")
        if field.get("type") not in (["string", "null"], ["null", "string"]):
            raise InvalidForm(f"{key}: поддерживается только type: ['string', 'null']")
        if not isinstance(field.get("title"), str) or not field["title"].strip() or len(field["title"]) > 200:
            raise InvalidForm(f"{key}: требуется title длиной 1–200 символов")
        if "description" in field and (not isinstance(field["description"], str) or len(field["description"]) > 4000):
            raise InvalidForm(f"{key}: description должен быть строкой до 4000 символов")
        length = field.get("maxLength", MAX_TEXT_LENGTH)
        if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= MAX_TEXT_LENGTH:
            raise InvalidForm(f"{key}: maxLength должен быть от 1 до {MAX_TEXT_LENGTH}")
        if "x-ui" in field and field["x-ui"] != "textarea":
            raise InvalidForm(f"{key}: поддерживается только x-ui: textarea")
        if "enum" in field:
            options = field["enum"]
            if not isinstance(options, list) or not 2 <= len(options) <= 21:
                raise InvalidForm(f"{key}: enum должен содержать null и 1–20 вариантов")
            if not all(item is None or (isinstance(item, str) and 0 < len(item) <= 100) for item in options):
                raise InvalidForm(f"{key}: варианты enum должны быть строками или null")
            if None not in options or len(set(options)) != len(options):
                raise InvalidForm(f"{key}: enum должен включать null и не содержать повторов")
            if "x-ui" in field:
                raise InvalidForm(f"{key}: enum отображается радиокнопками, x-ui не нужен")
            labels = field.get("x-enumLabels", {})
            if not isinstance(labels, dict) or any(
                code not in options or not isinstance(label, str) or not 0 < len(label) <= 200
                for code, label in labels.items()
            ):
                raise InvalidForm(f"{key}: x-enumLabels должен сопоставлять коды enum с подписями")
        elif "x-enumLabels" in field:
            raise InvalidForm(f"{key}: x-enumLabels требует enum")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise InvalidForm("Некорректная JSON Schema") from exc


def validate_values(schema: dict[str, Any], values: Any) -> None:
    if not isinstance(values, dict):
        raise InvalidForm("Ожидается объект значений")
    errors = sorted(Draft202012Validator(schema).iter_errors(values), key=lambda error: str(error.path))
    if errors:
        error = errors[0]
        field = ".".join(str(part) for part in error.path)
        # Do not include actual patient text in errors/logs.
        raise InvalidForm(f"Значение поля {field or '(форма)'} не соответствует схеме ({error.validator})")
    if any(isinstance(value, str) and len(value) > MAX_TEXT_LENGTH for value in values.values()):
        raise InvalidForm(f"Значение превышает {MAX_TEXT_LENGTH} символов")
