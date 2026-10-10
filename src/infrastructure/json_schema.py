"""Shared strict JSON object schemas for model transports."""


def object_schema(properties: dict) -> dict:
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}
