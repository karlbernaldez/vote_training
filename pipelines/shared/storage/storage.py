import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from google.cloud import storage

TRUE_VALUES = {'1', 'true', 'yes', 'y', 'on'}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def split_s3_bucket_config(raw_bucket_config: str) -> tuple[str | None, str]:
    value = raw_bucket_config.strip()
    parsed = urlparse(value)
    if parsed.scheme in {'http', 'https'}:
        bucket_name = parsed.path.strip('/')
        return f'{parsed.scheme}://{parsed.netloc}', bucket_name
    return None, value


def gcs_client() -> storage.Client:
    return storage.Client()


def s3_client(endpoint_url: str | None = None):
    client_kwargs = {
        'config': Config(signature_version='s3v4', s3={'addressing_style': 'path'})
    }
    if endpoint_url:
        client_kwargs['endpoint_url'] = endpoint_url

    access_key = env_value('CEPH_ACCESS_KEY') or env_value('AWS_ACCESS_KEY_ID')
    secret_key = env_value('CEPH_SECRET_KEY') or env_value('AWS_SECRET_ACCESS_KEY')
    if access_key and secret_key:
        client_kwargs['aws_access_key_id'] = access_key
        client_kwargs['aws_secret_access_key'] = secret_key

    return boto3.client('s3', **client_kwargs)


def storage_uri(storage_backend: str, bucket_name: str, object_path: str) -> str:
    scheme = 'gs' if storage_backend == 'gcs' else 's3'
    return f'{scheme}://{bucket_name}/{object_path}'


def object_exists(storage_backend: str, bucket_name: str, object_path: str) -> bool:
    if storage_backend == 'gcs':
        client = gcs_client()
        return client.bucket(bucket_name).blob(object_path).exists(client)

    if env_flag('S3_SKIP_EXISTS_CHECK'):
        return False

    try:
        s3_client().head_object(Bucket=bucket_name, Key=object_path)
        return True
    except ClientError as exc:
        code = exc.response.get('Error', {}).get('Code')
        if code in {'404', 'NoSuchKey', 'NotFound'}:
            return False
        raise


def upload_file(storage_backend: str, bucket_name: str, local_file: Path, object_path: str) -> str:
    if storage_backend == 'gcs':
        blob = gcs_client().bucket(bucket_name).blob(object_path)
        blob.upload_from_filename(str(local_file))
    else:
        s3_client().upload_file(str(local_file), bucket_name, object_path)
    return storage_uri(storage_backend, bucket_name, object_path)


def upload_json(storage_backend: str, bucket_name: str, data: Any, object_path: str) -> str:
    body = json.dumps(data, indent=2)
    if storage_backend == 'gcs':
        blob = gcs_client().bucket(bucket_name).blob(object_path)
        blob.upload_from_string(body, content_type='application/json')
    else:
        s3_client().put_object(Bucket=bucket_name, Key=object_path, Body=body.encode('utf-8'))
    return storage_uri(storage_backend, bucket_name, object_path)
