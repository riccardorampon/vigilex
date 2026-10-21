from typing import Any

import boto3
from botocore.config import Config

from vigilex_core.settings import Settings


def runtime_client(settings: Settings) -> Any:
    """One bedrock-runtime client for embed, chat and rerank (one VPC endpoint in AWS)."""
    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )
