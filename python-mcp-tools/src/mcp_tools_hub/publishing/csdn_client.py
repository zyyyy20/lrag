from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

from .markdown import render_markdown

CSDN_SAVE_ARTICLE_URL = "https://bizapi.csdn.net/blog-console-api/v3/mdeditor/saveArticle"


class CsdnConfigError(ValueError):
    """Raised when the CSDN publishing tool is not configured."""


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _redact_cookie(cookie: str) -> str:
    pairs = []
    for item in cookie.split(";"):
        key = item.strip().split("=", 1)[0].strip()
        if key:
            pairs.append(f"{key}=<redacted>")
    return "; ".join(pairs)


@dataclass(frozen=True)
class CsdnConfig:
    cookie: str
    category: str = "后端开发"
    default_tags: tuple[str, ...] = ("AI", "RAG", "MCP")
    publish_status: str = "draft"
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not self.cookie.strip():
            raise CsdnConfigError("CSDN_COOKIE is required to publish CSDN articles")

    @classmethod
    def from_env(cls) -> "CsdnConfig":
        return cls(
            cookie=os.getenv("CSDN_COOKIE", ""),
            category=os.getenv("CSDN_CATEGORY", "后端开发"),
            default_tags=_split_csv(os.getenv("CSDN_DEFAULT_TAGS", "AI,RAG,MCP")),
            publish_status=os.getenv("CSDN_PUBLISH_STATUS", "draft"),
            timeout_seconds=float(os.getenv("CSDN_TIMEOUT_SECONDS", "20")),
        )


@dataclass(frozen=True)
class CsdnArticle:
    title: str
    markdown_content: str
    description: str
    tags: tuple[str, ...] = ()
    category: str | None = None
    publish_status: str = "draft"


@dataclass(frozen=True)
class CsdnPublishResult:
    ok: bool
    code: int | None
    message: str
    article_id: str | None = None
    url: str | None = None
    title: str | None = None
    description: str | None = None
    publish_status: str = "draft"
    cookie_preview: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "article_id": self.article_id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "publish_status": self.publish_status,
            "cookie_preview": self.cookie_preview,
        }


class CsdnClient:
    def __init__(
        self,
        config: CsdnConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client or httpx.Client(timeout=config.timeout_seconds)

    def save_article(self, article: CsdnArticle) -> CsdnPublishResult:
        payload = self._build_payload(article)
        try:
            response = self._http_client.post(
                CSDN_SAVE_ARTICLE_URL,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            return CsdnPublishResult(
                ok=False,
                code=None,
                message=f"CSDN request failed: {exc}",
                publish_status=payload["pubStatus"],
                cookie_preview=_redact_cookie(self.config.cookie),
            )

        body = self._response_json(response)
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        code = body.get("code")
        message = str(body.get("msg") or response.reason_phrase or "")
        ok = response.is_success and (code in (None, 0, 200))
        return CsdnPublishResult(
            ok=ok,
            code=int(code) if isinstance(code, int) else response.status_code,
            message=message,
            article_id=str(data.get("id")) if data.get("id") is not None else None,
            url=str(data.get("url")) if data.get("url") else None,
            title=str(data.get("title")) if data.get("title") else None,
            description=(
                str(data.get("description")) if data.get("description") else None
            ),
            publish_status=payload["pubStatus"],
            cookie_preview=_redact_cookie(self.config.cookie),
        )

    def _build_payload(self, article: CsdnArticle) -> dict[str, Any]:
        title = article.title.strip()
        markdown_content = article.markdown_content.strip()
        if not title:
            raise ValueError("title is required")
        if not markdown_content:
            raise ValueError("markdown_content is required")

        tags = article.tags or self.config.default_tags
        publish_status = article.publish_status or self.config.publish_status
        category = article.category or self.config.category
        return {
            "title": title,
            "markdowncontent": markdown_content,
            "content": render_markdown(markdown_content),
            "readType": "public",
            "level": "0",
            "tags": ",".join(tags),
            "status": 0,
            "categories": category,
            "type": "original",
            "original_link": "",
            "authorized_status": True,
            "description": article.description.strip(),
            "resource_url": "",
            "not_auto_saved": "0",
            "source": "pc_mdeditor",
            "cover_images": [],
            "cover_type": 0,
            "is_new": 1,
            "vote_id": 0,
            "resource_id": "",
            "pubStatus": publish_status,
            "sync_git_code": 0,
        }

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9",
            "content-type": "application/json",
            "origin": "https://editor.csdn.net",
            "referer": "https://editor.csdn.net/",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari/537.36"
            ),
            "x-ca-key": "203803574",
            "x-ca-nonce": "a70ca99e-8bfa-46d1-8d12-363c72707ebe",
            "x-ca-signature": "NGLzlIyvH7BuQgGJrgfGOzao0SVpzdTs4aTcw3hio6Y=",
            "x-ca-signature-headers": "x-ca-key,x-ca-nonce",
            "cookie": self.config.cookie,
        }

    @staticmethod
    def _response_json(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError:
            return {"code": response.status_code, "msg": response.text}
        return body if isinstance(body, dict) else {"code": response.status_code}
