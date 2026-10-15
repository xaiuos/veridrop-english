"""Veridrop FastAPI app — POST /api/detect, GET /r/{id}, GET /r/{id}.jpg."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi import Response as FastAPIResponse
from fastapi.responses import (
    HTMLResponse, JSONResponse, RedirectResponse, Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import jobs, leaderboard
from .faq_data import FAQ_CATEGORIES, faqpage_jsonld, total_question_count
from .image_report import render_report_jpg
from .probe import probe_model_alive, probe_relay
from .ratelimit import check_rate


HERE = Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

logger = logging.getLogger("veridrop")
logger.setLevel(logging.INFO)

app = FastAPI(title="Veridrop", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.middleware("http")
async def no_html_cache(request: Request, call_next):
    """Prevent browsers from caching HTML responses.

    HTML pages reference versioned CSS/JS via ?v=N query strings, but the
    cache-bust only works if the browser refetches the HTML in the first place.
    Without an explicit Cache-Control, browsers apply a heuristic that can hold
    HTML for hours — making style/template iterations invisible to returning
    visitors. `no-cache` (NOT no-store) lets the browser keep a copy but
    forces revalidation against the server every time, so updates appear
    immediately while still serving 304s for unchanged pages.
    """
    response = await call_next(request)
    ctype = response.headers.get("content-type", "")
    if ctype.startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


_VALID_MODES = {"quick", "standard", "full"}
_VALID_WISHLIST_PROTOCOLS = {"openai", "gemini"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
WISHLIST_PATH = Path(
    os.environ.get("VERIDROP_WISHLIST_PATH", "/opt/veridrop/web_data/wishlist.txt")
)


def _protocol_from_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith(("gemini-", "models/gemini-")):
        return "gemini"
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return "anthropic"


def _model_choices() -> list[dict[str, str]]:
    """Curated dropdown — 4 most-tested model names. Free-form input still
    accepts anything; lookup_model() prefix-matches snapshot/alias forms.

    We keep it short on purpose:
    - Opus 4.7 + Opus 4.6: top-tier choices
    - Sonnet 4.6: most popular
    - Haiku 4.5 in snapshot form (-20251001) because some relays only route
      the snapshot ID, not the bare alias.
    """
    suggestions = [
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]
    return [{"id": s, "label": s} for s in suggestions]


def _openai_model_choices() -> list[dict[str, str]]:
    from relay_detector.protocols.openai import model_choices

    return [{"id": s, "label": s} for s in model_choices()]


def _gemini_model_choices() -> list[dict[str, str]]:
    from relay_detector.protocols.gemini import model_choices

    return [{"id": s, "label": s} for s in model_choices()]


@app.get("/", response_class=HTMLResponse)
async def hub(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "hub.html")


@app.get("/claude", response_class=HTMLResponse)
async def claude_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "index.html",
        {"models": _model_choices()},
    )


@app.get("/openai", response_class=HTMLResponse)
async def openai_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "openai.html",
        {"models": _openai_model_choices()},
    )


@app.get("/gemini", response_class=HTMLResponse)
async def gemini_index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "gemini.html",
        {"models": _gemini_model_choices()},
    )


_LEADERBOARD_TOP_N = 10
_LEADERBOARD_PER_PAGE = 25


@app.get("/leaderboard", response_class=HTMLResponse)
async def leaderboard_page(
    request: Request,
    page: int = 1,
) -> HTMLResponse:
    """Relay leaderboard — aggregates all public detection reports by domain.

    SEO/GEO killer feature: any "is relay XX any good" search lands directly
    on this page.

    Layout:
      - Top 10 (main board): always visible, no pagination, sorted by Bayesian-
        weighted ranking score so consistently-tested relays beat fluky-
        single-pass ones.
      - Rest (full list): paginated 25/page via ?page=N. Each page indexable
        for SEO long-tail (`?page=2` etc.).
    """
    all_relays, summary = leaderboard.aggregate()
    top = all_relays[:_LEADERBOARD_TOP_N]
    rest = all_relays[_LEADERBOARD_TOP_N:]

    total_rest = len(rest)
    total_pages = max(1, (total_rest + _LEADERBOARD_PER_PAGE - 1) // _LEADERBOARD_PER_PAGE)
    page = max(1, min(page, total_pages))
    rest_start = (page - 1) * _LEADERBOARD_PER_PAGE
    rest_page_items = rest[rest_start:rest_start + _LEADERBOARD_PER_PAGE]

    return templates.TemplateResponse(
        request,
        "leaderboard.html",
        {
            "top_relays": top,
            "rest_relays": rest_page_items,
            "rest_start_rank": _LEADERBOARD_TOP_N + rest_start + 1,
            "summary": summary,
            "page": page,
            "total_pages": total_pages,
            "has_rest": total_rest > 0,
            "protocol_labels": leaderboard.PROTOCOL_LABELS,
            "verdict_labels": leaderboard.VERDICT_LABELS,
        },
    )


@app.get("/leaderboard/{domain}", response_class=HTMLResponse)
async def leaderboard_domain_page(request: Request, domain: str) -> HTMLResponse:
    """Per-domain detail page — the key lever for SEO long-tail.

    When users search "{domain} relay review" / "{domain} fake or real" /
    "{domain} test", Google lands directly on this page. It contains every
    historical detection for that domain, protocol coverage, the most
    frequently failing detectors, and links to each specific /r/{job_id}
    report.
    """
    if not leaderboard.is_valid_domain(domain):
        raise HTTPException(status_code=404, detail="invalid domain")
    result = leaderboard.aggregate_one(domain)
    if result is None:
        raise HTTPException(status_code=404, detail="no reports for this domain")
    relay, history = result

    # Top 5 most-failed detectors across all protocols — the headline issues.
    failed_summary: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    for ps in relay.by_protocol.values():
        for name, cnt in ps.failed_detectors.most_common(5):
            if name not in seen_names:
                failed_summary.append((name, cnt))
                seen_names.add(name)
    failed_summary.sort(key=lambda x: x[1], reverse=True)

    return templates.TemplateResponse(
        request,
        "leaderboard_detail.html",
        {
            "relay": relay,
            "history": history,
            "failed_summary": failed_summary[:8],
            "protocol_labels": leaderboard.PROTOCOL_LABELS,
            "verdict_labels": leaderboard.VERDICT_LABELS,
        },
    )


@app.get("/faq", response_class=HTMLResponse)
async def faq_index(request: Request) -> HTMLResponse:
    """Standalone FAQ page — single source of truth in faq_data.py drives
    both the rendered HTML and the FAQPage JSON-LD schema in <head>."""
    return templates.TemplateResponse(
        request,
        "faq.html",
        {
            "categories": FAQ_CATEGORIES,
            "total_count": total_question_count(),
            "jsonld": json.dumps(faqpage_jsonld(), ensure_ascii=False),
        },
    )


@app.post("/api/wishlist")
async def api_wishlist(
    email: str = Form(...),
    protocol: str = Form(...),
) -> JSONResponse:
    email = email.strip()
    protocol = protocol.strip().lower()
    if protocol not in _VALID_WISHLIST_PROTOCOLS:
        raise HTTPException(status_code=400, detail="unsupported protocol")
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="email looks invalid")

    WISHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {"ts": int(time.time()), "protocol": protocol, "email": email},
        ensure_ascii=False,
    )
    with WISHLIST_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return JSONResponse({"ok": True})


def _client_ip(request: Request) -> str:
    """Resolve the originating client IP.

    uvicorn is started with --proxy-headers --forwarded-allow-ips=* so by the
    time FastAPI sees the request, request.client.host is already the leftmost
    X-Forwarded-For value (the real client). Fall back to "unknown" if the
    request somehow lacks a client (shouldn't happen behind Caddy).
    """
    return request.client.host if request.client else "unknown"


# Probe is cheap (one upstream GET) but accepts arbitrary base_url + api_key,
# making it a tempting key-scanning oracle. 15/min/IP is generous enough for
# a real user editing/correcting fields and tight enough to slow scanners.
# Preflight (per-model alive check) shares the same bucket — both are "fan
# one cheap upstream call out from this proxy" operations.
_PROBE_RATE_LIMIT = 15
_PROBE_RATE_WINDOW_S = 60.0


async def _preflight_or_422(
    request: Request,
    base_url: str,
    api_key: str,
    model: str,
    protocol: str,
) -> None:
    """Run model-alive preflight; raise HTTPException 422 with structured
    detail if the model is dead so the frontend can offer a one-click swap
    to the protocol's recommended model.

    Honors the same per-IP probe rate limit. force=true on the request body
    skips preflight entirely (escape hatch when relays false-negative on
    our 4-token ping).
    """
    form = await request.form()
    if str(form.get("force") or "").lower() in ("1", "true", "yes"):
        return

    ip = _client_ip(request)
    allowed, retry_after = check_rate(
        ip, limit=_PROBE_RATE_LIMIT, window_s=_PROBE_RATE_WINDOW_S
    )
    if not allowed:
        # Rate limited preflight — DON'T block submission. The detector
        # itself will surface the model_not_found if there is one; we just
        # skip the early-warning path until backoff expires.
        return

    alive, err = await probe_model_alive(base_url, api_key, model, protocol)
    if alive:
        return

    raise HTTPException(
        status_code=422,
        detail={
            "code": "model_not_alive",
            "message": (
                f"Model {model} is actually unavailable on this relay. It is "
                "listed in /v1/models, but real requests are rejected by the "
                "upstream."
            ),
            "model": model,
            "protocol": protocol,
            "upstream_error": err or "",
        },
    )


@app.post("/api/probe")
async def api_probe(
    request: Request,
    base_url: str = Form(...),
    api_key: str = Form(...),
) -> JSONResponse:
    """Probe a relay's /v1/models for the form's pre-submit pill.

    Always returns 200 with a structured payload — the frontend renders any
    upstream error inline rather than as a fetch failure. Exception: a 429
    is returned (with Retry-After) when the per-IP rate limit is exhausted,
    so the browser can back off and the frontend can show a clear "too
    many probes" pill instead of a generic upstream error.
    """
    ip = _client_ip(request)
    allowed, retry_after = check_rate(
        ip, limit=_PROBE_RATE_LIMIT, window_s=_PROBE_RATE_WINDOW_S
    )
    if not allowed:
        wait = int(retry_after) + 1
        return JSONResponse(
            {
                "ok": False,
                "auth_ok": True,
                "error": f"Probing too frequently. Please retry in {wait} seconds",
                "rate_limited": True,
            },
            status_code=429,
            headers={"Retry-After": str(wait)},
        )

    base_url = base_url.strip()
    api_key = api_key.strip()

    if not base_url.startswith(("http://", "https://")):
        return JSONResponse(
            {"ok": False, "error": "base_url must start with http(s)://"},
            status_code=200,
        )
    if not api_key or len(api_key) < 8:
        return JSONResponse(
            {"ok": False, "error": "api_key looks invalid"},
            status_code=200,
        )

    payload = await probe_relay(base_url, api_key)
    return JSONResponse(payload)


@app.post("/api/detect")
@app.post("/api/detect/claude")
async def api_detect_claude(
    request: Request,
    response: FastAPIResponse,
    base_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(...),
    mode: str = Form("full"),
    include_long_context: bool = Form(False),
    include_long_context_extreme: bool = Form(False),
) -> JSONResponse:
    base_url = base_url.strip()
    api_key = api_key.strip()
    model = model.strip()
    mode = mode.strip().lower()

    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="base_url must start with http(s)://")
    if not api_key or len(api_key) < 8:
        raise HTTPException(status_code=400, detail="api_key looks invalid")
    # Permissive model validation: relays often expose custom names like
    # "claude-opus-4-7-thinking" or vendor-prefixed variants. Detectors
    # use lookup_model() which does double-prefix matching and gracefully
    # skips thinking/PDF probes for unknown models, so we let anything
    # reasonable through and let the upstream relay decide what's valid.
    if not model or len(model) > 200:
        raise HTTPException(status_code=400, detail="model must be 1–200 chars")
    if mode not in _VALID_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {_VALID_MODES}")

    if request.url.path == "/api/detect":
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = '</api/detect/claude>; rel="successor-version"'
        inferred = _protocol_from_model(model)
        if inferred != "anthropic":
            await _preflight_or_422(request, base_url, api_key, model, inferred)
            job_id = await jobs.submit(
                base_url, api_key, model, mode,
                protocol=inferred,
                include_long_context=include_long_context,
                include_long_context_extreme=include_long_context_extreme,
            )
            return JSONResponse({"job_id": job_id, "status_url": f"/api/status/{job_id}"})
    elif _protocol_from_model(model) == "gemini":
        raise HTTPException(
            status_code=400,
            detail="This is a Gemini model. Please submit the detection on the /gemini page.",
        )
    elif _protocol_from_model(model) == "openai":
        raise HTTPException(
            status_code=400,
            detail="This is an OpenAI model. Please submit the detection on the /openai page.",
        )

    await _preflight_or_422(request, base_url, api_key, model, "anthropic")
    job_id = await jobs.submit(
        base_url, api_key, model, mode,
        protocol="anthropic",
        include_long_context=include_long_context,
        include_long_context_extreme=include_long_context_extreme,
    )
    # NOTE: never echo api_key back in the response
    return JSONResponse({"job_id": job_id, "status_url": f"/api/status/{job_id}"})


@app.post("/api/detect/openai")
async def api_detect_openai(
    request: Request,
    base_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(...),
    mode: str = Form("standard"),
    include_long_context: bool = Form(False),
    include_long_context_extreme: bool = Form(False),
) -> JSONResponse:
    base_url = base_url.strip()
    api_key = api_key.strip()
    model = model.strip()
    mode = mode.strip().lower()

    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="base_url must start with http(s)://")
    if not api_key or len(api_key) < 8:
        raise HTTPException(status_code=400, detail="api_key looks invalid")
    if not model or len(model) > 200:
        raise HTTPException(status_code=400, detail="model must be 1–200 chars")
    if mode not in _VALID_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {_VALID_MODES}")

    await _preflight_or_422(request, base_url, api_key, model, "openai")
    job_id = await jobs.submit(
        base_url, api_key, model, mode,
        protocol="openai",
        include_long_context=include_long_context,
        include_long_context_extreme=include_long_context_extreme,
    )
    return JSONResponse({"job_id": job_id, "status_url": f"/api/status/{job_id}"})


@app.post("/api/detect/gemini")
async def api_detect_gemini(
    request: Request,
    base_url: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(...),
    mode: str = Form("standard"),
) -> JSONResponse:
    base_url = base_url.strip()
    api_key = api_key.strip()
    model = model.strip()
    mode = mode.strip().lower()

    if not base_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="base_url must start with http(s)://")
    if not api_key or len(api_key) < 8:
        raise HTTPException(status_code=400, detail="api_key looks invalid")
    if not model or len(model) > 200:
        raise HTTPException(status_code=400, detail="model must be 1–200 chars")
    if mode not in _VALID_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {_VALID_MODES}")

    await _preflight_or_422(request, base_url, api_key, model, "gemini")
    job_id = await jobs.submit(base_url, api_key, model, mode, protocol="gemini")
    return JSONResponse({"job_id": job_id, "status_url": f"/api/status/{job_id}"})


@app.get("/api/status/{job_id}")
async def api_status(job_id: str) -> JSONResponse:
    j = await jobs.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="job not found")
    payload = {
        "job_id": j.id,
        "protocol": j.protocol,
        "status": j.status,
        "base_url": j.base_url,
        "target_model": j.target_model,
        "mode": j.mode,
        "created_at": j.created_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
    }
    if j.status == "done":
        payload["result_url"] = f"/r/{j.id}"
        payload["image_url"] = f"/r/{j.id}.jpg"
        payload["json_url"] = f"/api/result/{j.id}.json"
    elif j.status == "error":
        payload["error"] = j.error
    return JSONResponse(payload)


@app.get("/api/result/{job_id}.json")
async def api_result_json(job_id: str) -> JSONResponse:
    j = await jobs.get(job_id)
    if j is None or j.report is None:
        raise HTTPException(status_code=404, detail="result not ready")
    return JSONResponse(j.report)


# NOTE: declare .jpg route BEFORE the bare /r/{job_id} HTML route. Starlette
# path params match `[^/]+` greedily so `/r/{job_id}` would otherwise swallow
# `/r/foo.jpg` (job_id="foo.jpg") and shadow the image endpoint.
@app.get("/r/{job_id}.jpg")
async def result_jpg(job_id: str) -> Response:
    j = await jobs.get(job_id)
    if j is None or j.status != "done" or j.report is None:
        raise HTTPException(status_code=404, detail="result not ready")

    # Cache the JPG next to the JSON. Detection report is immutable, so once
    # generated we always serve from disk to spare the CPU.
    cache_path = jobs.image_path(job_id, j.protocol)
    if not cache_path.exists():
        png_bytes = render_report_jpg(j.report)
        cache_path.write_bytes(png_bytes)
    return Response(
        content=cache_path.read_bytes(),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'inline; filename="veridrop-{job_id}.jpg"',
            "Cache-Control": "public, max-age=86400",
        },
    )


_PROTOCOL_LABELS = {"anthropic": "Claude", "openai": "OpenAI", "gemini": "Gemini"}
_VERDICT_LABELS = {"passed": "Passed", "marginal": "At Risk", "failed": "Failed"}


def _seo_meta_for_report(report: dict) -> dict[str, str]:
    """Compute SEO title + description from a finished report.

    Each `/r/{job_id}` is a permanent landing page; without per-report meta,
    every report shows the same generic description and Google can't tell
    them apart for long-tail "what does relay X look like" queries. With
    domain + score + verdict in the meta, each one becomes its own indexable
    page.
    """
    base_url = str(report.get("base_url") or "")
    domain = base_url
    if "://" in base_url:
        domain = base_url.split("://", 1)[1].split("/", 1)[0]
    domain = domain or "relay"

    protocol = str(report.get("protocol") or "anthropic")
    proto_label = _PROTOCOL_LABELS.get(protocol, protocol)

    model = str(report.get("target_model") or "")
    score = float(report.get("total_score") or 0)
    verdict = str(report.get("verdict") or "failed")
    verdict_label = _VERDICT_LABELS.get(verdict, verdict)

    results = report.get("results") or []
    pass_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "pass")
    fail_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "fail")
    total = len(results)

    # Title diversity drives long-tail SEO: each report becomes its own
    # indexable page with a distinct query target ("X relay gpt-5.5 test").
    # Without the model, every report on the same domain shared one title.
    if model:
        title = (
            f"{domain} {proto_label} Relay {model} Detection: "
            f"{score:.0f}/100 {verdict_label} | Veridrop"
        )
    else:
        title = (
            f"{domain} {proto_label} Relay Detection: {score:.0f}/100 {verdict_label} | Veridrop"
        )
    description = (
        f"Full report of the {proto_label} relay detection for {domain}: "
        f"model {model}, total score {score:.0f}/100, verdict \"{verdict_label}\". "
        f"{pass_count} of {total} checks passed, {fail_count} failed. "
        f"Veridrop's field-level inspection identifies relay authenticity and quality."
    )
    og_description = (
        f"{domain} detection report: {score:.0f}/100 {verdict_label} ({pass_count}/{total} checks passed)"
    )
    return {
        "seo_title": title[:155],
        "seo_description": description[:160],
        "seo_og_description": og_description[:155],
    }


@app.get("/r/{job_id}", response_class=HTMLResponse)
async def result_page(request: Request, job_id: str) -> HTMLResponse:
    j = await jobs.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="job not found")
    if j.status != "done" or j.report is None:
        return templates.TemplateResponse(
            request, "running.html", {"job_id": job_id, "job": j},
        )
    # Domain feeds the breadcrumb (Home › Leaderboard › {domain} › Report).
    # Goes through is_valid_domain so a malformed base_url doesn't produce
    # a broken /leaderboard/{garbage} link.
    base_url = str(j.report.get("base_url") or "")
    domain = leaderboard._extract_domain(base_url)  # noqa: SLF001
    if domain and not leaderboard.is_valid_domain(domain):
        domain = ""
    return templates.TemplateResponse(
        request, "result.html",
        {
            "job_id": job_id,
            "report": j.report,
            "rows": _result_rows(j.report),
            "report_notes": _report_notes(j.report),
            "breadcrumb_domain": domain,
            **_seo_meta_for_report(j.report),
        },
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "ts": time.time()})


# SEO + AI GEO surface — see docs/SEO_AI_GEO_PLAN.md §3.1.A/B/C. These three
# files MUST be served at site root (not under /static) for crawlers to find
# them by convention.
@app.get("/robots.txt")
async def robots_txt() -> Response:
    return Response(
        content=(STATIC_DIR / "robots.txt").read_bytes(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/llms.txt")
async def llms_txt() -> Response:
    return Response(
        content=(STATIC_DIR / "llms.txt").read_bytes(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# (loc, changefreq, priority, template_filename) — lastmod resolved at
# request time from the template file's mtime. /leaderboard is special:
# its content changes whenever a new report lands, so we override its
# lastmod with the most recent report timestamp.
_STATIC_SITEMAP_URLS = [
    ("https://veridrop.org/",            "weekly",  "1.0",  "hub.html"),
    ("https://veridrop.org/claude",      "weekly",  "0.9",  "index.html"),
    ("https://veridrop.org/openai",      "weekly",  "0.9",  "openai.html"),
    ("https://veridrop.org/gemini",      "weekly",  "0.9",  "gemini.html"),
    ("https://veridrop.org/leaderboard", "daily",   "0.85", "leaderboard.html"),
    ("https://veridrop.org/faq",         "monthly", "0.8",  "faq.html"),
]

_SITEMAP_REPORT_DIRS = [
    Path("/opt/veridrop/web_data/jobs/anthropic"),
    Path("/opt/veridrop/web_data/jobs/openai"),
    Path("/opt/veridrop/web_data/jobs/gemini"),
    Path("/opt/veridrop/web_data/jobs"),  # legacy top-level
]


def _template_lastmod(filename: str) -> str:
    """Date string from a template's mtime, or '' if missing/unreadable."""
    try:
        mtime = (TEMPLATE_DIR / filename).stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")


@app.get("/sitemap.xml")
async def sitemap_xml() -> Response:
    """Dynamic sitemap — static product pages + reports + per-domain pages.

    `lastmod` is included for every URL so search engines know when to
    revisit. Static pages use the underlying template's mtime (≈ deploy
    time); /leaderboard and /leaderboard/{domain} use the most recent
    report timestamp since their content is data-driven.
    """
    # Aggregate once: powers both /leaderboard's lastmod (max across all
    # relays) and each /leaderboard/{domain} page's per-relay lastmod.
    relays, _ = leaderboard.aggregate()
    relay_last_checked = [r.last_checked for r in relays if r.last_checked]
    leaderboard_lastmod = (
        max(relay_last_checked).strftime("%Y-%m-%d")
        if relay_last_checked else ""
    )

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']

    for loc, freq, prio, tpl in _STATIC_SITEMAP_URLS:
        if loc.endswith("/leaderboard"):
            # Data-driven page: lastmod follows the data, not the template.
            lastmod = leaderboard_lastmod or _template_lastmod(tpl)
        else:
            lastmod = _template_lastmod(tpl)
        line = f"  <url><loc>{loc}</loc>"
        if lastmod:
            line += f"<lastmod>{lastmod}</lastmod>"
        line += f"<changefreq>{freq}</changefreq><priority>{prio}</priority></url>"
        lines.append(line)

    seen: set[str] = set()
    for dir_path in _SITEMAP_REPORT_DIRS:
        if not dir_path.is_dir():
            continue
        for json_path in sorted(dir_path.glob("*.json")):
            job_id = json_path.stem
            if job_id in seen:
                continue
            seen.add(job_id)
            try:
                lastmod = datetime.fromtimestamp(
                    json_path.stat().st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            except OSError:
                continue
            lines.append(
                f"  <url><loc>https://veridrop.org/r/{job_id}</loc>"
                f"<lastmod>{lastmod}</lastmod>"
                f"<changefreq>monthly</changefreq>"
                f"<priority>0.6</priority></url>"
            )

    # Per-domain detail pages — primary long-tail SEO surface. lastmod is
    # the relay's most recent report so Google revisits whenever new data
    # lands for that domain.
    for r in relays:
        if not leaderboard.is_valid_domain(r.domain):
            continue
        line = f"  <url><loc>https://veridrop.org/leaderboard/{r.domain}</loc>"
        if r.last_checked:
            line += f"<lastmod>{r.last_checked.strftime('%Y-%m-%d')}</lastmod>"
        line += "<changefreq>weekly</changefreq><priority>0.75</priority></url>"
        lines.append(line)

    lines.append("</urlset>\n")
    return Response(
        content="\n".join(lines),
        media_type="application/xml; charset=utf-8",
        headers={"Cache-Control": "public, max-age=600"},
    )


# ----- helpers shared with templates -----


_DETECTOR_DISPLAY = {
    "anthropic": [
        ("identity", "Identity Consistency"),
        ("behavioral_signature", "Behavioral Signature"),
        ("thinking_signature", "Thinking Signature"),
        ("consistency", "Model Consistency"),
        ("knowledge", "Knowledge Accuracy"),
        ("pdf", "PDF Document Recognition"),
        ("structured_output", "Structured Output"),
        ("protocol", "Protocol Conformance"),
        ("integrity", "Response Integrity"),
        ("token_usage", "Token Usage"),
        ("message_id", "Message ID Conformance"),
        ("long_context", "Long-Context Authenticity"),
    ],
    "openai": [
        ("basic_request", "Basic Request"),
        ("model_consistency", "Model Consistency"),
        ("function_calling", "Function Calling"),
        ("structured_output", "Structured Output"),
        ("protocol", "Protocol Conformance"),
        ("integrity", "Streaming Consistency"),
        ("token_billing", "Token Billing"),
        ("long_context", "Long-Context Authenticity"),
    ],
    "gemini": [
        ("basic_request", "Basic Request"),
        ("model_info", "Model Response Shape"),
        ("function_calling", "Function Calling"),
        ("structured_output", "Structured Output"),
        ("protocol", "Protocol Conformance"),
        ("integrity", "Streaming Consistency"),
        ("token_usage", "Token Usage"),
    ],
}


def _result_rows(report: dict) -> list[dict]:
    """Flatten results into the order/labels the result template expects."""
    by_name = {
        r.get("name"): r for r in report.get("results") or []
        if isinstance(r, dict)
    }
    out = []
    protocol = str(report.get("protocol") or "anthropic")
    display = _DETECTOR_DISPLAY.get(protocol, _DETECTOR_DISPLAY["anthropic"])
    for name, label in display:
        r = by_name.get(name) or {"status": "skip", "score": 0.0}
        status = str(r.get("status") or "skip")
        score = float(r.get("score") or 0.0)
        if status == "pass":
            label_short, css = "Passed", "ok"
        elif status == "skip" and name == "long_context":
            # long_context is opt-in. Skip can mean "user didn't check the
            # box", "model context too small", or "all tiers rate-limited".
            # The detail's summary clarifies which; the badge stays neutral.
            label_short, css = "Not Enabled", "muted"
        elif status == "skip" and name in {"token_billing", "token_usage"}:
            label_short, css = "Inconclusive", "muted"
        elif status == "skip":
            label_short, css = "Skipped", "muted"
        elif status == "error":
            label_short, css = "Error", "fail"
        elif score >= 70:
            label_short, css = "Warning", "warn"
        else:
            label_short, css = "Failed", "fail"
        out.append({
            "name": name,
            "label": label,
            "status": status,
            "label_short": label_short,
            "css": css,
            "score": score,
        })
    return out


def _report_notes(report: dict) -> list[dict[str, str]]:
    """Human-readable report notes for the result page.

    These are deliberately plain-English explanations. The raw detector JSON
    is useful for debugging, but public reports need to say what happened in
    terms a non-implementer can act on.
    """
    protocol = str(report.get("protocol") or "anthropic")
    if protocol == "gemini":
        return _gemini_report_notes(report)
    if protocol == "anthropic":
        return _anthropic_report_notes(report)
    if protocol != "openai":
        return []

    results = {
        r.get("name"): r for r in report.get("results") or []
        if isinstance(r, dict)
    }
    notes: list[dict[str, str]] = []

    structured = results.get("structured_output") or {}
    sd = structured.get("details") if isinstance(structured.get("details"), dict) else {}
    if structured.get("status") != "pass":
        text = str(sd.get("response_text") or "").strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:180] + "..."
        # Two distinct failure modes — different actionable signals:
        # - Markdown-fenced JSON: relay forwarded response_format and the
        #   model attempted JSON, but produced a code-fenced block instead of
        #   raw JSON. Likely a model behavior / weak strict-mode honoring.
        # - No JSON at all: the response_format was probably stripped by the
        #   relay's adapter layer.
        markdown_seen = bool(sd.get("markdown_json_seen"))
        if markdown_seen:
            message = (
                "The underlying model wrapped the JSON in a Markdown code block "
                "(```json ... ```). OpenAI strict mode should return bare JSON. "
                "This means response_format was forwarded, but the underlying "
                "model did not truly honor strict mode — usually a sign the "
                "relay routed the request to a non-GPT model."
            )
        else:
            message = (
                "The response was neither JSON nor code-block-wrapped. The "
                "request sent response_format=json_schema strict=true, but the "
                "relay most likely never forwarded this parameter to the backend."
            )
        if text:
            message += f" Actual response excerpt: {text}"
        notes.append({
            "title": "Structured output is not actually in effect",
            "body": message,
        })

    protocol_result = results.get("protocol") or {}
    pd = (
        protocol_result.get("details")
        if isinstance(protocol_result.get("details"), dict)
        else {}
    )
    issue_codes: set[str] = set()
    for issue in pd.get("issues") or []:
        if isinstance(issue, dict) and isinstance(issue.get("code"), str):
            issue_codes.add(issue["code"])
    impersonation_codes = issue_codes & {
        "usage_contains_claude_fields",
        "usage_contains_gemini_fields",
        "usage_source_non_openai",
    }
    if impersonation_codes:
        # Critical-severity adapter fingerprints — the relay is almost
        # certainly translating from a different upstream backend.
        notes.append({
            "title": "Relay suspected of impersonating OpenAI",
            "body": (
                "The usage field in the response contains fields only used by "
                "Anthropic / Google backends (such as claude_cache_creation_*, "
                "gemini_*, or usage_source self-reporting non-openai). "
                "This strongly suggests the relay forwarded your request to a "
                "different vendor's backend and repackaged it as an OpenAI "
                "response — the so-called GPT output may not be generated by a "
                "genuine OpenAI model."
            ),
        })
    elif "usage_mixed_token_fields" in issue_codes:
        # Lower-severity: just naming residue, may or may not mean translation.
        notes.append({
            "title": "Traces of an adapter layer in the response",
            "body": (
                "The returned usage contains both OpenAI's prompt_tokens/"
                "completion_tokens and Anthropic/Responses-style input_tokens/"
                "output_tokens. This usually indicates a translation layer in "
                "between, but is not enough to confirm a backend swap."
            ),
        })
    token_billing = results.get("token_billing") or {}
    td = (
        token_billing.get("details")
        if isinstance(token_billing.get("details"), dict)
        else {}
    )
    if token_billing:
        if token_billing.get("status") == "skip":
            notes.append({
                "title": "Cannot determine yet whether tokens are over-reported",
                "body": (
                    "The API did not provide complete enough token usage "
                    "information, so over-counting cannot be confirmed this time."
                ),
            })
        elif token_billing.get("status") != "pass":
            notes.append({
                "title": "Token billing at risk",
                "body": td.get("evaluation_zh") or (
                    "Token statistics show a significant deviation. Watch out "
                    "for over-counting or accounting errors."
                ),
            })
        # Note: when token_billing passes, the green check in the detector
        # list already conveys this. We deliberately don't add a redundant
        # "Token billing OK" note — report notes should only carry signal that
        # needs user attention.
    return notes


def _anthropic_report_notes(report: dict) -> list[dict[str, str]]:
    results = {
        r.get("name"): r for r in report.get("results") or []
        if isinstance(r, dict)
    }
    token_usage = results.get("token_usage") or {}
    td = (
        token_usage.get("details")
        if isinstance(token_usage.get("details"), dict)
        else {}
    )
    if not token_usage:
        return []
    if token_usage.get("status") == "skip":
        return [{
            "title": "Cannot determine yet whether tokens are over-reported",
            "body": "The API did not return a complete usage field, so "
                    "over-counting cannot be confirmed this time.",
        }]
    if token_usage.get("status") != "pass":
        return [{
            "title": "Token usage at risk",
                "body": td.get("evaluation_zh") or (
                    "The usage field is missing or internally inconsistent. "
                    "Do not rely directly on this relay's token counts for "
                    "billing reconciliation."
            ),
        }]
    return []


def _gemini_report_notes(report: dict) -> list[dict[str, str]]:
    results = {
        r.get("name"): r for r in report.get("results") or []
        if isinstance(r, dict)
    }
    notes: list[dict[str, str]] = []

    structured = results.get("structured_output") or {}
    sd = structured.get("details") if isinstance(structured.get("details"), dict) else {}
    if structured.get("status") != "pass":
        notes.append({
            "title": "Structured output is not actually in effect",
            "body": sd.get("evaluation_zh") or (
                "The request sent the OpenAI-compatible "
                "response_format=json_schema strict=true, but the Gemini relay's "
                "returned content could not be parsed against the schema."
            ),
        })

    token_usage = results.get("token_usage") or {}
    td = (
        token_usage.get("details")
        if isinstance(token_usage.get("details"), dict)
        else {}
    )
    if token_usage and token_usage.get("status") != "pass":
        notes.append({
            "title": "Token usage at risk",
                "body": td.get("evaluation_zh") or (
                    "The usage field is incomplete or internally inconsistent. "
                    "Do not rely directly on it for billing reconciliation."
            ),
        })

    integrity = results.get("integrity") or {}
    idetails = (
        integrity.get("details")
        if isinstance(integrity.get("details"), dict)
        else {}
    )
    if integrity and integrity.get("status") != "pass":
        notes.append({
            "title": "Streaming response deviation detected",
                "body": idetails.get("evaluation_zh") or (
                    "The text, finish reason, or usage fields do not align "
                    "between stream and non-stream responses."
            ),
        })

    if not notes:
        notes.append({
            "title": "Gemini OpenAI-compatible protocol performs well",
            "body": (
                "Basic requests, model fields, tool calls, structured output, "
                "streaming responses, and token usage fields largely conform to "
                "the OpenAI Chat Completions specification."
            ),
        })
    return notes
