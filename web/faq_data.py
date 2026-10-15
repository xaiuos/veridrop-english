"""FAQ content for /faq.

Single source of truth: drives both the rendered HTML page AND the FAQPage
JSON-LD schema embedded in <head>. Editing one place keeps the two in sync.

Authoring rules (see docs/SEO_AI_GEO_PLAN.md §3.2 for why):

* Each answer is **definition-first**: state the concrete answer in the
  first sentence, then expand with specifics or numbers.
* Keep answers ≤ ~280 Chinese chars / ~200 English words — long enough to give an LLM something
  to cite, short enough that the FAQPage schema doesn't bloat.
* Use real numbers (token counts, percentages, signature lengths) wherever
  possible — pages with concrete numbers get cited 33% more by AI engines.
* Anchors (id) become deep-linkable via /faq#<category>-<question> — never
  rename them after publication; that breaks external links.
* The order inside a category is the order users see; put the highest-
  intent / most-searched-for question first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FAQEntry:
    id: str
    q: str
    a: str


@dataclass(frozen=True)
class FAQCategory:
    id: str
    title: str
    intro: str
    entries: tuple[FAQEntry, ...]


FAQ_CATEGORIES: tuple[FAQCategory, ...] = (
    FAQCategory(
        id="basics",
        title="Relay Basics",
        intro=(
            "An \"AI API relay\" is a third-party API forwarding service that proxies your requests to "
            "the official OpenAI / Anthropic / Google endpoints and returns the responses to you. "
            "Relays vary wildly — some forward traffic 1:1, while others rewrite fields, swap models, "
            "or inject system prompts. This set of questions explains what relays are, why people use them, "
            "and where the risks lie."
        ),
        entries=(
            FAQEntry(
                "what-is-relay",
                "What is an AI API relay?",
                "A relay (also called a \"mirror\" or \"proxy\") is a third-party API forwarding service "
                "that sends your requests to the official OpenAI / Anthropic / Google endpoints and "
                "returns the responses to you as-is. Common reasons they exist: local networks cannot reach "
                "the official domains directly, cheaper wholesale pricing, and one unified interface for many models.",
            ),
            FAQEntry(
                "why-use-relay",
                "Why do people use relays? Isn't the official API enough?",
                "Three common reasons: ① accessing anthropic.com / openai.com requires a VPN in some regions; "
                "② some relays repackage wholesale quotas from multiple vendors at retail, 30-70% cheaper than official; "
                "③ some relays unify Claude / GPT / Gemini behind a single OpenAI-compatible interface, so you maintain only one codebase.",
            ),
            FAQEntry(
                "relay-vs-official",
                "What's the difference between a relay and the official API?",
                "Official APIs have deterministic fields, signatures, and error formats; relays differ enormously — "
                "some forward 1:1, others rewrite usage fields, strip thinking blocks, substitute different models, "
                "or even inject extra system prompts. What Veridrop detects is exactly \"what was tampered with in the middle\".",
            ),
            FAQEntry(
                "relay-legal",
                "Are relays legal? Will using one cause problems?",
                "Legally a gray area that depends on the partnership model. The real practical risks are three: "
                "① <strong>runaway risk</strong> — a relay can shut down, raise prices, or vanish with your balance at any time; "
                "think twice before topping up large amounts. "
                "② <strong>model responses inconsistent with official</strong> — the relay may tamper in the middle "
                "(model substitution, capability stripping, inflated token counts), affecting your downstream business stability. "
                "③ <strong>prompt content being logged</strong> — conversations involving trade secrets or user data are best kept "
                "away from relays of unknown provenance. Start small, pick reputable operators, and go direct to the official API for high-stakes workloads.",
            ),
            FAQEntry(
                "relay-steal-key",
                "What are the risks of using a relay's API key?",
                "Veridrop has tested many relays in practice; there are 5 common pitfalls (all with real observed cases): "
                "① <strong>silent model substitution</strong> — you pay for GPT but what actually runs may be Claude or another "
                "cheaper model. It's invisible on the surface, but the writing style, reasoning, and problem-solving habits are off, "
                "and real work easily goes wrong. "
                "② <strong>advanced features broken</strong> — things like strict formatted output, PDF reading, and tool calling "
                "frequently fail on \"swapped\" relays — because the cheap model swapped in behind the scenes simply can't do them. "
                "③ <strong>billing numbers you can't trust</strong> — ask the same question a few times and the charge differs each time; "
                "some relays never tell you how many tokens were used, so you can't reconcile the account at all. "
                "Your top-up may well cost more than official with no receipts to check. "
                "④ <strong>models on the menu that are out of stock</strong> — the relay claims dozens of models, "
                "but try some of them and you get \"model not found\". Veridrop tests this for you before you submit a detection, "
                "so you don't wait half a minute only to get an all-zeros report. "
                "⑤ <strong>rules changed without notice</strong> — the same key works fine today, and tomorrow suddenly demands "
                "\"upgrade your plan\", \"group expired\", or gets banned outright. Relay backend rules can change without any notice, "
                "so try a small amount first and run Veridrop once to see actual behavior before any large top-up.",
            ),
        ),
    ),
    FAQCategory(
        id="authenticity",
        title="Authenticity",
        intro=(
            "Relay \"counterfeits\" are the industry's biggest pain point — you think you're using Claude while Kiro / Amazon Q "
            "actually runs behind it; you think you're using GPT-4o while the backend is Claude Haiku. "
            "This set of questions explains the fakery techniques and how to detect them."
        ),
        entries=(
            FAQEntry(
                "verify-authentic",
                "How do I know I'm getting real Claude / GPT / Gemini and not a stand-in?",
                "Three directions: ① protocol fields (whether id prefix, object, finish_reason conform to official specs); "
                "② capability fingerprints (thinking signature cryptographic signatures, PDF multimodal, function calling shape); "
                "③ usage fields (whether usage contains foreign-origin traces like claude_cache_creation_*). "
                "Veridrop combines these three into 7-10 detections.",
            ),
            FAQEntry(
                "common-fake-tactics",
                "What are the common \"fakery\" tactics relays use?",
                "5 common ones: ① running Claude stand-ins via Kiro / Amazon Q / Bedrock; "
                "② routing GPT requests to a Claude backend (usage fields expose claude_* residue); "
                "③ quietly swapping GPT-4o for GPT-4o-mini to save cost; ④ stripping advanced capabilities like thinking blocks and PDF multimodal; "
                "⑤ injecting extra system prompts to rewrite identity answers.",
            ),
            FAQEntry(
                "ask-who-are-you",
                "Can I just ask \"who are you\" to verify the model?",
                "No. Modern models are trained to answer identity questions properly, and relays can inject a system prompt "
                "to make the model say \"I am Claude\". Veridrop assigns only 5% weight to the \"identity consistency\" detection, "
                "treating it as the weakest auxiliary signal — what's truly trustworthy is a cryptographic-grade metric like thinking signature.",
            ),
            FAQEntry(
                "kiro-amazon-q",
                "How do I expose Kiro / Amazon Q impersonating Claude?",
                "Kiro and Amazon Q are Amazon's Claude stand-in gateways; their responses carry no Claude thinking signature "
                "(they use AWS Bedrock's simplified interface, which does not return server-side signatures). "
                "Veridrop's thinking_signature detection scores them 0 outright and fails the overall detection.",
            ),
            FAQEntry(
                "gpt-becomes-claude",
                "How do I catch a relay routing GPT requests to a Claude backend?",
                "Look at the response's usage field. Native OpenAI has only prompt_tokens / completion_tokens / total_tokens. "
                "If fields like claude_cache_creation_5_m_tokens or usage_source: anthropic appear, "
                "you can essentially conclude the relay is doing protocol conversion. Veridrop flags such fingerprints as critical severity.",
            ),
        ),
    ),
    FAQCategory(
        id="claude",
        title="Claude API Relays",
        intro=(
            "Claude is the protocol Veridrop detects most deeply — because Claude has a cryptographic-grade "
            "verifiable metric in the thinking signature, and Veridrop is currently the only one in the industry using this signal."
        ),
        entries=(
            FAQEntry(
                "thinking-signature",
                "What is a thinking signature? Why is it the gold standard for authenticity verification?",
                "A thinking signature is the server-side cryptographic artifact returned in the signature field "
                "of Claude responses when extended thinking is enabled, 500-2000 characters long. It is generated "
                "by Anthropic's servers and cryptographically verifiable — relays theoretically cannot forge it. "
                "Veridrop treats it as a core detection with 25% weight.",
            ),
            FAQEntry(
                "claude-code-relay",
                "How do I choose a Claude Code relay?",
                "Four key things: ① run Veridrop once and check whether thinking signature scores 100; "
                "② check whether usage contains foreign-origin fields beyond claude_*; "
                "③ check stream / non-stream consistency (relays often handle the two paths differently); "
                "④ check whether message ids follow the msg_ / toolu_ / srvtoolu_ prefix spec.",
            ),
            FAQEntry(
                "claude-detector-coverage",
                "Which dimensions does Veridrop's Claude detection cover?",
                "11 items, by weight: thinking signature (25%) > behavioral signature (15%) > "
                "structured output (12%) > consistency / knowledge / token usage (10% each) > pdf (8%) > "
                "identity / protocol / integrity / message_id (5% each). "
                "standard mode runs 9 items in ~45 seconds; full mode runs all 11 in ~75 seconds.",
            ),
            FAQEntry(
                "claude-stream-no-thinking",
                "Why does a Claude relay return no thinking in stream mode?",
                "This is a known API drift in Claude Opus 4.7 — with the adaptive thinking + streaming + summarized "
                "combination, no thinking block appears in the SSE stream (non-stream works normally). "
                "Veridrop's thinking_signature detector has switched to non-streaming for this case, ensuring reliable detection.",
            ),
            FAQEntry(
                "claude-cost",
                "Why does detection consume tokens? How much does it cost?",
                "Detection sends real requests — roughly 12 API calls per standard run, ~3000-5000 total tokens "
                "(depending on whether thinking is triggered). At Haiku prices that's ~$0.012, Sonnet ~$0.05, "
                "Opus ~$0.20. Veridrop itself charges nothing — you pay the real token costs to the upstream API.",
            ),
            FAQEntry(
                "claude-zero-score",
                "Does a 0 score mean the relay is completely useless?",
                "It depends. Common zero-score causes: ① the API key is invalid or the quota is exhausted (an invalid-detection banner will say so); "
                "② the relay doesn't support the model you picked (preflight rejects it early with a 422); ③ the relay is genuinely terrible. "
                "Read each detector's details rather than only the total score.",
            ),
        ),
    ),
    FAQCategory(
        id="openai",
        title="OpenAI Relays",
        intro=(
            "The OpenAI protocol has no cryptographic-grade metric like the thinking signature, so Veridrop focuses on "
            "\"protocol conformance + adapter-layer fingerprinting\" on OpenAI — catching traces of relays disguising "
            "Anthropic / Google backends as GPT."
        ),
        entries=(
            FAQEntry(
                "openai-fake-evidence",
                "What's the hard evidence that an OpenAI relay swapped GPT for Claude?",
                "The response's usage field is the most direct evidence. Truly native OpenAI has only prompt_tokens / "
                "completion_tokens / total_tokens. If you find claude_cache_creation_5_m_tokens, "
                "usage_source: anthropic, or input_tokens / output_tokens (Anthropic naming), "
                "you can essentially conclude the relay is doing protocol conversion. Veridrop marks this critical and fails the detector outright.",
            ),
            FAQEntry(
                "openai-75-marginal",
                "My GPT-4o scored 75 but the verdict is marginal — how should I read that?",
                "75 falls in the \"pass\" band on its own, but if there are any critical-severity issues "
                "(especially the protocol detector reporting multiple criticals), Veridrop cuts the verdict down to marginal — "
                "the ring turns yellow with an \"at risk\" notice. This usually means the interface works but the relay is doing "
                "protocol conversion, and what actually runs may not be a real GPT model.",
            ),
            FAQEntry(
                "openai-strict-json",
                "What does it mean when response_format=json_schema strict doesn't take effect?",
                "OpenAI strict mode should return pure JSON, not wrapped in a Markdown code block. "
                "If the response comes back as ```json {...} ```, there are two possibilities: ① the relay didn't forward the response_format parameter; "
                "② it was forwarded "
                "but the underlying model (very likely not GPT) doesn't understand strict mode. "
                "Veridrop distinguishes the two cases via the markdown_json_seen flag.",
            ),
            FAQEntry(
                "openai-model-mismatch",
                "My relay's model field returns gpt-4o but the responses clearly don't look like GPT — what now?",
                "The model field is just a string; a relay can fill in anything. Authenticity has to come from other metrics: "
                "① Veridrop's protocol conformance detector scans usage-field fingerprints; "
                "② behavioral differences (GPT is terse, Claude is more polite and verbose); "
                "③ token-count CV across repeated calls (the model consistency detector).",
            ),
            FAQEntry(
                "openai-no-models",
                "What if an OpenAI relay shows 0 usable models?",
                "Veridrop's pre-probe lists how many of the key's models on the relay belong to this protocol. "
                "If the current page is OpenAI but there are 0 GPT models, a yellow card is shown with a notice"
                "\"This key has no GPT models on this relay, but has X Claude models / Y Gemini models → "
                "jump over and test those in one click\".",
            ),
        ),
    ),
    FAQCategory(
        id="gemini",
        title="Gemini API Relays",
        intro=(
            "Veridrop's Gemini detection only uses the OpenAI-compatible path (POST /chat/completions), "
            "because 99% of third-party Gemini relays use exactly that. The Gemini 3 series enables thinking by default, "
            "which places special requirements on max_tokens settings and token usage accounting."
        ),
        entries=(
            FAQEntry(
                "gemini-model-not-found",
                "Why do Gemini relays so often return model_not_found?",
                "Three possibilities: ① the relay only proxies part of the Gemini lineup (many relays carry only the 3.x preview series, not 2.5); "
                "② model-name spelling differences (gemini-2.5-flash vs models/gemini-2.5-flash); ③ the model has been retired. "
                "Veridrop runs a preflight before submission, identifying dead models within 500ms and offering the usable list instead.",
            ),
            FAQEntry(
                "gemini-thinking-default",
                "What should max_completion_tokens be when testing Gemini 3.x preview?",
                "At least 64; 128+ recommended. Gemini 3 has thinking on by default and burns 30-60 reasoning_tokens before any text. "
                "If max is too small (< 32), thinking fills the budget with no room for text, and the response is an empty string plus "
                "finish_reason=length. Veridrop's detector already sets max to 64-384.",
            ),
            FAQEntry(
                "gemini-no-native",
                "Why doesn't Veridrop support the native Gemini /v1beta/models/X:generateContent path?",
                "99% of third-party Gemini relays expose only the OpenAI-compatible protocol, and Google's official side also offers "
                "the OpenAI-compatible endpoint /v1beta/openai. Maintaining two independent detection logic sets is costly and easily "
                "introduces a translation layer that distorts results, so Veridrop concentrates on doing the OpenAI-compatible path thoroughly.",
            ),
            FAQEntry(
                "gemini-pick",
                "How do I pick a Gemini relay?",
                "Three tiers: ① must support gemini-2.5-flash or gemini-3-flash-preview (2026 stable); "
                "② scores ≥ 80 on Veridrop's protocol detection; "
                "③ no foreign-origin residue in the usage field beyond gemini_* (meaning no protocol conversion). "
                "Veridrop automatically sorts the relay's model list in recommended order.",
            ),
            FAQEntry(
                "gemini-google-official",
                "Can Veridrop test Google's official Gemini API too?",
                "Yes. Set base_url to https://generativelanguage.googleapis.com/v1beta/openai "
                "and use the AIza... key from Google AI Studio as the api_key. The official Google endpoint should score 100 on all 7 items, "
                "and can serve as a baseline for comparing third-party relays.",
            ),
        ),
    ),
    FAQCategory(
        id="veridrop",
        title="Using Veridrop",
        intro="Questions about the tool itself: usage, reading reports, self-hosting, and more.",
        entries=(
            FAQEntry(
                "how-to-use",
                "How do I use Veridrop? What's the minimal flow?",
                "Three steps: ① open veridrop.org/claude (or /openai, /gemini); "
                "② enter the relay's base_url + api_key and pick a model (the dropdown auto-lists the models available to that key on that relay); "
                "③ click \"Start detection\" and get the report in 30-60 seconds. Report URLs are permanently shareable (/r/{job_id}), and a JPG can be downloaded.",
            ),
            FAQEntry(
                "modes-difference",
                "What's the difference between the standard / quick / full modes?",
                "quick (~15s, 3-5 items) is for fast triage; standard (~40s, 7-8 items) is the recommended default; "
                "full (~70s, all 10 items) is the complete detection, recommended for 1:1 comparison against the official baseline. "
                "Every mode runs core items like the thinking signature.",
            ),
            FAQEntry(
                "how-to-read-score",
                "How do I read the report score?",
                "Weighted total: ≥85 excellent (green) / 70-84 pass (light green) / 50-69 marginal (yellow) / <50 failed (red). "
                "But any detector reporting a critical-severity issue cuts the verdict down to marginal (yellow ring) — "
                "even a 70+ score stays non-green. Open each detector's details and sub_checks to find what lost points.",
            ),
            FAQEntry(
                "report-link-public",
                "Who can see my detection report link /r/xxx?",
                "Anyone who gets the link can view it (anonymous, unencrypted). Reports never contain your API key in plaintext, "
                "only a masked form (sk-y7xU••••••0h). If you tested a private relay you don't want public, don't share the URL. "
                "An opt-in \"private report\" toggle is planned.",
            ),
            FAQEntry(
                "self-host",
                "Can I self-host Veridrop locally?",
                "Yes, Veridrop is fully open source. Clone the repo → pip install -e .[web] → "
                "run uvicorn web.server:app and you're done. The CLI also works standalone: relay-detector detect "
                "--base-url ... --api-key ... --mode full. See the README for details.",
            ),
        ),
    ),
    FAQCategory(
        id="privacy",
        title="Privacy & Security",
        intro=(
            "Handing an API key to a detection tool is inherently risky, so Veridrop's data-handling promises need "
            "to be verifiable — our code is fully open source; you can audit it and self-host it."
        ),
        entries=(
            FAQEntry(
                "key-stored",
                "Does Veridrop log my API key?",
                "No. The API key lives only in the in-memory job object and is wiped as soon as the job finishes "
                "(success or failure) — never written to report JSON, logs, or disk. In the report the key appears only "
                "in masked form (sk-y7xU••••••0h); the code is open source and verifiable.",
            ),
            FAQEntry(
                "vs-cctest-privacy",
                "How does this differ from cctest.ai on data security?",
                "cctest.ai also claims not to upload keys, but the code is closed-source and unverifiable. Veridrop is fully open source — "
                "you can clone the code and run it yourself (git clone ... && uvicorn web.server:app), "
                "or audit the server-side code (fully public on GitHub).",
            ),
            FAQEntry(
                "upstream-cost",
                "Does the detection process incur upstream costs?",
                "Yes, but very little. standard mode makes about 12 real requests with 3000-5000 total tokens. "
                "That's roughly $0.012 on Haiku, $0.05 on Sonnet, $0.20 on Opus. "
                "That money goes to the upstream (relay or official); Veridrop itself charges nothing.",
            ),
            FAQEntry(
                "personal-info",
                "What personal information is in a report? Can it be deleted?",
                "Reports contain only: ① a masked base_url; ② a masked api_key; ③ detection results. "
                "No names, emails, or IPs. If you need a specific report deleted, just open a GitHub issue.",
            ),
        ),
    ),
    FAQCategory(
        id="compare",
        title="Tool Comparison",
        intro=(
            "Veridrop isn't the only relay-detection tool around. This set explains how Veridrop differs "
            "from similar tools, and how to pick a relay."
        ),
        entries=(
            FAQEntry(
                "vs-cctest",
                "How is Veridrop different from cctest.ai?",
                "cctest only covers Claude (single protocol) and fights evasion with \"black-box detection\", but with limited dimensions. "
                "Veridrop offers three protocols (Claude / OpenAI / Gemini) + cryptographic thinking signature + "
                "cross-protocol auto-jump + pre-submission dead-model identification + fully open source, and on OpenAI it can identify "
                "the adapter-layer fingerprint of \"GPT that's actually Claude\".",
            ),
            FAQEntry(
                "vs-hvoy",
                "How is Veridrop different from hvoy.ai?",
                "hvoy also supports three protocols, but its detection dimensions are shallow (mainly protocol compliance). "
                "Veridrop has cryptographic thinking-signature verification on Claude (exclusive), "
                "protocol-conversion fingerprinting on OpenAI (usage_source and other critical-level signals), "
                "and thinking-by-default-aware handling on Gemini — deeper across the board.",
            ),
            FAQEntry(
                "vs-leaderboards",
                "How do relay comparison leaderboards (aiapipk.com / Zhihu reviews) relate to Veridrop?",
                "Those are \"subjective reviews\" or \"ad-driven\"; Veridrop is \"objective technical detection\". "
                "They complement each other: shortlist reputable relays via leaderboards, then verify them technically with Veridrop.",
            ),
            FAQEntry(
                "how-to-pick-relay",
                "How do I pick a relay? What are the hard criteria?",
                "Veridrop recommends 5 hard criteria: ① thinking signature ≥ 100; "
                "② protocol critical_issue_count = 0; ③ stream / non-stream usage consistent; "
                "④ model id field matches the requested model (model_consistency pass); "
                "⑤ completion_tokens CV < 0.10 across repeated requests (stability).",
            ),
        ),
    ),
)


def total_question_count() -> int:
    return sum(len(c.entries) for c in FAQ_CATEGORIES)


def faqpage_jsonld() -> dict:
    """Build a single FAQPage schema.org JSON-LD covering every entry.

    Schema spec recommends one FAQPage per page with all Q/A as mainEntity.
    Multiple FAQPages on the same URL are NOT recommended — they confuse
    the LLM extractor about which set is canonical.
    """
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": entry.q,
                "acceptedAnswer": {"@type": "Answer", "text": entry.a},
            }
            for cat in FAQ_CATEGORIES
            for entry in cat.entries
        ],
    }


def find_entry(anchor: str) -> tuple[FAQCategory, FAQEntry] | None:
    """Look up a Q by its anchor (e.g. 'claude-thinking-signature').

    Anchor format: '<category-id>-<entry-id>'. Used by product pages that
    deep-link to specific FAQ entries.
    """
    for cat in FAQ_CATEGORIES:
        prefix = cat.id + "-"
        if not anchor.startswith(prefix):
            continue
        entry_id = anchor[len(prefix):]
        for e in cat.entries:
            if e.id == entry_id:
                return cat, e
    return None
