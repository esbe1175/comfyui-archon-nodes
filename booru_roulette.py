import io
import json
import os
import random
import re
from urllib.parse import urlencode
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "booru_roulette_config.json")
GELBOORU_API_BASE = "https://gelbooru.com/index.php"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:108.0) Gecko/20100101 Firefox/108.0"

# Source: Gelbooru forum thread "Is it possible to search by date?" (id=2184),
# values shared by user (verified Feb 2026).
YEAR_START_IDS = {
    2007: 1,
    2008: 132204,
    2009: 407581,
    2010: 676805,
    2011: 1020436,
    2012: 1382384,
    2013: 1757245,
    2014: 2116687,
    2015: 2536659,
    2016: 2994708,
    2017: 3497715,
    2018: 4038325,
    2019: 4550300,
    2020: 5065398,
    2021: 5781187,
    2022: 6790766,
    2023: 8084028,
    2024: 9425769,
    2025: 11229486,
    2026: 13222588,
}
FIRST_KNOWN_YEAR = min(YEAR_START_IDS.keys())
LATEST_KNOWN_YEAR = max(YEAR_START_IDS.keys())
LATEST_KNOWN_ID = YEAR_START_IDS[LATEST_KNOWN_YEAR]


class QueryError(RuntimeError):
    def __init__(self, message, raw_request=""):
        super().__init__(message)
        self.raw_request = str(raw_request or "")


BAD_TAGS = {
    "mixed-language_text",
    "watermark",
    "text",
    "english_text",
    "speech_bubble",
    "signature",
    "artist_name",
    "translation",
    "twitter_username",
    "twitter_logo",
    "patreon_username",
    "commentary_request",
    "tagme",
    "commentary",
    "mosaic_censoring",
    "instagram_username",
    "text_focus",
    "translation_request",
    "fake_text",
    "text_bubble",
    "qr_code",
}

_PERSON_RE = re.compile(r"^(\d+)(girl|girls|boy|boys|other|others)$", re.IGNORECASE)


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _cfg_or_env(config, key, env):
    value = str(config.get(key, "")).strip()
    if value:
        return value
    return os.getenv(env, "").strip()


def _split_csv(text):
    if not text:
        return []
    s = str(text).strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]


def _normalize_query_tag(tag):
    return tag.strip().replace(" ", "_")


def _escape_prompt_tag(tag):
    return str(tag).replace("(", r"\(").replace(")", r"\)")


def _normalize_person(tag):
    compact = tag.lower().replace("_", "").replace(" ", "")
    m = _PERSON_RE.match(compact)
    if not m:
        return None
    n = int(m.group(1))
    kind = m.group(2)
    if kind.startswith("girl"):
        suffix = "girl" if n == 1 else "girls"
    elif kind.startswith("boy"):
        suffix = "boy" if n == 1 else "boys"
    else:
        suffix = "other" if n == 1 else "others"
    return f"{n}{suffix}"


def _extract_person_counts(tags):
    counts = []
    remaining = []
    seen = set()
    for tag in tags:
        parsed = _normalize_person(tag)
        if parsed is None:
            remaining.append(tag)
            continue
        if parsed not in seen:
            counts.append(parsed)
            seen.add(parsed)
    return counts, remaining


def _compare_tag_key(tag):
    s = str(tag).strip().lower()
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _split_space_tags(text):
    if not text:
        return []
    return [t.strip() for t in str(text).split(" ") if t.strip()]


def _normalize_rating_tag(value):
    v = str(value or "").strip().lower()
    return {
        "g": "safe",
        "general": "general",
        "s": "sensitive",
        "sensitive": "sensitive",
        "safe": "general",
        "q": "questionable",
        "questionable": "questionable",
        "e": "explicit",
        "explicit": "explicit",
    }.get(v, v)


def _to_anima_rating(value):
    v = _normalize_rating_tag(value)
    return {
        "general": "safe",
        "sensitive": "sensitive",
        "questionable": "nsfw",
        "explicit": "explicit",
    }.get(v, v)


def _candidate_image_urls(post, preview_only):
    # Prefer preview, but allow fallback when preview is missing/unreadable.
    keys = ("preview_url", "sample_url", "file_url") if preview_only else ("file_url", "sample_url", "preview_url")

    out = []
    seen = set()
    for key in keys:
        u = str(post.get(key, "")).strip()
        if not u:
            continue
        if u.startswith("//"):
            u = "https:" + u
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _http_get_with_cf_fallback(url, headers, timeout):
    res = requests.get(url, headers=headers, timeout=timeout)
    body_start = (res.text or "")[:200].lower() if hasattr(res, "text") else ""
    if res.status_code in (403, 503) and ("just a moment" in body_start or "cloudflare" in body_start):
        try:
            import cloudscraper  # type: ignore

            scraper = cloudscraper.create_scraper(browser={"browser": "firefox", "platform": "windows", "mobile": False})
            res = scraper.get(url, headers=headers, timeout=timeout)
        except Exception:
            pass
    return res


def _fetch_post_image(post, preview_only):
    urls = _candidate_image_urls(post, preview_only)
    if not urls:
        raise RuntimeError("No usable image URL in post data.")

    last_error = None
    for url in urls:
        try:
            parsed = urlparse(url)
            headers = {
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": f"{parsed.scheme}://{parsed.netloc}/",
            }
            res = _http_get_with_cf_fallback(url, headers=headers, timeout=30)
            if res.status_code >= 400:
                raise RuntimeError(f"HTTP {res.status_code}")
            ctype = (res.headers.get("content-type") or "").lower()
            if ctype and "image" not in ctype:
                raise RuntimeError(f"Non-image response content-type={ctype}")
            img = Image.open(io.BytesIO(res.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img, url
        except Exception as e:
            last_error = f"{url} -> {e}"
            continue
    raise RuntimeError(f"Failed to download/parse post image. Last error: {last_error}")


def _clean_tag_list(raw_tags, remove_tags, shuffle_tags):
    tags = [t for t in str(raw_tags or "").split(" ") if t]
    if shuffle_tags:
        random.shuffle(tags)
    remove_set = {x.lower() for x in _split_csv(remove_tags)}
    out = []
    for tag in tags:
        if tag in BAD_TAGS:
            continue
        if tag.lower() in remove_set:
            continue
        out.append(_escape_prompt_tag(tag))
    return out


def _build_gelbooru_tags(tags, exclude_tags, rating, year_id_filters, random_seed):
    chunks = ["-animated"]
    chunks.extend([_normalize_query_tag(t) for t in _split_csv(tags)])
    chunks.extend([f"-{_normalize_query_tag(t)}" for t in _split_csv(exclude_tags)])
    if rating != "All":
        chunks.append(f"rating:{ {'General':'general','Sensitive':'sensitive','Questionable':'questionable','Explicit':'explicit'}[rating] }")
    chunks.extend(year_id_filters)
    seed = int(random_seed)
    if 0 <= seed <= 10000:
        chunks.append(f"sort:random:{seed}")
    else:
        chunks.append("sort:random")
    return " ".join(chunks).strip()


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _gelbooru_lookup_tag_types(config, raw_tags):
    user_id = _cfg_or_env(config, "gelbooru_user_id", "GELBOORU_USER_ID")
    api_key = _cfg_or_env(config, "gelbooru_api_key", "GELBOORU_API_KEY")
    if not user_id or not api_key:
        return {}

    tags = [t for t in raw_tags if t]
    if not tags:
        return {}

    out = {}
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    for batch in _chunked(tags, 90):
        params = {
            "page": "dapi",
            "s": "tag",
            "q": "index",
            "json": "1",
            # Gelbooru expects names separated by spaces, not '+'.
            "names": " ".join(batch),
            "user_id": user_id,
            "api_key": api_key,
        }
        try:
            res = requests.get(GELBOORU_API_BASE, params=params, headers=headers, timeout=30)
            if res.status_code >= 400:
                continue
            data = res.json()
            entries = data.get("tag") if isinstance(data, dict) else None
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for entry in entries:
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                try:
                    out[name] = int(entry.get("type", 0))
                except Exception:
                    out[name] = 0
        except Exception:
            continue
    return out


def _get_gelbooru_auth(config):
    user_id = _cfg_or_env(config, "gelbooru_user_id", "GELBOORU_USER_ID")
    api_key = _cfg_or_env(config, "gelbooru_api_key", "GELBOORU_API_KEY")
    if not user_id or not api_key:
        raise RuntimeError(
            "Gelbooru credentials missing. Set gelbooru_user_id/gelbooru_api_key in config "
            "or GELBOORU_USER_ID/GELBOORU_API_KEY env vars."
        )
    return user_id, api_key


def _gelbooru_posts_query(config, tags, limit):
    user_id, api_key = _get_gelbooru_auth(config)
    params = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "json": "1",
        "limit": str(limit),
        "tags": tags,
        "user_id": user_id,
        "api_key": api_key,
    }
    # Expose the exact post query without leaking credentials.
    request_preview_params = {k: v for k, v in params.items() if k not in ("user_id", "api_key")}
    request_preview = f"{GELBOORU_API_BASE}?{urlencode(request_preview_params)}"

    res = requests.get(
        GELBOORU_API_BASE,
        params=params,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=30,
    )
    if res.status_code >= 400:
        raise QueryError(f"Gelbooru API error {res.status_code}: {res.text[:200]!r}", request_preview)
    data = res.json()
    posts = data.get("post", []) if isinstance(data, dict) else []
    if isinstance(posts, dict):
        posts = [posts]
    return posts, request_preview


def _year_start_id(year):
    y = int(year)
    if y <= FIRST_KNOWN_YEAR:
        return YEAR_START_IDS[FIRST_KNOWN_YEAR]
    if y in YEAR_START_IDS:
        return YEAR_START_IDS[y]
    # No guessing above known table: clamp to latest known-year boundary.
    return LATEST_KNOWN_ID


def _normalize_year_filter_mode(mode):
    m = str(mode or "").strip()
    if m in ("Any",):
        return "Any"
    if m in ("Between", "Between (year_start-year_end)"):
        return "Between"
    if m in ("After", "After (uses year_start)"):
        return "After"
    if m in ("Before", "Before (uses year_start)"):
        return "Before"
    return "Any"


def _build_year_id_filters(year_filter_mode, year_start, year_end):
    mode = _normalize_year_filter_mode(year_filter_mode)
    y_start = int(year_start)
    y_end = int(year_end)

    if mode == "Any":
        return []
    if mode == "Between":
        lo_year = min(y_start, y_end)
        hi_year = max(y_start, y_end)
        if hi_year < FIRST_KNOWN_YEAR:
            return []
        lo_id = _year_start_id(lo_year)
        if hi_year >= LATEST_KNOWN_YEAR:
            return [f"id:>={lo_id}"]
        hi_id = _year_start_id(hi_year + 1)
        return [f"id:>={lo_id}", f"id:<{hi_id}"]
    if mode == "After":
        if y_start >= LATEST_KNOWN_YEAR:
            # No known next-year boundary yet; use strict greater-than latest known start ID.
            return [f"id:>{LATEST_KNOWN_ID}"]
        lo_id = _year_start_id(y_start + 1)
        return [f"id:>={lo_id}"]
    if mode == "Before":
        if y_start <= FIRST_KNOWN_YEAR:
            return []
        hi_id = _year_start_id(y_start)
        return [f"id:<{hi_id}"]
    return []


def _describe_year_filter(mode, year_start, year_end):
    mode = _normalize_year_filter_mode(mode)
    if mode == "Any":
        return "year:any"
    if mode == "Between":
        lo = min(int(year_start), int(year_end))
        hi = max(int(year_start), int(year_end))
        return f"year:between {lo}-{hi}"
    if mode == "After":
        return f"year:after {int(year_start)}"
    if mode == "Before":
        return f"year:before {int(year_start)}"
    return "year:any"


def _query_gelbooru(config, tags, exclude_tags, rating, year_filter_mode, year_start, year_end, random_seed):
    year_id_filters = _build_year_id_filters(year_filter_mode, year_start, year_end)

    final_tags = _build_gelbooru_tags(tags, exclude_tags, rating, year_id_filters, random_seed)
    posts, request_preview = _gelbooru_posts_query(config, final_tags, 1)
    if not posts:
        raise QueryError("No Gelbooru posts matched your query.", request_preview)
    post = posts[0]

    raw_tag_list = [t for t in str(post.get("tags", "")).split(" ") if t]
    type_map = _gelbooru_lookup_tag_types(config, raw_tag_list)
    artist_tags = [t for t in raw_tag_list if type_map.get(t) == 1]
    series_tags = [t for t in raw_tag_list if type_map.get(t) == 3]
    character_tags = [t for t in raw_tag_list if type_map.get(t) == 4]

    return {
        "booru": "gelbooru",
        "id": str(post.get("id", "")),
        "post_url": f"https://gelbooru.com/index.php?page=post&s=view&id={post.get('id', '')}",
        "tags": str(post.get("tags", "")),
        "rating": str(post.get("rating", "")),
        "file_url": post.get("file_url", ""),
        "sample_url": post.get("sample_url", ""),
        "preview_url": post.get("preview_url", ""),
        "width": int(post.get("width", 0) or 0),
        "height": int(post.get("height", 0) or 0),
        "character_tags": " ".join(character_tags),
        "series_tags": " ".join(series_tags),
        "artist_tags": " ".join(artist_tags),
        "year_filter_meta": _describe_year_filter(year_filter_mode, year_start, year_end),
        "query_tags": final_tags,
        "raw_request": request_preview,
    }


class BooruRouletteNode:
    def __init__(self):
        self.last = None
        self.last_prompt = ""
        self.last_signature = None
        self.last_image = None
        self.last_image_url = ""
        self.last_preview_only = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"multiline": False, "default": "1girl"}),
                "exclude_tags": ("STRING", {"multiline": False, "default": ""}),
                "remove_tags": ("STRING", {"multiline": False, "default": ""}),
                "rating": (["All", "General", "Sensitive", "Questionable", "Explicit"], {"default": "General"}),
                "anima_rating_format": ("BOOLEAN", {"default": False}),
                "year_filter_mode": (
                    [
                        "Any",
                        "Between (year_start-year_end)",
                        "After (uses year_start)",
                        "Before (uses year_start)",
                    ],
                    {"default": "Any"},
                ),
                "year_start": ("INT", {"default": 2020, "min": 2007, "max": 2100, "step": 1}),
                "year_end": ("INT", {"default": 2026, "min": 2007, "max": 2100, "step": 1}),
                "random_seed": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "shuffle_tags": ("BOOLEAN", {"default": True}),
                "preview_only": ("BOOLEAN", {"default": True}),
                "return_picture": ("BOOLEAN", {"default": True}),
                "use_last_result": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "image",
        "meta",
        "rating",
        "person_count",
        "character",
        "series",
        "artist",
        "general",
        "post_id",
        "post_url",
        "image_url",
        "raw_tags",
        "width",
        "height",
        "raw_request",
    )
    FUNCTION = "run"
    CATEGORY = "Booru Roulette"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(
        self,
        tags,
        exclude_tags,
        remove_tags,
        rating,
        anima_rating_format,
        year_filter_mode,
        year_start,
        year_end,
        random_seed,
        shuffle_tags,
        preview_only,
        return_picture,
        use_last_result,
    ):
        signature = (
            tags,
            exclude_tags,
            remove_tags,
            rating,
            bool(anima_rating_format),
            year_filter_mode,
            int(year_start),
            int(year_end),
            int(random_seed),
            bool(shuffle_tags),
        )

        if use_last_result and self.last is not None and self.last_signature == signature:
            post = self.last
            prompt = self.last_prompt
        else:
            cfg = _load_config()
            try:
                post = _query_gelbooru(cfg, tags, exclude_tags, rating, year_filter_mode, year_start, year_end, random_seed)
            except QueryError as e:
                if e.raw_request:
                    raise RuntimeError(f"{e} | raw_request:{e.raw_request}") from e
                raise
            prompt_tags = _clean_tag_list(post.get("tags", ""), remove_tags, shuffle_tags)
            prompt = ",".join(prompt_tags)
            self.last = post
            self.last_prompt = prompt
            self.last_signature = signature

        all_prompt_tags = [t.strip() for t in prompt.split(",") if t.strip()]
        person_count_tags, general_tags = _extract_person_counts(all_prompt_tags)
        person_count = ", ".join(person_count_tags)

        meta = (
            f"{post.get('year_filter_meta', '')} | "
            f"query_tags:{post.get('query_tags', '')}"
        ).strip()
        final_rating = _normalize_rating_tag(post.get("rating", "")) or (rating.lower() if rating != "All" else "")
        if anima_rating_format:
            final_rating = _to_anima_rating(final_rating)

        character_tokens = _split_space_tags(post.get("character_tags", ""))
        series_tokens = _split_space_tags(post.get("series_tags", ""))
        artist_tokens = _split_space_tags(post.get("artist_tags", ""))

        character = ", ".join([t.replace("_", " ") for t in character_tokens])
        series = ", ".join([t.replace("_", " ") for t in series_tokens])
        artist = ", ".join([t.replace("_", " ") for t in artist_tokens])

        categorized_keys = {
            _compare_tag_key(t)
            for t in (character_tokens + series_tokens + artist_tokens)
        }
        general_filtered = [t for t in general_tags if _compare_tag_key(t) not in categorized_keys]
        general = ", ".join(general_filtered)

        width = int(post.get("width", 0) or 0)
        height = int(post.get("height", 0) or 0)

        image_url = ""
        if return_picture:
            if use_last_result and self.last_image is not None and self.last_preview_only == preview_only:
                img = self.last_image
            else:
                try:
                    img, image_url = _fetch_post_image(post, preview_only=preview_only)
                    self.last_image = img
                    self.last_image_url = image_url
                    self.last_preview_only = preview_only
                    if width <= 0 or height <= 0:
                        width, height = img.size
                except Exception as e:
                    print(
                        f"[BooruRoulette] Image fetch failed for Gelbooru post {post.get('id', '')}: {e}. "
                        "Using metadata-size fallback image."
                    )
                    # Keep execution alive; return metadata dimensions when image is unavailable.
                    w = width if width > 0 else 1
                    h = height if height > 0 else 1
                    img = Image.new("RGB", (w, h), color=(0, 0, 0))
                    image_url = ""
        else:
            w = width if width > 0 else 1
            h = height if height > 0 else 1
            img = Image.new("RGB", (w, h), color=(0, 0, 0))
            image_url = self.last_image_url if use_last_result else ""

        return (
            pil2tensor(img),
            meta,
            final_rating,
            person_count,
            character,
            series,
            artist,
            general,
            str(post.get("id", "")),
            str(post.get("post_url", "")),
            image_url,
            str(post.get("tags", "")),
            int(width if width > 0 else img.size[0]),
            int(height if height > 0 else img.size[1]),
            str(post.get("raw_request", "")),
        )


NODE_CLASS_MAPPINGS = {
    "BooruRouletteNode": BooruRouletteNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooruRouletteNode": "Booru Roulette",
}
