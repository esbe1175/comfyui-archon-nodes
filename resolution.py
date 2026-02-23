import math
import re


def _clamp(value, minimum):
    return value if value >= minimum else minimum


def _nearest_multiple(value, divisor, minimum):
    rounded = int(round(value / divisor)) * divisor
    return _clamp(rounded, minimum)


def _best_fit_resolution(width, height, target_pixels, divisor):
    width = max(1, int(width))
    height = max(1, int(height))
    divisor = max(1, int(divisor))
    target_pixels = max(1.0, float(target_pixels))

    min_dim = divisor
    aspect_target = width / height
    ideal_w = math.sqrt(target_pixels * aspect_target)
    ideal_h = math.sqrt(target_pixels / aspect_target)

    base_w = _nearest_multiple(ideal_w, divisor, min_dim)
    base_h = _nearest_multiple(ideal_h, divisor, min_dim)

    max_steps = 80
    best = None

    for dw in range(-max_steps, max_steps + 1):
        cand_w = base_w + (dw * divisor)
        if cand_w < min_dim:
            continue
        for dh in range(-max_steps, max_steps + 1):
            cand_h = base_h + (dh * divisor)
            if cand_h < min_dim:
                continue

            pixels = cand_w * cand_h
            aspect = cand_w / cand_h
            pixel_err = abs(pixels - target_pixels) / target_pixels
            aspect_err = abs(aspect - aspect_target) / aspect_target
            score = (pixel_err * 10.0) + aspect_err

            candidate = (score, pixel_err, aspect_err, abs(dw) + abs(dh), cand_w, cand_h, pixels)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return min_dim, min_dim, float(min_dim * min_dim)

    return best[4], best[5], float(best[6])


_PERSON_COUNT_RE = re.compile(r"^(\d+)\s*(girl|girls|boy|boys|other|others)$", re.IGNORECASE)


def _split_tags(text):
    if text is None:
        return []
    return [tag.strip() for tag in str(text).split(",") if tag.strip()]


def _escape_parentheses(text):
    return re.sub(r"(?<!\\)([()])", r"\\\1", text)


def _normalize_tag(tag):
    normalized = tag.replace("_", " ").strip()
    return _escape_parentheses(normalized)


def _parse_person_count_tag(tag):
    collapsed = tag.strip().lower().replace("_", "").replace(" ", "")
    match = _PERSON_COUNT_RE.match(collapsed)
    if not match:
        return None

    number = int(match.group(1))
    noun = match.group(2)

    if noun.startswith("girl"):
        suffix = "girl" if number == 1 else "girls"
    elif noun.startswith("boy"):
        suffix = "boy" if number == 1 else "boys"
    else:
        suffix = "other" if number == 1 else "others"

    return f"{number}{suffix}"


def _collect_person_count_tags(general_tags):
    person_count = []
    remaining = []
    seen = set()

    for tag in general_tags:
        parsed = _parse_person_count_tag(tag)
        if parsed is None:
            remaining.append(tag)
            continue

        if parsed not in seen:
            person_count.append(parsed)
            seen.add(parsed)

    return person_count, remaining


def _prefix_artist_tag(tag):
    cleaned = _normalize_tag(tag)
    cleaned = cleaned.lstrip("@").strip()
    if not cleaned:
        return ""
    return f"@{cleaned}"


def _join_non_empty_sections(sections):
    parts = []
    for section in sections:
        for tag in section:
            if tag:
                parts.append(tag)
    return ", ".join(parts)


def _tag_dedupe_key(tag):
    key = str(tag).strip().lower()
    key = key.replace("\\(", "(").replace("\\)", ")")
    key = key.replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    return key


class ScaleToBestFitResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 1, "max": 65535, "step": 1}),
                "height": ("INT", {"default": 1216, "min": 1, "max": 65535, "step": 1}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 100.0, "step": 0.01}),
                "divisor": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "applied_scale", "megapixels")
    FUNCTION = "calculate"
    CATEGORY = "Archon/Resolution"

    def calculate(self, width, height, scale, divisor):
        original_pixels = float(width * height)
        target_pixels = original_pixels * float(scale)
        out_w, out_h, out_pixels = _best_fit_resolution(width, height, target_pixels, divisor)
        applied_scale = out_pixels / original_pixels
        megapixels = out_pixels / 1_000_000.0
        return (int(out_w), int(out_h), float(applied_scale), float(megapixels))


class MegapixelsToBestFitResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 1, "max": 65535, "step": 1}),
                "height": ("INT", {"default": 1216, "min": 1, "max": 65535, "step": 1}),
                "megapixels": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 100.0, "step": 0.01}),
                "divisor": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("width", "height", "applied_scale", "actual_megapixels")
    FUNCTION = "calculate"
    CATEGORY = "Archon/Resolution"

    def calculate(self, width, height, megapixels, divisor):
        original_pixels = float(width * height)
        target_pixels = float(megapixels) * 1_000_000.0
        out_w, out_h, out_pixels = _best_fit_resolution(width, height, target_pixels, divisor)
        applied_scale = out_pixels / original_pixels
        actual_megapixels = out_pixels / 1_000_000.0
        return (int(out_w), int(out_h), float(applied_scale), float(actual_megapixels))


class PromptTagAssembler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta": ("STRING", {"default": "masterpiece, year 2025, new", "multiline": False}),
                "rating": ("STRING", {"default": "safe", "multiline": False}),
                "person_count": ("STRING", {"default": "", "multiline": False}),
                "character": ("STRING", {"default": "", "multiline": False}),
                "series": ("STRING", {"default": "", "multiline": False}),
                "artist": ("STRING", {"default": "", "multiline": False}),
                "general": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "assemble"
    CATEGORY = "Archon/Prompt"

    def assemble(self, meta, rating, person_count, character, series, artist, general):
        meta_tags = [_normalize_tag(tag) for tag in _split_tags(meta)]
        rating_tags = [_normalize_tag(tag) for tag in _split_tags(rating)]

        general_raw = _split_tags(general)
        detected_person_count, general_remaining = _collect_person_count_tags(general_raw)

        # Manual person_count is optional; detected values from general are always included.
        manual_person_count = []
        manual_seen = set()
        for tag in _split_tags(person_count):
            parsed = _parse_person_count_tag(tag)
            if parsed is None:
                continue
            if parsed not in manual_seen:
                manual_person_count.append(parsed)
                manual_seen.add(parsed)

        final_person_count = []
        seen_counts = set()
        for tag in manual_person_count + detected_person_count:
            if tag not in seen_counts:
                final_person_count.append(tag)
                seen_counts.add(tag)

        character_tags = [_normalize_tag(tag) for tag in _split_tags(character)]
        series_tags = [_normalize_tag(tag) for tag in _split_tags(series)]
        artist_tags = [_prefix_artist_tag(tag) for tag in _split_tags(artist)]
        general_tags = [_normalize_tag(tag) for tag in general_remaining]

        prompt = _join_non_empty_sections(
            [
                meta_tags + rating_tags,
                final_person_count,
                character_tags,
                series_tags,
                artist_tags,
                general_tags,
            ]
        )
        return (prompt,)


class MergeGeneralTags:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "general_a": ("STRING", {"default": "", "multiline": True}),
                "general_b": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("general",)
    FUNCTION = "merge"
    CATEGORY = "Archon/Prompt"

    def merge(self, general_a, general_b):
        merged = []
        seen = set()

        for raw_tag in _split_tags(general_a) + _split_tags(general_b):
            key = _tag_dedupe_key(raw_tag)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(_normalize_tag(raw_tag))

        return (", ".join(merged),)


NODE_CLASS_MAPPINGS = {
    "ScaleToBestFitResolution": ScaleToBestFitResolution,
    "MegapixelsToBestFitResolution": MegapixelsToBestFitResolution,
    "PromptTagAssembler": PromptTagAssembler,
    "MergeGeneralTags": MergeGeneralTags,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScaleToBestFitResolution": "Scale to Best Fit Resolution",
    "MegapixelsToBestFitResolution": "Megapixels to Best Fit Resolution",
    "PromptTagAssembler": "Prompt Tag Assembler",
    "MergeGeneralTags": "Merge General Tags",
}
