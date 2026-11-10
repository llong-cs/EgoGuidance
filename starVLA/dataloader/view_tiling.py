# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Unified view-role management for multi-camera observations and gen targets.

Standard StarVLA-PI feeds multiple camera views as N separate images in the fixed
``video_keys`` order, distinguished only by that implicit order — fragile, and a wrong
order silently corrupts the semantics. This module instead manages views by SEMANTIC
ROLE. Each dataset maps every camera to one canonical role (see DataConfig.
video_view_roles); everything downstream is driven by role, not raw key order:

  * VIEW_ROLES — the four canonical roles (central_global / auxiliary_global /
    primary_effector / secondary_effector), in the fixed order that IS both the
    multi-image order (view_tiling='off') and the 2x2 tile grid.
  * tile_roles(role->frame, cell) — composite up to four role-keyed frames into the
    fixed 2x2 canvas; a missing OR dropped role leaves its cell black (zero-pad).
  * sample_dropped_roles(...) — per-sample random view drop over the non-anchor roles
    (central_global is never dropped); shared by obs and the gen target of a sample.
  * layout_text(active) — the four-view layout paragraph interleaved into the prompt so
    the model reads which cell is which view (view_tiling='tile').
  * resolve_image_size / resize_cover_crop — the single obs_image_size resize (aspect-
    preserving cover + center-crop) used for every image the model perceives.

view_tiling only controls whether the observation is one 2x2 canvas ('tile') or N
role-ordered images ('off'); the gen target is chosen INDEPENDENTLY by gen_video_key
(a role, or 'tile').
"""

import random

from omegaconf import OmegaConf
from PIL import Image, ImageOps


def normalize_obs_resize_mode(value) -> str:
    """Return the observation resize mode, preserving ``fixed`` as the default."""
    mode = "fixed" if value is None else str(value).strip().lower()
    if mode not in ("fixed", "native"):
        raise ValueError(f"obs_resize_mode must be 'fixed' or 'native', got {value!r}")
    return mode


def normalize_view_tiling_mode(value) -> str:
    """Normalize YAML/dotlist ``off`` while keeping accidental truthy values loud.

    OmegaConf's dotlist parser follows YAML boolean spelling and may turn the
    literal ``off`` into ``False`` even when the source was a quoted string.
    ``False`` and an unset value therefore mean ``off``. ``True`` is deliberately
    rejected rather than guessed to mean ``tile``.
    """
    if value is None or value is False:
        return "off"
    if value is True:
        raise ValueError("view_tiling must be 'off' or 'tile'; boolean True is ambiguous")
    mode = str(value).strip().lower()
    if mode not in ("off", "tile"):
        raise ValueError(f"view_tiling must be 'off' or 'tile', got {value!r}")
    return mode


def normalize_view_prompt_mode(value) -> str:
    """Validate the observation-prompt contract.

    Every observation is role-aware: each image is paired with an explicit
    semantic-view description. ``role`` is the only supported mode, and an
    omitted value resolves to it.
    """
    mode = "role" if value is None else str(value).strip().lower()
    if mode != "role":
        raise ValueError(f"view_prompt_mode must be 'role', got {value!r}")
    return "role"


def resolve_image_size(size, default=None):
    """Resolve an image resize target as a ``(W, H)`` int tuple (or ``None`` = keep raw).

    ONE knob — ``obs_image_size`` — governs the fixed-mode resolution of every image the
    model perceives (robot & ego observations AND the pre-encode gen-target frames). Pass
    ``size`` (``obs_image_size``) and an optional ``default``; there is deliberately no
    separate resize knob for observation canvases or generation targets.

    Raises on a malformed non-empty size (wrong arity or a non-positive dimension) so a bad
    ``[0, H]`` / ``[W, H, D]`` config fails here with a clear message.
    """
    chosen = size or default
    if not chosen:
        return None
    wh = tuple(int(v) for v in chosen)
    if len(wh) != 2 or any(v <= 0 for v in wh):
        raise ValueError(f"image size must be two positive ints [W, H], got {list(chosen)}")
    return wh


def resize_cover_crop(image: Image.Image, size) -> Image.Image:
    """Resize preserving aspect ratio to COVER ``size``, then center-crop to exactly it.

    Reaches a fixed (W, H) WITHOUT the geometric distortion of a plain stretch — the same
    recipe the ``auto`` gen-target encoder uses (``QWen3_5._resize_to_cover_center_crop``),
    so the observation and the gen target share one aspect-preserving normalization.
    """
    w, h = int(size[0]), int(size[1])
    return ImageOps.fit(image.convert("RGB"), (w, h), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))


# ── Unified view-role system ────────────────────────────────────────────────
# Every camera view a dataset ships is mapped (in its DataConfig, video_view_roles)
# to ONE of these four canonical roles, so context assembly is driven by role
# SEMANTICS instead of the fragile implicit ordering of raw video_keys. This order
# IS the fixed multi-image order (view_tiling=off) and the 2x2 tile grid order.
VIEW_ROLES = ("central_global", "auxiliary_global", "primary_effector", "secondary_effector")

# Role -> (row, col) in the 2x2 tile canvas:
#   central_global   | auxiliary_global      (top-left  | top-right)
#   primary_effector | secondary_effector    (bottom-left | bottom-right)
_ROLE_GRID = {
    "central_global": (0, 0),
    "auxiliary_global": (0, 1),
    "primary_effector": (1, 0),
    "secondary_effector": (1, 1),
}
# The primary task-context role: must always be present and is NEVER dropped.
ANCHOR_ROLE = "central_global"


def normalize_view_role(role) -> str:
    """Validate a canonical view role name."""
    r = str(role).strip().lower()
    if r not in VIEW_ROLES:
        raise ValueError(f"unknown view role {role!r}; expected one of {list(VIEW_ROLES)}")
    return r


def order_roles(present) -> list:
    """Return the given roles in canonical VIEW_ROLES order (drops unknown/duplicate)."""
    present = {normalize_view_role(r) for r in present}
    return [r for r in VIEW_ROLES if r in present]


def tile_roles(role_to_frame: dict, cell_size) -> Image.Image:
    """Composite up to four role-keyed frames into the fixed 2x2 role canvas.

    ``role_to_frame`` maps a canonical role -> PIL frame, only for roles that are
    PRESENT and NOT dropped. Any role absent from the dict (missing camera OR a dropped
    view) leaves its cell BLACK (zero-pad, DreamZero-style). ``cell_size`` is ONE cell's
    ``(W, H)``; the returned canvas is ``(2W, 2H)``.
    """
    cell_w, cell_h = int(cell_size[0]), int(cell_size[1])
    canvas = Image.new("RGB", (2 * cell_w, 2 * cell_h))  # all-black -> missing cells stay black
    for role, frame in role_to_frame.items():
        if frame is None:
            continue
        r, c = _ROLE_GRID[normalize_view_role(role)]
        canvas.paste(frame.convert("RGB").resize((cell_w, cell_h), Image.Resampling.BILINEAR), (c * cell_w, r * cell_h))
    return canvas


def sample_dropped_roles(present_roles, drop_prob, protect=(), rng=None) -> set:
    """Randomly pick roles to drop for ONE training sample (shared by obs + gen target).

    Candidates = present roles minus ``ANCHOR_ROLE`` (central_global, never dropped) minus
    ``protect`` (e.g. the role a specific gen target predicts — dropping it would make the
    gen target all-black). Each candidate is dropped independently with prob ``drop_prob``.
    """
    if not drop_prob or float(drop_prob) <= 0:
        return set()
    blocked = {ANCHOR_ROLE, *(normalize_view_role(r) for r in protect)}
    candidates = [r for r in order_roles(present_roles) if r not in blocked]
    picker = rng if rng is not None else random
    return {r for r in candidates if picker.random() < float(drop_prob)}


_ROLE_TILE_POSITION = {
    "central_global": "Top-left",
    "auxiliary_global": "Top-right",
    "primary_effector": "Bottom-left",
    "secondary_effector": "Bottom-right",
}


def layout_text(active: dict) -> str:
    """Build the four-view layout prompt paragraph (view_tiling='tile').

    ``active`` maps each PRESENT-and-not-dropped role -> its natural-language description.
    Every role missing from ``active`` (absent camera OR dropped view) is labelled an
    ``inactive view``, matching the black cell it occupies in the 2x2 canvas.
    """
    lines = ["The video uses a semantic four-view layout."]
    for role in VIEW_ROLES:
        position = _ROLE_TILE_POSITION[role]
        if role in active:
            lines.append(f"{position}: {active[role]}.")
        else:
            lines.append(f"{position}: inactive view.")
    return "\n".join(lines)


def build_view_prompt_contract(role_descriptions: dict, view_tiling) -> dict:
    """Build the serializable role-prompt contract saved with a checkpoint.

    ``role_descriptions`` maps active canonical roles to their training text.
    The returned text list is exactly what inference must pair with its image
    list when no training-time view drop is active.
    """
    mode = normalize_view_tiling_mode(view_tiling)
    normalized = {normalize_view_role(role): str(description) for role, description in role_descriptions.items()}
    roles = order_roles(normalized)
    if ANCHOR_ROLE not in roles:
        raise ValueError(f"view prompt contract requires anchor role {ANCHOR_ROLE!r}; " f"active roles={roles}.")
    invalid = [role for role in roles if not normalized[role].strip()]
    if invalid:
        raise ValueError(f"view prompt descriptions must be non-empty; invalid roles={invalid}.")
    if mode == "tile":
        image_roles = ["semantic_2x2_role_canvas"]
        texts = [layout_text(normalized)]
    else:
        image_roles = roles
        texts = [normalized[role] for role in roles]
    return {
        "version": 1,
        "mode": "role",
        "view_tiling": mode,
        "image_roles": image_roles,
        "texts": texts,
    }


def publish_view_prompt_contract(data_cfg, contract: dict, *, context: str) -> None:
    """Persist one deterministic role-prompt contract into a stream config."""
    if data_cfg is None or not hasattr(data_cfg, "get"):
        raise ValueError(
            f"{context}: role-aware observations require a mutable dataset config "
            "so resolved_view_prompt can be saved with the checkpoint."
        )
    existing = data_cfg.get("resolved_view_prompt", None)
    if existing is not None:
        existing = OmegaConf.to_container(existing, resolve=True) if OmegaConf.is_config(existing) else existing
        if existing != contract:
            raise ValueError(
                f"{context}: one data stream resolved incompatible view prompt "
                f"contracts: existing={existing}, new={contract}."
            )
        return
    data_cfg["resolved_view_prompt"] = contract


def configure_robot_view_contract(dataset, data_config, data_cfg, *, context: str):
    """Resolve and attach a robot dataset's camera-role observation contract.

    ``data_config`` owns the raw-camera -> semantic-role/description declarations;
    ``dataset`` receives the active maps used while packing samples. The deterministic
    no-drop prompt is also written into ``data_cfg`` for checkpointed inference.
    """
    normalize_view_prompt_mode(data_cfg.get("view_prompt_mode", None) if data_cfg is not None else None)

    # Imported lazily to keep view_tiling independent from the LeRobot config module
    # at import time (that module itself uses the role constants defined above).
    from starVLA.dataloader.gr00t_lerobot.data_config import resolve_view_roles

    view_roles, view_descriptions = resolve_view_roles(data_config)
    role_descriptions = {role: view_descriptions[key] for key, role in view_roles.items()}
    contract = build_view_prompt_contract(
        role_descriptions,
        data_cfg.get("view_tiling", None) if data_cfg is not None else None,
    )
    publish_view_prompt_contract(data_cfg, contract, context=context)
    dataset.view_roles = view_roles
    dataset.view_descriptions = view_descriptions
    return dataset


def assemble_role_observation(
    frames_by_video_key: dict,
    view_roles: dict,
    view_descriptions: dict,
    data_cfg,
    *,
    context: str,
) -> tuple[list[Image.Image], list[str]]:
    """Build role-ordered images and their 1:1 semantic prompt texts.

    This is the shared robot observation path for action-only and action+generation
    datasets. It maps physical camera keys to canonical roles, orders them, applies
    the configured resize/tile policy, and emits descriptions aligned with the final
    image list.
    """
    cfg = {} if data_cfg is None else data_cfg
    normalize_view_prompt_mode(cfg.get("view_prompt_mode", None))
    if not frames_by_video_key:
        raise ValueError(f"{context}: role-aware observations require at least one camera frame.")

    missing_roles = [key for key in frames_by_video_key if key not in view_roles]
    if missing_roles:
        raise ValueError(f"{context}: active cameras have no semantic role: {missing_roles}.")
    missing_descriptions = [
        key
        for key in frames_by_video_key
        if key not in view_descriptions
        or not isinstance(view_descriptions[key], str)
        or not view_descriptions[key].strip()
    ]
    if missing_descriptions:
        raise ValueError(f"{context}: active cameras have no non-empty view description: {missing_descriptions}.")

    role_to_frame = {}
    role_to_description = {}
    for key, frame in frames_by_video_key.items():
        role = normalize_view_role(view_roles[key])
        if role in role_to_frame:
            raise ValueError(f"{context}: multiple active cameras map to role {role!r}.")
        if frame is None:
            raise ValueError(f"{context}: camera {key!r} produced no frame.")
        role_to_frame[role] = frame
        role_to_description[role] = view_descriptions[key].strip()

    roles = order_roles(role_to_frame)
    if ANCHOR_ROLE not in roles:
        raise ValueError(
            f"{context}: role-aware observations require anchor role {ANCHOR_ROLE!r}; " f"active roles={roles}."
        )

    view_tiling = normalize_view_tiling_mode(cfg.get("view_tiling", None))
    obs_resize_mode = normalize_obs_resize_mode(cfg.get("obs_resize_mode", None))
    obs_size = resolve_image_size(cfg.get("obs_image_size"), (224, 224)) if obs_resize_mode == "fixed" else None
    cell_size = resolve_image_size(cfg.get("tile_cell_size"), obs_size or (256, 256))

    if view_tiling == "tile":
        images = [tile_roles(role_to_frame, cell_size)]
        texts = [layout_text(role_to_description)]
    else:
        images = [
            resize_cover_crop(role_to_frame[role], obs_size) if obs_size is not None else role_to_frame[role]
            for role in roles
        ]
        texts = [role_to_description[role] for role in roles]

    if len(images) != len(texts):
        raise AssertionError(f"{context}: internal image/view_text alignment failure.")
    return images, texts
