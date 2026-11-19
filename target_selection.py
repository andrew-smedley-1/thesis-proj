"""Resolve an operator's choice against boxes in one wrist-camera image."""
import re


def centers(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def numbered_boxes(boxes):
    return sorted(boxes, key=centers)


def resolve_choice(reply, target, boxes, *, spatial_only=False):
    """Return one box, or None if the description doesn't identify one."""
    if not isinstance(reply, str):
        return None
    # Hyphenated directions are common, but a minus sign on a box number matters.
    words = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", reply.strip().casefold().rstrip(".!?"))
    words = " ".join(words.split())
    words = re.sub(r"^(?:please )?(?:pick up |pick |take |choose |select )?(?:the )?", "", words)
    number = re.fullmatch(r"(?:box (?:number )?)?(\d{1,5})", words)
    ordered = numbered_boxes(boxes)
    if number and spatial_only:
        return None
    if number:
        index = int(number[1]) - 1
        return ordered[index] if 0 <= index < len(ordered) else None
    target = target.casefold().removeprefix("the ")
    noun = rf"(?:{re.escape(target)}|{re.escape(target.split()[-1])}|one)"
    words = re.sub(rf"^{noun} (?:on|in|at) the ", "", words)
    words = re.sub(rf" {noun}$", "", words)
    direction = words.replace(" ", "")
    if direction.endswith("most"):
        direction = direction[:-4]
    if direction in ("middle", "center", "centre"):
        if len(ordered) % 2 == 0:
            return None
        chosen = ordered[len(ordered) // 2]
        axis, value = 0, centers(chosen)[0]
    elif direction in ("left", "right", "top", "bottom"):
        axis = int(direction in ("top", "bottom"))
        extreme = min if direction in ("left", "top") else max
        value = extreme(centers(box)[axis] for box in boxes)
    else:
        return None
    matches = [box for box in boxes if centers(box)[axis] == value]
    return matches[0] if len(matches) == 1 else None
