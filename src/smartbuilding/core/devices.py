"""Device ids: how lifts, doors, detectors, safety equipment and alarms are named.

Convention: ``DV_<kind>_<n>``. Lifts and doors of a Layout map to devices with the same
number: lift ``L_2`` is device ``DV_lift_2``, door ``D_7`` is device ``DV_door_7``.
"""

from smartbuilding.core.primitives import DeviceKind


def device_kind(device_id: str) -> DeviceKind:
    """Kind encoded in a device id such as ``DV_lift_1``."""
    parts = device_id.split("_")
    if len(parts) != 3 or parts[0] != "DV":
        raise ValueError(f"not a device id: {device_id!r}")
    return DeviceKind(parts[1])


def make_device_id(kind: DeviceKind, number: int) -> str:
    if number < 0:
        raise ValueError("device number must be non-negative")
    return f"DV_{kind.value}_{number}"


def device_for_region(region_id: str) -> str:
    """Device id for a Layout lift (``L_n``) or door (``D_n``)."""
    prefix, _, number = region_id.partition("_")
    if not number.isdigit():
        raise ValueError(f"cannot map {region_id!r} to a device")
    if prefix == "L":
        return make_device_id(DeviceKind.lift, int(number))
    if prefix == "D":
        return make_device_id(DeviceKind.door, int(number))
    raise ValueError(f"only lifts (L_n) and doors (D_n) map to devices, got {region_id!r}")
