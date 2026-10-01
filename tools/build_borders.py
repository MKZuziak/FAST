#!/usr/bin/env python3
"""Regenerate the country borders the Day 4 triangulation lab draws its maps on.

Downloads Natural Earth admin-0 boundaries, clips them to the two extents the lab uses, simplifies
them, and writes GeoJSON into `src/packages/fast/fast/labs/day4_verification/borders/`.

Natural Earth is public domain, which is why these ship inside the package rather than being
fetched at runtime. The lab originally drew on hosted map tiles and was blocked for it: twenty
participants pulling from volunteer-run tile servers is what the OpenStreetMap tile usage policy
exists to prevent, and the commercial alternatives want an API key. Bundled vectors need no tile
server, no key, and no network.

    uv run python tools/build_borders.py

Only needed if the extent or resolution changes; the outputs are committed.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src/packages/fast/fast/labs/day4_verification/borders"
SOURCE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_{}_admin_0_countries.geojson"

sys.setrecursionlimit(10000)


def _simplify_chain(points, tolerance):
    """Douglas-Peucker on an open chain."""
    if len(points) < 3:
        return points
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    span = math.hypot(dx, dy)
    furthest, index = -1.0, 0
    for i in range(1, len(points) - 1):
        x0, y0 = points[i]
        # Perpendicular distance to the chord, or radial distance when the chord is a point.
        distance = (
            abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / span
            if span
            else math.hypot(x0 - x1, y0 - y1)
        )
        if distance > furthest:
            furthest, index = distance, i
    if furthest > tolerance:
        return _simplify_chain(points[: index + 1], tolerance)[:-1] + _simplify_chain(points[index:], tolerance)
    return [points[0], points[-1]]


def simplify_ring(ring, tolerance):
    """Simplify a closed ring, or None if nothing survives.

    Split at the point furthest from the start first. Run Douglas-Peucker straight at a closed
    ring and its chord has zero length, so every point measures as zero distance from it and the
    whole country collapses to two points.
    """
    points = ring[:-1] if ring[0] == ring[-1] else list(ring)
    if len(points) < 4:
        return None
    start = points[0]
    far = max(range(len(points)), key=lambda i: math.hypot(points[i][0] - start[0], points[i][1] - start[1]))
    simplified = (
        _simplify_chain(points[: far + 1], tolerance)[:-1]
        + _simplify_chain(points[far:] + [start], tolerance)[:-1]
    )
    return simplified + [simplified[0]] if len(simplified) >= 3 else None


def build(resolution, bbox, tolerance, digits, name):
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    print(f"downloading Natural Earth {resolution}…")
    with urllib.request.urlopen(SOURCE.format(resolution)) as response:
        source = json.load(response)

    features = []
    for feature in source["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        polygons = (
            geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        )
        kept = []
        for polygon in polygons:
            outer = [tuple(point[:2]) for point in polygon[0]]
            if not any(lo_lon <= x <= hi_lon and lo_lat <= y <= hi_lat for x, y in outer):
                continue
            simplified = simplify_ring(outer, tolerance)
            if simplified:
                kept.append([[round(x, digits), round(y, digits)] for x, y in simplified])
        if kept:
            features.append({
                "type": "Feature",
                "properties": {"name": feature["properties"].get("NAME")},
                "geometry": {"type": "MultiPolygon", "coordinates": [[ring] for ring in kept]},
            })

    path = OUT / name
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))
    points = sum(len(ring[0]) for f in features for ring in f["geometry"]["coordinates"])
    print(f"  {name}: {len(features)} countries, {points} points, {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build("50m", (65, -35, 155, 45), 0.08, 2, "borders_region.geojson")
    build("10m", (102.0, 0.2, 105.5, 2.6), 0.002, 4, "borders_strait.geojson")
