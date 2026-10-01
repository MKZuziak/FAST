# Country borders

Two clipped, simplified extracts of [Natural Earth](https://www.naturalearthdata.com/) admin-0
country boundaries, used as the map background in `location-triangulation`.

- `borders_region.geojson` — 1:50m, clipped to 65–155°E / 35°S–45°N, simplified to 0.08°.
- `borders_strait.geojson` — 1:10m, clipped to the Singapore Strait, simplified to 0.002°.

Natural Earth is **public domain**, so these ship with the package rather than being downloaded at
runtime. That matters here: the lab used to draw its maps on hosted tiles, and a room of twenty
participants pulling tiles from volunteer-run servers is exactly what the OpenStreetMap tile usage
policy prohibits. Bundled vectors need no tile server, no API key, and no network at all.

Regenerated with `tools/build_borders.py` if the extent or resolution ever needs to change.
