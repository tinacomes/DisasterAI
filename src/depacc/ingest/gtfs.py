"""GTFS feed ingest (Tier 2): download the feeds listed in the city config
and optionally clip them to the FUA bounding box (smaller feeds keep R5's
memory footprint manageable for country-wide aggregations like gtfs.de)."""

from __future__ import annotations

from pathlib import Path

from depacc.provenance import download


def fetch_gtfs(cfg: dict, root: Path, fua=None) -> list[Path]:
    feeds = (cfg.get("sources", {}).get("gtfs", {}) or {}).get("feeds", [])
    raw = root / cfg["output"]["raw_root"] / "gtfs"
    out = []
    for feed in feeds:
        path = download(feed["url"], raw / f"{feed['id']}.zip",
                        licence=feed.get("licence", ""))
        if fua is not None:
            clipped = raw / f"{feed['id']}_clipped.zip"
            path = _maybe_clip(path, clipped, fua)
        out.append(path)
    return out


def _maybe_clip(feed_zip: Path, out: Path, fua) -> Path:
    """Clip a feed to the FUA bounding box with gtfs-kit; on any failure the
    full feed is used (R5 just works harder)."""
    if out.exists():
        return out
    try:
        import gtfs_kit as gk

        pad = 0.1  # degrees of slack around the FUA
        minx, miny, maxx, maxy = fua.to_crs("EPSG:4326").total_bounds
        feed = gk.read_feed(feed_zip, dist_units="km")
        clipped = feed.restrict_to_area(area=_bbox_gdf(minx - pad, miny - pad,
                                                       maxx + pad, maxy + pad))
        clipped.write(out)
        return out
    except Exception as err:  # noqa: BLE001 - degrade gracefully to full feed
        print(f"WARNING: GTFS clip failed ({err}); using full feed {feed_zip.name}")
        return feed_zip


def _bbox_gdf(minx, miny, maxx, maxy):
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326")
