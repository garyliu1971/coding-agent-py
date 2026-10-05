"""srdp_map_ext_content: map every ExtContent .bin to its slide position(s).

Algorithm (reverse-engineered from real SRDPs):

1. index.xml  ExternalContent:
      bin_id  ←→  content_id  (the Seismic content UUID)

2. SharedComponentObject/*.xml:
      ObjectId == content_id  →  ContentName, ContentType

3. InstanceLiveDoc.pptx:
   a) ppt/presentation.xml   sldIdLst   → ordered slide positions
   b) ppt/_rels/presentation.xml.rels   → rId → slide file path
   c) ppt/slides/slideN.xml             → search each slide for the
      SharedComponent ContentName as a shape name attribute

4. Correlate: ContentName found in slideN → slide position P
   Then: ExtContent/bin_id.bin → Slide P (because bin → content_id → name → slide)

Notes:
- "Morningstar Rating" might appear in MULTIPLE slides (same component reused
  on different pages); all positions are reported.
- StyleBox and similar components that have no ExtContent bin are also reported
  (they are embedded SharedComponents without a separate bin).
- The same name search is done in MainLiveDoc.pptx when InstanceLiveDoc is absent.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _cfg, _root, _within
from .srdp import _read_capped


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    return _read_capped(zf, name).decode("utf-8", errors="replace")


def _parse_rels(rels_xml: str, rel_type_suffix: str) -> dict[str, str]:
    """Return {rId: target} for relationships whose Type ends with rel_type_suffix."""
    result: dict[str, str] = {}
    for m in re.finditer(r"<Relationship\b([^>]+)/>", rels_xml):
        attrs = m.group(1)
        rid  = re.search(r'\bId="([^"]+)"', attrs)
        tgt  = re.search(r'\bTarget="([^"]+)"', attrs)
        typ  = re.search(r'\bType="([^"]+)"', attrs)
        if rid and tgt and typ and typ.group(1).endswith(rel_type_suffix):
            result[rid.group(1)] = tgt.group(1).lstrip("/")
    return result


def _normalise_slide_path(raw: str) -> str:
    """Normalise a slide Target path to the form 'ppt/slides/slideN.xml'.

    Different pptx files use:
      - 'slides/slide1.xml'        (relative to ppt/)
      - '/ppt/slides/slide1.xml'   (absolute)
      - 'ppt/slides/slide1.xml'    (already normalised)
    """
    # Strip leading slash
    p = raw.lstrip("/")
    # If it starts with 'slides/' it is relative to 'ppt/'
    if p.startswith("slides/"):
        p = "ppt/" + p
    return p


def _slide_position_map(pptx: zipfile.ZipFile) -> dict[str, int]:
    """Return {slide_file_path: 1-based_position} from presentation.xml + rels."""
    prs = _read_text(pptx, "ppt/presentation.xml")
    rels_xml = _read_text(pptx, "ppt/_rels/presentation.xml.rels")

    sld_order = re.findall(r'<p:sldId\s+id="(\d+)"\s+r:id="([^"]+)"', prs)
    rid_to_file_raw = _parse_rels(rels_xml, "/slide")
    rid_to_file = {rid: _normalise_slide_path(tgt) for rid, tgt in rid_to_file_raw.items()}

    result: dict[str, int] = {}
    for pos, (_, rid) in enumerate(sld_order, 1):
        f = rid_to_file.get(rid)
        if f:
            result[f] = pos
    return result


# ---------------------------------------------------------------------------
# core analysis
# ---------------------------------------------------------------------------

def _analyse(srdp_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(str(srdp_path)) as srdp:
        names = srdp.namelist()

        # ── 1. index.xml: bin_id → content_id ────────────────────────────
        index_xml = _read_text(srdp, "index.xml")
        bin_to_cid: dict[str, str] = {}
        for m in re.finditer(
            r'<Content\b[^>]+Key="[^"]*contentId=([0-9a-f-]{36})[^"]*"[^>]+Id="([0-9a-f-]{36})"',
            index_xml, re.I,
        ):
            cid, bin_id = m.group(1), m.group(2)
            bin_to_cid[bin_id] = cid
        cid_to_bin = {v: k for k, v in bin_to_cid.items()}

        # ── 2. SharedComponentObject XMLs: obj_id → name, type ───────────
        obj_to_info: dict[str, dict] = {}   # obj_id → {name, type}
        for entry in names:
            if "SharedComponentObject/" not in entry or not entry.endswith(".xml"):
                continue
            xml  = _read_text(srdp, entry)
            oid  = re.search(r"<ObjectId>([^<]+)</ObjectId>", xml)
            name = re.search(r"<ContentName>([^<]+)</ContentName>", xml)
            typ  = re.search(r"<ContentType>([^<]+)</ContentType>", xml)
            if oid:
                obj_to_info[oid.group(1)] = {
                    "name": name.group(1) if name else "?",
                    "type": typ.group(1)  if typ  else "?",
                }

        # Unique SC names to search for.
        # Also search MainLiveDoc (template) since instance slide XML often has
        # generic shape names like 'object 12' — the ContentName appears in the
        # template slide's descr= attributes or shape names instead.
        sc_names: dict[str, dict] = {}   # name → {obj_id, type}
        for oid, info in obj_to_info.items():
            n = info["name"]
            if n not in sc_names:
                sc_names[n] = {"obj_id": oid, "type": info["type"]}

        # ── 3. InstanceLiveDoc (or MainLiveDoc) slides ────────────────────
        # Search both InstanceLiveDoc AND MainLiveDoc for SC names.
        # Template (Main) tends to have ContentNames in shape name/descr attributes.
        # Instance tends to have them in slide XML text content after rendering.
        pptx_entries: list[str] = []
        for candidate in ("InstanceLiveDoc.pptx", "MainLiveDoc.pptx"):
            if candidate in names:
                pptx_entries.append(candidate)
        pptx_used = ", ".join(pptx_entries) if pptx_entries else "none"

        name_to_positions: dict[str, set[int]] = {}  # sc_name → {slide positions}

        for pptx_entry in pptx_entries:
            raw = _read_capped(srdp, pptx_entry)
            with zipfile.ZipFile(io.BytesIO(raw)) as pptx:
                pos_map = _slide_position_map(pptx)

                slide_files = sorted(
                    n for n in pptx.namelist()
                    if re.match(r"ppt/slides/slide\d+\.xml$", n)
                )
                # Build a lookup: slide_file → all customXml item content
                # (SharedComponent names often appear in customXml items
                # that are referenced from the slide rels, not in the slide XML itself)
                cx_items: dict[str, str] = {}  # path → text
                for n in pptx.namelist():
                    if re.match(r"customXml/item\d+\.xml$", n, re.I):
                        cx_items[n] = _read_text(pptx, n)

                slide_cx_rels: dict[str, list[str]] = {}  # slide_file → [cx paths]
                for sf in slide_files:
                    slide_num = re.search(r"slide(\d+)\.xml$", sf)
                    if not slide_num:
                        continue
                    rels_path = f"ppt/slides/_rels/slide{slide_num.group(1)}.xml.rels"
                    if rels_path not in pptx.namelist():
                        continue
                    rels_xml = _read_text(pptx, rels_path)
                    # Match Target paths like '../../customXml/item149.xml'
                    # or '/customXml/item149.xml' — extract the normalised path
                    cx_paths: list[str] = []
                    for m in re.finditer(r'Target="([^"]+customXml/item\d+\.xml)"', rels_xml, re.I):
                        raw_tgt = m.group(1)
                        # Normalise to 'customXml/itemN.xml'
                        norm = re.sub(r'^.*customXml/', 'customXml/', raw_tgt, flags=re.I)
                        cx_paths.append(norm)
                    slide_cx_rels[sf] = cx_paths

                for sf in slide_files:
                    content = _read_text(pptx, sf)
                    pos = pos_map.get(sf)
                    if pos is None:
                        continue

                    # Combine slide XML + referenced customXml items for searching
                    combined = content.lower()
                    for cx_path in slide_cx_rels.get(sf, []):
                        if cx_path in cx_items:
                            combined += "\n" + cx_items[cx_path].lower()

                    for sc_name in sc_names:
                        if sc_name.lower() in combined:
                            name_to_positions.setdefault(sc_name, set()).add(pos)

        # ── 4. Build result rows ───────────────────────────────────────────
        # One row per ExtContent bin.
        # Also include SharedComponents with no bin if they appear in slides.
        rows: list[dict] = []
        seen_names: set[str] = set()

        # Bins first (in index.xml order)
        for bin_id, cid in bin_to_cid.items():
            oid_for_bin = cid  # content_id == obj_id in most cases
            # Find by matching cid_to_bin
            info = obj_to_info.get(oid_for_bin, {})
            name = info.get("name", "?")
            typ  = info.get("type", "?")
            positions = sorted(name_to_positions.get(name, set()))
            rows.append({
                "bin_file":  f"ExtContent/{bin_id}.bin",
                "bin_id":    bin_id,
                "name":      name,
                "type":      typ,
                "positions": positions or ["?"],
            })
            seen_names.add(name)

        # SCs with no bin
        for sc_name, sc_info in sc_names.items():
            if sc_name in seen_names:
                continue
            oid = sc_info["obj_id"]
            if cid_to_bin.get(oid):
                continue  # has a bin, already covered
            positions = sorted(name_to_positions.get(sc_name, set()))
            rows.append({
                "bin_file":  "(no ExtContent bin)",
                "bin_id":    None,
                "name":      sc_name,
                "type":      sc_info["type"],
                "positions": positions or ["?"],
            })

        # Count slides from InstanceLiveDoc (authoritative for output)
        total_slides = 0
        inst_entry = "InstanceLiveDoc.pptx" if "InstanceLiveDoc.pptx" in names else (
            "MainLiveDoc.pptx" if "MainLiveDoc.pptx" in names else None
        )
        if inst_entry:
            raw = _read_capped(srdp, inst_entry)
            with zipfile.ZipFile(io.BytesIO(raw)) as pptx:
                prs = _read_text(pptx, "ppt/presentation.xml")
                total_slides = len(re.findall(r"<p:sldId\b", prs))

        return {
            "pptx_used":   pptx_used,
            "total_slides": total_slides,
            "ext_bins":    len(bin_to_cid),
            "rows":        rows,
        }


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

@tool
def srdp_map_ext_content(
    srdp_path: str,
    config: RunnableConfig = None,
) -> str:
    """Map every ExtContent .bin file in an SRDP to its slide position(s).

    For each ExtContent/*.bin (and any SharedComponents without a bin), reports:
    - The bin file name (or '(no ExtContent bin)')
    - The SharedComponent name and type
    - Which slide number(s) in the generated document contain it

    Algorithm:
    1. index.xml ExternalContent: bin ID → content ID (Seismic UUID)
    2. SharedComponentObject/*.xml: content ID → component name
    3. InstanceLiveDoc.pptx slide order + slide XML: search each slide for the
       component name as a shape/element, record which slide position it falls on

    Args:
        srdp_path: Path to the SRDP .zip file, relative to project root.
    """
    root = _root(config)
    p = (root / srdp_path).resolve()
    if not _within(root, p):
        return f"Error: path escapes project root: {srdp_path}"
    if not p.is_file():
        return f"Error: file not found: {srdp_path}"

    try:
        result = _analyse(p)
    except zipfile.BadZipFile:
        return f"Error: {srdp_path} is not a valid zip file."
    except Exception as exc:
        return f"Error analysing SRDP: {exc}"

    lines = [
        f"SRDP: {srdp_path}",
        f"  Source pptx:   {result['pptx_used']}",
        f"  Total slides:  {result['total_slides']}",
        f"  ExtContent bins: {result['ext_bins']}",
        "",
        "ExtContent bin → Slide mapping:",
        "─" * 60,
    ]

    if not result["rows"]:
        lines.append("  (no SharedComponents found)")
    else:
        for row in result["rows"]:
            positions = row["positions"]
            if positions == ["?"]:
                pos_str = "Slide ? (not found in slide XML)"
            else:
                pos_str = ", ".join(f"Slide {p}" for p in positions)

            lines += [
                f"  {row['bin_file']}",
                f"    Name:  {row['name']}",
                f"    Type:  {row['type']}",
                f"    → {pos_str}",
                "",
            ]

    return "\n".join(lines)
