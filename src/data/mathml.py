"""
Parsing and canonicalisation of ARQMath MathML (SLT = Presentation MathML,
OPT = Content MathML).

The same formula arrives in slightly different shapes depending on its source:

  corpus (v3)      <math alttext=… class=… display=…><semantics>…</semantics></math>
  ARQMath-2/3 topics  same as the corpus, but ARQMath-3 SLT has unescaped '<'
                   (in alttext and as <mo><</mo>), which is not well-formed XML
  ARQMath-1 topics XML declaration, pretty-printed, no <semantics> wrapper,
                   self-closing empty elements

`canonical()` maps all of these to one string form so that identical formulas
compare equal and downstream tokenisers and graph builders see the same input:
namespaces, the root <math> attributes, <semantics>/annotation wrappers and
whitespace between elements are removed. Symbol text and meaningful attributes
(mathvariant, cd, stretchy, …) are kept unchanged.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional

_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>")
_ALTTEXT_RE = re.compile(r"""\salttext=(?:"[^"]*"|'[^']*')""")
_MPADDED_RE = re.compile(r"</?mpadded\b[^>]*>")
_BARE_AMP_RE = re.compile(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z][A-Za-z0-9]*;)")  # '&' not starting an entity
_BARE_LT_RE = re.compile(r"<(?![A-Za-z/!?])")  # a '<' that cannot start a tag or comment

_WRAPPERS = {"semantics"}
_DROPPED = {"annotation", "annotation-xml"}


def sanitize(xml: str) -> str:
    """
    Make ARQMath MathML well-formed. Fixes, in order, the defects found in the corpus
    and topic files (scripts/inspect_graph_failures.py):
      - XML declaration (ARQMath-1 topics)
      - alttext attribute, double- or single-quoted, which may hold raw '<' and '&'
      - <mpadded> tags, often left unclosed (≈5% of corpus SLT); mpadded only adds
        spacing, so its tags are dropped and its content kept
      - bare '&' and '<' in text (e.g. <mo>&</mo>, <mo><</mo>)
    """
    xml = _XML_DECL_RE.sub("", xml.strip())
    xml = _ALTTEXT_RE.sub("", xml)
    xml = _MPADDED_RE.sub("", xml)
    xml = _BARE_AMP_RE.sub("&amp;", xml)
    return _BARE_LT_RE.sub("&lt;", xml)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean(elem: ET.Element) -> ET.Element:
    """Copy of `elem` without namespaces, wrappers, annotations or inter-element whitespace."""
    out = ET.Element(_local(elem.tag), {_local(k): v for k, v in elem.attrib.items()})
    out.text = elem.text if elem.text and elem.text.strip() else None
    for child in _flatten(elem):
        out.append(_clean(child))
    return out


def _flatten(elem: ET.Element):
    """Children of `elem`, with wrapper elements replaced by their own children."""
    for child in elem:
        tag = _local(child.tag)
        if tag in _DROPPED:
            continue
        if tag in _WRAPPERS:
            yield from _flatten(child)
        else:
            yield child


def parse(xml: Optional[str]) -> Optional[ET.Element]:
    """Canonical element tree of an ARQMath MathML string, or None if it is empty or unparseable."""
    if not xml or not xml.strip():
        return None
    try:
        root = ET.fromstring(sanitize(xml))
    except ET.ParseError:
        return None
    root = _clean(root)
    root.attrib.clear()  # display/class/encoding on <math> vary by source and carry no content
    return root


def canonical(xml: Optional[str]) -> Optional[str]:
    """Canonical string form (see module docstring), or None if the input cannot be parsed."""
    root = parse(xml)
    return None if root is None else ET.tostring(root, encoding="unicode")
