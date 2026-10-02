"""
Fixed deterministic transition templates between subtopic groups within sections.
These are the ONLY non-extracted strings in the survey output — all other text
comes directly from the source PDFs.
"""

SECTION_TRANSITIONS: dict = {
    "introduction": [
        "The following section reviews the current state of research in this area.",
        "Across the reviewed studies, several foundational themes emerge.",
        "Prior work has established important context for understanding these topics.",
    ],
    "methods": [
        "With respect to methodology, the reviewed papers adopt diverse approaches.",
        "Across the reviewed studies, the following methodological patterns are observed.",
        "The methods employed in these works can be categorized as follows.",
    ],
    "findings": [
        "In terms of reported findings, the evidence suggests the following.",
        "The results across multiple studies demonstrate consistent patterns.",
        "Empirical outcomes from the reviewed literature highlight the following.",
    ],
    "limitations": [
        "Common limitations identified across the reviewed works include the following.",
        "Despite reported advances, several challenges remain.",
        "The reviewed studies acknowledge the following constraints.",
    ],
    "research_gaps": [
        "Several open problems and research gaps emerge from this survey.",
        "Future work directions identified across the literature include the following.",
        "The reviewed papers collectively point to the following under-explored areas.",
    ],
    "conclusion": [
        "In summary, the reviewed literature highlights the following key contributions.",
        "This survey synthesizes evidence from multiple sources as follows.",
        "The collective findings support the following conclusions.",
    ],
}

_FALLBACK = "Furthermore, the reviewed literature notes the following."


def get_transition(section: str, subtopic_idx: int) -> str:
    """
    Return the transition template for a section at a given subtopic index.
    Cycles through available templates deterministically.
    """
    templates = SECTION_TRANSITIONS.get(section, [_FALLBACK])
    return templates[subtopic_idx % len(templates)]