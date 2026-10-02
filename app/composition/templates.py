"""Static section header templates for survey composition."""

SECTION_HEADER_TEMPLATES = {
    "introduction":  "## 1. Introduction and Background\n\n",
    "methods":       "## 2. Methods and Approaches\n\n",
    "findings":      "## 3. Findings and Results\n\n",
    "limitations":   "## 4. Limitations and Challenges\n\n",
    "research_gaps": "## 5. Research Gaps and Future Work\n\n",
    "conclusion":    "## 6. Conclusion\n\n",
}

SURVEY_PREAMBLE = (
    "This literature survey was generated algorithmically using extractive "
    "summarization, topic modeling, section classification, and coherence-optimized "
    "sentence ordering. All statements are directly attributed to their source "
    "documents. No generative AI was used.\n\n"
)