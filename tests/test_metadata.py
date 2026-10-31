from ingestion.metadata import assign_metadata

TAXONOMY = {
    "department": ["support", "product", "engineering", "hr"],
    "access_level": ["public", "internal", "restricted"],
    "product": ["core", "billing", "integrations", "security"],
}


def meta(path, dept="support"):
    return assign_metadata("src", path, dept, TAXONOMY)


def test_deterministic():
    assert meta("a/b.md") == meta("a/b.md")


def test_values_always_in_taxonomy():
    for i in range(200):
        m = meta(f"folder/doc-{i}.md")
        assert m["department"] in TAXONOMY["department"]
        assert m["access_level"] in TAXONOMY["access_level"]
        assert m["product"] in TAXONOMY["product"]


def test_hr_path_is_hr_and_restricted():
    m = meta("hr/leave-policy.md")
    assert m["department"] == "hr" and m["access_level"] == "restricted"


def test_keyword_hints():
    assert meta("support/billing-refunds.md")["product"] == "billing"
    assert meta("product/webhooks-integration.md")["product"] == "integrations"
    assert meta("support/password-reset.md")["product"] == "security"
    assert meta("product/internal-sla-tiers.md")["access_level"] == "internal"


def test_hint_matches_whole_words_only():
    # "hr" must not fire inside "three", "api" not inside "capital"
    m = meta("guides/three-capital-ideas.md")
    assert m["department"] == "support"


def test_default_department_used_when_no_hint():
    assert meta("misc/notes.md", dept="product")["department"] == "product"


def test_unknown_default_department_falls_back():
    assert meta("misc/notes.md", dept="not-a-dept")["department"] == "support"


def test_access_mix_is_mostly_public():
    levels = [meta(f"docs/page-{i}.md")["access_level"] for i in range(500)]
    assert levels.count("public") > levels.count("internal") > levels.count("restricted") > 0


def test_folder_names_set_department_and_access():
    m = meta("product/internal/sla-tiers.md")
    assert (m["department"], m["access_level"]) == ("product", "internal")
    m = meta("support/public/password-reset.md")
    assert (m["department"], m["access_level"]) == ("support", "public")
    m = meta("hr/restricted/leave-policy.md")
    assert (m["department"], m["access_level"]) == ("hr", "restricted")
