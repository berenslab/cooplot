import csv
import json

import pytest

from cooplot import api
from cooplot.aggregate import aggregate_publications
from cooplot.metrics import (
    _CitationFetcher,
    _CrossrefDetails,
    _PubMedDetails,
    cross_group_publications,
    cross_group_report,
)
from cooplot.scrape import Publications


@pytest.fixture
def sample_people():
    return [
        {"name": "Alice Alpha", "team": "Group 1"},
        {"name": "Bob Beta", "team": "Group 1"},
        {"name": "Cara Gamma", "team": "Group 2"},
    ]


@pytest.fixture
def sample_publications():
    return {
        "Alice Alpha": [
            {"title": "Deep Learning", "norm_title": "deep learning", "year": 2021},
            {"title": "Shared Paper", "norm_title": "shared paper", "year": 2020},
        ],
        "Bob Beta": [
            {"title": "Shared Paper", "norm_title": "shared paper", "year": 2020},
        ],
        "Cara Gamma": [
            {"title": "Shared Paper", "norm_title": "shared paper", "year": 2020},
        ],
        # Dana is missing from people to exercise the unlabeled bucket
        "Dana Delta": [
            {"title": "Shared Paper", "norm_title": "shared paper", "year": 2020},
        ],
    }


def test_aggregate_publications_deduplicates(
    tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
    )

    assert set(grouped.by_group.keys()) == {"Group 1", "Group 2", "Unlabeled"}

    group_one_records = grouped.by_group["Group 1"]
    titles = {record["title"] for record in group_one_records}
    assert titles == {"Deep Learning", "Shared Paper"}
    shared_record = next(
        record for record in group_one_records if record["title"] == "Shared Paper"
    )
    assert shared_record["authors"] == ["Alice Alpha", "Bob Beta"]

    # JSON files are written using slugified group names
    expected_path = tmp_path / "Group_1.json"
    assert expected_path.exists()
    saved_data = json.loads(expected_path.read_text(encoding="utf-8"))
    assert len(saved_data) == len(group_one_records)

    unlabeled_path = grouped.paths["Unlabeled"]
    assert unlabeled_path == tmp_path / "Unlabeled.json"


def test_cross_group_publications_filters(tmp_path, sample_publications, sample_people):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    records = cross_group_publications(grouped)
    assert len(records) == 1
    record = records[0]
    assert record["groups"] == ["Group 1", "Group 2"]

    records_with_unlabeled = cross_group_publications(grouped, include_unlabeled=True)
    assert records_with_unlabeled[0]["groups"] == ["Group 1", "Group 2", "Unlabeled"]

    filtered_out = cross_group_publications(grouped, year_from=2021)
    assert filtered_out == []

    filtered_in = cross_group_publications(grouped, year_from=2019, year_to=2020)
    assert len(filtered_in) == 1


def test_build_from_grouped_data(sample_publications, sample_people):
    grouped = api.aggregate(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        save_json=False,
    )

    mats = api.build(
        grouped,
        ["2019-2020"],
    )
    window = mats["2019-2020"]
    assert window["labels"] == ["Group 1", "Group 2", "Unlabeled"]
    matrix = window["matrix"]
    assert matrix[0][1] == 1  # Group 1 vs Group 2 share "Shared Paper"
    assert matrix[0][0] == 0


def test_cross_group_publications_year_resolution(tmp_path):
    people = [
        {"name": "Author Missing", "team": "Group 1"},
        {"name": "Author Early", "team": "Group 2"},
        {"name": "Author Frequent A", "team": "Group 2"},
        {"name": "Author Frequent B", "team": "Group 2"},
    ]
    publications = {
        "Author Missing": [
            {
                "title": "Advanced Search Search",
                "norm_title": "advanced search search",
                "year": None,
            }
        ],
        "Author Early": [
            {
                "title": "Advanced Search Search",
                "norm_title": "advanced search search",
                "year": 2020,
            }
        ],
        "Author Frequent A": [
            {
                "title": "Advanced Search Search",
                "norm_title": "advanced search search",
                "year": 2021,
            }
        ],
        "Author Frequent B": [
            {
                "title": "Advanced Search Search",
                "norm_title": "advanced search search",
                "year": 2021,
            }
        ],
    }

    grouped = aggregate_publications(
        publications,
        people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    records = cross_group_publications(grouped)
    assert len(records) == 1
    record = records[0]
    assert record["year"] == 2021
    assert record["groups"] == ["Group 1", "Group 2"]
    expected_authors = [
        "Author Early",
        "Author Frequent A",
        "Author Frequent B",
    ]
    expected_authors.sort(key=lambda name: name.split()[-1].lower())
    assert record["authors"]["Group 2"] == expected_authors


def test_build_with_author_publications_wrapper(sample_publications, sample_people):
    author_data = Publications.from_data(
        sample_publications,
        sample_people,
        group_col="team",
    )
    mats = api.build(author_data, ["2019-2020"])
    window = mats["2019-2020"]
    assert window["label_to_group"]["Alice Alpha"] == "Group 1"
    matrix = window["matrix"]
    assert matrix[0][1] == 1


def test_cross_group_publications_export(tmp_path, sample_publications, sample_people):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )
    out_file = tmp_path / "cross.json"
    records = cross_group_publications(
        grouped,
        out_path=out_file,
    )
    assert out_file.exists()
    assert records == cross_group_publications(grouped)

    out_csv = tmp_path / "cross.csv"
    cross_group_publications(grouped, out_path=out_csv)
    assert out_csv.exists()
    with out_csv.open() as fh:
        header = fh.readline().strip()
        assert header == "title,norm_title,year,groups,authors"
        rows = [line.strip() for line in fh if line.strip()]
        assert len(rows) == len(records)


def test_cross_group_publications_reuses_existing_file(
    tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    out_path = tmp_path / "cross.json"
    cross_group_publications(grouped, out_path=out_path, include_unlabeled=True)
    assert out_path.exists()

    override_data = [
        {
            "title": "From Cache",
            "norm_title": "from cache",
            "year": 2022,
            "groups": ["Group 1", "Group 2"],
            "authors": {"Group 1": ["Alice Alpha"], "Group 2": ["Cara Gamma"]},
        }
    ]
    out_path.write_text(json.dumps(override_data), encoding="utf-8")

    reused = cross_group_publications(
        grouped, out_path=out_path, include_unlabeled=True
    )
    assert reused and reused[0]["title"] == "From Cache"
    assert json.loads(out_path.read_text(encoding="utf-8")) == override_data


def test_cross_group_publications_overwrite_existing_file(
    tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    out_path = tmp_path / "cross.json"
    sentinel = [
        {
            "title": "Stale Record",
            "norm_title": "stale record",
            "year": 1999,
            "groups": ["Old Group"],
            "authors": {"Old Group": ["Old Author"]},
        }
    ]
    out_path.write_text(json.dumps(sentinel), encoding="utf-8")

    records = cross_group_publications(
        grouped,
        out_path=out_path,
        include_unlabeled=True,
        overwrite=True,
    )

    assert records
    assert records[0]["title"] != "Stale Record"
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == records


def test_publications_exclude_authors(sample_publications, sample_people):
    pubs = Publications.from_data(
        sample_publications,
        sample_people,
        group_col="team",
    )
    filtered = pubs.exclude_authors(["Bob Beta"])
    assert "Bob Beta" not in filtered.mapping()
    name_key = filtered.resolve_name_column("name")
    remaining_names = {row[name_key] for row in filtered.people_rows()}
    assert "Bob Beta" not in remaining_names
    assert "Alice Alpha" in remaining_names


def test_cross_group_publications_enrich_pubmed(
    monkeypatch, tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    calls = []

    def fake_identifiers(self, title, year):
        calls.append((title, year))
        return _PubMedDetails(
            pubmed_id="PM12345",
            doi="10.1000/example",
            authors=["Alice Alpha", "Bob Beta", "Cara Gamma"],
            journal="Journal of Testing",
        )

    monkeypatch.setattr(
        "cooplot.metrics._PubMedLookup.identifiers_for_title",
        fake_identifiers,
    )

    out_csv = tmp_path / "cross_enriched.csv"
    records = cross_group_publications(
        grouped,
        enrich_pubmed=True,
        out_path=out_csv,
    )

    assert calls == [("Shared Paper", 2020)]

    assert records
    record = records[0]
    assert record["pubmed_id"] == "PM12345"
    assert record["doi"] == "10.1000/example"
    assert record["pubmed_authors"] == [
        "Alice Alpha",
        "Bob Beta",
        "Cara Gamma",
    ]
    assert record["pubmed_journal"] == "Journal of Testing"

    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    assert reader.fieldnames == [
        "title",
        "norm_title",
        "year",
        "groups",
        "authors",
        "pubmed_id",
        "doi",
        "pubmed_authors",
        "pubmed_journal",
    ]
    assert row["pubmed_id"] == "PM12345"
    assert row["doi"] == "10.1000/example"
    assert json.loads(row["pubmed_authors"]) == [
        "Alice Alpha",
        "Bob Beta",
        "Cara Gamma",
    ]
    assert row["pubmed_journal"] == "Journal of Testing"


def test_cross_group_publications_enrich_crossref(
    monkeypatch, tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    captured: dict = {"calls": []}

    class DummyCrossrefLookup:
        def __init__(self, *, mailto, min_delay, session=None):
            captured["mailto"] = mailto
            captured["min_delay"] = min_delay

        def identifiers_for_title(self, title, year):
            captured["calls"].append((title, year))
            return _CrossrefDetails(
                doi="10.2000/crossref",
                title=title,
                authors=["Alice Alpha", "Bob Beta"],
                container="Testing Journal",
                year=year,
            )

    monkeypatch.setattr("cooplot.metrics._CrossrefLookup", DummyCrossrefLookup)
    monkeypatch.setenv("CROSSREF_MAILTO", "cross@example.com")
    monkeypatch.setattr("cooplot.metrics._DOTENV_LOADED", False)

    out_csv = tmp_path / "crossref.csv"
    records = cross_group_publications(
        grouped,
        enrich_crossref=True,
        out_path=out_csv,
    )

    assert captured["calls"] == [("Shared Paper", 2020)]
    assert captured["mailto"] == "cross@example.com"
    assert captured["min_delay"] is None

    assert records
    record = records[0]
    assert record["doi"] == "10.2000/crossref"
    assert record["crossref_title"] == "Shared Paper"
    assert record["crossref_authors"] == ["Alice Alpha", "Bob Beta"]
    assert record["crossref_container"] == "Testing Journal"

    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    assert reader.fieldnames == [
        "title",
        "norm_title",
        "year",
        "groups",
        "authors",
        "doi",
        "crossref_title",
        "crossref_authors",
        "crossref_container",
    ]
    assert row["doi"] == "10.2000/crossref"
    assert json.loads(row["crossref_authors"]) == ["Alice Alpha", "Bob Beta"]
    assert row["crossref_container"] == "Testing Journal"


def test_cross_group_publications_deduplicates_pubmed_preprints(monkeypatch, tmp_path):
    people = [
        {"name": "Alice Alpha", "team": "Group 1"},
        {"name": "Cara Gamma", "team": "Group 2"},
    ]
    publications = {
        "Alice Alpha": [
            {"title": "Joint Study", "norm_title": "joint study", "year": 2020},
            {
                "title": "Joint Study (Preprint)",
                "norm_title": "joint study (preprint)",
                "year": 2019,
            },
        ],
        "Cara Gamma": [
            {"title": "Joint Study", "norm_title": "joint study", "year": 2020},
            {
                "title": "Joint Study (Preprint)",
                "norm_title": "joint study (preprint)",
                "year": 2019,
            },
        ],
    }

    grouped = aggregate_publications(
        publications,
        people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=False,
        save_json=False,
    )

    baseline = cross_group_publications(grouped)
    assert {record["norm_title"] for record in baseline} == {
        "joint study",
        "joint study (preprint)",
    }

    call_titles: list[str] = []

    class DuplicateLookup:
        def __init__(self, *, api_key, email, min_delay, session=None):
            pass

        def identifiers_for_title(self, title, year):
            call_titles.append(title)
            return _PubMedDetails(
                pubmed_id="PMSHARED",
                doi=None,
                authors=None,
                journal="Testing Journal",
                title="Joint Study",
                year=2020,
            )

    monkeypatch.setattr("cooplot.metrics._PubMedLookup", DuplicateLookup)

    records = cross_group_publications(grouped, enrich_pubmed=True)
    assert len(records) == 1
    record = records[0]
    assert record["norm_title"] == "joint study"
    assert record["pubmed_id"] == "PMSHARED"
    assert record.get("pubmed_title") == "Joint Study"
    assert any("joint study (preprint)" in title.lower() for title in call_titles)


def test_cross_group_publications_env_defaults(
    monkeypatch, tmp_path, sample_publications, sample_people
):
    project_dir = tmp_path / "env_project"
    project_dir.mkdir()
    env_text = "NCBI_API_KEY=ENV_KEY\nNCBI_EMAIL=env@example.com\n"
    (project_dir / ".env").write_text(env_text, encoding="utf-8")

    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr("cooplot.metrics._DOTENV_LOADED", False)

    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=project_dir / "cache",
        include_unlabeled=True,
        save_json=False,
    )

    captured = {}

    class DummyLookup:
        def __init__(self, *, api_key, email, min_delay, session=None):
            captured["api_key"] = api_key
            captured["email"] = email

        def identifiers_for_title(self, title, year):
            captured["title"] = title
            captured["year"] = year
            return _PubMedDetails(
                pubmed_id="PMENV",
                doi="DOIENV",
                authors=["Alice Alpha", "Bob Beta", "Cara Gamma"],
                journal="Env Journal",
            )

    monkeypatch.setattr("cooplot.metrics._PubMedLookup", DummyLookup)

    records = cross_group_publications(grouped, enrich_pubmed=True)

    assert captured["api_key"] == "ENV_KEY"
    assert captured["email"] == "env@example.com"
    assert records[0]["pubmed_id"] == "PMENV"
    assert records[0]["doi"] == "DOIENV"
    assert records[0]["pubmed_authors"] == [
        "Alice Alpha",
        "Bob Beta",
        "Cara Gamma",
    ]
    assert records[0]["pubmed_journal"] == "Env Journal"


def test_cross_group_publications_enrich_pubmed_mismatch(
    monkeypatch, tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    class MismatchLookup:
        def __init__(self, *, api_key, email, min_delay, session=None):
            pass

        def identifiers_for_title(self, title, year):
            return _PubMedDetails(
                pubmed_id="PMBAD",
                doi="10.9999/mismatch",
                authors=["Different Person"],
                journal="Mismatch Journal",
            )

    monkeypatch.setattr("cooplot.metrics._PubMedLookup", MismatchLookup)

    records = cross_group_publications(grouped, enrich_pubmed=True)
    record = records[0]
    assert record["pubmed_id"] is None
    assert record["doi"] is None
    assert record["pubmed_authors"] is None
    assert record["pubmed_journal"] is None


def test_cross_group_publications_norm_title_fallback(
    monkeypatch, tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )

    calls: list[str] = []

    class FallbackLookup:
        def __init__(self, *, api_key, email, min_delay, session=None):
            pass

        def identifiers_for_title(self, title, year):
            calls.append(title)
            if title == "shared paper":
                return _PubMedDetails(
                    pubmed_id="PMLOWER",
                    doi="DOILOWER",
                    authors=None,
                    journal=None,
                )
            return None

    monkeypatch.setattr("cooplot.metrics._PubMedLookup", FallbackLookup)

    records = cross_group_publications(grouped, enrich_pubmed=True)
    assert records[0]["pubmed_id"] == "PMLOWER"
    assert "Shared Paper" in calls
    assert "shared paper" in calls


def test_grouped_publications_exclude_groups(
    tmp_path, sample_publications, sample_people
):
    grouped = aggregate_publications(
        sample_publications,
        sample_people,
        name_col="name",
        group_col="team",
        cache_dir=tmp_path,
        include_unlabeled=True,
        save_json=False,
    )
    filtered = grouped.exclude_groups(["Unlabeled"])
    assert "Unlabeled" not in filtered.by_group
    assert all(g != "Unlabeled" for g in filtered.sorted_groups())


def test_cross_group_report_from_json(monkeypatch, tmp_path, capsys):
    data = [
        {
            "title": "Shared Discoveries",
            "norm_title": "shared discoveries",
            "year": 2021,
            "groups": ["Group A", "Group B"],
            "authors": {"Group A": ["Alice Alpha"], "Group B": ["Bob Beta"]},
            "doi": "10.1234/example",
            "pubmed_id": "PM12345",
        }
    ]
    path = tmp_path / "cross.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        "cooplot.metrics._CitationFetcher._fetch_via_doi",
        lambda self, doi: f"Citation for {doi}",
    )
    monkeypatch.setattr(
        "cooplot.metrics._CitationFetcher._fetch_from_pubmed",
        lambda self, pmid: None,
    )

    out_file = tmp_path / "report.txt"
    report = cross_group_report(path, verbose=True, out_path=out_file)
    assert "Citation for 10.1234/example" in report
    assert "Collaboration: Group A (Alice Alpha) and Group B (Bob Beta)." in report
    captured = capsys.readouterr().out
    assert "Processing record" in captured
    assert "Retrieved citation" in captured
    data = out_file.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM


def test_cross_group_report_from_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "cross.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "title",
                "norm_title",
                "year",
                "groups",
                "authors",
                "pubmed_id",
                "doi",
                "pubmed_authors",
                "pubmed_journal",
            ]
        )
        writer.writerow(
            [
                "Neural Collaboration",
                "neural collaboration",
                "2020",
                "Group X;Group Y",
                json.dumps({"Group X": ["Xavier"], "Group Y": ["Yara"]}),
                "32132905",
                "10.5678/example",
                "[]",
                "Journal",
            ]
        )

    monkeypatch.setattr(
        "cooplot.metrics._CitationFetcher._fetch_via_doi",
        lambda self, doi: None,
    )

    monkeypatch.setattr(
        "cooplot.metrics._CitationFetcher._fetch_from_pubmed",
        lambda self, pmid: f"Citation from PubMed {pmid}",
    )

    out_file = tmp_path / "report.txt"
    report = cross_group_report(csv_path, out_path=out_file, ensure_ascii=True)
    assert "Citation from PubMed 32132905" in report
    assert "Collaboration: Group X (Xavier) and Group Y (Yara)." in report
    assert out_file.read_text(encoding="ascii") == report


def test_cross_group_report_uses_crossref_fallback(monkeypatch, tmp_path):
    data = [
        {
            "title": "Local Title",
            "norm_title": "local title",
            "year": 2022,
            "groups": ["Group Crossref", "Group Other"],
            "authors": {
                "Group Crossref": ["Alice Alpha"],
                "Group Other": ["Bob Beta"],
            },
            "doi": "10.4242/crossref",
            "crossref_title": "Crossref Title",
            "crossref_authors": ["Alice Alpha", "Bob Beta"],
            "crossref_container": "Journal of Crossref",
        }
    ]
    path = tmp_path / "cross.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        "cooplot.metrics._CitationFetcher._fetch_via_doi",
        lambda self, doi: None,
    )
    monkeypatch.setattr(
        "cooplot.metrics._CitationFetcher._fetch_from_pubmed",
        lambda self, pmid: None,
    )

    report = cross_group_report(path)
    assert "Crossref Title" in report
    assert "Journal of Crossref" in report
    assert "https://doi.org/10.4242/crossref" in report
    assert "Collaboration: Group Crossref (Alice Alpha) and Group Other (Bob Beta)." in report


def test_cross_group_report_skips_when_output_exists(monkeypatch, tmp_path):
    input_path = tmp_path / "cross.json"
    input_path.write_text("[]", encoding="utf-8")

    out_file = tmp_path / "report.txt"
    out_file.write_text("Existing report content", encoding="utf-8")

    def boom(path):
        raise AssertionError("Should not load records")

    monkeypatch.setattr("cooplot.metrics._load_cross_group_records", boom)

    report = cross_group_report(
        input_path,
        out_path=out_file,
        overwrite=False,
    )

    assert report == "Existing report content"
    assert out_file.read_text(encoding="utf-8") == "Existing report content"


def test_cross_group_report_overwrite_existing_file(monkeypatch, tmp_path):
    data = [
        {
            "title": "Example Title",
            "norm_title": "example title",
            "groups": ["Group 1", "Group 2"],
            "authors": {"Group 1": ["Alice"], "Group 2": ["Bob"]},
            "doi": "10.5555/example",
        }
    ]
    input_path = tmp_path / "cross.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")

    out_file = tmp_path / "report.txt"
    out_file.write_text("Stale report", encoding="utf-8")

    calls = {"count": 0}

    class DummyFetcher:
        def __init__(self, *, style, locale, pubmed_min_delay, verbose, printer):
            calls["count"] += 1

        def citation_for(self, record):
            return "Dummy Citation"

    monkeypatch.setattr("cooplot.metrics._CitationFetcher", DummyFetcher)

    report = cross_group_report(
        input_path,
        out_path=out_file,
        overwrite=True,
    )

    assert calls["count"] == 1
    assert "Dummy Citation" in report
    assert "Collaboration: Group 1 (Alice) and Group 2 (Bob)." in report
    saved = out_file.read_text(encoding="utf-8-sig")
    assert saved == report


def test_citation_fetcher_decodes_utf8(monkeypatch):
    fetcher = _CitationFetcher(
        style="apa",
        locale="en-US",
        pubmed_min_delay=None,
        verbose=False,
        printer=None,
    )

    def fake_get(url, headers=None, timeout=None):
        class Dummy:
            status_code = 200
            content = "Kühlewein, L. (2025).".encode("utf-8")

        return Dummy()

    monkeypatch.setattr(fetcher._session, "get", fake_get)
    citation = fetcher._fetch_via_doi("10.1000/test")
    assert citation == "Kühlewein, L. (2025)."
