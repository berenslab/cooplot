import csv
import json

import pytest

from cooplot import api
from cooplot.aggregate import aggregate_publications
from cooplot.metrics import _PubMedDetails, cross_group_publications
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
            authors=["Author One", "Author Two"],
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
    assert record["pubmed_authors"] == ["Author One", "Author Two"]
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
    assert json.loads(row["pubmed_authors"]) == ["Author One", "Author Two"]
    assert row["pubmed_journal"] == "Journal of Testing"


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
                authors=["Env Author"],
                journal="Env Journal",
            )

    monkeypatch.setattr("cooplot.metrics._PubMedLookup", DummyLookup)

    records = cross_group_publications(grouped, enrich_pubmed=True)

    assert captured["api_key"] == "ENV_KEY"
    assert captured["email"] == "env@example.com"
    assert records[0]["pubmed_id"] == "PMENV"
    assert records[0]["doi"] == "DOIENV"
    assert records[0]["pubmed_authors"] == ["Env Author"]
    assert records[0]["pubmed_journal"] == "Env Journal"


def test_grouped_publications_exclude_groups(tmp_path, sample_publications, sample_people):
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
