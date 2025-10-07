import pytest

from cooplot.metrics import _PubMedLookup


@pytest.mark.network
def test_pubmed_lookup_real_cases():
    lookup = _PubMedLookup(min_delay=0.5)

    cases = [
        (
            "A gaze-triggered downbeat nystagmus persisting in primary position in a "
            "patient with hypomagnesemia combined with anti-SOX1",
            2020,
            "32105977",
            "10.1016/j.jns.2020.116732",
        ),
        (
            "A Tactile Virtual Reality for the Study of Active Somatosensation",
            2020,
            "32132905",
            "10.3389/fnint.2020.00005",
        ),
    ]

    for title, year, pmid, doi in cases:
        details = lookup.identifiers_for_title(title, year)
        assert details is not None
        assert details.pubmed_id == pmid
        assert details.doi and details.doi.lower() == doi.lower()
