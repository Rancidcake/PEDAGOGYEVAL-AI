"""Capture presentation screenshots of the running app (dev tool; needs `pip install playwright` + Google Chrome).

    bash scripts/run_demo.sh                       # in one terminal
    python scripts/capture_screenshots.py presentation/screenshots [http://localhost:8501]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "presentation/screenshots")
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8501"
SLIDE = {"width": 1600, "height": 900}  # 16:9 slide
TALL = {"width": 1600, "height": 2700}  # whole verdict column in one image


def settle(pg: Page, ms: int = 2500) -> None:
    pg.mouse.move(2, 2)  # keep hover styles off buttons
    pg.wait_for_timeout(ms)


def shot(pg: Page, name: str, tall: bool = False) -> None:
    pg.set_viewport_size(TALL if tall else SLIDE)
    settle(pg, 1500)
    pg.screenshot(path=str(OUT / f"{name}.png"))
    pg.set_viewport_size(SLIDE)
    print("  ", name)


def grade(pg: Page, quick: str) -> None:
    pg.get_by_role("button", name=quick, exact=True).click()
    pg.wait_for_timeout(1000)
    pg.get_by_role("button", name="GRADE", exact=True).click()
    pg.wait_for_selector("[data-testid=stGraphVizChart] svg", timeout=60000)
    settle(pg)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        pg = b.new_page(viewport=SLIDE, device_scale_factor=2)

        pg.goto(BASE)
        pg.wait_for_selector(".pe-title", timeout=90000)
        settle(pg, 3000)
        shot(pg, "01_home")
        shot(pg, "01_home_full", tall=True)

        pg.goto(f"{BASE}/Grade")
        pg.wait_for_selector("text=GOOD PARAPHRASE", timeout=60000)
        settle(pg)
        shot(pg, "02_grade_empty")

        grade(pg, "GOOD PARAPHRASE")
        shot(pg, "03_grade_good_paraphrase")
        shot(pg, "03_grade_good_paraphrase_full", tall=True)
        pg.locator("[data-testid=stGraphVizChart]").last.screenshot(path=str(OUT / "03_concept_graph_paraphrase.png"))

        pg.get_by_text("Semantic matching").click()
        settle(pg)
        shot(pg, "04_grade_paraphrase_semantic_off_full", tall=True)
        pg.get_by_text("Semantic matching").click()
        settle(pg)

        grade(pg, "INCOMPLETE")
        shot(pg, "05_grade_incomplete")
        shot(pg, "05_grade_incomplete_full", tall=True)
        pg.locator("[data-testid=stGraphVizChart]").last.screenshot(path=str(OUT / "05_concept_graph_incomplete.png"))

        grade(pg, "WRONG BUT TOPICAL")
        shot(pg, "06_grade_wrong_but_topical")
        shot(pg, "06_grade_wrong_but_topical_full", tall=True)

        grade(pg, "NEGATION TRAP")
        shot(pg, "07_grade_negation_trap")
        shot(pg, "07_grade_negation_trap_full", tall=True)
        pg.locator(".pe-response").screenshot(path=str(OUT / "07_marked_script_negation.png"))

        for label, name in (("A · TF-IDF", "08_grade_model_A_tfidf"), ("B · MINILM", "08_grade_model_B_minilm")):
            pg.get_by_test_id("stRadioGroup").get_by_text(label, exact=True).click()
            settle(pg)
            shot(pg, name)
        pg.get_by_test_id("stRadioGroup").get_by_text("C · CONCEPT GRAPH", exact=True).click()
        settle(pg)

        pg.goto(f"{BASE}/Rubric")
        pg.wait_for_selector("text=Validation", timeout=60000)
        settle(pg, 3000)
        shot(pg, "09_rubric_editor")
        shot(pg, "09_rubric_editor_full", tall=True)
        num = pg.locator("[data-testid=stNumberInput] input").first
        num.fill("10")
        num.press("Enter")
        settle(pg)
        shot(pg, "10_rubric_validation_error")

        pg.goto(f"{BASE}/Results")
        pg.wait_for_selector(".pe-ledger", timeout=60000)
        settle(pg, 4000)
        shot(pg, "11_results_pilot")
        shot(pg, "11_results_pilot_full", tall=True)
        pg.get_by_role("tab", name="SCIENTSBANK").click()
        settle(pg, 3500)
        shot(pg, "12_results_scientsbank")
        shot(pg, "12_results_scientsbank_full", tall=True)
        pg.get_by_test_id("stRadioGroup").get_by_text("3WAY", exact=True).click()
        settle(pg, 3500)
        shot(pg, "13_results_scientsbank_3way_full", tall=True)
        pg.get_by_role("tab", name="ABLATIONS").click()
        settle(pg, 3500)
        shot(pg, "14_results_ablations")
        shot(pg, "14_results_ablations_full", tall=True)
        b.close()


if __name__ == "__main__":
    main()
