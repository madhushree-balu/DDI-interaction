"""
polyguard.core.data_loader
==========================
Responsible for loading, normalising, and caching every dataset the pipeline
needs.  All file I/O is isolated here so the rest of the code never touches
pandas / CSV paths directly.

Usage::

    loader = DataLoader(data_dir="./datasets")
    loader.load()                       # call once at startup

    brands   = loader.search_brands("Aug", limit=5)
    ings     = loader.get_ingredients("Augmentin 625 Duo Tablet")
    pairs    = loader.lookup_interactions(["amoxicillin", "clarithromycin"])
    id_map   = loader.ingredient_id_map
"""

from __future__ import annotations

import gc
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and normalises all PolyGuard datasets from a given directory.

    Parameters
    ----------
    data_dir : str | Path
        Directory that contains the CSV files.  Defaults to ``./datasets``.
    """

    _PHARMA_FILE      = "indian_pharmaceutical_products_clean.csv"
    _DRUGBANK_FILE    = "drugbank_drugs.csv"
    _DDI_COMPLETE     = "ddi_complete.csv"
    _DDI_LABELED      = "ddi_labeled.csv"
    _DRUGBANK_INTER   = "drugbank_interactions.csv"

    def __init__(self, data_dir: str | Path = "./datasets") -> None:
        self._data_dir: Path = Path(data_dir)
        self._pharma_db: pd.DataFrame          = pd.DataFrame()
        self._drugbank_drugs: pd.DataFrame     = pd.DataFrame()
        self._interactions: pd.DataFrame       = pd.DataFrame()
        self._ingredient_id_map: Dict[str, str] = {}
        self._loaded: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "DataLoader":
        """Load all datasets.  Safe to call multiple times (idempotent)."""
        if self._loaded:
            return self
        logger.info("DataLoader: loading datasets from %s", self._data_dir)
        self._pharma_db       = self._load_csv(self._PHARMA_FILE)
        self._drugbank_drugs  = self._load_csv(self._DRUGBANK_FILE)
        self._interactions    = self._build_interactions_table()
        self._ingredient_id_map = self._build_ingredient_id_map()
        self._loaded = True
        logger.info(
            "DataLoader: ready — %d pharma rows, %d interaction pairs, %d ID entries",
            len(self._pharma_db),
            len(self._interactions),
            len(self._ingredient_id_map),
        )
        return self

    @property
    def ingredient_id_map(self) -> Dict[str, str]:
        self._assert_loaded()
        return self._ingredient_id_map

    @property
    def interactions_table(self) -> pd.DataFrame:
        self._assert_loaded()
        return self._interactions

    def search_brands(self, prefix: str, limit: int = 10) -> List[str]:
        """Return brand names containing *prefix* (case-insensitive), or ingredient names."""
        self._assert_loaded()
        if self._pharma_db.empty or "brand_name" not in self._pharma_db.columns:
            return []
            
        q = prefix.lower().strip()
        
        # 1. Startswith match
        mask_starts = self._pharma_db["brand_name"].str.lower().str.startswith(q, na=False)
        starts_results = self._pharma_db[mask_starts]["brand_name"].drop_duplicates().tolist()
        
        # 2. Contains match
        mask_contains = self._pharma_db["brand_name"].str.lower().str.contains(q, na=False, regex=False)
        contains_results = self._pharma_db[mask_contains & ~mask_starts]["brand_name"].drop_duplicates().tolist()
        
        # 3. Ingredient match
        ing_mask = self._pharma_db["primary_ingredient"].str.lower().str.contains(q, na=False, regex=False)
        ing_results = self._pharma_db[ing_mask]["primary_ingredient"].dropna().drop_duplicates().tolist()
        
        # Combine
        # Return starts_results first, then contains_results, then ingredients
        results = starts_results + contains_results + ing_results
        
        seen = set()
        final_results = []
        for r in results:
            clean_r = str(r).strip()
            if clean_r.lower() not in seen:
                seen.add(clean_r.lower())
                final_results.append(clean_r)
                if len(final_results) >= limit:
                    break
        
        # 4. Fallback for OCR cases: "Some Name Aspirine" -> Match "Aspirine"
        if not final_results and len(q.split()) > 1:
            words = [w for w in q.split() if len(w) > 3]
            fallback_results = []
            for w in words:
                w_mask = self._pharma_db["brand_name"].str.lower().str.contains(w, na=False, regex=False)
                fallback_results.extend(self._pharma_db[w_mask]["brand_name"].drop_duplicates().tolist())
                
                w_ing = self._pharma_db["primary_ingredient"].str.lower().str.contains(w, na=False, regex=False)
                fallback_results.extend(self._pharma_db[w_ing]["primary_ingredient"].dropna().drop_duplicates().tolist())
                
            for r in fallback_results:
                clean_r = str(r).strip()
                if clean_r.lower() not in seen:
                    seen.add(clean_r.lower())
                    final_results.append(clean_r)
                    if len(final_results) >= limit:
                        break

        return final_results

    def get_ingredients(self, brand_name: str) -> List[str]:
        """Return all active ingredients for *brand_name*."""
        self._assert_loaded()
        if self._pharma_db.empty:
            return []

        rows = self._pharma_db[
            self._pharma_db["brand_name"].str.lower() == brand_name.lower()
        ]

        # Fuzzy fallback: contains match
        if rows.empty:
            rows = self._pharma_db[
                self._pharma_db["brand_name"].str.lower().str.contains(
                    re.escape(brand_name.lower()), na=False
                )
            ]

        if rows.empty:
            search_name = brand_name.lower().strip()
            # Check if it is an ingredient
            mask_ing = self._pharma_db["primary_ingredient"].str.lower() == search_name
            if mask_ing.any():
                return [brand_name.title()]
                
            # Check in interactions directly
            if not self._interactions.empty:
                in_drug1 = (self._interactions["drug1_name"] == search_name).any()
                in_drug2 = (self._interactions["drug2_name"] == search_name).any()
                if in_drug1 or in_drug2:
                    return [brand_name.title()]
                    
            logger.warning("Brand or Ingredient '%s' not found in database.", brand_name)
            return []

        result: set = set()

        if "primary_ingredient" in rows.columns:
            for val in rows["primary_ingredient"].dropna():
                result.add(str(val).strip())

        if "active_ingredients" in rows.columns:
            for raw in rows["active_ingredients"].dropna():
                try:
                    parsed = eval(raw) if isinstance(raw, str) else raw  # noqa: S307
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                result.add(str(item["name"]).strip())
                except Exception:
                    pass

        return sorted(result)

    def lookup_interactions(self, ingredients: List[str]) -> List[Dict]:
        """
        Return all pairwise drug-drug interactions for *ingredients*.

        Strategy (priority order):
          1. Exact name match
          2. Partial / contains match as fallback
        """
        self._assert_loaded()
        if self._interactions.empty:
            return []

        from itertools import combinations

        norm  = [ing.lower().strip() for ing in ingredients]
        pairs = list(combinations(norm, 2))
        found: List[Dict] = []

        for drug_a, drug_b in pairs:
            mask = (
                (
                    (self._interactions["drug1_name"] == drug_a)
                    & (self._interactions["drug2_name"] == drug_b)
                )
                | (
                    (self._interactions["drug1_name"] == drug_b)
                    & (self._interactions["drug2_name"] == drug_a)
                )
            )
            matches = self._interactions[mask]

            if matches.empty:
                mask = (
                    (
                        self._interactions["drug1_name"].str.contains(
                            drug_a, na=False, regex=False
                        )
                        & self._interactions["drug2_name"].str.contains(
                            drug_b, na=False, regex=False
                        )
                    )
                    | (
                        self._interactions["drug1_name"].str.contains(
                            drug_b, na=False, regex=False
                        )
                        & self._interactions["drug2_name"].str.contains(
                            drug_a, na=False, regex=False
                        )
                    )
                )
                matches = self._interactions[mask]

            if not matches.empty:
                row = matches.iloc[0]
                found.append(
                    {
                        "drug_a":      drug_a.title(),
                        "drug_b":      drug_b.title(),
                        "description": row.get("description", "No description available"),
                        "severity":    row.get("severity", "Unknown"),
                        "mechanism":   row.get("mechanism", "Unknown"),
                        "source":      row.get("source", "DrugBank"),
                    }
                )

        return found

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "DataLoader has not been initialised. Call .load() first."
            )

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = self._data_dir / filename
        if not path.exists():
            logger.warning("Dataset not found: %s", path)
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]
            logger.debug("Loaded %s (%d rows)", filename, len(df))
            return df
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return pd.DataFrame()

    def _build_interactions_table(self) -> pd.DataFrame:
        """Merge all interaction CSV sources into one canonical table."""
        frames: List[pd.DataFrame] = []

        _source_priority = {
            self._DDI_COMPLETE:   ("ddi_complete",   0),
            self._DDI_LABELED:    ("ddi_labeled",    1),
            self._DRUGBANK_INTER: ("drugbank_raw",   2),
        }

        for filename, (source_tag, _) in _source_priority.items():
            df = self._load_csv(filename)
            if df.empty or "drug1_name" not in df.columns:
                del df
                gc.collect()
                continue

            cols = ["drug1_name", "drug2_name", "description"]
            for col in ("severity", "mechanism"):
                if col not in df.columns:
                    df[col] = "Unknown"
                cols.append(col)

            chunk = df[cols].copy()
            chunk["source"] = source_tag
            frames.append(chunk)
            del df
            gc.collect()

        if not frames:
            logger.warning("No interaction databases loaded.")
            return pd.DataFrame(
                columns=["drug1_name", "drug2_name", "description", "severity", "mechanism", "source"]
            )

        merged = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

        merged["drug1_name"]  = merged["drug1_name"].str.strip().str.lower()
        merged["drug2_name"]  = merged["drug2_name"].str.strip().str.lower()
        merged["description"] = merged["description"].fillna("No description available")
        merged["severity"]    = merged["severity"].fillna("Unknown")
        merged["mechanism"]   = merged["mechanism"].fillna("Unknown")

        source_prio = {"ddi_complete": 0, "ddi_labeled": 1, "drugbank_raw": 2}
        merged["_prio"] = merged["source"].map(source_prio).fillna(9)
        merged.sort_values("_prio", inplace=True)
        merged.drop_duplicates(subset=["drug1_name", "drug2_name"], keep="first", inplace=True)
        merged.drop(columns="_prio", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info("Interaction table: %d unique pairs", len(merged))
        return merged

    def _build_ingredient_id_map(self) -> Dict[str, str]:
        if self._drugbank_drugs.empty:
            return {}
        name_col = next(
            (c for c in self._drugbank_drugs.columns if "name" in c), None
        )
        id_col = next(
            (c for c in self._drugbank_drugs.columns if "id" in c), None
        )
        if not (name_col and id_col):
            return {}
        return dict(
            zip(
                self._drugbank_drugs[name_col].str.strip().str.lower(),
                self._drugbank_drugs[id_col].str.strip(),
            )
        )