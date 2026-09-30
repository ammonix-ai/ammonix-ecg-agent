"""The three endpoints the Universe (Lattice) page needs.

Paths match the private application exactly so the frontend needs no rewrite:

    GET /api/admin/cv/lattice-status
    GET /api/admin/cv/projection/{method}?cohort_id=-60603&perplexity=&n_neighbors=
    GET /api/admin/cv/source-safe-roc?diagnosis=<name>
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.source_safe_universe import (
    SOURCE_SAFE_COHORT_ID,
    UniverseDataMissing,
    get_source_safe_projection,
    get_source_safe_roc,
    source_safe_lattice_status,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/cv", tags=["universe"])


# ---------------------------------------------------------------------------
# response models
# ---------------------------------------------------------------------------

class CVLatticeStatusResponse(BaseModel):
    """Response for GET /api/admin/cv/lattice-status.

    One row per selectable cohort. The open build publishes exactly one: the
    source-safe universe.
    """
    model_config = ConfigDict(populate_by_name=True)

    cohorts: List[Dict[str, Any]]


class CVProjectionResponse(BaseModel):
    """Response for GET /api/admin/cv/projection/{method}.

    The per-point payload is intentionally loose — each row carries 20+ fields
    and the viewer consumes them defensively. Top-level scalars are typed.
    """
    model_config = ConfigDict(populate_by_name=True)

    method: str  # 'pca' | 'clinical' | 'tsne' | 'umap'
    points: List[Dict[str, Any]]
    total_patients: int = Field(alias="totalPatients")
    cohorts_included: List[str] = Field(alias="cohortsIncluded")
    hyperparameters: Dict[str, Any]
    diagnosis_colors: Dict[str, str] = Field(alias="diagnosisColors")
    cached: bool
    axis_labels: List[str] = Field(default_factory=list, alias="axisLabels")
    source_safe: Optional[Dict[str, Any]] = Field(default=None, alias="sourceSafe")


class CVROCPoint(BaseModel):
    fpr: float
    tpr: float


class SourceSafeROCResponse(BaseModel):
    """Per-diagnosis ROC computed on the displayed universe. `auroc` is the area
    over THIS population; `heldOutFinalAuroc` / `heldOutPrimaryAuroc` are the
    model package's held-out numbers, returned for reference and shown
    distinctly so the on-universe value is never mistaken for the held-out one.
    """
    model_config = ConfigDict(populate_by_name=True)

    diagnosis: str
    points: List[CVROCPoint]
    auroc: float
    threshold: float
    operating_fpr: Optional[float] = Field(default=None, alias="operatingFpr")
    operating_tpr: Optional[float] = Field(default=None, alias="operatingTpr")
    n_positive: int = Field(alias="nPositive")
    n_negative: int = Field(alias="nNegative")
    held_out_final_auroc: Optional[float] = Field(default=None, alias="heldOutFinalAuroc")
    held_out_primary_auroc: Optional[float] = Field(default=None, alias="heldOutPrimaryAuroc")


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@router.get("/lattice-status", response_model=CVLatticeStatusResponse)
async def lattice_status() -> dict[str, Any]:
    """Cohorts available to the Universe page. Populates the cohort dropdown;
    the page auto-selects the first row that has training."""
    try:
        row = await asyncio.to_thread(source_safe_lattice_status)
    except UniverseDataMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"cohorts": [row]}


@router.get("/projection/{method}", response_model=CVProjectionResponse)
async def cv_projection(
    method: str = "pca",
    perplexity: int = 100,
    n_neighbors: int = 15,
    cohort_id: Optional[int] = None,
) -> dict[str, Any]:
    """The 3-D scene: every record's coordinates, labels, predictions and colour.

    `cohort_id` is accepted for wire compatibility with the private app, which
    routes other ids to locally trained cohorts. This build serves only the
    published universe, so any other id is a 404.
    """
    if cohort_id is not None and cohort_id != SOURCE_SAFE_COHORT_ID:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown cohort_id {cohort_id}. This build serves only the published "
                f"source-safe universe (cohort_id={SOURCE_SAFE_COHORT_ID})."
            ),
        )
    try:
        return await asyncio.to_thread(
            get_source_safe_projection,
            method,
            perplexity=perplexity,
            n_neighbors=n_neighbors,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except UniverseDataMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Universe projection failed")
        raise HTTPException(status_code=500, detail=f"Universe projection failed: {exc}")


@router.get("/source-safe-roc", response_model=SourceSafeROCResponse)
async def source_safe_roc_curve(diagnosis: str) -> dict[str, Any]:
    """ROC for one diagnosis on the displayed universe.

    `diagnosis` is a QUERY parameter, not a path segment: several class names
    contain a '/' (e.g. "left bundle branch block / variations"), and a slash in
    a path parameter — even percent-encoded as %2F — breaks the route match and
    404s. A query value carries '/' and other specials cleanly.
    """
    try:
        return await asyncio.to_thread(get_source_safe_roc, diagnosis)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Diagnosis '{diagnosis}' is not in the universe.")
    except UniverseDataMissing as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Universe ROC computation failed")
        raise HTTPException(status_code=500, detail=f"Universe ROC failed: {exc}")
