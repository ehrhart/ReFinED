import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from refined.data_types.base_types import Span
from refined.inference.processor import Refined

MODEL_NAME = os.environ.get("MODEL_NAME", "wikipedia_model_with_numbers")
ENTITY_SET = os.environ.get("ENTITY_SET", "wikipedia")
DATA_DIR = os.environ.get("REFINED_DATA_DIR", "/data/refined")
USE_PRECOMPUTED_DESCRIPTIONS = os.environ.get("USE_PRECOMPUTED_DESCRIPTIONS", "true").lower() == "true"
DEVICE = os.environ.get("REFINED_DEVICE", "cpu")

refined = Refined.from_pretrained(
    model_name=MODEL_NAME,
    entity_set=ENTITY_SET,
    data_dir=DATA_DIR,
    use_precomputed_descriptions=USE_PRECOMPUTED_DESCRIPTIONS,
    download_files=True,
    device=DEVICE,
)

app = FastAPI(
    title="ReFinED entity linking",
    description="Entity linking over Wikidata. Given text, returns entity mentions with Wikidata QIDs.",
    version="1.0.0",
)


class AnnotateTextRequest(BaseModel):
    text: str = Field(..., description="Text to annotate")


def entity_to_json(entity):
    if entity is None:
        return None
    return {
        "qid": entity.wikidata_entity_id,
        "label": entity.human_readable_name,
        "wikipedia_title": entity.wikipedia_entity_title,
    }


def span_to_json(span: Span):
    top_entities = []
    if span.top_k_predicted_entities is not None:
        top_entities = [entity_to_json(e) for e in span.top_k_predicted_entities[0] or []]
    return {
        "text": span.text,
        "start": span.start,
        "end": span.start + span.ln,
        "coarse_type": span.coarse_type,
        "mention_type": span.coarse_mention_type,
        "date": str(span.date) if span.date is not None else None,
        "confidence": span.entity_linking_model_confidence_score,
        "entity": entity_to_json(span.predicted_entity),
        "top_entities": top_entities,
        "types": [
            {"id": type_id, "label": type_label, "confidence": type_confidence}
            for type_id, type_label, type_confidence in (span.predicted_entity_types or [])
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "entity_set": ENTITY_SET, "device": DEVICE}


@app.post("/annotate_text")
def annotate_text(request: AnnotateTextRequest):
    try:
        spans = refined.process_text(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"text": request.text, "spans": [span_to_json(s) for s in spans]}
