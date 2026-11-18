from fastapi import APIRouter, Depends
from sqlalchemy import select
from .database import session
from .auth import current, operator, supervisor
from .models import CalibrationBatch, Complaint, NoiseEvent, TemplateCandidate
from .common import get
from . import calibration_service as service
from .calibration_schemas import BatchInput, CorrectionInput, EvaluateInput
router = APIRouter()


@router.get('/points')
def points(building: str, db=Depends(session), u=Depends(current)):
    # 已登记事件但还没校准的测点也要列出，让页面显示“沿用原模板”。
    positions = set(db.scalars(
        select(NoiseEvent.position)
        .join(Complaint, Complaint.id == NoiseEvent.complaint_id)
        .where(Complaint.building == building.strip()).distinct()
    ))
    return service.list_points(db, building, positions)


@router.get('/points/{building}/{position}')
def point(building: str, position: str, db=Depends(session), u=Depends(current)):
    view = service.get_point(db, building, position)
    return service.point_view(db, view) if view else service.point_view_placeholder(building, position)


@router.post('/calibration/batches')
def create_batch(data: BatchInput, db=Depends(session), u=Depends(operator)):
    return service.create_batch(db, u, data)


@router.get('/calibration/batches')
def batches(building: str = '', position: str = '', db=Depends(session), u=Depends(current)):
    query = select(CalibrationBatch).order_by(CalibrationBatch.id.desc())
    if building:
        query = query.where(CalibrationBatch.building == building)
    if position:
        query = query.where(CalibrationBatch.position == position)
    return [service.batch_view(db, b) for b in db.scalars(query)]


@router.post('/calibration/samples/{id}/correct')
def correct(id: int, data: CorrectionInput, db=Depends(session), u=Depends(operator)):
    return service.correct_sample(db, u, id, data)


@router.post('/calibration/candidates')
def evaluate(data: EvaluateInput, db=Depends(session), u=Depends(supervisor)):
    return service.evaluate(db, u, data)


@router.get('/calibration/candidates')
def candidates(building: str = '', position: str = '', db=Depends(session), u=Depends(current)):
    query = select(TemplateCandidate).order_by(TemplateCandidate.id.desc())
    if building:
        query = query.where(TemplateCandidate.building == building)
    if position:
        query = query.where(TemplateCandidate.position == position)
    return [service.candidate_view(db, c) for c in db.scalars(query)]


@router.post('/calibration/candidates/{id}/activate')
def activate(id: int, db=Depends(session), u=Depends(supervisor)):
    return service.activate(db, u, id)


@router.get('/calibration/labels')
def labels(db=Depends(session), u=Depends(current)):
    return {'states': service.TEMPLATE_STATE_LABELS}
