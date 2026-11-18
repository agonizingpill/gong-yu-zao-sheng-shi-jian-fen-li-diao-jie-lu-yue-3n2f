"""测点模板校准：整批训练/留出、留出验收、更正后失效重算、并发审批复比。

规则要点：
- 同一次采集的所有样本属于一个批次，整批只能用作训练或留出，不能拆分。
- 主管只能启用“在候选自己的留出批次上，比现用模板更准”的候选；默认模板视为基线。
- 留出样本（或训练样本）被更正后，依赖它的候选结论失效并重算，已生效模板退回原模板。
- 审批时锁定测点行并重新读取刚生效的模板作比较，后到者在同批留出样本上重新比较。
"""
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from .models import (
    CalibrationBatch, CalibrationSample, MeasurementPoint, TemplateCandidate,
)
from .common import require, get, record, now, dump
from .constants import CALIBRATION_SOURCES
from . import templates as math

EPSILON = 1e-12


def point_key(building, position):
    return {'building': str(building).strip(), 'position': str(position).strip()}


def get_point(db, building, position):
    key = point_key(building, position)
    return db.scalar(select(MeasurementPoint).where(
        MeasurementPoint.building == key['building'],
        MeasurementPoint.position == key['position'],
    ))


def lock_point(db, building, position):
    key = point_key(building, position)
    return db.scalar(select(MeasurementPoint).where(
        MeasurementPoint.building == key['building'],
        MeasurementPoint.position == key['position'],
    ).with_for_update().execution_options(populate_existing=True))


def ensure_point(db, building, position):
    point = get_point(db, building, position)
    if point:
        return point
    key = point_key(building, position)
    savepoint = db.begin_nested()
    try:
        point = MeasurementPoint(**key)
        db.add(point)
        savepoint.commit()
        return point
    except IntegrityError:
        savepoint.rollback()
        return get_point(db, building, position)


def active_template(db, point):
    """返回该测点当前生效模板；未通过校准或已失效时返回原模板。"""
    if point and point.active_template_id:
        # 审批锁后重新读取，确保与刚提交的生效模板比较，而不是会话里的旧对象。
        candidate = db.get(TemplateCandidate, point.active_template_id, populate_existing=True)
        if candidate and candidate.state == 'active':
            return {
                'origin': 'calibration',
                'candidate': candidate,
                'id': candidate.id,
                'revision': candidate.revision,
                'point_revision': point.revision,
                'templates': math.as_dict(candidate.templates),
            }
    return {
        'origin': 'default',
        'candidate': None,
        'id': None,
        'revision': 0,
        'point_revision': point.revision if point else 0,
        'templates': math.as_dict(None),
    }


def classify_for_point(db, building, position, energies):
    """按测点当前模板判读一条事件，返回可直接落库的快照字段。"""
    point = get_point(db, building, position)
    active = active_template(db, point)
    source, score = math.classify(energies, active['templates'])
    return {
        'template_origin': active['origin'],
        'template_id': active['id'],
        'template_revision': active['revision'],
        'predicted_source': source,
        'predicted_score': score,
    }


def samples_of(db, batch_ids):
    if not batch_ids:
        return []
    return list(db.scalars(select(CalibrationSample).where(
        CalibrationSample.batch_id.in_(batch_ids)
    ).order_by(CalibrationSample.batch_id, CalibrationSample.id)))


def holdout_checksum(db, batch_ids):
    rows = samples_of(db, batch_ids)
    return math.checksum([{'source': s.source, 'energies': s.energies} for s in rows])


def _validate_batches(db, building, position, train_ids, holdout_ids):
    require(train_ids, '至少选择一个训练批次')
    require(holdout_ids, '至少选择一个留出批次')
    require(not set(train_ids) & set(holdout_ids), '同一次采集不能同时用于训练和留出')
    batches = list(db.scalars(select(CalibrationBatch).where(
        CalibrationBatch.id.in_(set(train_ids) | set(holdout_ids))
    )))
    by_id = {b.id: b for b in batches}
    require(len(by_id) == len(set(train_ids) | set(holdout_ids)), '存在无效的校准批次')
    key = point_key(building, position)
    for batch_id in train_ids:
        batch = by_id[batch_id]
        require((batch.building, batch.position) == (key['building'], key['position']), '批次不属于该测点')
        require(batch.role == 'train', f'批次 #{batch_id} 登记为留出，不能用于训练')
    for batch_id in holdout_ids:
        batch = by_id[batch_id]
        require((batch.building, batch.position) == (key['building'], key['position']), '批次不属于该测点')
        require(batch.role == 'holdout', f'批次 #{batch_id} 登记为训练，不能用于留出验收')
    holdout_sources = {s.source for s in samples_of(db, holdout_ids)}
    require(set(CALIBRATION_SOURCES) <= holdout_sources, '留出批次须覆盖全部已知声源，否则无法验收')
    train_sources = {s.source for s in samples_of(db, train_ids)}
    require(set(CALIBRATION_SOURCES) <= train_sources, '训练批次须覆盖全部已知声源，否则无法生成候选模板')
    return by_id


def _build(db, point, train_ids, holdout_ids, actor, revision, state='evaluated'):
    train_samples = samples_of(db, train_ids)
    holdout_samples = samples_of(db, holdout_ids)
    try:
        candidate_templates = math.derive(train_samples)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    active = active_template(db, point)
    baseline = active['templates']
    train_accuracy = math.accuracy(train_samples, candidate_templates)
    holdout_accuracy = math.accuracy(holdout_samples, candidate_templates)
    baseline_accuracy = math.accuracy(holdout_samples, baseline)
    candidate = TemplateCandidate(
        building=point.building, position=point.position,
        state=state, revision=revision,
        templates=candidate_templates,
        train_batch_ids=[int(x) for x in train_ids],
        holdout_batch_ids=[int(x) for x in holdout_ids],
        train_accuracy=train_accuracy,
        holdout_accuracy=holdout_accuracy,
        baseline_kind=active['origin'],
        baseline_candidate_id=active['id'],
        baseline_accuracy=baseline_accuracy,
        baseline_revision=active['revision'],
        holdout_checksum=holdout_checksum(db, holdout_ids),
        created_by=actor,
    )
    db.add(candidate)
    db.flush()
    return candidate


def create_batch(db, user, data):
    point = ensure_point(db, data.building, data.position)
    require(not db.scalar(select(CalibrationBatch).where(
        CalibrationBatch.building == point.building,
        CalibrationBatch.position == point.position,
        CalibrationBatch.captured_at == data.captured_at,
    )), '该测点同一采集时刻已登记批次，整批不可重复登记')
    require(data.captured_at <= now(db), '采集时刻不可晚于演示时钟')
    batch = CalibrationBatch(
        building=point.building, position=point.position,
        captured_at=data.captured_at, role=data.role, note=data.note,
        created_by=user.username,
    )
    db.add(batch)
    db.flush()
    for sample in data.samples:
        db.add(CalibrationSample(batch_id=batch.id, source=sample.source, energies=sample.energies))
    record(db, user, 'calibration.batch', batch.id,
           f'{point.building}/{point.position};role={data.role};samples={len(data.samples)}')
    db.commit()
    return batch_view(db, get(db, CalibrationBatch, batch.id))


def batch_view(db, batch):
    samples = list(db.scalars(select(CalibrationSample).where(
        CalibrationSample.batch_id == batch.id
    ).order_by(CalibrationSample.id)))
    return {**dump(batch), 'samples': [dump(s) for s in samples]}


def candidate_view(db, candidate):
    return {**dump(candidate), 'template_label': label(candidate.id, candidate.revision)}


def label(candidate_id, revision):
    return f'QTC-{candidate_id}/v{revision}'


def correct_sample(db, user, sample_id, data):
    sample = get(db, CalibrationSample, sample_id)
    batch = get(db, CalibrationBatch, sample.batch_id)
    point = lock_point(db, batch.building, batch.position)
    if point is None:
        point = ensure_point(db, batch.building, batch.position)
    require(sample.source != data.source or list(sample.energies) != list(data.energies),
            '更正后的声源与频带与原件相同，无需登记更正')
    sample.source = data.source
    sample.energies = data.energies
    sample.corrected_at = now(db)
    sample.correction_reason = data.reason
    record(db, user, 'calibration.correct', sample.id,
           f'batch={batch.id};role={batch.role};source={data.source};reason={data.reason}')
    recomputed = recompute(db, point, user, batch.id,
                           f'批次 #{batch.id}（{batch.role}）的样本 #{sample.id} 被更正')
    db.commit()
    return {'sample': dump(sample), 'recomputed': [candidate_view(db, c) for c in recomputed]}


def recompute(db, point, user, batch_id, reason):
    """更正后：引用该批次的候选结论失效；若已生效则退回原模板，并按当前数据重算新候选。"""
    candidates = list(db.scalars(select(TemplateCandidate).where(
        TemplateCandidate.building == point.building,
        TemplateCandidate.position == point.position,
        TemplateCandidate.state.in_(['evaluated', 'active']),
    ).order_by(TemplateCandidate.id)))
    affected = [
        c for c in candidates
        if int(batch_id) in list(c.train_batch_ids or []) + list(c.holdout_batch_ids or [])
    ]
    if not affected:
        return []
    was_active = point.active_template_id
    if was_active and any(c.id == was_active for c in affected):
        point.active_template_id = None
        point.revision += 1
        db.flush()
        record(db, user, 'template.invalidate', was_active, reason)
    fresh = []
    rebuilt = set()
    for candidate in affected:
        candidate.state = 'invalidated'
        candidate.invalidated_reason = reason
        db.flush()
        combo = (tuple(candidate.train_batch_ids or []), tuple(candidate.holdout_batch_ids or []))
        if combo in rebuilt:
            continue
        rebuilt.add(combo)
        revision = (db.scalar(select(TemplateCandidate.revision).where(
            TemplateCandidate.building == point.building,
            TemplateCandidate.position == point.position,
        ).order_by(TemplateCandidate.revision.desc()).limit(1)) or 0) + 1
        try:
            fresh.append(_build(db, point, candidate.train_batch_ids,
                                candidate.holdout_batch_ids, 'system', revision))
        except HTTPException:
            # 更正后训练批可能已不足以生成模板；保留更正记录，等补齐样本再重算。
            db.flush()
            record(db, user, 'template.recompute_skipped',
                   f'{point.building}/{point.position}',
                   f'train={candidate.train_batch_ids};holdout={candidate.holdout_batch_ids}')
    db.flush()
    record(db, user, 'template.recompute', f'{point.building}/{point.position}',
           f'affected={len(affected)};reason={reason}')
    return fresh


def evaluate(db, user, data):
    point = ensure_point(db, data.building, data.position)
    point = lock_point(db, point.building, point.position) or point
    _validate_batches(db, point.building, point.position, data.train_batch_ids, data.holdout_batch_ids)
    revision = (db.scalar(select(TemplateCandidate.revision).where(
        TemplateCandidate.building == point.building,
        TemplateCandidate.position == point.position,
    ).order_by(TemplateCandidate.revision.desc()).limit(1)) or 0) + 1
    candidate = _build(db, point, data.train_batch_ids, data.holdout_batch_ids,
                       user.username, revision)
    record(db, user, 'template.evaluate', candidate.id,
           f'holdout={candidate.holdout_accuracy:.4f};baseline={candidate.baseline_accuracy:.4f}')
    db.commit()
    return candidate_view(db, candidate)


def activate(db, user, candidate_id):
    candidate = get(db, TemplateCandidate, candidate_id)
    # 测点行是模板审批的锁根：后到的审批在此等待，拿到刚生效模板后再比较。
    point = lock_point(db, candidate.building, candidate.position)
    require(point is not None, '测点记录不存在')
    db.refresh(candidate)
    require(candidate.state == 'evaluated', '该候选已生效、已被取代或已失效，不能再次启用')
    require(holdout_checksum(db, candidate.holdout_batch_ids) == candidate.holdout_checksum,
            '留出样本已变更，验收结论失效，请重算后再启用')
    active = active_template(db, point)
    baseline_accuracy = math.accuracy(
        samples_of(db, candidate.holdout_batch_ids), active['templates'])
    holdout_accuracy = math.accuracy(
        samples_of(db, candidate.holdout_batch_ids),
        math.as_dict(candidate.templates))
    if holdout_accuracy <= baseline_accuracy + EPSILON:
        record(db, user, 'template.activate_reject', candidate.id,
               f'holdout={holdout_accuracy:.4f};baseline={baseline_accuracy:.4f};'
               f'baseline_kind={active["origin"]};candidate={active["id"]}')
        db.commit()
        raise HTTPException(
            409,
            f'候选 {label(candidate.id, candidate.revision)} 在同一批留出样本上判对率 '
            f'{holdout_accuracy:.2%}，未高于现用模板的 {baseline_accuracy:.2%}，主管不能启用')
    if point.active_template_id:
        previous = db.get(TemplateCandidate, point.active_template_id)
        if previous and previous.id != candidate.id:
            previous.state = 'superseded'
    candidate.state = 'active'
    candidate.activated_at = now(db)
    candidate.holdout_accuracy = holdout_accuracy
    candidate.baseline_kind = active['origin']
    candidate.baseline_candidate_id = active['id']
    candidate.baseline_accuracy = baseline_accuracy
    candidate.baseline_revision = active['revision']
    point.active_template_id = candidate.id
    point.revision += 1
    db.flush()
    record(db, user, 'template.activate', candidate.id,
           f'holdout={holdout_accuracy:.4f};baseline={baseline_accuracy:.4f};'
           f'baseline_kind={active["origin"]};revision={point.revision}')
    db.commit()
    return {**candidate_view(db, candidate),
            'point': point_view(db, point)}


def event_snapshot_view(db, event):
    """给事件补充所用模板的可读标签与当前状态（历史快照本身不改）。"""
    view = dump(event)
    view['template_label'] = '原模板' if not event.template_id else label(event.template_id, event.template_revision)
    view['template_state'] = 'default'
    if event.template_id:
        candidate = db.get(TemplateCandidate, event.template_id)
        view['template_state'] = candidate.state if candidate else 'missing'
    return view


def point_view(db, point):
    batches = list(db.scalars(select(CalibrationBatch).where(
        CalibrationBatch.building == point.building,
        CalibrationBatch.position == point.position,
    ).order_by(CalibrationBatch.captured_at, CalibrationBatch.id)))
    candidates = list(db.scalars(select(TemplateCandidate).where(
        TemplateCandidate.building == point.building,
        TemplateCandidate.position == point.position,
    ).order_by(TemplateCandidate.id.desc())))
    active = active_template(db, point)
    return {
        **dump(point),
        'active': None if active['origin'] == 'default' else {
            'id': active['id'],
            'label': label(active['id'], active['revision']),
            'revision': active['revision'],
            'templates': active['templates'],
            'holdout_accuracy': active['candidate'].holdout_accuracy,
            'baseline_accuracy': active['candidate'].baseline_accuracy,
            'summary': '已通过留出验收，正在使用校准模板',
        },
        'using_default': active['origin'] == 'default',
        'batches': [batch_view(db, b) for b in batches],
        'candidates': [candidate_view(db, c) for c in candidates],
    }


def list_points(db, building, positions=()):
    building = str(building).strip()
    rows = list(db.scalars(select(MeasurementPoint).where(
        MeasurementPoint.building == building
    ).order_by(MeasurementPoint.position)))
    seen = {p.position for p in rows}
    views = [point_view(db, p) for p in rows]
    # 只在页面上展示尚未校准的测点，沿用原模板。
    for position in sorted(set(positions) - seen):
        views.append(point_view_placeholder(building, position))
    return views


def point_view_placeholder(building, position):
    return {
        'id': None, 'building': building, 'position': position,
        'active_template_id': None, 'revision': 0,
        'active': None, 'using_default': True,
        'batches': [], 'candidates': [],
    }


def template_state_label(state):
    return TEMPLATE_STATE_LABELS.get(state, state)


TEMPLATE_STATE_LABELS = {
    'evaluated': '待审批', 'active': '已生效',
    'superseded': '已取代', 'invalidated': '已失效',
    'default': '原模板',
}
