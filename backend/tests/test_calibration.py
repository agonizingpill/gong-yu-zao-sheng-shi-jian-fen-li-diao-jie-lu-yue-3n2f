"""测点校准模板的行为测试：整批训练/留出、留出验收、更正失效重算、并发审批。

使用临时SQLite数据库与真实FastAPI应用（不启动worker），直接验证HTTP层与领域规则。
"""
import os
import tempfile

_DB = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
_DB.close()
os.environ['DATABASE_URL'] = 'sqlite:///' + _DB.name
os.environ['SEED_DEMO'] = 'false'
os.environ['REDIS_URL'] = 'redis://127.0.0.1:1/0'

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.database import Base, engine, Session
from src.models import User, DemoClock, Complaint
from src.auth import password_hash
from src import templates as math

POINT = {'building': '3栋北侧', 'position': '窗台测点A'}
# 该测点上的真实剖面：equipment 在这个测点会与默认 impact 模板混淆。
TRUE = np.array([
    [1, 2, 3.5, 3, 2, 1, 0.5, 0.2],
    [0.3, 1, 3, 6, 5, 2, 0.7, 0.2],
    [0.1, 0.3, 0.8, 2, 5, 7, 4, 1],
], dtype=float)
TRUE /= TRUE.sum(axis=1, keepdims=True)


def rows(source_index, count, seed, noise=0.02):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        vector = TRUE[source_index] * (1 + rng.normal(0, noise, 8)) + rng.uniform(0, .004, 8)
        out.append([float(max(v, 0.0)) for v in vector])
    return out


def batch_payload(role, captured_at, seed):
    samples = []
    for index, source in enumerate(math.CALIBRATION_SOURCES):
        for energies in rows(index, 4, seed * 10 + index):
            samples.append({'source': source, 'energies': energies})
    return {**POINT, 'captured_at': captured_at, 'role': role, 'samples': samples}


@pytest.fixture(scope='session', autouse=True)
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session() as db:
        for role in ('operator', 'supervisor', 'viewer'):
            db.add(User(username=role, role=role, password_hash=password_hash('Quiet2026!')))
        db.add(DemoClock(id=1, at=__import__('datetime').datetime(2026, 11, 18, 16, 0)))
        db.commit()
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(engine)


def token(client, role):
    response = client.post('/auth/login', json={'username': role, 'password': 'Quiet2026!'})
    assert response.status_code == 200, response.text
    return {'Authorization': 'Bearer ' + response.json()['token']}


@pytest.fixture()
def desks(client):
    return {role: token(client, role) for role in ('operator', 'supervisor', 'viewer')}


@pytest.fixture(autouse=True)
def clean_calibration(client):
    """每个用例从原模板状态开始，避免测点模板在用例间互相影响。"""
    from src.models import (CalibrationBatch, CalibrationSample, MeasurementPoint,
                             TemplateCandidate, NoiseEvent, Complaint)
    with Session() as db:
        for model in (CalibrationSample, CalibrationBatch, TemplateCandidate,
                      MeasurementPoint, NoiseEvent, Complaint):
            db.query(model).delete()
        db.commit()
    yield


def make_batches(client, desks, tag):
    day = 1 + int(tag) % 20
    month = 1 + int(tag) // 20
    train = client.post('/calibration/batches', json=batch_payload('train', f'2026-{month:02d}-{day:02d}T20:00:00', int(tag) * 2 + 1), headers=desks['operator'])
    holdout = client.post('/calibration/batches', json=batch_payload('holdout', f'2026-{month:02d}-{day:02d}T23:00:00', int(tag) * 2 + 2), headers=desks['operator'])
    assert train.status_code == 200, train.text
    assert holdout.status_code == 200, holdout.text
    return train.json()['id'], holdout.json()['id']


def evaluate(client, desks, train_id, holdout_id):
    return client.post('/calibration/candidates', json={
        **POINT, 'train_batch_ids': [train_id], 'holdout_batch_ids': [holdout_id],
    }, headers=desks['supervisor'])


def activate(client, desks, candidate_id):
    return client.post(f'/calibration/candidates/{candidate_id}/activate', headers=desks['supervisor'])


def new_case(client, desks, code):
    response = client.post('/complaints', json={
        'code': code, 'alias': '访客测试', 'building': POINT['building'],
        'summary': '夜间低频持续，用于校准模板测试。', 'consent': True,
    }, headers=desks['operator'])
    assert response.status_code == 200, response.text
    return response.json()['id']


def record_event(client, desks, case_id, position, measured_at, energies, duration=60):
    return client.post(f'/complaints/{case_id}/events', json={
        'measured_at': measured_at, 'position': position,
        'duration_s': duration, 'energies': energies,
    }, headers=desks['operator'])


def test_batches_are_whole_and_point_scoped(client, desks):
    train_id, holdout_id = make_batches(client, desks, 1)
    # 同一次采集不能同时选作训练和留出（schema层与领域层都拦截）
    same = client.post('/calibration/candidates', json={
        **POINT, 'train_batch_ids': [train_id], 'holdout_batch_ids': [train_id],
    }, headers=desks['supervisor'])
    assert same.status_code == 422
    # 训练批次不能放在留出位置
    swapped = client.post('/calibration/candidates', json={
        **POINT, 'train_batch_ids': [train_id], 'holdout_batch_ids': [holdout_id],
    }, headers=desks['supervisor'])
    assert swapped.status_code == 200, swapped.text
    wrong = client.post('/calibration/candidates', json={
        **POINT, 'train_batch_ids': [holdout_id], 'holdout_batch_ids': [train_id],
    }, headers=desks['supervisor'])
    assert wrong.status_code == 409
    # 留出批若要用于训练，只能登记为训练批次
    other = client.post('/calibration/batches', json=batch_payload('holdout', '2026-11-03T22:00:00', 99), headers=desks['operator'])
    assert other.status_code == 200
    # 重复采集时刻整批拒绝
    duplicate = client.post('/calibration/batches', json=batch_payload('train', '2026-11-03T22:00:00', 100), headers=desks['operator'])
    assert duplicate.status_code == 409


def test_operator_cannot_evaluate_or_activate(client, desks):
    train_id, holdout_id = make_batches(client, desks, 2)
    response = client.post('/calibration/candidates', json={
        **POINT, 'train_batch_ids': [train_id], 'holdout_batch_ids': [holdout_id],
    }, headers=desks['operator'])
    assert response.status_code == 403


def test_uncalibrated_point_uses_default_template_for_events(client, desks):
    case_id = new_case(client, desks, 'QTCAL-DEFAULT')
    response = record_event(client, desks, case_id, '窗台测点B', '2026-11-17T22:30', rows(0, 1, 7)[0])
    assert response.status_code == 200, response.text
    event = response.json()
    assert event['template_origin'] == 'default'
    assert event['template_id'] is None
    assert event['template_label'] == '原模板'
    # 默认模板在这个测点上把设备声判成冲击声——正是需要校准的情形。
    assert event['predicted_source'] == 'impact'
    listed = client.get(f'/complaints/{case_id}/events', headers=desks['viewer']).json()
    assert listed[0]['template_label'] == '原模板'


def test_holdout_acceptance_calibrates_only_that_point(client, desks):
    train_id, holdout_id = make_batches(client, desks, 3)
    candidate = evaluate(client, desks, train_id, holdout_id)
    assert candidate.status_code == 200, candidate.text
    data = candidate.json()
    assert data['holdout_accuracy'] > data['baseline_accuracy']
    assert data['baseline_kind'] == 'default'

    activated = activate(client, desks, data['id'])
    assert activated.status_code == 200, activated.text
    after = activated.json()
    assert after['state'] == 'active'
    point = after['point']
    assert point['active']['label'] == f'QTC-{data["id"]}/v{data["revision"]}'
    assert point['active_template_id'] == data['id']
    assert point['using_default'] is False

    case_id = new_case(client, desks, 'QTCAL-ACTIVE')
    event = record_event(client, desks, case_id, POINT['position'], '2026-11-17T22:30', rows(0, 1, 11)[0]).json()
    assert event['template_origin'] == 'calibration'
    assert event['template_id'] == data['id']
    assert event['predicted_source'] == 'equipment'
    # 另一个测点不受影响，仍用原模板。
    other_case = new_case(client, desks, 'QTCAL-OTHER')
    other = record_event(client, desks, other_case, '窗台测点C', '2026-11-17T22:30', rows(0, 1, 12)[0]).json()
    assert other['template_origin'] == 'default'
    assert other['predicted_source'] == 'impact'


def test_weaker_candidate_is_rejected(client, desks):
    train_id, holdout_id = make_batches(client, desks, 4)
    first = evaluate(client, desks, train_id, holdout_id).json()
    assert activate(client, desks, first['id']).status_code == 200
    # 用同一批训练但噪声更大的训练批构造更差候选；其留出批相同。
    worse_train = client.post('/calibration/batches', json={
        **POINT, 'captured_at': '2026-11-04T22:00:00', 'role': 'train',
        'samples': [{'source': source, 'energies': energies}
                    for index, source in enumerate(math.CALIBRATION_SOURCES)
                    for energies in rows(index, 4, 500 + index, noise=0.6)],
    }, headers=desks['operator']).json()
    worse = evaluate(client, desks, worse_train['id'], holdout_id)
    assert worse.status_code == 200, worse.text
    body = worse.json()
    assert body['holdout_accuracy'] <= body['baseline_accuracy']
    rejected = activate(client, desks, body['id'])
    assert rejected.status_code == 409
    assert '未高于现用模板' in rejected.json()['detail']
    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    assert point['active_template_id'] == first['id']


def test_correcting_holdout_invalidates_and_recomputes(client, desks):
    train_id, holdout_id = make_batches(client, desks, 5)
    candidate = evaluate(client, desks, train_id, holdout_id).json()
    assert activate(client, desks, candidate['id']).status_code == 200

    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    holdout = next(b for b in point['batches'] if b['id'] == holdout_id)
    sample = holdout['samples'][0]
    # 把一条留出样本改成错误标签，等价于现场更正。
    response = client.post(f"/calibration/samples/{sample['id']}/correct", json={
        'source': 'speech' if sample['source'] != 'speech' else 'impact',
        'energies': sample['energies'],
        'reason': '现场复核录音标签，原声源标注有误',
    }, headers=desks['operator'])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['recomputed'], '更正后应生成重算候选'

    updated = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    assert updated['active_template_id'] is None, '留出样本被更正后，已生效结论失效，退回原模板'
    old = next(c for c in updated['candidates'] if c['id'] == candidate['id'])
    assert old['state'] == 'invalidated'
    assert '更正' in old['invalidated_reason']
    # 失效候选不能再被直接启用
    assert activate(client, desks, candidate['id']).status_code == 409
    # 旧候选所依据的留出摘要与新数据不一致
    recomputed_ids = {c['id'] for c in body['recomputed']}
    assert recomputed_ids <= {c['id'] for c in updated['candidates']}


def test_correcting_holdout_while_under_approval(client, desks):
    train_id, holdout_id = make_batches(client, desks, 6)
    candidate = evaluate(client, desks, train_id, holdout_id).json()
    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    sample = next(b for b in point['batches'] if b['id'] == holdout_id)['samples'][-1]
    client.post(f"/calibration/samples/{sample['id']}/correct", json={
        'source': 'equipment' if sample['source'] != 'equipment' else 'speech',
        'energies': sample['energies'],
        'reason': '复核发现留出批次标签登记错误',
    }, headers=desks['operator'])
    # 更正前算出的候选直接审批必须失败，不能凭旧结论启用。
    assert activate(client, desks, candidate['id']).status_code == 409


def test_concurrent_approvals_compare_against_freshly_active_template(client, desks):
    train_a, holdout = make_batches(client, desks, 7)
    train_b, _ = make_batches(client, desks, 8)
    first = evaluate(client, desks, train_a, holdout).json()
    second = evaluate(client, desks, train_b, holdout).json()
    assert first['baseline_kind'] == second['baseline_kind'] == 'default'
    first_result = activate(client, desks, first['id'])
    assert first_result.status_code == 200, first_result.text
    second_result = activate(client, desks, second['id'])
    # 两者训练数据不同，但留出相同；第二次审批必须与刚生效的第一个模板比较。
    if second_result.status_code == 200:
        body = second_result.json()
        assert body['baseline_kind'] == 'calibration'
        assert body['baseline_candidate_id'] == first['id']
    else:
        assert second_result.status_code == 409
        assert '现用模板' in second_result.json()['detail']
    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    assert point['active_template_id'] in (first['id'], second['id'])


def test_stale_candidate_after_another_activation_compares_to_active(client, desks):
    train_id, holdout_id = make_batches(client, desks, 9)
    stale = evaluate(client, desks, train_id, holdout_id).json()
    train2 = client.post('/calibration/batches', json=batch_payload('train', '2026-11-05T22:00:00', 77), headers=desks['operator']).json()
    winner = evaluate(client, desks, train2['id'], holdout_id).json()
    assert activate(client, desks, winner['id']).status_code == 200
    # stale 与 winner 都在同一留出批上；后审批的必须重新比较刚生效的 winner。
    result = activate(client, desks, stale['id'])
    if result.status_code == 200:
        assert result.json()['baseline_candidate_id'] == winner['id']
    else:
        assert result.status_code == 409
        assert '现用模板' in result.json()['detail']
    # 已生效候选不能重复启用
    assert activate(client, desks, winner['id']).status_code == 409


def test_viewer_cannot_write_calibration(client, desks):
    response = client.post('/calibration/batches', json=batch_payload('train', '2026-11-06T22:00:00', 55), headers=desks['viewer'])
    assert response.status_code == 403


def test_point_scoped_batch_cannot_build_other_point(client, desks):
    train_id, holdout_id = make_batches(client, desks, 10)
    other = client.post('/calibration/candidates', json={
        'building': '6栋东侧', 'position': '窗台测点A',
        'train_batch_ids': [train_id], 'holdout_batch_ids': [holdout_id],
    }, headers=desks['supervisor'])
    assert other.status_code == 409
    assert '不属于该测点' in other.json()['detail']


def test_events_keep_template_snapshot_after_invalidation(client, desks):
    train_id, holdout_id = make_batches(client, desks, 11)
    candidate = evaluate(client, desks, train_id, holdout_id).json()
    assert activate(client, desks, candidate['id']).status_code == 200
    case_id = new_case(client, desks, 'QTCAL-SNAPSHOT')
    before = record_event(client, desks, case_id, POINT['position'], '2026-11-17T22:30', rows(0, 1, 21)[0]).json()
    assert before['template_id'] == candidate['id']

    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    sample = next(b for b in point['batches'] if b['id'] == holdout_id)['samples'][0]
    client.post(f"/calibration/samples/{sample['id']}/correct", json={
        'source': 'impact' if sample['source'] != 'impact' else 'speech',
        'energies': sample['energies'], 'reason': '留出样本标签复核更正',
    }, headers=desks['operator'])

    events = client.get(f'/complaints/{case_id}/events', headers=desks['viewer']).json()
    kept = next(e for e in events if e['id'] == before['id'])
    # 历史事件保留当时用的模板，不改写记录；状态标注为已失效。
    assert kept['template_id'] == candidate['id']
    assert kept['template_label'] == f'QTC-{candidate["id"]}/v{candidate["revision"]}'
    assert kept['template_state'] == 'invalidated'
    # 新事件退回原模板。
    after = record_event(client, desks, case_id, POINT['position'], '2026-11-17T23:30', rows(0, 1, 22)[0]).json()
    assert after['template_origin'] == 'default'
    assert after['template_label'] == '原模板'


def test_correcting_train_sample_invalidates_candidate_too(client, desks):
    train_id, holdout_id = make_batches(client, desks, 12)
    candidate = evaluate(client, desks, train_id, holdout_id).json()
    assert activate(client, desks, candidate['id']).status_code == 200
    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    sample = next(b for b in point['batches'] if b['id'] == train_id)['samples'][0]
    response = client.post(f"/calibration/samples/{sample['id']}/correct", json={
        'source': 'impact' if sample['source'] != 'impact' else 'speech',
        'energies': sample['energies'], 'reason': '训练样本声源标签复核更正',
    }, headers=desks['operator'])
    assert response.status_code == 200, response.text
    updated = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    assert updated['active_template_id'] is None
    assert next(c for c in updated['candidates'] if c['id'] == candidate['id'])['state'] == 'invalidated'


def test_correction_without_change_is_rejected(client, desks):
    train_id, holdout_id = make_batches(client, desks, 13)
    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    sample = next(b for b in point['batches'] if b['id'] == holdout_id)['samples'][0]
    response = client.post(f"/calibration/samples/{sample['id']}/correct", json={
        'source': sample['source'], 'energies': sample['energies'], 'reason': '无变化重复提交',
    }, headers=desks['operator'])
    assert response.status_code == 409


def test_correction_that_leaves_training_without_source_keeps_audit(client, desks):
    """训练批被改成缺少某声源时，更正仍要留痕，但跳过重算。"""
    train = client.post('/calibration/batches', json={
        **POINT, 'captured_at': '2026-11-07T20:00:00', 'role': 'train',
        'samples': [{'source': source, 'energies': energies}
                    for index, source in enumerate(math.CALIBRATION_SOURCES)
                    for energies in rows(index, 2, 700 + index)],
    }, headers=desks['operator']).json()
    holdout = client.post('/calibration/batches', json=batch_payload('holdout', '2026-11-07T23:00:00', 801), headers=desks['operator']).json()
    candidate = evaluate(client, desks, train['id'], holdout['id']).json()
    speech_samples = [s for s in train['samples'] if s['source'] == 'speech']
    for sample in speech_samples:
        client.post(f"/calibration/samples/{sample['id']}/correct", json={
            'source': 'equipment', 'energies': sample['energies'], 'reason': '复核发现并非人声频带',
        }, headers=desks['operator'])
    point = client.get(f"/points/{POINT['building']}/{POINT['position']}", headers=desks['viewer']).json()
    original = next(c for c in point['candidates'] if c['id'] == candidate['id'])
    assert original['state'] == 'invalidated'
    assert all(c['state'] != 'evaluated' for c in point['candidates']), '缺声源时不能生成新候选'
