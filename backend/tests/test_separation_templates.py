"""分离计算按各事件所用测点模板分解，并逐事件标注模板来源。"""
import numpy as np
from src import templates as math
from src.separation import calculate

TRUE = np.array([
    [1, 2, 3.5, 3, 2, 1, 0.5, 0.2],
    [0.3, 1, 3, 6, 5, 2, 0.7, 0.2],
    [0.1, 0.3, 0.8, 2, 5, 7, 4, 1],
], dtype=float)
TRUE /= TRUE.sum(axis=1, keepdims=True)


def test_separation_labels_each_event_template():
    default = math.as_dict(None)
    calibrated = {source: TRUE[i].tolist() for i, source in enumerate(math.CALIBRATION_SOURCES)}
    events = [
        {'id': 1, 'energies': TRUE[0].tolist(), 'templates': default,
         'template': {'origin': 'default', 'id': None, 'revision': 0, 'label': '原模板'}},
        {'id': 2, 'energies': TRUE[0].tolist(), 'templates': calibrated,
         'template': {'origin': 'calibration', 'id': 9, 'revision': 2, 'label': 'QTC-9/v2'}},
    ]
    result = calculate(events)
    first, second = result['events']
    assert first['template']['label'] == '原模板'
    assert second['template']['label'] == 'QTC-9/v2'
    # 同一剖面在换测点后换模板：原模板判成冲击，校准模板判回设备。
    assert first['dominant'] == 'impact'
    assert second['dominant'] == 'equipment'
    assert set(result['sources']) == set(math.CALIBRATION_SOURCES)
    assert 0 <= result['heldout_mixture_mae'] <= 1


def test_decompose_weights_sum_to_one():
    weights = math.decompose(TRUE[2].tolist(), math.as_dict(None))
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert max(weights, key=weights.get) == 'speech'
