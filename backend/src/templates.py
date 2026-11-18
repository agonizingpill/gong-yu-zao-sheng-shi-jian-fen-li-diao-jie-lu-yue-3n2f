"""测点模板的公共数学：默认原模板、归一化、余弦判读与训练批求质心。

这里只处理8频带相对能量，不涉及录音、分贝或身份判断。
"""
from functools import lru_cache
import numpy as np
from .constants import CALIBRATION_SOURCES

# 合成已知声源的原始剖面；顺序与 CALIBRATION_SOURCES 一致。
_RAW_DEFAULT = np.array([
    [9, 7, 2, 0.3, 0.1, 0.1, 0.05, 0.02],   # equipment
    [0.2, 0.5, 1.5, 3, 4, 3, 1, 0.3],       # impact
    [0.05, 0.2, 0.7, 3, 6, 8, 5, 1],        # speech
], dtype=float)


def normalize(values):
    v = np.asarray(values, dtype=float)
    total = float(v.sum())
    if not np.isfinite(v).all() or v.min() < 0 or total <= 0:
        raise ValueError('频带能量须为非负有限值且总能量大于0')
    return v / total


@lru_cache(maxsize=1)
def default_templates():
    """原模板：内置合成剖面，未通过校准的测点一直使用它。"""
    return tuple(tuple(normalize(row).tolist()) for row in _RAW_DEFAULT)


def as_dict(templates):
    """把模板集合转成 source -> 8频带列表；兼容数据库JSON与ndarray。"""
    if templates is None:
        return {source: list(row) for source, row in zip(CALIBRATION_SOURCES, default_templates())}
    if isinstance(templates, dict):
        return {source: normalize(templates[source]).tolist() for source in CALIBRATION_SOURCES}
    return {source: normalize(row).tolist() for source, row in zip(CALIBRATION_SOURCES, templates)}


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def classify(energies, templates):
    """按余弦相似度最高的已知声源模板判读；同分时按固定顺序取先者。"""
    row = normalize(energies)
    best_source = None
    best_score = -2.0
    for source in CALIBRATION_SOURCES:
        score = cosine(row, templates[source])
        if score > best_score:
            best_source, best_score = source, score
    return best_source, best_score


def accuracy(samples, templates):
    """samples: 带 source / energies 的对象或字典列表。返回留出样本判对比例。"""
    if not samples:
        return 0.0
    correct = 0
    for sample in samples:
        source = sample['source'] if isinstance(sample, dict) else sample.source
        energies = sample['energies'] if isinstance(sample, dict) else sample.energies
        predicted, _ = classify(energies, templates)
        correct += int(predicted == source)
    return correct / len(samples)


def derive(samples):
    """由同一次采集的训练批求每个已知声源的归一化质心。"""
    result = {}
    for source in CALIBRATION_SOURCES:
        rows = [
            normalize(s['energies'] if isinstance(s, dict) else s.energies)
            for s in samples
            if (s['source'] if isinstance(s, dict) else s.source) == source
        ]
        if not rows:
            raise ValueError(f'训练批缺少已知声源 {source} 的样本，不能生成候选模板')
        result[source] = np.mean(np.vstack(rows), axis=0)
    return {source: normalize(row).tolist() for source, row in result.items()}


def decompose(energies, templates):
    """固定模板基底上的非负最小二乘权重；再按各基底总能量归一。"""
    from scipy.optimize import nnls
    h = np.vstack([np.asarray(templates[source], dtype=float) for source in CALIBRATION_SOURCES])
    w, _ = nnls(h.T, np.asarray(energies, dtype=float))
    scaled = w * h.sum(axis=1)
    total = float(scaled.sum())
    if total <= 0:
        return {source: 1.0 / len(CALIBRATION_SOURCES) for source in CALIBRATION_SOURCES}
    scaled = scaled / total
    return {source: float(v) for source, v in zip(CALIBRATION_SOURCES, scaled)}


def checksum(samples):
    """留出/训练样本内容的稳定摘要，用于识别更正后结论所依据的数据。"""
    import hashlib, json
    rows = []
    for sample in samples:
        rows.append({
            'source': sample['source'] if isinstance(sample, dict) else sample.source,
            'energies': list(sample['energies'] if isinstance(sample, dict) else sample.energies),
        })
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
