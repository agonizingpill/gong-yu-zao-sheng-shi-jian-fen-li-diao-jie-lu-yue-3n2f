"""NMF只分解合成频带能量，不识别人、不提供分贝/声源物证结论。

训练/留出指标仍来自固定合成样本；实际事件按各自测点当前生效模板（未校准则为原模板）
作非负分解，并逐条记录所用模板，便于在分离页面追溯。
"""
import hashlib,json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from .constants import CALIBRATION_SOURCES
from . import templates as math
SEED=1040

def _synthetic():
    rng=np.random.default_rng(SEED)
    templates=np.array([[9,7,2,0.3,0.1,0.1,0.05,0.02],[0.2,0.5,1.5,3,4,3,1,0.3],[0.05,0.2,0.7,3,6,8,5,1]],dtype=float)
    templates/=templates.sum(axis=1,keepdims=True)
    weights=rng.gamma(.7,20,size=(360,3));weights[:30]=np.eye(3)[np.arange(30)%3]*40+0.001
    x=weights@templates+rng.uniform(0,.015,size=(360,8))
    return templates,weights,x

def calculate(events):
    templates,weights,x=_synthetic()
    # 原始样本先划分，fit仅触达前280行；后80行是固定留出集。
    train=x[:280];test=x[280:];truth=weights[280:]
    model=NMF(n_components=3,init='nndsvda',random_state=SEED,max_iter=2500,tol=1e-5)
    w=model.fit_transform(train);h=model.components_
    normal=h/h.sum(axis=1,keepdims=True)
    cosine=normal@templates.T/(np.linalg.norm(normal,axis=1)[:,None]*np.linalg.norm(templates,axis=1)[None,:])
    a,b=linear_sum_assignment(-cosine);order=[int(a[list(b).index(i)]) for i in range(3)]
    wt=model.transform(test);pred=wt*h.sum(axis=1)[None,:];pred=pred[:,order]
    pred/=pred.sum(axis=1,keepdims=True);truth/=truth.sum(axis=1,keepdims=True)
    reconstruction=wt@h
    out=[]
    for e in events:
        basis=math.as_dict(e.get('templates'))
        weights_e=math.decompose(e['energies'],basis)
        dominant=max(CALIBRATION_SOURCES,key=lambda s:weights_e[s])
        out.append({
            'event_id':e['id'],
            'weights':weights_e,
            'dominant':dominant,
            'template':e.get('template') or {'origin':'default','id':None,'revision':0,'label':'原模板'},
        })
    return {'seed':SEED,'input_checksum':hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),'training_checksum':hashlib.sha256(x.tobytes()).hexdigest(),'training_rows':280,'heldout_rows':80,'heldout_relative_rmse':float(np.linalg.norm(test-reconstruction)/np.linalg.norm(test)),'heldout_mixture_mae':float(np.mean(np.abs(pred-truth))),'train_reconstruction':float(model.reconstruction_err_),'iterations':int(model.n_iter_),'bands':['63Hz','125Hz','250Hz','500Hz','1kHz','2kHz','4kHz','8kHz'],'templates':normal[order].tolist(),'sources':list(CALIBRATION_SOURCES),'events':out,'template_note':'逐事件按所在测点当前生效模板分解；未通过校准的测点沿用原模板。','limitation':'固定合成样本仅验证分解链路；频带能量不能证明真实声源或身份，须人工现场证据复核。'}
