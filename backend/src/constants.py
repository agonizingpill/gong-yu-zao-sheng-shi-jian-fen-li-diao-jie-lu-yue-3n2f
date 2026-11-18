ROLES=('operator','supervisor','viewer')
CASE_STATES=('monitoring','mediation','remedy','observation','closed')
SOURCE_TYPES=('equipment','impact','speech','undetermined')
SOURCE_LABELS={'equipment':'低频设备','impact':'冲击声','speech':'人声频带','undetermined':'待核实'}
BANDS=('63Hz','125Hz','250Hz','500Hz','1kHz','2kHz','4kHz','8kHz')
# 校准只接受已知声源；undetermined 不能作为校准标签。
CALIBRATION_SOURCES=('equipment','impact','speech')
TEMPLATE_STATES=('evaluated','active','superseded','invalidated')
