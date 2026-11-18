# 001 初始领域结构

由 `src/models.py` SQLAlchemy metadata 建立领域/支撑表，首次启动执行create_all。投诉是所有同卷写入的锁根；协议动作先锁投诉再读取协议。MySQL显式READ COMMITTED，防止锁等待后读取旧聚合快照。事件同卷/同地点/同时间唯一；签字同协议/同人唯一；复验事件和复发事件一对一约束。

测点校准部分：`measurement_points` 以楼栋+测点为自然键，是模板审批的锁根并指向当前生效候选；`calibration_batches` 以楼栋+测点+采集时刻唯一，整批样本（`calibration_samples`）只登记训练或留出一种用途；`template_candidates` 保存候选模板、训练/留出批次、留出判对率、比较基线与留出摘要。`noise_events` 新增 `template_origin/template_id/template_revision/predicted_source/predicted_score`，记录判读当时的模板快照。

本小型示例不附自动升级迁移器；修改列定义时需编写显式迁移，不能对已有库依赖create_all修改。已有库升级到含测点校准的结构需给 `noise_events` 增加上述5列（默认值见模型）并新建4张校准表。
