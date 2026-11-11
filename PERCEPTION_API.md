# 面板识别接口 v2

识别节点面向相机和面板，不绑定手臂。执行器负责选择目标、坐标变换、可达性、碰撞检查和动作授权。

## 编号

- ROS1/ROS2 统一 `id = tag_id`；T00→0、T02→2、T03→3、T07→7。
- 编号与相机、话题前缀及执行器无关。局部视野不从0重新编号。
- 无有效Tag关联时保留目标，`id=null`、`tag_id=null`、`identity_valid=false`。旧布局推断不再作为公开编号。
- 短时跟踪保留身份时，`identity_source=apriltag_track`；直接关联为Tag来源。跟踪过期后清空编号。
- 私有负数匹配键仅在单帧内部使用，不发布，也不进入跨帧位置/角度滤波。
- `registry_target_ids` / `allowed_ids` 已弃用并忽略；旧launch传入 `[5,6]` 不会限制识别结果。下游必须自行选择执行目标。
- 以上是面板编号规则；螺栓/螺母分组及阀门任务的独立编号不作替换。

## 消息

```json
{
  "schema_version": 2,
  "stamp": 1792936800.123,
  "camera_id": "camera_a",
  "frame_id": "camera_a_color_optical_frame",
  "targets": [
    {
      "id": 0,
      "tag_id": 0,
      "identity_valid": true,
      "identity_source": "apriltag_36h11",
      "class": "button",
      "confidence": 0.94,
      "position": {"x": 0.1, "y": 0.02, "z": 0.5},
      "position_valid": true,
      "measurement_ready": true
    }
  ]
}
```

位置单位米，属于顶层 `frame_id`；同一目标在两台相机下坐标可以不同，编号必须相同。
`camera_id` 用于来源标识，默认回退到 `frame_id`。ROS1可设置私有参数 `camera_id`；ROS2可在节点配置中设置 `camera_id`。
不同相机应使用各自的光学坐标系；下游使用各自标定/TF转换。保留按相机划分的话题，不赋予话题任何手臂权限含义。

识别到目标但深度无效时仍保留类别、编号和bbox，`position=null`、`position_valid=false`、`measurement_ready=false`；不会回填旧坐标。
`measurement_ready` 仅表示当前测量通过已有有效性检查。板卡同时需要已确认状态及完整几何；空槽也可以是有效测量。
该字段不表示可抓取、允许执行或达到毫米级绝对精度。公开面板消息不再发布 `graspable` / `operation_ready`；内部旧适配字段在发布边界去除。
旧 `/panel/knob_angles` 数值规则保持，编号改为原始Tag ID；ROS2 `/panel/knob_observations` 同时包含无编号的视觉角度。

## 迁移

1. 下游移除Tag加1/减1转换，直接使用原始Tag ID。特别处理ID0，判断 `id is not None`，不要使用 `if id`。
2. 下游按任务清单筛选目标。左右执行器的分工不再通过识别节点的编号范围配置。
3. 用 `position_valid` / `measurement_ready` 判断测量可用性；动作授权留在控制层。
4. 保留传感器时间戳、相机来源和frame_id，禁止把不同光学坐标系的位置直接混用。
5. 旧v4精度包是独立交付，本分支基于仓库main的48188ea，仅修改编号与识别接口；没有合入v4精度算法。

## 本地验证范围

两个平台分别运行编号、Tag、角度、状态及接口测试，并用实际回调代码的AST与替代传输验证公开消息。
ROS1独立仓库不包含visual_servo/runtime_workflow，相关CPU集成测试不适用于此检出；旧ROS跨平台整节点相同测试已与当前双适配器结构不符。测试报告单列这些未执行项。
本机两个bag的406张真实RGB、已有1020模型与两端编号通路用于回放；不把该编号验证当作位置精度验证。
