"""
铭牌文字识别模块

通过规则方法定位器件上方的铭牌区域，使用 PaddleOCR 识别文字，
并将识别结果与下方的 button/knob 关联。

设计思路：
  - 铭牌是静态标签，注册阶段多帧投票确认后缓存，不需要每帧跑 OCR
  - 对每个已编号的 button/knob，向其上方裁剪 ROI 送入 OCR
  - 多帧投票后输出稳定的 label

用法:
    from .nameplate_ocr import NameplateRecognizer
    recognizer = NameplateRecognizer()
    recognizer.update(matched, color_image)
    label = recognizer.get_label(target_id)
"""
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class NameplateRecognizer:
    """铭牌文字识别器：规则定位 + OCR + 多帧投票"""

    def __init__(
        self,
        confirm_frames: int = 5,
        ocr_interval: int = 10,
        roi_above_ratio: float = 1.8,
        roi_gap_ratio: float = 0.2,
        roi_width_ratio: float = 2.5,
        roi_min_height: int = 20,
        use_gpu: bool = False,
        lang: str = 'ch',
    ):
        """
        Args:
            confirm_frames: 同一文字出现多少帧后确认
            ocr_interval: 每隔多少帧执行一次 OCR（注册阶段）
            roi_above_ratio: 铭牌 ROI 高度 = 器件高度 × ratio
            roi_gap_ratio: 铭牌与器件之间的间隙 = 器件高度 × ratio
            roi_width_ratio: 铭牌 ROI 宽度 = 器件宽度 × ratio
            roi_min_height: ROI 最小高度（像素），太小不做 OCR
            use_gpu: PaddleOCR 是否使用 GPU
            lang: OCR 语言
        """
        self.confirm_frames = confirm_frames
        self.ocr_interval = ocr_interval
        self.roi_above_ratio = roi_above_ratio
        self.roi_gap_ratio = roi_gap_ratio
        self.roi_width_ratio = roi_width_ratio
        self.roi_min_height = roi_min_height

        self._ocr = None
        self._ocr_backend = None
        self._ocr_init_args = {'use_gpu': use_gpu, 'lang': lang}
        self._ocr_lock = threading.Lock()

        # 投票与确认状态
        self._votes: Dict[int, Counter] = defaultdict(Counter)
        self._confirmed: Dict[int, str] = {}

        self._frame_count = 0

    def _ensure_ocr(self):
        """懒加载 OCR 引擎，首次调用时初始化。优先使用 RapidOCR (ONNX Runtime)。"""
        if self._ocr is not None:
            return True
        with self._ocr_lock:
            if self._ocr is not None:
                return True
            # 优先使用 RapidOCR（基于 ONNX Runtime，aarch64 兼容）
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
                self._ocr_backend = 'rapidocr'
                return True
            except ImportError:
                pass
            # 回退到 PaddleOCR
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang=self._ocr_init_args['lang'],
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    show_log=False,
                )
                self._ocr_backend = 'paddleocr'
                return True
            except (ImportError, Exception):
                pass
            return False

    @property
    def all_confirmed(self) -> bool:
        """是否所有已知 target 都有确认的 label（外部可据此停止 OCR）"""
        return len(self._votes) > 0 and all(
            tid in self._confirmed for tid in self._votes
        )

    def get_label(self, target_id: int) -> Optional[str]:
        """获取已确认的铭牌文字"""
        return self._confirmed.get(target_id)

    @property
    def all_labels(self) -> Dict[int, str]:
        """获取所有已确认的 {target_id: label} 映射"""
        return dict(self._confirmed)

    def update(
        self,
        matched: List[Tuple[int, object]],
        color_image: np.ndarray,
    ) -> Dict[int, str]:
        """
        每帧调用，在注册阶段定期做 OCR 并投票。

        Args:
            matched: [(target_id, FrameDetection), ...] 已编号的器件列表
            color_image: 当前帧彩色图 (BGR)

        Returns:
            当前已确认的 {target_id: label} 映射
        """
        self._frame_count += 1

        # 所有 label 已确认，不再做 OCR
        if self.all_confirmed and self._confirmed:
            return dict(self._confirmed)

        # 按间隔执行 OCR
        if self._frame_count % self.ocr_interval != 0:
            return dict(self._confirmed)

        if not self._ensure_ocr():
            return dict(self._confirmed)

        for target_id, det in matched:
            if target_id in self._confirmed:
                continue
            if det.class_name not in ('button', 'knob'):
                continue

            roi = self._extract_nameplate_roi(det, color_image)
            if roi is None:
                continue

            text = self._recognize_text(roi)
            if text:
                self._votes[target_id][text] += 1
                top_text, count = self._votes[target_id].most_common(1)[0]
                if count >= self.confirm_frames:
                    self._confirmed[target_id] = top_text

        return dict(self._confirmed)

    def _extract_nameplate_roi(self, det, color_image: np.ndarray) -> Optional[np.ndarray]:
        """
        在器件上方裁剪铭牌 ROI。

        铭牌通常位于 button/knob 正上方，是一块金属小板。
        """
        img_h, img_w = color_image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in det.bbox]
        det_w = x2 - x1
        det_h = y2 - y1

        if det_w < 5 or det_h < 5:
            return None

        cx = (x1 + x2) / 2.0

        # 铭牌 ROI 尺寸
        roi_w = det_w * self.roi_width_ratio
        roi_h = det_h * self.roi_above_ratio
        gap = det_h * self.roi_gap_ratio

        # 铭牌在器件上方
        roi_x1 = int(round(cx - roi_w / 2))
        roi_x2 = int(round(cx + roi_w / 2))
        roi_y2 = int(round(y1 - gap))
        roi_y1 = int(round(roi_y2 - roi_h))

        # 边界裁剪
        roi_x1 = max(0, roi_x1)
        roi_x2 = min(img_w, roi_x2)
        roi_y1 = max(0, roi_y1)
        roi_y2 = min(img_h, roi_y2)

        if roi_x2 - roi_x1 < 10 or roi_y2 - roi_y1 < self.roi_min_height:
            return None

        return color_image[roi_y1:roi_y2, roi_x1:roi_x2].copy()

    @staticmethod
    def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
        """对铭牌 ROI 做预处理以提升 OCR 识别率"""
        h, w = roi.shape[:2]
        # 放大至最短边不小于 64px，提升小文字辨识度
        min_side = 64
        if min(h, w) < min_side:
            scale = min_side / min(h, w)
            roi = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        # 转灰度 → CLAHE 增强对比 → 转回 BGR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _recognize_text(self, roi: np.ndarray) -> Optional[str]:
        """对 ROI 执行 OCR 并返回识别文字"""
        if self._ocr is None:
            return None
        roi = self._preprocess_roi(roi)

        try:
            if self._ocr_backend == 'rapidocr':
                result, _ = self._ocr(roi)
                if not result:
                    return None
                # RapidOCR 返回 [(box, text, conf), ...]
                texts = []
                for item in result:
                    box, text, conf = item
                    if conf >= 0.5:
                        texts.append(text)
            else:
                # PaddleOCR 3.7 predict API
                results = self._ocr.predict(roi)
                if not results:
                    return None
                texts = []
                for r in results:
                    if hasattr(r, 'rec_text'):
                        if r.rec_score >= 0.5:
                            texts.append(r.rec_text)
                    elif isinstance(r, dict):
                        text = r.get('text', r.get('rec_text', ''))
                        conf = r.get('confidence', r.get('rec_score', 0))
                        if conf >= 0.5 and text:
                            texts.append(text)
        except Exception:
            return None

        if not texts:
            return None

        # 合并为单行文字，去除多余空白
        merged = ''.join(texts).strip()
        # 过滤掉纯数字/单字符等噪声结果
        if len(merged) < 2:
            return None

        return merged

    def reset(self):
        """重置所有状态（如需重新识别）"""
        self._votes.clear()
        self._confirmed.clear()
        self._frame_count = 0

    def draw_debug(self, canvas: np.ndarray, matched: List[Tuple[int, object]]):
        """在画面上绘制铭牌 ROI 区域和识别结果（调试用）"""
        for target_id, det in matched:
            if det.class_name not in ('button', 'knob'):
                continue

            img_h, img_w = canvas.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in det.bbox]
            det_w = x2 - x1
            det_h = y2 - y1
            cx = (x1 + x2) / 2.0

            roi_w = det_w * self.roi_width_ratio
            roi_h = det_h * self.roi_above_ratio
            gap = det_h * self.roi_gap_ratio

            roi_x1 = int(round(max(0, cx - roi_w / 2)))
            roi_x2 = int(round(min(img_w, cx + roi_w / 2)))
            roi_y2 = int(round(min(img_h, y1 - gap)))
            roi_y1 = int(round(max(0, roi_y2 - roi_h)))

            # 绘制 ROI 框
            label = self._confirmed.get(target_id)
            if label:
                color = (0, 255, 128)  # 绿色 = 已确认
                cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2, roi_y2),
                              color, 2, cv2.LINE_AA)
                # 文字放在 ROI 上方
                text_y = max(15, roi_y1 - 5)
                cv2.putText(canvas, label, (roi_x1, text_y),
                            0, 0.5, color, 1, cv2.LINE_AA)
            else:
                color = (0, 200, 255)  # 橙色 = 待确认
                cv2.rectangle(canvas, (roi_x1, roi_y1), (roi_x2, roi_y2),
                              color, 1, cv2.LINE_AA)
                # 显示当前投票状态
                votes = self._votes.get(target_id)
                if votes:
                    top_text, count = votes.most_common(1)[0]
                    text_y = max(15, roi_y1 - 5)
                    cv2.putText(canvas, f'{top_text}({count}/{self.confirm_frames})',
                                (roi_x1, text_y), 0, 0.4, color, 1, cv2.LINE_AA)
