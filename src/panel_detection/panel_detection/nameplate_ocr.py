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
                # 编辑距离合并：如果新文字与已有投票中某个条目相似，合并到该条目
                merged_key = self._find_similar_vote(target_id, text)
                self._votes[target_id][merged_key] += 1
                top_text, count = self._votes[target_id].most_common(1)[0]
                if count >= self.confirm_frames:
                    self._confirmed[target_id] = top_text

        return dict(self._confirmed)

    def _find_similar_vote(self, target_id: int, text: str) -> str:
        """
        在已有投票中找编辑距离最近的条目，距离 <= 阈值时合并。

        策略：较长的文字更可能是完整识别结果，合并时保留更长的那个。
        """
        votes = self._votes.get(target_id)
        if not votes:
            return text

        best_match = None
        best_dist = float('inf')
        threshold = max(1, len(text) // 3)  # 允许 1/3 字符差异

        for existing in votes:
            dist = self._edit_distance(text, existing)
            if dist < best_dist:
                best_dist = dist
                best_match = existing

        if best_match is not None and best_dist <= threshold and best_dist > 0:
            # 保留更长的文字作为 key（更可能是完整结果）
            return text if len(text) > len(best_match) else best_match

        return text

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """计算两个字符串的编辑距离（Levenshtein）"""
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

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
    def _locate_nameplate_in_roi(roi: np.ndarray) -> Optional[np.ndarray]:
        """
        在粗 ROI 内精定位铭牌金属板区域。

        铭牌特征：灰色面板背景上的浅色/银色矩形小板，宽高比 2:1~6:1。
        通过自适应阈值 + 轮廓检测找到最可能的铭牌矩形并裁剪。
        """
        h, w = roi.shape[:2]
        if h < 10 or w < 10:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 自适应阈值：铭牌在背景上通常偏亮（银色金属）
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, -8)

        # 形态学闭合填充文字间隙
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_plate = None
        best_area = 0
        min_area = h * w * 0.05
        max_area = h * w * 0.85

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < 1 or rh < 1:
                continue
            # 确保 rw > rh
            if rw < rh:
                rw, rh = rh, rw
            aspect = rw / rh
            # 铭牌宽高比约 2~7
            if not (1.5 <= aspect <= 8.0):
                continue
            if area > best_area:
                best_area = area
                x, y, bw, bh = cv2.boundingRect(cnt)
                best_plate = roi[y:y+bh, x:x+bw]

        return best_plate

    @staticmethod
    def _upscale(roi: np.ndarray, min_height: int = 48) -> np.ndarray:
        """放大图像使短边不小于 min_height"""
        h, w = roi.shape[:2]
        if h < min_height:
            scale = min_height / h
            roi = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        return roi

    @staticmethod
    def _preprocess_clahe(roi: np.ndarray) -> np.ndarray:
        """CLAHE 对比度增强"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _preprocess_sharpen(roi: np.ndarray) -> np.ndarray:
        """锐化 + 对比拉伸"""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        sharpened = cv2.filter2D(roi, -1, kernel)
        # 对比拉伸
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _preprocess_binary(roi: np.ndarray) -> np.ndarray:
        """自适应二值化（适合刻字铭牌）"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _ocr_single(self, image: np.ndarray) -> Tuple[str, float]:
        """对单张图执行 OCR，返回 (合并文字, 平均置信度)"""
        try:
            if self._ocr_backend == 'rapidocr':
                result, _ = self._ocr(image)
                if not result:
                    return ('', 0.0)
                texts = []
                confs = []
                for item in result:
                    box, text, conf = item
                    if conf >= 0.4:
                        texts.append(text)
                        confs.append(conf)
            else:
                results = self._ocr.predict(image)
                if not results:
                    return ('', 0.0)
                texts = []
                confs = []
                for r in results:
                    if hasattr(r, 'rec_text'):
                        if r.rec_score >= 0.4:
                            texts.append(r.rec_text)
                            confs.append(r.rec_score)
                    elif isinstance(r, dict):
                        text = r.get('text', r.get('rec_text', ''))
                        conf = r.get('confidence', r.get('rec_score', 0))
                        if conf >= 0.4 and text:
                            texts.append(text)
                            confs.append(conf)
        except Exception:
            return ('', 0.0)

        if not texts:
            return ('', 0.0)

        merged = ''.join(texts).strip()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return (merged, avg_conf)

    def _recognize_text(self, roi: np.ndarray) -> Optional[str]:
        """
        对 ROI 执行多策略 OCR，取置信度最高的结果。

        流程：
          1. 尝试在 ROI 内精定位铭牌矩形
          2. 对定位后的图像尝试多种预处理
          3. 取置信度最高且长度 >= 2 的结果
        """
        if self._ocr is None:
            return None

        # 精定位铭牌区域
        plate = self._locate_nameplate_in_roi(roi)
        # 用精定位结果和原始 ROI 都尝试
        candidates = []
        if plate is not None and plate.size > 0:
            candidates.append(plate)
        candidates.append(roi)

        best_text = ''
        best_conf = 0.0

        for img in candidates:
            img = self._upscale(img, min_height=48)
            # 多种预处理策略
            variants = [
                img,                             # 原图（放大后）
                self._preprocess_clahe(img),     # CLAHE
                self._preprocess_sharpen(img),   # 锐化
                self._preprocess_binary(img),    # 二值化
            ]
            for variant in variants:
                text, conf = self._ocr_single(variant)
                if len(text) >= 2 and conf > best_conf:
                    best_text = text
                    best_conf = conf

        if len(best_text) < 2:
            return None
        # 铭牌应以中文为主，过滤表盘数字/英文噪声
        cjk_count = sum(1 for c in best_text if '一' <= c <= '鿿')
        if cjk_count < 2 or cjk_count < len(best_text) * 0.5:
            return None
        return best_text

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
