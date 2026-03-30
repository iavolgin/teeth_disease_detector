from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from pkg.metadata import compute_metadata
from pkg.teeth_classifier import load_classifier, predict_teeth
from pkg.teeth_detector import detect_teeth


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageProcessingResult:
    image_name: str
    image_path: str
    detections_count: int
    result_image_path: str


class TeethPipeline:
    def __init__(self) -> None:
        try:
            self._model, self._scaler, self._ohe, self._label_encoder = load_classifier()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Не удалось загрузить веса классификатора. "
                "Проверьте наличие файла model.pkl в pkg/weights/classificator."
            ) from exc

    def run(self, input_dir: str | Path) -> Path:
        source_dir = Path(input_dir).expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Папка не найдена: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Ожидалась папка с изображениями: {source_dir}")

        image_paths = self._collect_images(source_dir)
        if not image_paths:
            raise ValueError(
                f"В папке {source_dir} не найдено изображений. "
                f"Поддерживаются: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

        results_dir = source_dir / "results"
        results_dir.mkdir(exist_ok=True)

        for image_path in image_paths:
            self._process_image(image_path, results_dir)

        return results_dir

    def _collect_images(self, source_dir: Path) -> list[Path]:
        return sorted(
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    def _process_image(self, image_path: Path, results_dir: Path) -> ImageProcessingResult:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Не удалось прочитать изображение: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = detect_teeth(image_rgb)

        bboxes = [bbox[:4] for bbox in detections]
        metadata = compute_metadata(bboxes, image_rgb.shape[1])
        labels = predict_teeth(
            metadata,
            self._model,
            self._scaler,
            self._ohe,
            self._label_encoder,
        )

        annotated = image_bgr.copy()

        for index, detection in enumerate(detections):
            x1, y1, x2, y2, confidence = detection
            label = labels[index] if index < len(labels) else None

            self._draw_detection(
                annotated,
                (int(x1), int(y1), int(x2), int(y2)),
                label=label,
            )

        result_image_path = results_dir / f"{image_path.stem}_result{image_path.suffix}"

        if not cv2.imwrite(str(result_image_path), annotated):
            raise IOError(f"Не удалось сохранить результат: {result_image_path}")

        return ImageProcessingResult(
            image_name=image_path.name,
            image_path=str(image_path),
            detections_count=len(detections),
            result_image_path=str(result_image_path),
        )

    def _draw_detection(
        self,
        image,
        bbox: tuple[int, int, int, int],
        label: Any,
    ) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 180, 40), 1)

        if label is None:
            return

        text = str(label)

        text_origin_y = y1 - 10 if y1 > 25 else y1 + 20
        cv2.putText(
            image,
            text,
            (x1, text_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 180, 40),
            1,
            cv2.LINE_AA,
        )


def run_pipeline(input_dir: str | Path) -> Path:
    pipeline = TeethPipeline()
    return pipeline.run(input_dir)
