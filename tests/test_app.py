import importlib
import importlib.util
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class FakeImage:
    def __init__(self, width=120, height=80):
        self.shape = (height, width, 3)

    def copy(self):
        return FakeImage(self.shape[1], self.shape[0])


def purge_modules(*prefixes: str) -> None:
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(module_name, None)


class MainCliTests(unittest.TestCase):
    def tearDown(self) -> None:
        purge_modules("cmd.main", "usecase", "usecase.pipe")

    def test_main_prompts_for_directory_when_argument_is_missing(self) -> None:
        fake_pipe = types.ModuleType("usecase.pipe")
        captured = {}

        def fake_run_pipeline(input_dir):
            captured["input_dir"] = str(input_dir)
            return Path("C:/tmp/results")

        fake_pipe.run_pipeline = fake_run_pipeline
        sys.modules["usecase.pipe"] = fake_pipe

        module_path = Path(__file__).resolve().parents[1] / "cmd" / "main.py"
        spec = importlib.util.spec_from_file_location("test_cmd_main", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        stdout = io.StringIO()

        with patch.object(sys, "argv", ["main.py"]), patch("builtins.input", return_value="C:/images"), redirect_stdout(stdout):
            exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(Path(captured["input_dir"]), Path("C:/images"))
        output = stdout.getvalue()
        self.assertIn("Укажите путь к папке", output)
        self.assertIn("Результаты сохранены", output)


class PipelineSmokeTests(unittest.TestCase):
    def tearDown(self) -> None:
        purge_modules("usecase", "usecase.pipe", "pkg", "pkg.metadata", "pkg.teeth_classifier", "pkg.teeth_detector", "cv2")

    def test_pipeline_creates_result_images_without_json_files(self) -> None:
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.COLOR_BGR2RGB = 1
        fake_cv2.FONT_HERSHEY_SIMPLEX = 0
        fake_cv2.LINE_AA = 16
        fake_cv2.imread = lambda _: FakeImage()
        fake_cv2.cvtColor = lambda image, _: image
        fake_cv2.rectangle = lambda *args, **kwargs: None
        fake_cv2.putText = lambda *args, **kwargs: None

        def fake_imwrite(path, _image):
            Path(path).write_bytes(b"fake-image")
            return True

        fake_cv2.imwrite = fake_imwrite
        sys.modules["cv2"] = fake_cv2

        fake_metadata = types.ModuleType("pkg.metadata")
        fake_metadata.compute_metadata = lambda bboxes, image_width: [{"quadrant": 1}] * len(bboxes)
        sys.modules["pkg.metadata"] = fake_metadata

        fake_classifier = types.ModuleType("pkg.teeth_classifier")
        fake_classifier.load_classifier = lambda: ("model", "scaler", "ohe", "encoder")
        fake_classifier.predict_teeth = lambda metadata, *_: [11 + index for index, _ in enumerate(metadata)]
        sys.modules["pkg.teeth_classifier"] = fake_classifier

        fake_detector = types.ModuleType("pkg.teeth_detector")
        fake_detector.detect_teeth = lambda image: [[10, 12, 50, 60, 0.98]]
        sys.modules["pkg.teeth_detector"] = fake_detector

        module = importlib.import_module("usecase.pipe")

        with TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            (source_dir / "scan1.jpg").write_bytes(b"raw")

            results_dir = module.run_pipeline(source_dir)

            self.assertTrue((results_dir / "scan1_result.jpg").exists())
            self.assertFalse(any(results_dir.glob("*.json")))


if __name__ == "__main__":
    unittest.main()
