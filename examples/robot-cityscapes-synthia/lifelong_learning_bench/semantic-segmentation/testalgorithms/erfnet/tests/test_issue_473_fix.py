"""
Regression tests for issue #473 (kubeedge/ianvs).

These tests exercise the ACTUAL committed fix in:
  - task_allocation_by_domain.py   (Bug 1)
  - ERFNet/dataloaders/datasets/cityscapes.py   (Bugs 2 & 3)

Heavy ML dependencies (torch, torchvision, sedna) are stubbed out with
lightweight fakes so the test runs in isolation without needing the full
Sedna/PyTorch environment installed -- it only exercises our own logic,
not sedna's or torch's internals.

Run with:  python -m pytest test_issue_473_fix.py -v
"""
import sys
import types
import os
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy / external modules BEFORE importing the real fixed source files
# ---------------------------------------------------------------------------

# --- stub torch / torch.utils.data ---
torch_mod = types.ModuleType("torch")
torch_utils_mod = types.ModuleType("torch.utils")


class _FakeDataset:
    """Stand-in for torch.utils.data.Dataset (only used as a base class)."""
    pass


torch_utils_data_mod = types.ModuleType("torch.utils.data")
torch_utils_data_mod.Dataset = _FakeDataset
torch_utils_mod.data = torch_utils_data_mod
torch_mod.utils = torch_utils_mod
sys.modules["torch"] = torch_mod
sys.modules["torch.utils"] = torch_utils_mod
sys.modules["torch.utils.data"] = torch_utils_data_mod

# --- stub torchvision.transforms ---
torchvision_mod = types.ModuleType("torchvision")
transforms_mod = types.ModuleType("torchvision.transforms")
transforms_mod.Compose = lambda transforms_list: (lambda sample: sample)
torchvision_mod.transforms = transforms_mod
sys.modules["torchvision"] = torchvision_mod
sys.modules["torchvision.transforms"] = transforms_mod

# --- stub mypath.Path ---
mypath_mod = types.ModuleType("mypath")


class _FakePath:
    @staticmethod
    def db_root_dir(dataset_name):
        return "/fake/dataset/root"


mypath_mod.Path = _FakePath
sys.modules["mypath"] = mypath_mod

# --- stub dataloaders.custom_transforms (only referenced, never called
#     in the __init__ paths we're testing) ---
custom_transforms_mod = types.ModuleType("dataloaders.custom_transforms")
for _name in ("CropBlackArea", "RandomHorizontalFlip", "RandomScaleCrop",
              "Normalize", "ToTensor"):
    setattr(custom_transforms_mod, _name, lambda *a, **k: (lambda x: x))
dataloaders_pkg = types.ModuleType("dataloaders")
dataloaders_pkg.custom_transforms = custom_transforms_mod
sys.modules["dataloaders"] = dataloaders_pkg
sys.modules["dataloaders.custom_transforms"] = custom_transforms_mod

# --- stub sedna.datasources.BaseDataSource ---
sedna_mod = types.ModuleType("sedna")
sedna_datasources_mod = types.ModuleType("sedna.datasources")


class BaseDataSource:
    """Minimal stand-in for Sedna's real BaseDataSource."""
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


sedna_datasources_mod.BaseDataSource = BaseDataSource
sedna_mod.datasources = sedna_datasources_mod

# --- stub sedna.common.class_factory.ClassFactory / ClassType ---
sedna_common_mod = types.ModuleType("sedna.common")
class_factory_mod = types.ModuleType("sedna.common.class_factory")


class ClassType:
    STP = "STP"


class ClassFactory:
    @staticmethod
    def register(class_type, alias=None):
        def decorator(cls):
            return cls
        return decorator


class_factory_mod.ClassFactory = ClassFactory
class_factory_mod.ClassType = ClassType
sedna_common_mod.class_factory = class_factory_mod
sedna_mod.common = sedna_common_mod

sys.modules["sedna"] = sedna_mod
sys.modules["sedna.datasources"] = sedna_datasources_mod
sys.modules["sedna.common"] = sedna_common_mod
sys.modules["sedna.common.class_factory"] = class_factory_mod

# ---------------------------------------------------------------------------
# Now import the REAL fixed source files
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ERFNet"))
sys.path.insert(0, os.path.join(HERE, "..", "ERFNet", "dataloaders", "datasets"))

from task_allocation_by_domain import TaskAllocationByOrigin  # noqa: E402
from cityscapes import CityscapesSegmentation  # noqa: E402


# ---------------------------------------------------------------------------
# Bug 1: TaskAllocationByOrigin call signature must match Sedna's convention
# ---------------------------------------------------------------------------
class TestBug1TaskAllocationCallingConvention:

    def test_sedna_style_construction_and_call_does_not_raise(self):
        """
        Reproduces Sedna's exact usage pattern:
            method_cls = TaskAllocationByOrigin(task_extractor=..., **extend_param)
            mappings = method_cls(samples=samples)
        Before the fix this raised:
            TypeError: __call__() missing 1 required positional argument: 'task_extractor'
        """
        extractor = {"Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2}
        method_cls = TaskAllocationByOrigin(task_extractor=extractor, default=None)

        samples = BaseDataSource(x=[["some/path/Cityscapes/img1.png"],
                                     ["some/path/Synthia/img2.png"]])

        # This is the exact call Sedna makes -- samples only, no task_extractor.
        result_samples, allocations = method_cls(samples=samples)

        assert result_samples is samples
        assert allocations == [1, 0]  # Cityscapes -> 1, Synthia -> 0

    def test_falls_back_to_default_mapping_when_no_extractor_given(self):
        """If task_extractor isn't provided, the hardcoded default mapping is used."""
        method_cls = TaskAllocationByOrigin()
        assert method_cls.task_extractor == {
            "Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2
        }

    def test_default_origin_short_circuit_path(self):
        """When a default_origin is configured, every sample gets that allocation."""
        method_cls = TaskAllocationByOrigin(
            task_extractor={"Synthia": 0, "Cityscapes": 1, "Cloud-Robotics": 2},
            default="Synthia",
        )
        samples = BaseDataSource(x=[["a"], ["b"], ["c"]])
        _, allocations = method_cls(samples=samples)
        assert allocations == [0, 0, 0]


# ---------------------------------------------------------------------------
# Helper to build a minimal args namespace CityscapesSegmentation needs
# ---------------------------------------------------------------------------
class _FakeArgs:
    base_size = 513
    crop_size = 513


# ---------------------------------------------------------------------------
# Bug 2: data.y is None during inference must not crash with
#        "TypeError: object of type 'NoneType' has no len()"
# ---------------------------------------------------------------------------
class TestBug2NoneLabelsDuringInference:

    def test_init_does_not_crash_when_labels_are_none(self):
        data = BaseDataSource(
            x=[["img1.png", "depth1.png"], ["img2.png", "depth2.png"]],
            y=None,  # <-- this is what crashed before the fix
        )
        ds = CityscapesSegmentation(_FakeArgs(), root="/fake/root", data=data,
                                     split="test")
        assert ds.labels["test"] is None
        assert len(ds.images["test"]) == 2
        assert len(ds.disparities["test"]) == 2

    def test_getitem_falls_back_to_image_as_label_when_none(self, monkeypatch):
        """
        __getitem__ should use the image itself as a placeholder label
        instead of crashing when there are no ground-truth labels.
        """
        data = BaseDataSource(
            x=[["img1.png", "depth1.png"]],
            y=None,
        )
        ds = CityscapesSegmentation(_FakeArgs(), root="/fake/root", data=data,
                                     split="test")

        fake_image = object()

        class _FakeImage:
            @staticmethod
            def open(path):
                return _FakeImageHandle()

        class _FakeImageHandle:
            def convert(self, mode):
                return fake_image

        import cityscapes as cityscapes_module
        monkeypatch.setattr(cityscapes_module, "Image", _FakeImage)

        ds.transform_ts = lambda s: s  # bypass real transform pipeline
        result, path = ds[0]  # split is "test" -> uses transform_ts internally
        assert result["label"] is fake_image  # fell back to the image, no crash


# ---------------------------------------------------------------------------
# Bug 3: data.y as a multi-element numpy array must not crash with
#        "ValueError: truth value of an array ... is ambiguous"
# ---------------------------------------------------------------------------
class TestBug3NumpyArrayLabels:

    def test_init_does_not_crash_with_numpy_array_labels(self):
        data = BaseDataSource(
            x=[["img1.png", "depth1.png"], ["img2.png", "depth2.png"]],
            y=np.array(["label1.png", "label2.png"]),  # multi-element ndarray
        )
        # Before the fix, truthiness checks on this array raised:
        #   ValueError: The truth value of an array with more than one
        #   element is ambiguous. Use a.any() or a.all()
        ds = CityscapesSegmentation(_FakeArgs(), root="/fake/root", data=data,
                                     split="train")
        assert ds.labels["train"] == ["label1.png", "label2.png"]

    def test_init_does_not_crash_with_plain_list_input(self):
        """Non-BaseDataSource plain list/tuple input path (else branch)."""
        data = [("img1.png", "depth1.png"), ("img2.png", "depth2.png")]
        ds = CityscapesSegmentation(_FakeArgs(), root="/fake/root", data=data,
                                     split="train")
        assert ds.labels["train"] is None
        assert ds.images["train"] == ["img1.png", "img2.png"]
        assert ds.disparities["train"] == ["depth1.png", "depth2.png"]

    def test_init_raises_clear_error_on_none_data(self):
        """data=None should raise a clear ValueError, not crash deep inside."""
        with pytest.raises(ValueError, match="requires a 'data' argument"):
            CityscapesSegmentation(_FakeArgs(), root="/fake/root", data=None,
                                    split="train")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
