import json

import numpy as np


from drawing_identifier.synthetic import DETECTOR_CLASSES, SynthConfig, generate_dataset, generate_sample
from drawing_identifier.vision.geometry import as_contour, fraction_inside


def test_sample_is_deterministic():
    a, ga = generate_sample(3)
    b, gb = generate_sample(3)
    assert np.array_equal(a, b)
    assert ga.model_dump() == gb.model_dump()


def test_ground_truth_is_consistent():
    for seed in range(8):
        img, g = generate_sample(seed, SynthConfig(p_rotate90=0.5))
        h, w = img.shape[:2]
        assert g.image.width == w and g.image.height == h
        for s in g.shapes:
            x0, y0, x1, y1 = s.bbox.xyxy
            assert -2 <= x0 and x1 <= w + 2 and -2 <= y0 and y1 <= h + 2
            if s.parent_id:
                parent = g.shape(s.parent_id)
                assert fraction_inside(s.polygon, as_contour(parent.polygon)) > 0.95
        for c in g.connections:
            for ep in c.endpoints:
                if ep.text_id:
                    assert g.text(ep.text_id) is not None
                if ep.shape_id:
                    assert g.shape(ep.shape_id) is not None
        assert all(t.text for t in g.texts)


def test_dataset_export(tmp_path):
    s = generate_dataset(tmp_path, 4, seed=1, val_fraction=0.25, workers=1, progress=False)
    assert s["n_train"] == 3 and s["n_val"] == 1
    assert (tmp_path / "data.yaml").exists()
    labels = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(labels) == 3
    for lf in labels:
        for line in lf.read_text().splitlines():
            parts = line.split()
            assert 0 <= int(parts[0]) < len(DETECTOR_CLASSES)
            vals = [float(v) for v in parts[1:]]
            assert len(vals) >= 6 and all(0 <= v <= 1 for v in vals)
    coco = json.loads((tmp_path / "coco_train.json").read_text())
    assert len(coco["images"]) == 3 and coco["annotations"]
    rec = json.loads((tmp_path / "vlm_val.jsonl").read_text().splitlines()[0])
    target = json.loads(rec["messages"][-1]["content"])
    assert {"shapes", "texts", "connections"} <= set(target)
    assert (tmp_path / "ocr" / "train.tsv").read_text().count("\n") > 1

