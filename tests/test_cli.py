from drawing_identifier.cli import build_parser, main


def test_parser_and_config(capsys):
    args = build_parser().parse_args(["analyze", "x.png", "--vlm", "ollama:qwen2.5vl:7b", "--agent-vlm", "verifier=claude", "--set", "shapes.fit_threshold=0.8"])
    from drawing_identifier.cli import _config

    cfg = _config(args)
    assert cfg.default_vlm == "ollama:qwen2.5vl:7b"
    assert cfg.agent_vlms["verifier"] == "claude"
    assert cfg.shapes.fit_threshold == 0.8
    assert main(["config"]) == 0
    assert "vlms:" in capsys.readouterr().out


def test_analyze_command(tmp_path, simple_drawing):
    import cv2

    p = tmp_path / "d.png"
    cv2.imwrite(str(p), simple_drawing)
    assert main(["analyze", str(p), "--no-vlm", "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "d.json").exists()
    assert (tmp_path / "out" / "d_overlay.png").exists()
    assert (tmp_path / "out" / "d.mmd").exists()
