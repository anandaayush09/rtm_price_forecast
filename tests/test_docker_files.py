from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = [".project-local", "*.local.md", "data", "optuna", "literature"]


def test_private_folders_stay_out_of_the_build_context() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for name in PRIVATE:
        assert name in ignored


def test_the_image_copies_no_data() -> None:
    copied = [
        line.split()[1]
        for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
        if line.startswith("COPY ") and "--from" not in line
    ]
    assert copied
    for source in copied:
        assert not source.lstrip("./").startswith(tuple(PRIVATE))


def test_the_app_is_published_on_this_computer_only() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    ports = compose["services"]["app"]["ports"]
    assert ports
    assert all(str(port).startswith("127.0.0.1:") for port in ports)
    assert compose["services"]["app"]["volumes"] == ["./data:/work/data:ro"]


def test_the_release_workflow_publishes_the_image_and_attaches_no_results() -> None:
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert workflow[True]["push"]["tags"] == ["v*"]  # PyYAML reads the key `on` as True
    assert "workflow_dispatch" in workflow[True]
    assert "gh release create" in text and "docker push" in text
    # a manual run on a branch must build and check the image only: both publishing steps are
    # guarded to tags
    steps = workflow["jobs"]["release"]["steps"]
    guarded = [s["name"] for s in steps if s.get("if") == "github.ref_type == 'tag'"]
    assert guarded == ["Push the image", "Create the Release"]
    command = text.split("gh release create")[1]
    for private in (*PRIVATE, "results", "reports", "--attach", "gh release upload"):
        assert private not in command
