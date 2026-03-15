from pathlib import Path


class TestProjectStructure:

    def test_project_structure_exists(self):

        assert Path("src").exists()
        assert Path("config/config.yaml").exists()
        assert Path("api").exists()
        assert Path("models").exists()
        assert Path("data").exists()

    def test_source_modules_exist(self):

        assert Path("src/data").exists()
        assert Path("src/models").exists()
        assert Path("src/features").exists()
        assert Path("src/evaluation").exists()

    def test_config_files_exist(self):

        assert Path("config/config.yaml").exists()
        assert Path("requirements.txt").exists()