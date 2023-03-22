import unittest


def setUpModule():
    unittest.defaultTestLoader.parallelism = 1


class TestClearMLLogger(unittest.TestCase):
    def test_task_init(self):
        from clearml import Task

        Task.set_offline(offline_mode=True)
        from parlai.core.logs import ClearMLLogger

        opt = {}
        try:
            self.clearml_callback = ClearMLLogger(opt)
        except Exception as exc:
            self.clearml_callback = None
            self.fail(exc)

        self.assertEqual(Task.current_task()._project_name[1], "ParlAI")


if __name__ == '__main__':
    unittest.main()
