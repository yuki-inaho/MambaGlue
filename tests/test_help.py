import unittest

from mambaglue.help import console_commands, module_summary


class HelpTest(unittest.TestCase):
    def test_lists_known_console_commands_with_summaries(self):
        commands = dict(console_commands())

        for name in (
            "mambaglue-camera",
            "mambaglue-demo",
            "mambaglue-help",
            "mambaglue-smoke",
        ):
            self.assertIn(name, commands)
            self.assertTrue(commands[name])

    def test_missing_module_has_no_summary(self):
        self.assertEqual(module_summary("mambaglue.missing"), "")


if __name__ == "__main__":
    unittest.main()
