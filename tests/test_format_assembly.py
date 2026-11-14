"""Formatting must preserve assembly literals, comments and macro arguments."""

import unittest

from tools.format_assembly import format_source


class AssemblyFormatTests(unittest.TestCase):
    def test_quoted_data_is_opaque(self):
        for literal in (r'"a,  b // \"quoted\" \0"', "' '", r'"https://example.com/a,b"'):
            with self.subTest(literal=literal):
                source = f"  value: .ascii {literal} // keep comment,  spacing\n"
                self.assertIn(literal + "  // keep comment,  spacing\n", format_source(source))

    def test_macro_names_and_arguments_remain_separate(self):
        source = ".macro check code\n find_expect #\\code\n adrp x9, _symbol@PAGE\n .endm\n"
        formatted = format_source(source)
        self.assertIn("find_expect #\\code\n", formatted)
        self.assertIn("_symbol@PAGE", formatted)

    def test_formatting_is_idempotent(self):
        source = "\n.text\nlabel:  .quad 0,1, 2\n  ldp x0,x1,[sp,#16] // comment\n ret\n\n"
        formatted = format_source(source)
        self.assertEqual(format_source(formatted), formatted)


if __name__ == "__main__":
    unittest.main()
