"""
    Functions that register sabor commands.

    John E. Miller, Apr 15, 2022
"""

from sabor import report


def register_common(parser):
    parser.add_argument(
        "--language",
        nargs="*",
        type=str,
        default=['all'],
        help="'all' or list of languages"
    )
    parser.add_argument(
        "--donor",
        nargs="*",
        type=str,
        default=["Spanish", "Portuguese"],
        help='Donor language(s).',
    )
    parser.add_argument(
        "--store",
        type=str,
        default="store",
        help='Directory to store analysis wordlist.'
    )
    parser.add_argument(
        "--series",
        type=str,
        default='',
        help='Filename affix for analysis, evaluation, report files.'
    )
    parser.add_argument(
        "--any",
        action="store_true",
        help='Any loan from any donor.'
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Family to report or None if report over all families."
    )


def register_analysis(parser):
    parser.add_argument(
        "--method",
        type=str,
        choices=["sca", "lexstat", "edit-dist"],
        default="lexstat",
        help='Scoring method (default: "lexstat").',
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["sca", "asjp"],
        default="sca",
        help='Sound class model to transform tokens.'
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        choices=["upgma", "infomap"],
        default="upgma",
        help='Method to use in clustering.',
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2000,
        help='Number of runs for lexstat scorer.',
    )


def register_report(parser):
    parser.add_argument(
        "--infile",
        type=str,
        default="",
        help="Filename of analysis wordlist."
    )


def register_diagnostic(parser):
    parser.add_argument(
        "--full",
        action="store_true",
        help='Full report of evaluation, cross family cognates, detail diagnostics.'
    )
    parser.add_argument(
        "--status",
        type=str,
        default='ntn',
        choices=[f"{s.name.lower()}" for s in report.PredStatus],
        # choices=["tn", "tp", "fp", "fn", "f", "t", "ntn", "all"],
        help='Code for reporting words for status.',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help='Directory to write output.'
    )


def register_evaluate(parser):
    parser.add_argument(
        "--cross",
        action="store_true",
        help='Show shared cross family cognate counts.'
    )


def register_rept_eval(parser):
    register_common(parser)
    register_report(parser)
    register_diagnostic(parser)
    register_evaluate(parser)
