"""
    Report borrowed word detection results by the method of cognate intruders
    using LingPy's LingRex add-on package which performs a dual analysis.
    Initially screening for cognate within language families, and then
    screening across language families. Language cognates crossing families
    indicate borrowings.
    Compare detected and known borrowed word results both overall and
    focusing specifically on donor languages as intruders.
    Report cross language concept and words, overall and donor focused
    detection performance, and give detail diagnostic report of borrowed
    word detection.

    John E. Miller, Apr 15, 2022
"""

import argparse

import sabor.arguments as argfns
import sabor.retrieve as retr
import sabor.evaluate as evaluate
import sabor.report as diag


def run(args):
    table = retr.get_ana_table(
        store=args.store,
        infile=args.infile,
        donors=args.donor,
        ref_template="AUTOBORID",
        ref_cnt=1,
        any_loan=args.any)

    ids_table = retr.get_ids_table_for(
        table,
        ref_id="AUTOBORID",
        loc_id="AUTOCOGID",
        donors=args.donor,
        any_loan=args.any)

    marked_forms, unmarked_forms = \
        diag.build_donor_forms_dict(ids_table, args.donor)

    if args.family:  # Filter on family.
        ids_table = [row for row in ids_table if row["FAMILY"] == args.family]

    retr.report_overall_ids(ids_table)

    if args.full:
        evaluate.report_id_counts(ids_table)
        evaluate.report_metrics(ids_table, args.donor)

    diag.report_detail_diagnostics(
        ids_table,
        donor_forms=marked_forms,
        unmarked_donor_forms=unmarked_forms,
        donors=args.donor,
        status=diag.PredStatus[args.status.upper()],
        output=args.output,
        series=args.series,
        module='lingrex')


def register(parser):
    argfns.register_rept_eval(parser)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
