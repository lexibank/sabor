"""
    Report on borrowed words detected by method of cognate intruders
    using LingPy's pairwise method to identify similarities across languages.
    Compare detected and known borrowed word results both overall and
    focusing specifically on donor languages as intruders.
    Report cross language concept and words, overall and donor focused
    detection performance, and give detail diagnostic report of borrowed
    word detection.

    Adapted from report cluster.

    John E. Miller, Apr 19, 2022
"""

import argparse

import sabor.arguments as argfns
import sabor.retrieve as retr
import sabor.evaluate as evaluate
import sabor.report as diag


def run(args):
    parameters, thresholds = retr.get_parameters(
        store=args.store,
        infile=args.infile)

    table = retr.get_ana_table(
        store=args.store,
        infile=args.infile,
        donors=args.donor,
        ref_template="CROSS_ID_{}",
        ref_cnt=len(thresholds),
        any_loan=args.any)

    for idx, threshold in enumerate(thresholds):

        ids_table = retr.get_ids_table_for(
            table=table,
            ref_id="CROSS_ID_{}".format(idx),
            donors=args.donor,
            any_loan=args.any)

        marked_forms, unmarked_forms = \
            diag.build_donor_forms_dict(ids_table, args.donor)

        if args.family:  # Filter on family.
            ids_table = [row for row in ids_table
                         if row["FAMILY"] == args.family]

        retr.report_overall_ids(ids_table, threshold)

        if args.full:
            evaluate.report_id_counts(
                ids_table,
                threshold=threshold)

            evaluate.report_metrics(
                ids_table,
                donors=args.donor,
                threshold=threshold)

        diag.report_detail_diagnostics(
            ids_table,
            donor_forms=marked_forms,
            unmarked_donor_forms=unmarked_forms,
            donors=args.donor,
            status=diag.PredStatus[args.status.upper()],
            threshold=threshold,
            output=args.output,
            series=args.series,
            module='pairwise')


def register(parser):
    argfns.register_rept_eval(parser)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
