"""
    Analyze language cognate and borrowing using LingRex and
    report quality metrics [precision, recall, F score, accuracy].

    John E. Miller, Apr 9, 2022
"""

import argparse

import sabor.arguments as argfns
import sabor.evaluate as evaluate
import sabor.retrieve as retr


# ==================
# Perform evaluation
# ==================
def run(args):
    table = retr.get_ana_table(
        store=args.store,
        infile=args.infile,
        donors=args.donor,
        ref_template="AUTOBORID",
        ref_cnt=1,
        any_loan=args.any)

    ids_table = retr.get_ids_table_for(
        table=table,
        ref_id="AUTOBORID",
        donors=args.donor,
        any_loan=args.any)

    if args.family:  # Filter on family.
        ids_table = [row for row in ids_table if row["FAMILY"] == args.family]

    if args.cross:
        evaluate.report_id_counts(
            ids_table)

    evaluate.report_metrics(
        ids_table,
        donors=args.donor)


def register(parser):
    argfns.register_common(parser)
    argfns.register_report(parser)
    argfns.register_evaluate(parser)


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
