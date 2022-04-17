"""
    Evaluation with precision, recall, F1 Score and accuracy for
    borrowed words detected by method of cognate intruders
    using LingPy's cluster methods to identify similarities across languages.
    Compare detected and known borrowed word results both overall and
    focusing specifically on donor languages as intruders.

    John E. Miller, Apr 14, 2022
"""

import argparse

import sabor.arguments as argfns
import sabor.evaluate as evaluate
import sabor.retrieve as retr


# ==================
# Perform evaluation
# ==================
def run(args):
    parameters, thresholds = retr.get_parameters(
        store=args.store,
        infile=args.infile)

    table = retr.get_ana_table(
        store=args.store,
        infile=args.infile,
        donors=args.donor,
        ref_template="SCALLID_{}",
        ref_cnt=len(thresholds),
        any_loan=args.any)

    for idx, threshold in enumerate(thresholds):

        ids_table = retr.get_ids_table_for(
            table=table,
            ref_id="SCALLID_{}".format(idx),
            donors=args.donor,
            any_loan=args.any)

        if args.family:  # Filter on family.
            ids_table = [row for row in ids_table
                         if row["FAMILY"] == args.family]

        if args.cross:
            evaluate.report_id_counts(
                ids_table,
                threshold=threshold)

        evaluate.report_metrics(
            ids_table,
            donors=args.donor,
            threshold=threshold)


def register(parser):
    argfns.register_common(parser)
    argfns.register_evaluate(parser)
    argfns.register_report(parser)


def get_exp_result(store, infile, family=None,
                   donors=None, index=0, any_loan=False):
    # Application interface to perform run based on invocation
    # by another application. Purpose is to automate experimentation.

    parameters, thresholds = retr.get_parameters(store=store, infile=infile)
    table = retr.get_ana_table(
        store=store, infile=infile, donors=donors,
        ref_template="SCALLID_{}", ref_cnt=len(thresholds),
        any_loan=any_loan)

    ids_table = retr.get_ids_table_for(
        table=table, ref_id="SCALLID_{}".format(index),
        donors=donors, any_loan=any_loan)

    if family:  # Filter on family.
        ids_table = [row for row in table if row["FAMILY"] == family]

    metrics = evaluate.calculate_overall_metrics(ids_table)
    return [round(num, 3) for num in metrics]


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()
    register(parser_)
    run(parser_.parse_args())
