import os


def add_line_to_csv(addr, cols, create=False):
    cols_str = ",".join(cols)
    with open(addr, 'wt' if create else 'at') as f:
        f.write(f"{cols_str}\n")
