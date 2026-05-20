"""Apply DiffSBDD-style train/val/test split + size filtering. M3."""


def main():
    raise NotImplementedError(
        "M3: read split_by_name.pt; filter ligands by size (<50 atoms typically); "
        "write data/processed/crossdocked/{train,val,test}.pt"
    )


if __name__ == "__main__":
    main()
