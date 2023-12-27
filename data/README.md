The following script will automatically download `ispd2005`, `ispd2015`, and `iccad2019` benchmarks in `./data/raw`. It also preprocesses `ispd2015` benchmark to fix some errors when routing them by Innovus®.
```bash
./download_data.sh
```

# Note of Fixing ISPD 2015
We provide a python scirpt `fix_ispd2015_route.py` to fix some errors in `ispd2015` benchmark. Thus, Innovus now can detailedly routed them.

## Limitations
**removeDefSNetVias**: Due to numerous DRVs caused by SNet Vias (spacing) after nanoroute routing, we have enabled `removeDefSNetVias` to remove these vias and address the above issue temporarily. It is likely that these vias are oversized, directly violating the spacing rule. While this adjustment has no significant impact on placement, it does result in open SNets. We sincerely encourage and appreciate contributions towards resolving this issue. Your contribution is highly valued and appreciated.