# Verdict vb-04455a

**Overall: upheld.** Every number, threshold and direction checked reproduces against the model; nothing refuted or weakened, three domain flags.

Genome `9c06b29f-4ca4-4a6a-b9ed-5d04ee037b66`. Started 2026-11-10T23:25:20+13:00, finished 2026-11-10T23:28:04+13:00 (3 min).

## Refuted and weakened findings

| Id | Location | Level | Grade | Stated | What the model does | Check |
|---|---|---|---|---|---|---|

## Parts upheld

- ann_120: upheld (checked to L4).
- ann_114: upheld (checked to L3).
- ann_113: upheld (checked to L3).
- ann_117: upheld (checked to L3).
- ann_118: upheld (checked to L3).
- ann_115: upheld (checked to L3).
- ann_119: upheld (checked to L4).
- ann_116: upheld (checked to L1).

## Flags (domain sense, not refutations)

- `drafted.worth_checking[3]`: 1st-4th grade lowers predicted chance of low income to about 10% -- very little schooling giving a lower chance of low income conflicts with what is known; arises because education level is held fixed while the category changes
- `drafted.worth_checking[4]`: census sampling weight lowers chance -- a survey weight is not a property of the person
- `drafted.readdown.ann_119.hypothesis[0]`: being born in China raises the chance of low income to 85% -- the observed low-income rate among people born in China (70.8%) is below the rest (76.1%); the model adds a large +1.24 logit; worth a domain look
