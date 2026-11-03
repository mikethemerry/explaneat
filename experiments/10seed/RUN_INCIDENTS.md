adult-s1,adult-s3,adult-s8: crashed at startup 2026-11-03 (~11:20 NZDT; corrected from an earlier "~10:05" note) during data load, before any Experiment row was written: psycopg2.OperationalError "server closed the connection unexpectedly" while 30 runs started concurrently. Re-run with the same seeds. Logs in logs/crashed/.

Adult runs (all 10) paused with SIGSTOP at 2026-11-03 14:13:54 NZDT at Mike's request (meetings); processes suspended in place at generations ~43-48. Wall-clock training time for Adult must subtract the pause interval (resume time logged below).
