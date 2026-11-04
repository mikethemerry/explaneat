adult-s1,adult-s3,adult-s8: crashed at startup 2026-11-03 (~11:20 NZDT; corrected from an earlier "~10:05" note) during data load, before any Experiment row was written: psycopg2.OperationalError "server closed the connection unexpectedly" while 30 runs started concurrently. Re-run with the same seeds. Logs in logs/crashed/.

Adult runs (all 10) paused with SIGSTOP at 2026-11-03 14:13:54 NZDT at Mike's request (meetings); processes suspended in place at generations ~43-48. Wall-clock training time for Adult must subtract the pause interval (resume time logged below).
Adult runs resumed with SIGCONT at 2026-11-05 08:41:37 NZDT.
Throttle (Mike: CPU load): adult-s7, s9, s4, s5, s8 suspended with SIGSTOP at 2026-11-05 08:50:18 NZDT; max 5 Adult seeds run at once, suspended seeds resumed one by one as others finish (times below). Per-seed training wall clock must subtract each seed's suspended intervals.
adult-s7 resumed (SIGCONT) at 2026-11-05 09:54:29 NZDT.
adult-s9 resumed (SIGCONT) at 2026-11-05 10:04:42 NZDT.
adult-s4 resumed (SIGCONT) at 2026-11-05 10:05:12 NZDT.
adult-s5 resumed (SIGCONT) at 2026-11-05 10:07:12 NZDT.
adult-s8 resumed (SIGCONT) at 2026-11-05 10:09:13 NZDT.
