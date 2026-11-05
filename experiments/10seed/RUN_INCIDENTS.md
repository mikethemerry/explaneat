adult-s1,adult-s3,adult-s8: crashed at startup 2026-11-03 (~11:20 NZDT; corrected from an earlier "~10:05" note) during data load, before any Experiment row was written: psycopg2.OperationalError "server closed the connection unexpectedly" while 30 runs started concurrently. Re-run with the same seeds. Logs in logs/crashed/.

Adult runs (all 10) paused with SIGSTOP at 2026-11-03 14:13:54 NZDT at Mike's request (meetings); processes suspended in place at generations ~43-48. Wall-clock training time for Adult must subtract the pause interval (resume time logged below).
Adult runs resumed with SIGCONT at 2026-11-05 08:41:37 NZDT.
Throttle (Mike: CPU load): adult-s7, s9, s4, s5, s8 suspended with SIGSTOP at 2026-11-05 08:50:18 NZDT; max 5 Adult seeds run at once, suspended seeds resumed one by one as others finish (times below). Per-seed training wall clock must subtract each seed's suspended intervals.
adult-s7 resumed (SIGCONT) at 2026-11-05 09:54:29 NZDT.
adult-s9 resumed (SIGCONT) at 2026-11-05 10:04:42 NZDT.
adult-s4 resumed (SIGCONT) at 2026-11-05 10:05:12 NZDT.
adult-s5 resumed (SIGCONT) at 2026-11-05 10:07:12 NZDT.
adult-s8 resumed (SIGCONT) at 2026-11-05 10:09:13 NZDT.

churn100mask-s0..s9 crashed 2026-11-05 ~16:46 NZDT: Postgres.app refused connections ("Postgres.app failed to verify \"trust\" authentication; configure app permissions in Postgres.app settings"). s0-s3 died mid-run after the training loop caught the DB error and set the population fitness to -1000 (a silent fallback in DatabaseBackpropPopulation._evaluate_population); s4-s9 died at startup. Those 4 experiment rows were renamed churn100mask-crashed-s* and marked failed, and all 10 were re-run with the same seeds. No reported run hit this fallback (checked all logs and the DB for fitness = -1000).
Re-run also failed: Postgres.app refused connections again at ~16:56 NZDT. The processes kept evolving after the -1000 fallback and were killed; rows renamed churn100mask-crashed2-s0..s3 and marked failed. Masked runs are on hold until Postgres.app app permissions are fixed.
