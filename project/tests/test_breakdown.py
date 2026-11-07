import json

from deadline_agent.breakdown import run_breakdown


def _write(tmp_path, events):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events))
    return p


def test_breakdown_splits_summaries_at_last_return_and_sums_to_completion(tmp_path):
    ev = [dict(t=0.0, event="task_start"), dict(t=2.0, event="tool_launch"), dict(t=2.0, event="tool_launch"),
          dict(t=3.0, event="tool_return"),
          dict(t=3.5, event="consume_start", inv=1), dict(t=6.0, event="consume_end", inv=1),   # hidden
          dict(t=5.0, event="tool_return"),                                                      # last return
          dict(t=6.0, event="consume_start", inv=2), dict(t=6.5, event="consume_end", inv=2),   # straddles? no: after
          dict(t=7.0, event="final_start", inv=3), dict(t=8.0, event="final_end", inv=3),
          dict(t=8.0, event="task_end", valid=True, completion_s=8.0)]
    b = run_breakdown(_write(tmp_path, ev))
    assert b["planning"] == 2.0 and b["tool_wait"] == 3.0
    assert b["hidden_summarize"] == 1.5 and b["post_summarize"] == 1.0 + 0.5
    assert abs(sum(b[k] for k in ("planning", "tool_wait", "post_summarize", "post_preload", "final")) - 8.0) < 1e-9


def test_missed_run_is_charged_the_deadline_with_cut_off_work_clipped(tmp_path):
    ev = [dict(t=0.0, event="task_start", deadline_s=10.0), dict(t=1.0, event="tool_launch"),
          dict(t=1.0, event="tool_launch"), dict(t=2.0, event="tool_return"),
          dict(t=2.5, event="consume_start", inv=1), dict(t=5.0, event="consume_end", inv=1),
          dict(t=7.0, event="tool_return"),
          dict(t=8.0, event="consume_start", inv=2),  # cut off by the deadline: no end event
          dict(t=10.0, event="deadline_miss"), dict(t=10.1, event="task_end", valid=False)]
    b = run_breakdown(_write(tmp_path, ev))
    assert b["completion"] == 10.0 and not b["valid"]
    assert b["planning"] == 1.0 and b["tool_wait"] == 6.0
    assert b["hidden_summarize"] == 2.5 and b["post_summarize"] == 2.0 and b["final"] == 1.0


def test_tools_still_running_at_a_miss_count_as_tool_wait(tmp_path):
    ev = [dict(t=0.0, event="task_start", deadline_s=10.0), dict(t=1.0, event="tool_launch"),
          dict(t=1.0, event="tool_launch"), dict(t=2.0, event="tool_return"),
          dict(t=10.1, event="task_end", valid=False)]
    b = run_breakdown(_write(tmp_path, ev))
    assert b["tool_wait"] == 9.0 and b["final"] == 0.0


def test_serving_metrics_split_gpu_time_around_tool_waits_and_count_cache_reuse(tmp_path):
    from deadline_agent.breakdown import run_serving_metrics

    ev = [dict(t=0.0, event="task_start", deadline_s=20.0),
          dict(t=0.0, event="plan_start", inv="m1"), dict(t=2.0, event="plan_end", inv="m1", prompt_tokens=50,
                                                          cached_tokens=0, completion_tokens=20),
          dict(t=2.0, event="tool_launch"), dict(t=2.0, event="tool_launch"), dict(t=3.0, event="tool_return"),
          dict(t=3.0, event="preload_start", inv="m2"), dict(t=3.5, event="preload_end", inv="m2", prompt_tokens=600,
                                                              cached_tokens=0, completion_tokens=1),
          dict(t=8.0, event="tool_return"),
          dict(t=8.0, event="final_start", inv="m3"), dict(t=9.0, event="final_end", inv="m3", prompt_tokens=1200,
                                                            cached_tokens=592, completion_tokens=10),
          dict(t=9.0, event="task_end", valid=True, completion_s=9.0)]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in ev))
    m = run_serving_metrics(p)
    assert m["gpu_busy_s"] == 3.5 and m["gpu_busy_during_tools_s"] == 0.5 and m["gpu_idle_during_tools_s"] == 5.5
    assert m["model_s_after_last_return"] == 1.0 and m["model_requests"] == 3
    assert m["prefill_tokens_reused"] == 592 and m["prefill_tokens_computed"] == 50 + 600 + 1200 - 592
