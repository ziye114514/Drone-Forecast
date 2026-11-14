"""Source-time acquisition cadence, separate from recent-loss recovery.

    A never-seen/long-absent target uses the ordinary detector stride.
    The current KLT tracker still runs at every source cutoff; a recently
    detected target that just lost tracking retains immediate retry.
    No predicted box is returned as a current image observation.
"""


def acquisition_due(t,tracked,last_success,last_request,hz,stride,original_due,source_start=None):
    cold=tracked is None and (last_success is None or t-last_success>.3+1e-8)
    initial_burst=bool(source_start is not None and t-source_start<=.3+1e-8)
    info=dict(cold_search=cold,deferred=False,ordinary_search_interval_s=stride/hz,
        recent_loss_recovery_delayed=False,new_detector_models=0,
        last_current_image_observation_t=last_success,initial_acquisition_burst=initial_burst,
        initial_burst_duration_s=.3,source_start_t=source_start)
    if not cold or initial_burst or last_request is None:return original_due,info
    due=bool(t-last_request>=stride/hz-.5/hz)
    info.update(deferred=bool(original_due and not due),time_since_last_request_s=t-last_request,
        current_search_only_no_fake_detection=True)
    return due,info
