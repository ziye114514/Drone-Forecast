"""Use existing current-NN recovery only when recent service fits this slot."""
from forecast_runtime import nn_reply_wait_budget


def decision(now,deadline,period,recent_costs,recent_services,pending_running=False):
    budget=nn_reply_wait_budget(now,deadline,period,recent_costs)
    expected=max(recent_services,default=.5*period)
    allowed=bool(not pending_running and budget>=expected and budget>0.)
    return allowed,dict(allowed=allowed,available_s=budget,expected_NN_service_s=expected,
        pending_worker_still_running=pending_running,
        reason='current_NN_service_fits_estimated_slot' if allowed else 'retain_asynchronous_control_until_slot_has_room',
        hard_latency_bound_claimed=False,position_error_or_video_identity_used=False)
