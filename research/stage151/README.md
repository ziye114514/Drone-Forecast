# Stage 151: independent event motion

Default forecasting remains stage 144. This experiment reuses the installed
Prophesee SparseOpticalFlowAlgorithm and the pinned FilterPy state update.
It adds a current RAW velocity observation, rather than another smoothed
derivative of detector boxes. Missing rotor states never become measurements.

Development measurements must distinguish cached-box observation probes from
RAW-only end-to-end runs. Freeze the rule before reading forecast scores.
