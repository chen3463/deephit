# deephit

Try to implement a deephit algorithm via pytorch based on https://github.com/chl8856/DeepHit/tree/master

```
+----------------------------+
|      DeepHit Overview     |
+----------------------------+
| GOAL: Predict WHEN an event happens (time-to-event)
|       Supports competing risks and nonlinear patterns
|
| INPUT: 
|   - Tabular features (e.g., age, balance, transaction history)
|   - Optional: embeddings, time series features
|
| ARCHITECTURE:
|   - Feedforward Neural Network
|       --> Shared layers
|       --> Output layer: [T x R] matrix
|          T = time intervals (e.g., days, weeks)
|          R = number of risk types (e.g., default, withdrawal)
|
| OUTPUT:
|   - A probability distribution P(t, r) over time & risk
|     ("Probability of event r at time t")
|
| LOSS FUNCTION:
|   - Log-likelihood loss --> accuracy of timing
|   - Ranking loss       --> preserves time ordering
|
| BENEFITS:
|   - Predicts full distribution, not just hazard or point estimate
|   - Captures nonlinear interactions + multiple event types
|
| USED FOR:
|   - Credit: Predict default time or early repayment
|   - Engagement: Predict when user takes financial action
|
| WHY NOT COX?:
|   - Cox assumes proportional hazards and linearity
|   - DeepHit = no restrictive assumptions
|
| RESULT:
|   - Timed outreach, better segmentation
|   - Higher engagement / reduced loss
+----------------------------+

```

